//! The compiler: a program's semantics, partially evaluated for its tree,
//! written as LLVM IR (ir.zig) and JIT-compiled by zgram's LLVM.
//!
//! Each semantic is run at compile time with its node as a known value:
//! whatever depends only on the tree (fields, kinds, texts, `node.op ==
//! "+"`, loops over a node's children) is decided here, and what depends on
//! the program's data becomes code. `rt.eval(child)` inlines the child's
//! semantic; `rt.load` / `rt.store` are the variable's slot; `raise
//! rt.Return(v)` is a jump. A language function becomes an LLVM function,
//! the top level another.
//!
//! Values known here are `SVal`s; values only known at run time are
//! `Dyn`: a tag and 64 bits in SSA registers, with the shape known of them
//! (an int, a str...), which lets int and float arithmetic be inlined.
//! Dynamic values are owned references: whoever ends up with one keeps it
//! or drops it.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const front = @import("front.zig");
const ir = @import("ir.zig");
const program_mod = @import("program.zig");
const grammar_mod = @import("grammar.zig");
const helpers = @import("helpers.zig");

const Allocator = std.mem.Allocator;
const NONE = program_mod.NONE;
const FunctionSpec = program_mod.FunctionSpec;

/// What the compiler needs of the language
pub const LangView = struct {
    grammar: *const grammar_mod.Grammar,
    eval_of: []const ?*PyObject,
    exec_of: []const ?*PyObject,
    functions: []const ?FunctionSpec,
    read: *const std.AutoHashMapUnmanaged(*PyObject, *front.Function),
    hosts: *PyObject,
    /// The State object of the program (its tree, for scalar field values)
    tree: *PyObject,
    analysis: ?*PyObject,
};

/// A compile failure: what and where (a source line of a semantic, or a
/// node of the program)
pub const Failure = struct {
    message: std.ArrayListUnmanaged(u8) = .empty,
};

pub const Error = error{ OutOfMemory, Python, Unsupported };

// ======================================================================
// Values at compile time
// ======================================================================

pub const Shape = enum { any, none, bool, int, float, str, list, tuple, dict, record, function, node };

/// A value only known at run time: operands (SSA names or constants)
pub const Dyn = struct {
    tag: []const u8,
    bits: []const u8,
    shape: Shape,

    fn heapish(self: Dyn) bool {
        return switch (self.shape) {
            .none, .bool, .int, .float, .node => false,
            else => true,
        };
    }
};

pub const RtMethod = enum { eval, exec, loop, load, store, function, call, @"error", kind, text, span, scope, symbol, type_of, node_at, Return, Break, Continue };

/// A list known at compile time (its items may be dynamic): mutable, with
/// identity (aliases see changes)
pub const SList = struct {
    items: std.ArrayListUnmanaged(SVal) = .empty,
};

pub const SVal = union(enum) {
    none,
    bool: bool,
    int: i64,
    float: f64,
    str: []const u8,
    node: u32,
    list: *SList,
    tuple: []const SVal,
    rt,
    rt_method: RtMethod,
    /// A Python object known at compile time (a module-level name)
    py: *PyObject,
    /// raise rt.Return(v) / rt.Break() / rt.Continue(): what was raised
    control: struct { kind: RtMethod, value: ?*const SVal },
    /// `obj.name` for a method call
    method: struct { recv: *const SVal, name: []const u8 },
    dyn: Dyn,

    fn isStatic(self: SVal) bool {
        return self != .dyn;
    }
};

// ======================================================================
// The compiler
// ======================================================================

/// Where a language function keeps its variables
const Layout = struct {
    /// The symbols living in it, by slot
    syms: std.ArrayListUnmanaged(u32) = .empty,
    /// In a heap frame (functions are made in it), or on the stack
    heap: bool = false,
};

pub const Compiler = struct {
    a: Allocator,
    data: *program_mod.Data,
    lang: LangView,
    m: ir.Module,
    failure: *Failure,
    /// Python objects the code refers to (owned references: the compiled
    /// program keeps them)
    objects: std.ArrayListUnmanaged(*PyObject) = .empty,
    /// Per function node (NONE: the top level): its variables' layout
    layouts: std.AutoHashMapUnmanaged(u32, *Layout) = .empty,
    /// Slot of each symbol in its layout
    slot_of: std.AutoHashMapUnmanaged(u32, u32) = .empty,
    /// Language functions compiled or to compile
    compiled_fns: std.AutoHashMapUnmanaged(u32, void) = .empty,
    queue: std.ArrayListUnmanaged(u32) = .empty,
    /// Helper functions (module-level Python functions semantics call),
    /// read once
    helpers_read: std.AutoHashMapUnmanaged(*PyObject, *front.Function) = .empty,

    pub fn init(a: Allocator, data: *program_mod.Data, lang: LangView, prefix: []const u8, failure: *Failure) Compiler {
        return .{ .a = a, .data = data, .lang = lang, .m = ir.Module.init(a, prefix), .failure = failure };
    }

    /// Release what isn't in the arena (the helpers read here).
    pub fn deinit(self: *Compiler, gpa: Allocator) void {
        var it = self.helpers_read.valueIterator();
        while (it.next()) |f| f.*.destroy(gpa);
    }

    fn unsupported(self: *Compiler, comptime fmt: []const u8, args: anytype) Error {
        self.failure.message.clearRetainingCapacity();
        self.failure.message.print(self.a, fmt, args) catch {};
        return error.Unsupported;
    }

    /// Unsupported at a semantic's line.
    fn unsupportedAt(self: *Compiler, f: *const front.Function, pos: front.Pos, comptime fmt: []const u8, args: anytype) Error {
        self.failure.message.clearRetainingCapacity();
        self.failure.message.print(self.a, "{s}:{d}:{d}: in {s}(): ", .{ f.file, pos.line, pos.col, f.name }) catch {};
        self.failure.message.print(self.a, fmt, args) catch {};
        return error.Unsupported;
    }

    /// The whole program: the top level as `@<prefix>_main`, each language
    /// function reached as `@<prefix>_f<node>`.
    pub fn compileProgram(self: *Compiler) Error!void {
        try self.declareRuntime();
        try self.computeLayouts();
        try self.genFunction(NONE);
        while (self.queue.pop()) |fnode| try self.genFunction(fnode);
    }

    fn declareRuntime(self: *Compiler) !void {
        const decls = [_][2][]const u8{
            .{ "zr_incref", "void @zr_incref(i64, i64)" },
            .{ "zr_decref", "void @zr_decref(i64, i64)" },
            .{ "zr_fail", "i1 @zr_fail(ptr, i32, ptr)" },
            .{ "zr_unset", "i1 @zr_unset(ptr, i32, ptr)" },
            .{ "zr_overflow", "i1 @zr_overflow(ptr, i32)" },
            .{ "zr_binary", "i1 @zr_binary(ptr, i32, i32, i64, i64, i64, i64, ptr)" },
            .{ "zr_compare", "i1 @zr_compare(ptr, i32, i32, i64, i64, i64, i64, ptr)" },
            .{ "zr_unary", "i1 @zr_unary(ptr, i32, i32, i64, i64, ptr)" },
            .{ "zr_truthy", "i1 @zr_truthy(i64, i64)" },
            .{ "zr_function", "i1 @zr_function(ptr, ptr, ptr, i32, ptr, i64, ptr)" },
            .{ "zr_call", "i1 @zr_call(ptr, i32, i64, i64, ptr, i64, ptr, ptr)" },
            .{ "zr_object", "void @zr_object(ptr, i64, ptr)" },
            .{ "zr_frame_new", "ptr @zr_frame_new(ptr, i64)" },
            .{ "zr_frame_release", "void @zr_frame_release(ptr)" },
        };
        for (decls) |d| try self.m.declare(d[0], d[1]);
        try self.m.declare("llvm.sadd.with.overflow.i64", "{ i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)");
        try self.m.declare("llvm.ssub.with.overflow.i64", "{ i64, i1 } @llvm.ssub.with.overflow.i64(i64, i64)");
        try self.m.declare("llvm.smul.with.overflow.i64", "{ i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)");
    }

    /// Each symbol's slot, in the function it lives in; which functions
    /// keep their variables in a heap frame (those functions are made in).
    fn computeLayouts(self: *Compiler) !void {
        const d = self.data;
        for (d.syms, 0..) |s, i| {
            if (s.builtin) continue;
            const home = d.homeOf(@intCast(i));
            const layout = try self.layoutOf(home);
            try self.slot_of.put(self.a, @intCast(i), @intCast(layout.syms.items.len));
            try layout.syms.append(self.a, @intCast(i));
        }
        // A function with a function inside it: heap (the inner one's env)
        for (d.nodes, 0..) |n, i| {
            if (!self.isFunctionNode(@intCast(i))) continue;
            _ = n;
            const outer = self.enclosingFunction(@intCast(i));
            const layout = try self.layoutOf(outer);
            layout.heap = true;
        }
        (try self.layoutOf(NONE)).heap = true;
    }

    fn layoutOf(self: *Compiler, fnode: u32) !*Layout {
        const entry = try self.layouts.getOrPut(self.a, fnode);
        if (!entry.found_existing) {
            entry.value_ptr.* = try self.a.create(Layout);
            entry.value_ptr.*.* = .{};
        }
        return entry.value_ptr.*;
    }

    pub fn isFunctionNode(self: *const Compiler, idx: u32) bool {
        const rid = self.data.rule(idx);
        return rid < self.lang.functions.len and self.lang.functions[rid] != null;
    }

    /// The function node around a node (NONE: the top level).
    pub fn enclosingFunction(self: *const Compiler, idx: u32) u32 {
        var n = self.data.parents[idx];
        while (n != NONE) : (n = self.data.parents[n]) {
            if (self.isFunctionNode(n)) return n;
        }
        return NONE;
    }

    fn fnName(self: *Compiler, fnode: u32) ![]const u8 {
        if (fnode == NONE) return std.fmt.allocPrint(self.a, "@{s}_main", .{self.m.prefix});
        return std.fmt.allocPrint(self.a, "@{s}_f{d}", .{ self.m.prefix, fnode });
    }

    /// The code of a language function, compiled (queued if it isn't yet).
    fn functionCode(self: *Compiler, fnode: u32) ![]const u8 {
        if (!self.compiled_fns.contains(fnode)) {
            try self.compiled_fns.put(self.a, fnode, {});
            try self.queue.append(self.a, fnode);
        }
        return self.fnName(fnode);
    }

    /// A Python object the code refers to: its index (a new reference kept).
    fn objectIndex(self: *Compiler, o: *PyObject) !usize {
        for (self.objects.items, 0..) |x, i| if (x == o) return i;
        py.Py_IncRef(o);
        try self.objects.append(self.a, o);
        return self.objects.items.len - 1;
    }

    /// Generate one function: the top level (NONE) or a language function.
    fn genFunction(self: *Compiler, fnode: u32) Error!void {
        var g = Gen{ .c = self, .f = ir.Function.init(&self.m), .fnode = fnode, .layout = try self.layoutOf(fnode) };
        try g.prologue();
        if (fnode == NONE) {
            try g.hoist(NONE);
            try g.execNode(0);
        } else {
            const spec = self.specOf(fnode).?;
            try g.hoist(fnode);
            try g.bindParams(fnode, spec);
            // the body: a field, or the child of a rule
            const body = if (spec.body != 0)
                try g.fieldValue(fnode, spec.body)
            else if (program_mod.childOfRule(self.data, fnode, spec.body_rule)) |b|
                try g.nodeValue(b)
            else
                SVal.none;
            try g.execValue(body);
        }
        try g.epilogue();
        const name = try self.fnName(fnode);
        const sig = if (fnode == NONE)
            try std.fmt.allocPrint(self.a, "i1 {s}(ptr %ctx, ptr %globals)", .{name})
        else
            try std.fmt.allocPrint(self.a, "internal i1 {s}(ptr %ctx, ptr %env, ptr %args, i64 %nargs, ptr %recv, ptr %result)", .{name});
        try g.f.finish(sig);
    }

    pub fn specOf(self: *const Compiler, idx: u32) ?FunctionSpec {
        const rid = self.data.rule(idx);
        return if (rid < self.lang.functions.len) self.lang.functions[rid] else null;
    }
};

// ======================================================================
// Generating a function
// ======================================================================

/// A semantic running inline: its function, node, locals
const Inst = struct {
    func: *const front.Function,
    /// The node it runs for (errors are reported at it)
    node: u32,
    locals: []Local,
    /// Its result: known (returned outside run-time control flow) or in a
    /// slot (returned from inside it)
    result: ?SVal = null,
    result_slot: ?[]const u8 = null,
    exit_label: []const u8,
    /// Run-time control flow depth within it (if, while...)
    dyn_depth: u32 = 0,
    /// Python loop targets of the semantic itself (break / continue)
    loops: std.ArrayListUnmanaged(struct { brk: []const u8, cont: []const u8 }) = .empty,
    /// It returned (outside run-time control flow): the rest is dead
    done: bool = false,
};

const Local = union(enum) {
    unset,
    static: SVal,
    /// In a stack slot (an alloca of {i64, i64}), with the shape last stored
    slot: struct { ptr: []const u8, shape: Shape },
};

/// An rt.loop's targets, and how many semantics ran when it started (those
/// above are left by a Break / Continue)
const LoopTarget = struct { brk: []const u8, cont: []const u8, depth: usize };

const Gen = struct {
    c: *Compiler,
    f: ir.Function,
    fnode: u32,
    layout: *Layout,
    /// The frame pointer (a heap frame) or null (stack slots)
    frame: ?[]const u8 = null,
    /// Stack slots of the variables, by layout slot (stack layouts)
    var_slots: std.ArrayListUnmanaged([]const u8) = .empty,
    /// Scratch space for helpers' results
    out: []const u8 = "",
    /// Where errors go (return false), and returns (rt.Return)
    err_label: []const u8 = "",
    ret_label: []const u8 = "",
    /// rt.loop targets, innermost last
    loops: std.ArrayListUnmanaged(LoopTarget) = .empty,
    /// Semantics running inline, innermost last
    insts: std.ArrayListUnmanaged(*Inst) = .empty,

    fn a(self: *Gen) Allocator {
        return self.c.a;
    }

    fn prologue(self: *Gen) Error!void {
        const f = &self.f;
        self.out = try f.alloca("{ i64, i64 }");
        self.err_label = try f.label("error");
        self.ret_label = try f.label("return");
        const n = self.layout.syms.items.len;
        if (self.fnode == NONE) {
            self.frame = "%globals";
        } else if (self.layout.heap) {
            self.frame = try f.value("call ptr @zr_frame_new(ptr %env, i64 {d})", .{n});
        } else {
            for (0..n) |_| {
                const slot = try f.alloca("{ i64, i64 }");
                try self.var_slots.append(self.a(), slot);
            }
            // (unset until assigned; the entry block runs once)
            for (self.var_slots.items) |slot| {
                try f.entry.print(self.a(), "  store i64 {d}, ptr {s}, align 8\n", .{ helpers.UNSET, slot });
            }
        }
        if (self.fnode != NONE) {
            // (None until a Return says otherwise)
            try f.emit("store {{ i64, i64 }} {{ i64 0, i64 0 }}, ptr %result, align 8", .{});
        }
    }

    fn epilogue(self: *Gen) Error!void {
        const f = &self.f;
        try f.br(self.ret_label);
        try f.block(self.ret_label);
        try self.releaseFrame();
        try f.ret("i1 true", .{});
        try f.block(self.err_label);
        try self.releaseFrame();
        try f.ret("i1 false", .{});
    }

    /// Drop the function's variables (its heap frame, or its stack slots'
    /// values).
    fn releaseFrame(self: *Gen) Error!void {
        const f = &self.f;
        if (self.fnode == NONE) return;
        if (self.frame) |fr| {
            try f.emit("call void @zr_frame_release(ptr {s})", .{fr});
            return;
        }
        for (self.var_slots.items) |slot| {
            const tag = try f.value("load i64, ptr {s}, align 8", .{slot});
            const bp = try f.value("getelementptr inbounds {{ i64, i64 }}, ptr {s}, i32 0, i32 1", .{slot});
            const bits = try f.value("load i64, ptr {s}, align 8", .{bp});
            // (an unset slot holds no reference: zr_decref ignores its tag)
            try f.emit("call void @zr_decref(i64 {s}, i64 {s})", .{ tag, bits });
        }
    }

    // ------------------------------------------------------------------
    // Errors
    // ------------------------------------------------------------------

    /// After a helper that can fail: on false, to the error exit.
    fn check(self: *Gen, ok: []const u8) Error!void {
        const cont = try self.f.label("ok");
        try self.f.condBr(ok, cont, self.err_label);
        try self.f.block(cont);
    }

    /// A runtime error at a node with a fixed message: to the error exit.
    fn failAt(self: *Gen, node: u32, message: []const u8) Error!void {
        const s = try self.c.m.string(message);
        _ = try self.f.value("call i1 @zr_fail(ptr %ctx, i32 {d}, ptr {s})", .{ node, s });
        try self.f.br(self.err_label);
        // (the code after it is unreachable: a fresh block keeps it valid)
        try self.f.block(try self.f.label("dead"));
    }

    // ------------------------------------------------------------------
    // Values: making them dynamic, dropping them
    // ------------------------------------------------------------------

    fn dyn(tag: []const u8, bits: []const u8, shape: Shape) SVal {
        return .{ .dyn = .{ .tag = tag, .bits = bits, .shape = shape } };
    }

    /// A value's run-time form (an owned reference).
    fn materialize(self: *Gen, v: SVal, at: u32) Error!Dyn {
        const f = &self.f;
        return switch (v) {
            .dyn => |d| d,
            .none => .{ .tag = "0", .bits = "0", .shape = .none },
            .bool => |b| .{ .tag = "1", .bits = if (b) "1" else "0", .shape = .bool },
            .int => |n| .{ .tag = "2", .bits = try std.fmt.allocPrint(self.a(), "{d}", .{n}), .shape = .int },
            .float => |x| .{ .tag = "3", .bits = try std.fmt.allocPrint(self.a(), "{d}", .{@as(i64, @bitCast(x))}), .shape = .float },
            .str => |s| blk: {
                const g = try self.c.m.string(s);
                break :blk .{ .tag = "4", .bits = try f.value("ptrtoint ptr {s} to i64", .{g}), .shape = .str };
            },
            .node => |n| .{ .tag = "11", .bits = try std.fmt.allocPrint(self.a(), "{d}", .{n}), .shape = .node },
            .py => |o| blk: {
                const idx = try self.c.objectIndex(o);
                try f.emit("call void @zr_object(ptr %ctx, i64 {d}, ptr {s})", .{ idx, self.out });
                break :blk try self.loadOut(.any);
            },
            else => self.c.unsupported("a {s} can't be kept in a variable or passed as a value (node {d})", .{ @tagName(v), at }),
        };
    }

    /// The helpers' result slot, read (an owned reference).
    fn loadOut(self: *Gen, shape: Shape) Error!Dyn {
        const f = &self.f;
        const tag = try f.value("load i64, ptr {s}, align 8", .{self.out});
        const bp = try f.value("getelementptr inbounds {{ i64, i64 }}, ptr {s}, i32 0, i32 1", .{self.out});
        const bits = try f.value("load i64, ptr {s}, align 8", .{bp});
        return .{ .tag = tag, .bits = bits, .shape = shape };
    }

    /// Give up a dynamic value (decref it).
    fn drop(self: *Gen, v: SVal) Error!void {
        switch (v) {
            .dyn => |d| if (d.heapish()) try self.f.emit("call void @zr_decref(i64 {s}, i64 {s})", .{ d.tag, d.bits }),
            else => {},
        }
    }

    fn increfDyn(self: *Gen, d: Dyn) Error!void {
        if (d.heapish()) try self.f.emit("call void @zr_incref(i64 {s}, i64 {s})", .{ d.tag, d.bits });
    }

    /// A slot ({i64, i64}) of the stack: its pointer.
    fn storeSlot(self: *Gen, slot: []const u8, d: Dyn) Error!void {
        const f = &self.f;
        try f.emit("store i64 {s}, ptr {s}, align 8", .{ d.tag, slot });
        const bp = try f.value("getelementptr inbounds {{ i64, i64 }}, ptr {s}, i32 0, i32 1", .{slot});
        try f.emit("store i64 {s}, ptr {s}, align 8", .{ d.bits, bp });
    }

    fn loadSlot(self: *Gen, slot: []const u8, shape: Shape) Error!Dyn {
        const f = &self.f;
        const tag = try f.value("load i64, ptr {s}, align 8", .{slot});
        const bp = try f.value("getelementptr inbounds {{ i64, i64 }}, ptr {s}, i32 0, i32 1", .{slot});
        const bits = try f.value("load i64, ptr {s}, align 8", .{bp});
        return .{ .tag = tag, .bits = bits, .shape = shape };
    }

    // ------------------------------------------------------------------
    // Truth
    // ------------------------------------------------------------------

    const Truth = union(enum) { known: bool, dyn: []const u8 };

    fn truth(self: *Gen, v: SVal, at: u32) Error!Truth {
        switch (v) {
            .none => return .{ .known = false },
            .bool => |b| return .{ .known = b },
            .int => |n| return .{ .known = n != 0 },
            .float => |x| return .{ .known = x != 0 },
            .str => |s| return .{ .known = s.len != 0 },
            .list => |l| return .{ .known = l.items.items.len != 0 },
            .tuple => |t| return .{ .known = t.len != 0 },
            .node, .rt, .rt_method, .method, .control => return .{ .known = true },
            .py => |o| {
                const r = py.c.PyObject_IsTrue(o);
                if (r < 0) return error.Python;
                return .{ .known = r == 1 };
            },
            .dyn => |d| {
                const f = &self.f;
                const t = switch (d.shape) {
                    .bool, .int => try f.value("icmp ne i64 {s}, 0", .{d.bits}),
                    .none => "false",
                    .float => blk: {
                        const x = try f.value("bitcast i64 {s} to double", .{d.bits});
                        break :blk try f.value("fcmp une double {s}, 0.0", .{x});
                    },
                    else => try f.value("call i1 @zr_truthy(i64 {s}, i64 {s})", .{ d.tag, d.bits }),
                };
                try self.drop(v);
                _ = at;
                return .{ .dyn = t };
            },
        }
    }

    // ------------------------------------------------------------------
    // Variables of the language
    // ------------------------------------------------------------------

    /// The slot pointer of a symbol's variable, from this function: its
    /// own, or one of the functions around it through the frames.
    fn varSlot(self: *Gen, sym: u32) Error![]const u8 {
        const c = self.c;
        const f = &self.f;
        const home = c.data.homeOf(sym);
        const slot = c.slot_of.get(sym).?;
        if (home == self.fnode) {
            if (self.frame) |fr| return f.value("getelementptr inbounds i8, ptr {s}, i64 {d}", .{ fr, 32 + 16 * slot });
            return self.var_slots.items[slot];
        }
        // Out through the frames: env is the frame of the function around
        var frame: []const u8 = if (self.fnode == NONE) "%globals" else "%env";
        var at = c.enclosingFunction(self.fnode);
        while (at != home) {
            if (at == NONE) return c.unsupported("a variable of another function isn't reachable from here (node {d})", .{sym});
            frame = try f.value("load ptr, ptr {s}, align 8", .{try f.value("getelementptr inbounds i8, ptr {s}, i64 16", .{frame})});
            at = c.enclosingFunction(at);
        }
        return f.value("getelementptr inbounds i8, ptr {s}, i64 {d}", .{ frame, 32 + 16 * slot });
    }

    /// rt.load(name): the variable (an owned reference); a builtin's host
    /// function.
    fn loadVar(self: *Gen, name_node: u32) Error!SVal {
        const c = self.c;
        const d = c.data;
        const si = d.symbolIndex(name_node) orelse return self.notAVariable(name_node);
        const sym = d.syms[si];
        if (sym.builtin) {
            const key = ph.newString(sym.name) orelse return error.Python;
            defer py.Py_DecRef(key);
            const h = py.c.PyDict_GetItem(c.lang.hosts, key) orelse {
                try self.failAt(name_node, try std.fmt.allocPrint(self.a(), "no host function for the builtin '{s}'", .{sym.name}));
                return SVal.none;
            };
            return SVal{ .py = h };
        }
        const f = &self.f;
        const slot = try self.varSlot(si);
        const v = try self.loadSlot(slot, .any);
        // Unset: an error at the name
        const is_unset = try f.value("icmp eq i64 {s}, {d}", .{ v.tag, helpers.UNSET });
        const bad = try f.label("unset");
        const good = try f.label("set");
        try f.condBr(is_unset, bad, good);
        try f.block(bad);
        const name = try c.m.string(sym.name);
        _ = try f.value("call i1 @zr_unset(ptr %ctx, i32 {d}, ptr {s})", .{ name_node, name });
        try f.br(self.err_label);
        try f.block(good);
        try self.increfDyn(v);
        return .{ .dyn = v };
    }

    /// rt.store(name, value), taking the value.
    fn storeVar(self: *Gen, name_node: u32, value: SVal) Error!void {
        const c = self.c;
        const d = c.data;
        const si = d.symbolIndex(name_node) orelse {
            _ = try self.notAVariable(name_node);
            return;
        };
        if (d.syms[si].builtin) {
            try self.failAt(name_node, try std.fmt.allocPrint(self.a(), "can't assign to the builtin '{s}'", .{d.syms[si].name}));
            return;
        }
        const v = try self.materialize(value, name_node);
        const slot = try self.varSlot(si);
        const old = try self.loadSlot(slot, .any);
        try self.storeSlot(slot, v);
        try self.f.emit("call void @zr_decref(i64 {s}, i64 {s})", .{ old.tag, old.bits });
    }

    fn notAVariable(self: *Gen, idx: u32) Error!SVal {
        const msg = if (self.c.lang.analysis == null)
            "variables need rules with a scopes() rule"
        else
            try std.fmt.allocPrint(self.a(), "'{s}' is not a variable", .{self.c.data.text(idx)});
        try self.failAt(idx, msg);
        return SVal.none;
    }

    // ------------------------------------------------------------------
    // Functions of the language
    // ------------------------------------------------------------------

    /// The frame a function made from `fnode` sees: the frame of the
    /// function around it (here, or through the frames).
    fn envFor(self: *Gen, fnode: u32) Error![]const u8 {
        const c = self.c;
        const outer = c.enclosingFunction(fnode);
        if (outer == self.fnode) return self.frame orelse c.unsupported("a function made in a function whose variables aren't in a frame (node {d})", .{fnode});
        var frame: []const u8 = if (self.fnode == NONE) "%globals" else "%env";
        var at = c.enclosingFunction(self.fnode);
        while (at != outer) {
            if (at == NONE) return c.unsupported("a function made outside the function around it (node {d})", .{fnode});
            frame = try self.f.value("load ptr, ptr {s}, align 8", .{try self.f.value("getelementptr inbounds i8, ptr {s}, i64 16", .{frame})});
            at = c.enclosingFunction(at);
        }
        return frame;
    }

    /// rt.function(node): a function value.
    fn makeFunction(self: *Gen, fnode: u32) Error!SVal {
        const c = self.c;
        const spec = c.specOf(fnode) orelse {
            try self.failAt(fnode, try std.fmt.allocPrint(self.a(), "{s} isn't a function kind (Language.function)", .{c.data.grammar.kind_names[c.data.rule(fnode)]}));
            return SVal.none;
        };
        const code = try c.functionCode(fnode);
        const env = try self.envFor(fnode);
        const name_text = if (spec.name != 0) if (program_mod.labelled(c.data, fnode, spec.name)) |nn| c.data.text(nn) else "<anonymous>" else "<anonymous>";
        const name = try c.m.string(name_text);
        const nparams = self.paramNodes(fnode, spec).len;
        const ok = try self.f.value("call i1 @zr_function(ptr %ctx, ptr {s}, ptr {s}, i32 {d}, ptr {s}, i64 {d}, ptr {s})", .{ code, env, fnode, name, nparams, self.out });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.function) };
    }

    /// A function node's parameters: the nodes their values are stored
    /// under.
    fn paramNodes(self: *Gen, fnode: u32, spec: FunctionSpec) []const u32 {
        const c = self.c;
        const d = c.data;
        var out: std.ArrayListUnmanaged(u32) = .empty;
        if (spec.params == 0) return &.{};
        var ch = fnode + 1;
        const stop = d.end(fnode);
        while (ch < stop) : (ch = d.end(ch)) {
            if (d.nodes[ch].fieldId() != spec.params) continue;
            // the parameter itself if it defines a variable, else its name
            var target = ch;
            if (d.symbolIndex(ch)) |si| {
                if (d.syms[si].node != ch) target = nameChild(d, ch) orelse ch;
            } else target = nameChild(d, ch) orelse ch;
            out.append(self.a(), target) catch return &.{};
        }
        return out.items;
    }

    fn nameChild(d: *const program_mod.Data, n: u32) ?u32 {
        const field = d.grammar.field_ids.get("name") orelse return null;
        return program_mod.labelled(d, n, field);
    }

    fn bindParams(self: *Gen, fnode: u32, spec: FunctionSpec) Error!void {
        const f = &self.f;
        for (self.paramNodes(fnode, spec), 0..) |p, i| {
            const ap = try f.value("getelementptr inbounds {{ i64, i64 }}, ptr %args, i64 {d}", .{i});
            const v = try self.loadSlot(ap, .any);
            try self.increfDyn(v);
            try self.storeVar(p, .{ .dyn = v });
        }
    }

    /// Define the hoisted functions of a scope just entered.
    fn hoist(self: *Gen, scope: u32) Error!void {
        const c = self.c;
        const list = c.data.hoisted.get(scope) orelse return;
        for (list.items) |fnode| {
            const spec = c.specOf(fnode).?;
            const name_node = program_mod.labelled(c.data, fnode, spec.name) orelse continue;
            const fv = try self.makeFunction(fnode);
            try self.storeVar(name_node, fv);
        }
    }

    // ------------------------------------------------------------------
    // Nodes: their fields, their semantics
    // ------------------------------------------------------------------

    /// A node's value as a field (what its action makes, or the node).
    fn nodeValue(self: *Gen, idx: u32) Error!SVal {
        const c = self.c;
        const d = c.data;
        const rid = d.rule(idx);
        const action: grammar_mod.Action = if (rid < d.grammar.actions.len) d.grammar.actions[rid] else .none;
        switch (action) {
            .none, .class => return .{ .node = idx },
            .str, .int, .float, .unquote => {
                // zgram's own conversion, as the reference mode
                const zn = py.c.PyObject_CallMethod(c.lang.tree, "node", "I", @as(c_uint, idx)) orelse return error.Python;
                defer py.Py_DecRef(zn);
                const o = py.c.PyObject_CallMethod(zn, "to_ast", null) orelse return error.Python;
                defer py.Py_DecRef(o);
                return self.constant(o, idx);
            },
            .true_ => return .{ .bool = true },
            .false_ => return .{ .bool = false },
            .none_ => return .none,
            .drop => return .none,
            .list, .tuple, .first, .dict => {
                const l = try self.a().create(SList);
                l.* = .{};
                var ch = idx + 1;
                const stop = d.end(idx);
                while (ch < stop) : (ch = d.end(ch)) {
                    const crid = d.rule(ch);
                    if (crid < d.grammar.actions.len and d.grammar.actions[crid] == .drop) continue;
                    try l.items.append(self.a(), try self.nodeValue(ch));
                }
                if (action == .tuple) return .{ .tuple = l.items.items };
                if (action == .first) return if (l.items.items.len > 0) l.items.items[0] else SVal.none;
                if (action == .dict) return c.unsupported("a -> dict field isn't supported in compiled code yet (node {d})", .{idx});
                return .{ .list = l };
            },
        }
    }

    /// A Python constant as a value known here.
    fn constant(self: *Gen, o: *PyObject, at: u32) Error!SVal {
        if (o == py.Py_None()) return .none;
        if (py.PyBool_Check(o)) return .{ .bool = o == py.Py_True() };
        if (py.PyLong_Check(o)) {
            var overflow: c_int = 0;
            const n = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
            if (overflow != 0) {
                // (as the reference mode: an error when it's evaluated, at
                // the node whose semantic reads it)
                const where = if (self.insts.items.len > 0) self.insts.items[self.insts.items.len - 1].node else at;
                try self.failAt(where, "integer overflow");
                return .none;
            }
            return .{ .int = n };
        }
        if (py.PyFloat_Check(o)) return .{ .float = py.c.PyFloat_AsDouble(o) };
        if (py.PyUnicode_Check(o)) {
            const s = ph.utf8(o, "str") orelse return error.Python;
            return .{ .str = try self.a().dupe(u8, s) };
        }
        if (py.PyTuple_Check(o)) {
            const n: usize = @intCast(py.c.PyTuple_Size(o));
            const items = try self.a().alloc(SVal, n);
            for (items, 0..) |*it, i| it.* = try self.constant(py.c.PyTuple_GetItem(o, @intCast(i)).?, at);
            return .{ .tuple = items };
        }
        return .{ .py = o };
    }

    /// A field of a node: the labelled child's value, a list of them for a
    /// label that repeats, None if absent.
    fn fieldValue(self: *Gen, idx: u32, field: u8) Error!SVal {
        const d = self.c.data;
        const label = d.grammar.labelOf(d.rule(idx), field);
        const many = label != null and label.?.many;
        const l = try self.a().create(SList);
        l.* = .{};
        var ch = idx + 1;
        const stop = d.end(idx);
        while (ch < stop) : (ch = d.end(ch)) {
            if (d.nodes[ch].fieldId() != field) continue;
            const crid = d.rule(ch);
            if (crid < d.grammar.actions.len and d.grammar.actions[crid] == .drop) continue;
            try l.items.append(self.a(), try self.nodeValue(ch));
        }
        if (many or l.items.items.len > 1) return .{ .list = l };
        if (l.items.items.len == 1) return l.items.items[0];
        return .none;
    }

    /// The children's values (`node.children`).
    fn childValues(self: *Gen, idx: u32) Error!SVal {
        const d = self.c.data;
        const l = try self.a().create(SList);
        l.* = .{};
        var ch = idx + 1;
        const stop = d.end(idx);
        while (ch < stop) : (ch = d.end(ch)) {
            const crid = d.rule(ch);
            if (crid < d.grammar.actions.len and d.grammar.actions[crid] == .drop) continue;
            try l.items.append(self.a(), try self.nodeValue(ch));
        }
        return .{ .list = l };
    }

    /// The semantic of a node, read by the front (null: none registered).
    fn semanticOf(self: *Gen, idx: u32, which: enum { eval, exec }) Error!?*const front.Function {
        const c = self.c;
        const rid = c.data.rule(idx);
        const table = if (which == .eval) c.lang.eval_of else c.lang.exec_of;
        if (rid >= table.len) return null;
        const fobj = table[rid] orelse return null;
        return c.lang.read.get(fobj) orelse return c.unsupported("the semantic of {s} is native=False: compiled code can't run it yet", .{c.data.grammar.kind_names[rid]});
    }

    /// rt.eval of a node.
    fn evalNode(self: *Gen, idx: u32) Error!SVal {
        if (try self.semanticOf(idx, .eval)) |func| return self.runSemantic(func, idx);
        // Defaults: a name's variable; an only child's value
        const d = self.c.data;
        if (d.symbolIndex(idx) != null) return self.loadVar(idx);
        const kids = try self.childValues(idx);
        if (kids.list.items.items.len == 1) return self.evalValue(kids.list.items.items[0]);
        try self.failAt(idx, try std.fmt.allocPrint(self.a(), "no semantics to evaluate {s}", .{d.grammar.kind_names[d.rule(idx)]}));
        return .none;
    }

    fn evalValue(self: *Gen, v: SVal) Error!SVal {
        switch (v) {
            .node => |n| return self.evalNode(n),
            .list => |l| {
                const out = try self.a().create(SList);
                out.* = .{};
                for (l.items.items) |item| try out.items.append(self.a(), try self.evalValue(item));
                return .{ .list = out };
            },
            else => return v,
        }
    }

    /// rt.exec of a node.
    fn execNode(self: *Gen, idx: u32) Error!void {
        const c = self.c;
        if (try self.semanticOf(idx, .exec)) |func| {
            try self.drop(try self.runSemantic(func, idx));
            return;
        }
        // Defaults: a function definition (unless hoisted), an expression
        // for its effect, the children in order
        if (c.specOf(idx)) |spec| {
            if (spec.hoist) return;
            const name_node = program_mod.labelled(c.data, idx, spec.name) orelse return;
            try self.storeVar(name_node, try self.makeFunction(idx));
            return;
        }
        if (try self.semanticOf(idx, .eval)) |func| {
            try self.drop(try self.runSemantic(func, idx));
            return;
        }
        try self.execValue(try self.childValues(idx));
    }

    fn execValue(self: *Gen, v: SVal) Error!void {
        switch (v) {
            .node => |n| try self.execNode(n),
            .list => |l| for (l.items.items) |item| try self.execValue(item),
            .tuple => |t| for (t) |item| try self.execValue(item),
            else => {},
        }
    }

    // ------------------------------------------------------------------
    // Running a semantic inline
    // ------------------------------------------------------------------

    /// Run a semantic for a node (its params: the node, rt); its result.
    fn runSemantic(self: *Gen, func: *const front.Function, idx: u32) Error!SVal {
        return self.runFunction(func, idx, &.{ .{ .node = idx }, .rt });
    }

    /// Run a front function inline with arguments.
    fn runFunction(self: *Gen, func: *const front.Function, at: u32, args: []const SVal) Error!SVal {
        if (self.insts.items.len > 200) return self.c.unsupported("semantics nest more than 200 deep (recursive helpers aren't compiled yet)", .{});
        const locals = try self.a().alloc(Local, func.locals.len);
        @memset(locals, .unset);
        if (args.len != func.param_count) return self.c.unsupportedAt(func, .{ .line = func.first_line }, "called with {d} arguments, takes {d}", .{ args.len, func.param_count });
        for (args, 0..) |arg, i| locals[i] = .{ .static = arg };
        const inst = try self.a().create(Inst);
        inst.* = .{ .func = func, .node = at, .locals = locals, .exit_label = try self.f.label("ret") };
        try self.insts.append(self.a(), inst);
        defer _ = self.insts.pop();

        try self.stmts(inst, func.body);
        // Falling off the end: None
        if (!inst.done) try self.setResult(inst, .none);
        // The exit: where returns from run-time control flow meet
        var result: SVal = inst.result orelse .none;
        if (inst.result_slot) |slot| {
            try self.f.block(inst.exit_label);
            result = .{ .dyn = try self.loadSlot(slot, .any) };
        }
        try self.releaseLocals(inst);
        return result;
    }

    /// The semantic's result: kept, or in its slot (and to its exit).
    fn setResult(self: *Gen, inst: *Inst, v: SVal) Error!void {
        if (inst.dyn_depth == 0 and inst.result_slot == null) {
            inst.result = v;
            inst.done = true;
            return;
        }
        const slot = inst.result_slot orelse blk: {
            const s = try self.f.alloca("{ i64, i64 }");
            inst.result_slot = s;
            break :blk s;
        };
        try self.storeSlot(slot, try self.materialize(v, inst.node));
        try self.f.br(inst.exit_label);
        if (inst.dyn_depth == 0) {
            inst.done = true;
        } else try self.f.block(try self.f.label("after_return"));
    }

    /// Drop the semantic's locals that hold run-time values.
    fn releaseLocals(self: *Gen, inst: *Inst) Error!void {
        for (inst.locals) |l| switch (l) {
            .slot => |s| {
                const v = try self.loadSlot(s.ptr, s.shape);
                try self.drop(.{ .dyn = v });
            },
            .static => |sv| try self.drop(sv),
            .unset => {},
        };
    }

    /// Release the locals of the semantics above `depth` (leaving them by a
    /// jump: rt.Return, rt.Break).
    fn releaseAbove(self: *Gen, depth: usize) Error!void {
        var i = self.insts.items.len;
        while (i > depth) {
            i -= 1;
            try self.releaseLocals(self.insts.items[i]);
        }
    }

    // ------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------

    fn stmts(self: *Gen, inst: *Inst, body: []const front.Stmt) Error!void {
        for (body) |s| {
            if (inst.done) return;
            try self.stmt(inst, s);
        }
    }

    fn stmt(self: *Gen, inst: *Inst, s: front.Stmt) Error!void {
        switch (s.kind) {
            .pass => {},
            .expr => |e| try self.drop(try self.expr(inst, e)),
            .assign => |a_| {
                const v = try self.expr(inst, a_.value);
                for (a_.targets, 0..) |t, i| {
                    // (each target after the first gets its own reference)
                    if (i > 0) if (v == .dyn) try self.increfDyn(v.dyn);
                    try self.assign(inst, t, v, s.pos);
                }
            },
            .aug => |a_| {
                const cur = try self.targetValue(inst, a_.target, s.pos);
                const rhs = try self.expr(inst, a_.value);
                const r = try self.binary(inst, a_.op, cur, rhs);
                try self.assign(inst, a_.target, r, s.pos);
            },
            .if_ => |i| {
                const t = try self.truth(try self.expr(inst, i.test_), inst.node);
                switch (t) {
                    .known => |b| try self.stmts(inst, if (b) i.body else i.else_),
                    .dyn => |cond| {
                        try self.prepareDynamic(inst, &.{ i.body, i.else_ });
                        const yes = try self.f.label("then");
                        const no = try self.f.label("else");
                        const join = try self.f.label("endif");
                        try self.f.condBr(cond, yes, no);
                        inst.dyn_depth += 1;
                        try self.f.block(yes);
                        try self.stmts(inst, i.body);
                        try self.f.br(join);
                        try self.f.block(no);
                        try self.stmts(inst, i.else_);
                        try self.f.br(join);
                        inst.dyn_depth -= 1;
                        try self.f.block(join);
                    },
                }
            },
            .while_ => |w| {
                try self.prepareDynamic(inst, &.{w.body});
                const head = try self.f.label("while");
                const body = try self.f.label("body");
                const els = try self.f.label("whileelse");
                const exit = try self.f.label("endwhile");
                try self.f.br(head);
                try self.f.block(head);
                inst.dyn_depth += 1;
                const t = try self.truth(try self.expr(inst, w.test_), inst.node);
                switch (t) {
                    .known => |b| try self.f.br(if (b) body else els),
                    .dyn => |cond| try self.f.condBr(cond, body, els),
                }
                try self.f.block(body);
                try inst.loops.append(self.a(), .{ .brk = exit, .cont = head });
                try self.stmts(inst, w.body);
                _ = inst.loops.pop();
                try self.f.br(head);
                try self.f.block(els);
                try self.stmts(inst, w.else_);
                inst.dyn_depth -= 1;
                try self.f.br(exit);
                try self.f.block(exit);
            },
            .for_ => |fr| try self.forLoop(inst, fr.target, fr.iter, fr.body, fr.else_, s.pos),
            .return_ => |r| {
                const v = if (r) |e| try self.expr(inst, e) else SVal.none;
                try self.setResult(inst, v);
            },
            .raise_ => |r| {
                const v = if (r) |e| try self.expr(inst, e) else return self.c.unsupportedAt(inst.func, s.pos, "a bare `raise` can't be compiled", .{});
                try self.raise(inst, v, s.pos);
            },
            .assert_ => |as| {
                const t = try self.truth(try self.expr(inst, as.test_), inst.node);
                const msg: []const u8 = if (as.msg) |m| blk: {
                    const mv = try self.expr(inst, m);
                    break :blk if (mv == .str) mv.str else return self.c.unsupportedAt(inst.func, s.pos, "an assert's message must be a string known when compiling", .{});
                } else "";
                const text = try std.fmt.allocPrint(self.a(), "AssertionError: {s}", .{msg});
                switch (t) {
                    .known => |b| if (!b) try self.failAt(inst.node, text),
                    .dyn => |cond| {
                        const bad = try self.f.label("assert_failed");
                        const good = try self.f.label("assert_ok");
                        try self.f.condBr(cond, good, bad);
                        try self.f.block(bad);
                        try self.failAt(inst.node, text);
                        try self.f.block(good);
                    },
                }
            },
            .break_, .continue_ => {
                if (inst.loops.items.len == 0) return self.c.unsupportedAt(inst.func, s.pos, "this break or continue can't be compiled", .{});
                const target = inst.loops.items[inst.loops.items.len - 1];
                try self.f.br(if (s.kind == .break_) target.brk else target.cont);
                try self.f.block(try self.f.label("after_jump"));
            },
        }
    }

    /// Before run-time control flow (an if on a run-time value, a loop): the
    /// locals its statements assign become slots, so each path sees them.
    fn prepareDynamic(self: *Gen, inst: *Inst, bodies: []const []const front.Stmt) Error!void {
        var set = std.AutoHashMapUnmanaged(u32, void).empty;
        for (bodies) |b| collectAssigned(b, &set, self.a()) catch return error.OutOfMemory;
        var it = set.keyIterator();
        while (it.next()) |slot| try self.toSlot(inst, slot.*);
    }

    /// Make a local a stack slot (its current value in it).
    fn toSlot(self: *Gen, inst: *Inst, slot: u32) Error!void {
        switch (inst.locals[slot]) {
            .slot => return,
            .unset => {
                const p = try self.f.alloca("{ i64, i64 }");
                try self.storeSlot(p, .{ .tag = "0", .bits = "0", .shape = .none });
                inst.locals[slot] = .{ .slot = .{ .ptr = p, .shape = .any } };
            },
            .static => |v| {
                const d = try self.materialize(v, inst.node);
                const p = try self.f.alloca("{ i64, i64 }");
                try self.storeSlot(p, d);
                inst.locals[slot] = .{ .slot = .{ .ptr = p, .shape = .any } };
            },
        }
    }

    fn assign(self: *Gen, inst: *Inst, t: front.Target, v: SVal, pos: front.Pos) Error!void {
        switch (t) {
            .local => |slot| switch (inst.locals[slot]) {
                .slot => |s| {
                    const d = try self.materialize(v, inst.node);
                    const old = try self.loadSlot(s.ptr, s.shape);
                    try self.storeSlot(s.ptr, d);
                    try self.drop(.{ .dyn = old });
                },
                .static => |old| {
                    if (inst.dyn_depth > 0) {
                        // (prepareDynamic makes such locals slots first)
                        try self.toSlot(inst, slot);
                        return self.assign(inst, t, v, pos);
                    }
                    try self.drop(old);
                    inst.locals[slot] = .{ .static = v };
                },
                .unset => {
                    if (inst.dyn_depth > 0) {
                        try self.toSlot(inst, slot);
                        return self.assign(inst, t, v, pos);
                    }
                    inst.locals[slot] = .{ .static = v };
                },
            },
            .tuple => |ts| {
                const items: []const SVal = switch (v) {
                    .tuple => |x| x,
                    .list => |l| l.items.items,
                    else => return self.c.unsupportedAt(inst.func, pos, "unpacking a value only known at run time can't be compiled yet", .{}),
                };
                if (items.len != ts.len) return self.c.unsupportedAt(inst.func, pos, "unpacking {d} values into {d} names", .{ items.len, ts.len });
                for (ts, items) |x, item| try self.assign(inst, x, item, pos);
            },
            .attr, .index => return self.c.unsupportedAt(inst.func, pos, "assigning to an attribute or an item isn't compiled yet", .{}),
        }
    }

    /// The current value of an assignment target (for `x += 1`).
    fn targetValue(self: *Gen, inst: *Inst, t: front.Target, pos: front.Pos) Error!SVal {
        switch (t) {
            .local => |slot| return self.readLocal(inst, slot, pos),
            else => return self.c.unsupportedAt(inst.func, pos, "augmented assignment to an attribute or an item isn't compiled yet", .{}),
        }
    }

    fn readLocal(self: *Gen, inst: *Inst, slot: u32, pos: front.Pos) Error!SVal {
        switch (inst.locals[slot]) {
            .unset => return self.c.unsupportedAt(inst.func, pos, "'{s}' may be read before it is assigned", .{inst.func.locals[slot]}),
            .static => |v| {
                if (v == .dyn) try self.increfDyn(v.dyn);
                return v;
            },
            .slot => |s| {
                const d = try self.loadSlot(s.ptr, s.shape);
                try self.increfDyn(d);
                return .{ .dyn = d };
            },
        }
    }

    /// raise rt.Return(v) / rt.Break() / rt.Continue()
    fn raise(self: *Gen, inst: *Inst, v: SVal, pos: front.Pos) Error!void {
        const ctl = switch (v) {
            .control => |x| x,
            else => return self.c.unsupportedAt(inst.func, pos, "only rt.Return, rt.Break and rt.Continue can be raised in compiled code", .{}),
        };
        switch (ctl.kind) {
            .Return => {
                if (self.fnode == NONE) {
                    try self.failAt(inst.node, "return outside a function");
                    return;
                }
                const d = try self.materialize(if (ctl.value) |x| x.* else .none, inst.node);
                try self.releaseAbove(0);
                try self.storeSlot("%result", d);
                try self.f.br(self.ret_label);
            },
            .Break, .Continue => {
                if (self.loops.items.len == 0) {
                    try self.failAt(inst.node, "break or continue outside a loop");
                    return;
                }
                const target = self.loops.items[self.loops.items.len - 1];
                try self.releaseAboveLoop(target);
                try self.f.br(if (ctl.kind == .Break) target.brk else target.cont);
            },
            else => unreachable,
        }
        if (inst.dyn_depth == 0) {
            inst.done = true;
            // (jumped: whatever follows is unreachable)
            try self.f.block(try self.f.label("after_raise"));
        } else try self.f.block(try self.f.label("after_raise"));
    }

    fn releaseAboveLoop(self: *Gen, target: LoopTarget) Error!void {
        // (the semantics run inside the loop's body)
        try self.releaseAbove(target.depth);
    }

    fn forLoop(self: *Gen, inst: *Inst, target: front.Target, iter_e: *const front.Expr, body: []const front.Stmt, else_: []const front.Stmt, pos: front.Pos) Error!void {
        const it = try self.expr(inst, iter_e);
        const items: []const SVal = switch (it) {
            .list => |l| l.items.items,
            .tuple => |t| t,
            else => return self.c.unsupportedAt(inst.func, pos, "a for loop over a value only known at run time isn't compiled yet", .{}),
        };
        // Known items: unrolled (break / continue jump within it)
        const exit = try self.f.label("endfor");
        var broke = false;
        for (items) |item| {
            const next = try self.f.label("next");
            if (item == .dyn) try self.increfDyn(item.dyn);
            try self.assign(inst, target, item, pos);
            try inst.loops.append(self.a(), .{ .brk = exit, .cont = next });
            try self.stmts(inst, body);
            _ = inst.loops.pop();
            try self.f.block(next);
            if (inst.done) {
                broke = true;
                break;
            }
        }
        if (!broke) try self.stmts(inst, else_);
        try self.f.block(exit);
    }

    // ------------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------------

    fn expr(self: *Gen, inst: *Inst, e: *const front.Expr) Error!SVal {
        const c = self.c;
        switch (e.kind) {
            .int => |n| return .{ .int = n },
            .float => |x| return .{ .float = x },
            .str => |s| return .{ .str = s },
            .bool => |b| return .{ .bool = b },
            .none => return .none,
            .local => |slot| return self.readLocal(inst, slot, e.pos),
            .global => |name| return self.global(inst, name, e.pos),
            .attr => |x| {
                const obj = try self.expr(inst, x.obj);
                return self.attr(inst, obj, x.name, e.pos);
            },
            .call => |x| return self.call(inst, x.func, x.args, x.keywords, e.pos),
            .binary => |x| {
                const l = try self.expr(inst, x.left);
                const r = try self.expr(inst, x.right);
                return self.binary(inst, x.op, l, r);
            },
            .unary => |x| {
                const v = try self.expr(inst, x.operand);
                return self.unary(inst, x.op, v);
            },
            .compare => |x| {
                // a < b < c: each pair, and-ed (each operand once)
                var left = try self.expr(inst, x.first);
                if (x.ops.len == 1) {
                    const right = try self.expr(inst, x.rest[0]);
                    return self.compare(inst, x.ops[0], left, right);
                }
                var result: SVal = .{ .bool = true };
                for (x.ops, x.rest, 0..) |op, rest_e, i| {
                    const right = try self.expr(inst, rest_e);
                    if (i + 1 < x.ops.len and right == .dyn) try self.increfDyn(right.dyn);
                    const r = try self.compare(inst, op, left, right);
                    result = try self.andValues(inst, result, r);
                    left = right;
                }
                return result;
            },
            .and_, .or_ => |items| return self.boolOp(inst, e.kind == .and_, items),
            .cond => |x| {
                const t = try self.truth(try self.expr(inst, x.test_), inst.node);
                switch (t) {
                    .known => |b| return self.expr(inst, if (b) x.then else x.else_),
                    .dyn => |cond| return self.branchValue(inst, cond, x.then, x.else_),
                }
            },
            .list, .tuple => |items| {
                const l = try self.a().create(SList);
                l.* = .{};
                for (items) |item| try l.items.append(self.a(), try self.expr(inst, item));
                if (e.kind == .tuple) return .{ .tuple = l.items.items };
                return .{ .list = l };
            },
            .list_comp => |comp| return self.listComp(inst, comp, e.pos),
            else => return c.unsupportedAt(inst.func, e.pos, "this expression ({s}) isn't compiled yet", .{@tagName(e.kind)}),
        }
    }

    /// A module-level name of the semantic: its value when compiling.
    fn global(self: *Gen, inst: *Inst, name: []const u8, pos: front.Pos) Error!SVal {
        const fobj = inst.func.py_function;
        const globals = ph.attr(fobj, "__globals__") orelse return error.Python;
        defer py.Py_DecRef(globals);
        const key = ph.newString(name) orelse return error.Python;
        defer py.Py_DecRef(key);
        // A captured variable first, then the module, then builtins
        if (try closureValue(fobj, name)) |cell_value| return self.constant(cell_value, inst.node);
        if (py.c.PyDict_GetItem(globals, key)) |v| return self.constant(v, inst.node);
        const builtins = py.c.PyImport_ImportModule("builtins") orelse return error.Python;
        defer py.Py_DecRef(builtins);
        if (py.c.PyObject_HasAttr(builtins, key) == 1) {
            const v = py.c.PyObject_GetAttr(builtins, key) orelse return error.Python;
            // (builtins live as long as the interpreter: borrowed is fine)
            py.Py_DecRef(v);
            return .{ .py = v };
        }
        return self.c.unsupportedAt(inst.func, pos, "name '{s}' is not defined", .{name});
    }

    fn closureValue(fobj: *PyObject, name: []const u8) Error!?*PyObject {
        const code = ph.attr(fobj, "__code__") orelse return error.Python;
        defer py.Py_DecRef(code);
        const freevars = ph.attr(code, "co_freevars") orelse return error.Python;
        defer py.Py_DecRef(freevars);
        const closure = ph.attr(fobj, "__closure__") orelse return error.Python;
        defer py.Py_DecRef(closure);
        if (closure == py.Py_None()) return null;
        const n: usize = @intCast(py.c.PyTuple_Size(freevars));
        for (0..n) |i| {
            const fv = ph.utf8(py.c.PyTuple_GetItem(freevars, @intCast(i)).?, "name") orelse return error.Python;
            if (!std.mem.eql(u8, fv, name)) continue;
            const cell = py.c.PyTuple_GetItem(closure, @intCast(i)).?;
            const contents = ph.attr(cell, "cell_contents") orelse return error.Python;
            // (the cell keeps it alive)
            py.Py_DecRef(contents);
            return contents;
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Attributes
    // ------------------------------------------------------------------

    fn attr(self: *Gen, inst: *Inst, obj: SVal, name: []const u8, pos: front.Pos) Error!SVal {
        const c = self.c;
        const eq = std.mem.eql;
        switch (obj) {
            .node => |idx| {
                const d = c.data;
                if (d.grammar.field_ids.get(name)) |field| return self.fieldValue(idx, field);
                const n = d.nodes[idx];
                const rid = n.ruleId();
                if (eq(u8, name, "kind")) return .{ .str = d.grammar.kind_names[rid] };
                if (eq(u8, name, "rule")) return .{ .str = d.grammar.rule_names[rid] };
                if (eq(u8, name, "text")) return .{ .str = d.text(idx) };
                if (eq(u8, name, "start")) return .{ .int = n.text_start };
                if (eq(u8, name, "end")) return .{ .int = n.text_end };
                if (eq(u8, name, "index")) return .{ .int = idx };
                if (eq(u8, name, "span")) {
                    const items = try self.a().alloc(SVal, 2);
                    items[0] = .{ .int = n.text_start };
                    items[1] = .{ .int = n.text_end };
                    return .{ .tuple = items };
                }
                if (eq(u8, name, "children")) return self.childValues(idx);
                if (eq(u8, name, "parent")) {
                    const p = d.parents[idx];
                    return if (p == NONE) SVal.none else SVal{ .node = p };
                }
                return c.unsupportedAt(inst.func, pos, "{s} has no field '{s}'", .{ d.grammar.kind_names[rid], name });
            },
            .rt => {
                inline for (@typeInfo(RtMethod).@"enum".fields) |fd| {
                    if (eq(u8, name, fd.name)) return .{ .rt_method = @enumFromInt(fd.value) };
                }
                if (eq(u8, name, "receiver")) {
                    if (self.fnode == NONE) return .none;
                    // (the receiver of the function being run)
                    const f = &self.f;
                    const has = try f.value("icmp ne ptr %recv, null", .{});
                    const slot = try f.alloca("{ i64, i64 }");
                    try self.storeSlot(slot, .{ .tag = "0", .bits = "0", .shape = .none });
                    const yes = try f.label("recv");
                    const join = try f.label("recv_end");
                    try f.condBr(has, yes, join);
                    try f.block(yes);
                    const v = try self.loadSlot("%recv", .any);
                    try self.increfDyn(v);
                    try self.storeSlot(slot, v);
                    try f.block(join);
                    return .{ .dyn = try self.loadSlot(slot, .any) };
                }
                return c.unsupportedAt(inst.func, pos, "rt has no '{s}' in compiled code", .{name});
            },
            .py => |o| {
                const key = ph.newString(name) orelse return error.Python;
                defer py.Py_DecRef(key);
                const v = py.c.PyObject_GetAttr(o, key) orelse return error.Python;
                // (kept alive by the object it belongs to, for constants: a
                // reference is kept with the compiled program when used)
                const sv = try self.constant(v, inst.node);
                if (sv == .py) _ = try c.objectIndex(v);
                py.Py_DecRef(v);
                return sv;
            },
            .str, .list, .dyn => return .{ .method = .{ .recv = try self.boxed(obj), .name = name } },
            else => return c.unsupportedAt(inst.func, pos, "'{s}' of a {s} isn't compiled yet", .{ name, @tagName(obj) }),
        }
    }

    fn boxed(self: *Gen, v: SVal) Error!*const SVal {
        const p = try self.a().create(SVal);
        p.* = v;
        return p;
    }

    // ------------------------------------------------------------------
    // Calls
    // ------------------------------------------------------------------

    fn call(self: *Gen, inst: *Inst, func_e: *const front.Expr, args_e: []const *const front.Expr, kws: []const front.Keyword, pos: front.Pos) Error!SVal {
        const c = self.c;
        const callee = try self.expr(inst, func_e);
        // (arguments in order, as Python evaluates them)
        const args = try self.a().alloc(SVal, args_e.len);
        for (args, args_e) |*slot, ae| slot.* = try self.expr(inst, ae);
        var receiver: ?SVal = null;
        for (kws) |kw| {
            if (callee == .rt_method and callee.rt_method == .call and std.mem.eql(u8, kw.name, "receiver")) {
                receiver = try self.expr(inst, kw.value);
            } else if (callee == .rt_method and callee.rt_method == .@"error" and std.mem.eql(u8, kw.name, "code")) {
                // (the code isn't kept by compiled errors yet: reported as runtime)
                try self.drop(try self.expr(inst, kw.value));
            } else return c.unsupportedAt(inst.func, pos, "the keyword argument {s}= isn't compiled", .{kw.name});
        }
        switch (callee) {
            .rt_method => |m| return self.rtCall(inst, m, args, receiver, pos),
            .py => |o| return self.pyCall(inst, o, args, pos),
            .dyn => {
                const l = try self.a().create(SList);
                l.* = .{};
                try l.items.appendSlice(self.a(), args);
                return self.dynCall(inst, callee, .{ .list = l }, null);
            },
            else => return c.unsupportedAt(inst.func, pos, "calling a {s} isn't compiled yet", .{@tagName(callee)}),
        }
    }

    fn rtCall(self: *Gen, inst: *Inst, m: RtMethod, args: []const SVal, receiver: ?SVal, pos: front.Pos) Error!SVal {
        const c = self.c;
        const want: usize = switch (m) {
            .eval, .exec, .loop, .load, .function, .kind, .text, .span, .scope, .symbol, .type_of, .node_at => 1,
            .store, .call => 2,
            .@"error" => 2,
            .Return => if (args.len == 0) 0 else 1,
            .Break, .Continue => 0,
        };
        if (args.len != want) return c.unsupportedAt(inst.func, pos, "rt.{s}() takes {d} arguments here", .{ @tagName(m), want });
        switch (m) {
            .eval => return self.evalValue(args[0]),
            .exec => {
                try self.execValue(args[0]);
                return .none;
            },
            .loop => return self.rtLoop(args[0]),
            .load => {
                const n = try self.nodeArg(inst, args[0], pos);
                return self.loadVar(n);
            },
            .store => {
                const n = try self.nodeArg(inst, args[0], pos);
                try self.storeVar(n, args[1]);
                return .none;
            },
            .function => return self.makeFunction(try self.nodeArg(inst, args[0], pos)),
            .call => return self.dynCall(inst, args[0], args[1], receiver),
            .@"error" => {
                const n = switch (args[0]) {
                    .node => |x| x,
                    .none => inst.node,
                    else => return c.unsupportedAt(inst.func, pos, "rt.error's node must be known when compiling", .{}),
                };
                const msg = switch (args[1]) {
                    .str => |s| s,
                    else => return c.unsupportedAt(inst.func, pos, "rt.error's message must be known when compiling (for now)", .{}),
                };
                try self.failAt(n, msg);
                return .none;
            },
            .kind => {
                const n = try self.nodeArg(inst, args[0], pos);
                return .{ .str = c.data.grammar.kind_names[c.data.rule(n)] };
            },
            .text => return .{ .str = c.data.text(try self.nodeArg(inst, args[0], pos)) },
            .span => {
                const n = c.data.nodes[try self.nodeArg(inst, args[0], pos)];
                const items = try self.a().alloc(SVal, 2);
                items[0] = .{ .int = n.text_start };
                items[1] = .{ .int = n.text_end };
                return .{ .tuple = items };
            },
            .scope => {
                const n = try self.nodeArg(inst, args[0], pos);
                const si = c.data.symbolIndex(n) orelse return .none;
                const s = c.data.syms[si].scope;
                return if (s == NONE or s >= c.data.nodes.len) SVal.none else SVal{ .node = s };
            },
            .symbol, .type_of => {
                const n = try self.nodeArg(inst, args[0], pos);
                const analysis = c.lang.analysis orelse return .none;
                const r = py.c.PyObject_CallMethod(analysis, if (m == .symbol) "resolve" else "type_of", "I", @as(c_uint, n)) orelse return error.Python;
                const sv = try self.constant(r, n);
                if (sv == .py) _ = try c.objectIndex(r);
                py.Py_DecRef(r);
                return sv;
            },
            .node_at => switch (args[0]) {
                .int => |i| return .{ .node = @intCast(i) },
                else => return c.unsupportedAt(inst.func, pos, "rt.node_at's index must be known when compiling", .{}),
            },
            .Return => return .{ .control = .{ .kind = .Return, .value = if (args.len == 1) try self.boxed(args[0]) else null } },
            .Break, .Continue => return .{ .control = .{ .kind = m, .value = null } },
        }
    }

    fn nodeArg(self: *Gen, inst: *Inst, v: SVal, pos: front.Pos) Error!u32 {
        return switch (v) {
            .node => |n| n,
            else => self.c.unsupportedAt(inst.func, pos, "a node only known at run time can't be used here", .{}),
        };
    }

    /// rt.loop(body): run the body; false if it broke out.
    fn rtLoop(self: *Gen, body: SVal) Error!SVal {
        const f = &self.f;
        const flag = try f.alloca("i1");
        try f.emit("store i1 true, ptr {s}", .{flag});
        const brk = try f.label("loop_break");
        const cont = try f.label("loop_continue");
        const done = try f.label("loop_done");
        try self.loops.append(self.a(), .{ .brk = brk, .cont = cont, .depth = self.insts.items.len });
        try self.execValue(body);
        _ = self.loops.pop();
        try f.br(done);
        try f.block(brk);
        try f.emit("store i1 false, ptr {s}", .{flag});
        try f.br(done);
        try f.block(cont);
        try f.br(done);
        try f.block(done);
        const v = try f.value("load i1, ptr {s}", .{flag});
        return dyn("1", try f.value("zext i1 {s} to i64", .{v}), .bool);
    }

    /// Call a run-time function value (or a host function) with arguments.
    fn dynCall(self: *Gen, inst: *Inst, fv: SVal, args_v: SVal, receiver: ?SVal) Error!SVal {
        const f = &self.f;
        const items: []const SVal = switch (args_v) {
            .list => |l| l.items.items,
            .tuple => |t| t,
            else => return self.c.unsupported("arguments only known as a list at run time aren't compiled yet (node {d})", .{inst.node}),
        };
        const fd = try self.materialize(fv, inst.node);
        // The arguments, in a stack array
        const n = items.len;
        const arr = try f.alloca(try std.fmt.allocPrint(self.a(), "[{d} x {{ i64, i64 }}]", .{@max(n, 1)}));
        const ds = try self.a().alloc(Dyn, n);
        for (items, 0..) |item, i| {
            ds[i] = try self.materialize(item, inst.node);
            const p = try f.value("getelementptr inbounds {{ i64, i64 }}, ptr {s}, i64 {d}", .{ arr, i });
            try self.storeSlot(p, ds[i]);
        }
        var recv_ptr: []const u8 = "null";
        var recv_d: ?Dyn = null;
        if (receiver) |r| {
            recv_d = try self.materialize(r, inst.node);
            const p = try f.alloca("{ i64, i64 }");
            try self.storeSlot(p, recv_d.?);
            recv_ptr = p;
        }
        const ok = try f.value("call i1 @zr_call(ptr %ctx, i32 {d}, i64 {s}, i64 {s}, ptr {s}, i64 {d}, ptr {s}, ptr {s})", .{ inst.node, fd.tag, fd.bits, arr, n, recv_ptr, self.out });
        // (the call borrowed them)
        for (ds) |d| try self.drop(.{ .dyn = d });
        if (recv_d) |r| try self.drop(.{ .dyn = r });
        try self.drop(.{ .dyn = fd });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.any) };
    }

    /// Calling a Python object known when compiling: a helper function of
    /// the semantics (compiled inline), or a builtin.
    fn pyCall(self: *Gen, inst: *Inst, o: *PyObject, args: []const SVal, pos: front.Pos) Error!SVal {
        const c = self.c;
        // A Python function of the semantics' module (a def: it has code):
        // compiled too
        if (py.c.PyObject_HasAttrString(o, "__code__") == 1 and py.c.PyObject_HasAttrString(o, "__globals__") == 1) {
            const func = try self.helperFunction(o);
            return self.runFunction(func, inst.node, args);
        }
        return c.unsupportedAt(inst.func, pos, "calling this (a builtin or a class) isn't compiled yet", .{});
    }

    fn helperFunction(self: *Gen, o: *PyObject) Error!*const front.Function {
        const c = self.c;
        if (c.helpers_read.get(o)) |f| return f;
        var failure = front.Failure{};
        const f = front.read(std.heap.c_allocator, o, &failure) catch |e| switch (e) {
            error.Unsupported => return c.unsupported("{s}", .{failure.text()}),
            else => |x| return x,
        };
        try c.helpers_read.put(c.a, o, f);
        return f;
    }

    // ------------------------------------------------------------------
    // Operators
    // ------------------------------------------------------------------

    fn binary(self: *Gen, inst: *Inst, op: front.BinOp, l: SVal, r: SVal) Error!SVal {
        // Known both: computed now, as Python does (an error stays an error
        // at run time, where it would happen)
        if (l.isStatic() and r.isStatic() and isScalar(l) and isScalar(r)) {
            if (try self.staticBinary(inst, op, l, r)) |v| return v;
        }
        const f = &self.f;
        const ld = try self.materialize(l, inst.node);
        const rd = try self.materialize(r, inst.node);
        // Ints: inline, checked
        if (ld.shape == .int and rd.shape == .int and (op == .add or op == .sub or op == .mul)) {
            const intrinsic = switch (op) {
                .add => "sadd",
                .sub => "ssub",
                else => "smul",
            };
            const pair = try f.value("call {{ i64, i1 }} @llvm.{s}.with.overflow.i64(i64 {s}, i64 {s})", .{ intrinsic, ld.bits, rd.bits });
            const res = try f.value("extractvalue {{ i64, i1 }} {s}, 0", .{pair});
            const ovf = try f.value("extractvalue {{ i64, i1 }} {s}, 1", .{pair});
            const bad = try f.label("overflow");
            const good = try f.label("no_overflow");
            try f.condBr(ovf, bad, good);
            try f.block(bad);
            _ = try f.value("call i1 @zr_overflow(ptr %ctx, i32 {d})", .{inst.node});
            try f.br(self.err_label);
            try f.block(good);
            return dyn("2", res, .int);
        }
        const ok = try f.value("call i1 @zr_binary(ptr %ctx, i32 {d}, i32 {d}, i64 {s}, i64 {s}, i64 {s}, i64 {s}, ptr {s})", .{ inst.node, @intFromEnum(op), ld.tag, ld.bits, rd.tag, rd.bits, self.out });
        try self.drop(.{ .dyn = ld });
        try self.drop(.{ .dyn = rd });
        try self.check(ok);
        const shape: Shape = if (ld.shape == .int and rd.shape == .int and op != .div and op != .pow)
            .int
        else if ((ld.shape == .float or rd.shape == .float) and (ld.shape == .int or ld.shape == .float) and (rd.shape == .int or rd.shape == .float))
            .float
        else
            .any;
        return .{ .dyn = try self.loadOut(shape) };
    }

    fn isScalar(v: SVal) bool {
        return switch (v) {
            .none, .bool, .int, .float, .str => true,
            .tuple => |t| for (t) |x| {
                if (!isScalar(x)) break false;
            } else true,
            else => false,
        };
    }

    /// A Python object for a known scalar (new reference).
    fn pyOf(self: *Gen, v: SVal) Error!*PyObject {
        return switch (v) {
            .none => blk: {
                py.Py_IncRef(py.Py_None());
                break :blk py.Py_None();
            },
            .bool => |b| blk: {
                const o = if (b) py.Py_True() else py.Py_False();
                py.Py_IncRef(o);
                break :blk o;
            },
            .int => |n| py.c.PyLong_FromLongLong(n) orelse error.Python,
            .float => |x| py.c.PyFloat_FromDouble(x) orelse error.Python,
            .str => |s| ph.newString(s) orelse error.Python,
            .tuple => |t| blk: {
                const out = py.c.PyTuple_New(@intCast(t.len)) orelse return error.Python;
                for (t, 0..) |x, i| _ = py.c.PyTuple_SetItem(out, @intCast(i), try self.pyOf(x));
                break :blk out;
            },
            else => error.Python,
        };
    }

    /// Python's result for known operands, or null (then it's an error:
    /// compiled as one at run time).
    fn staticBinary(self: *Gen, inst: *Inst, op: front.BinOp, l: SVal, r: SVal) Error!?SVal {
        const x = try self.pyOf(l);
        defer py.Py_DecRef(x);
        const y = try self.pyOf(r);
        defer py.Py_DecRef(y);
        const res = switch (op) {
            .add => py.c.PyNumber_Add(x, y),
            .sub => py.c.PyNumber_Subtract(x, y),
            .mul => py.c.PyNumber_Multiply(x, y),
            .div => py.c.PyNumber_TrueDivide(x, y),
            .floordiv => py.c.PyNumber_FloorDivide(x, y),
            .mod => py.c.PyNumber_Remainder(x, y),
            .pow => py.c.PyNumber_Power(x, y, py.Py_None()),
            .lshift => py.c.PyNumber_Lshift(x, y),
            .rshift => py.c.PyNumber_Rshift(x, y),
            .bitor => py.c.PyNumber_Or(x, y),
            .bitxor => py.c.PyNumber_Xor(x, y),
            .bitand => py.c.PyNumber_And(x, y),
        } orelse {
            // (the error happens when this runs: compiled as it)
            py.c.PyErr_Clear();
            return null;
        };
        defer py.Py_DecRef(res);
        return try self.constant(res, inst.node);
    }

    fn compare(self: *Gen, inst: *Inst, op: front.CmpOp, l: SVal, r: SVal) Error!SVal {
        // Known: decided now (None checks on fields are common)
        if (l.isStatic() and r.isStatic()) {
            if (op == .is or op == .is_not) {
                const same = std.meta.activeTag(l) == std.meta.activeTag(r) and switch (l) {
                    .none => true,
                    .bool => |b| b == r.bool,
                    .node => |n| n == r.node,
                    else => false,
                };
                // (anything known that isn't None is not None)
                if (l == .none or r == .none or same) return .{ .bool = if (op == .is) same else !same };
            }
            if (isScalar(l) and isScalar(r)) {
                const x = try self.pyOf(l);
                defer py.Py_DecRef(x);
                const y = try self.pyOf(r);
                defer py.Py_DecRef(y);
                const cmp_op: c_int = switch (op) {
                    .eq => py.c.Py_EQ,
                    .ne => py.c.Py_NE,
                    .lt => py.c.Py_LT,
                    .le => py.c.Py_LE,
                    .gt => py.c.Py_GT,
                    .ge => py.c.Py_GE,
                    else => -1,
                };
                if (cmp_op >= 0) {
                    const res = py.c.PyObject_RichCompareBool(x, y, cmp_op);
                    if (res >= 0) return .{ .bool = res == 1 };
                    py.c.PyErr_Clear();
                }
            }
            if (l == .node and r == .node and (op == .eq or op == .ne)) {
                const same = l.node == r.node;
                return .{ .bool = if (op == .eq) same else !same };
            }
        }
        // A run-time value against None: its tag
        if ((op == .is or op == .is_not) and (l == .none or r == .none)) {
            const other = if (l == .none) r else l;
            if (other.isStatic()) return .{ .bool = (other == .none) == (op == .is) };
            const d = other.dyn;
            const f = &self.f;
            const t = try f.value("icmp {s} i64 {s}, 0", .{ if (op == .is) "eq" else "ne", d.tag });
            try self.drop(other);
            return dyn("1", try f.value("zext i1 {s} to i64", .{t}), .bool);
        }
        const f = &self.f;
        const ld = try self.materialize(l, inst.node);
        const rd = try self.materialize(r, inst.node);
        if (ld.shape == .int and rd.shape == .int) {
            const pred: ?[]const u8 = switch (op) {
                .eq => "eq",
                .ne => "ne",
                .lt => "slt",
                .le => "sle",
                .gt => "sgt",
                .ge => "sge",
                else => null,
            };
            if (pred) |p| {
                const t = try f.value("icmp {s} i64 {s}, {s}", .{ p, ld.bits, rd.bits });
                return dyn("1", try f.value("zext i1 {s} to i64", .{t}), .bool);
            }
        }
        const ok = try f.value("call i1 @zr_compare(ptr %ctx, i32 {d}, i32 {d}, i64 {s}, i64 {s}, i64 {s}, i64 {s}, ptr {s})", .{ inst.node, @intFromEnum(op), ld.tag, ld.bits, rd.tag, rd.bits, self.out });
        try self.drop(.{ .dyn = ld });
        try self.drop(.{ .dyn = rd });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.bool) };
    }

    fn unary(self: *Gen, inst: *Inst, op: front.UnaryOp, v: SVal) Error!SVal {
        if (op == .not_) {
            return switch (try self.truth(v, inst.node)) {
                .known => |b| .{ .bool = !b },
                .dyn => |t| blk: {
                    const n = try self.f.value("xor i1 {s}, true", .{t});
                    break :blk dyn("1", try self.f.value("zext i1 {s} to i64", .{n}), .bool);
                },
            };
        }
        if (v.isStatic() and isScalar(v)) {
            const x = try self.pyOf(v);
            defer py.Py_DecRef(x);
            const res = switch (op) {
                .neg => py.c.PyNumber_Negative(x),
                .pos => py.c.PyNumber_Positive(x),
                else => py.c.PyNumber_Invert(x),
            };
            if (res) |r| {
                defer py.Py_DecRef(r);
                return self.constant(r, inst.node);
            }
            py.c.PyErr_Clear();
        }
        const f = &self.f;
        const d = try self.materialize(v, inst.node);
        const ok = try f.value("call i1 @zr_unary(ptr %ctx, i32 {d}, i32 {d}, i64 {s}, i64 {s}, ptr {s})", .{ inst.node, @intFromEnum(op), d.tag, d.bits, self.out });
        try self.drop(.{ .dyn = d });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(if (d.shape == .int or d.shape == .float) d.shape else .any) };
    }

    /// `a and b` / `a or b`: the deciding operand's value (short-circuit).
    fn boolOp(self: *Gen, inst: *Inst, is_and: bool, items: []const *const front.Expr) Error!SVal {
        var v = try self.expr(inst, items[0]);
        for (items[1..]) |next_e| {
            // Known: decided now
            if (v.isStatic()) {
                const t = (try self.truth(v, inst.node)).known;
                if (t != is_and) return v;
                v = try self.expr(inst, next_e);
                continue;
            }
            // At run time: the value, or the next one's
            const f = &self.f;
            const slot = try f.alloca("{ i64, i64 }");
            try self.storeSlot(slot, v.dyn);
            try self.increfDyn(v.dyn);
            const t = (try self.truth(v, inst.node)).dyn;
            const take_next = try f.label("boolop_next");
            const join = try f.label("boolop_end");
            if (is_and) try f.condBr(t, take_next, join) else try f.condBr(t, join, take_next);
            try f.block(take_next);
            // (the first value is dropped on the path taking the next)
            const first = try self.loadSlot(slot, v.dyn.shape);
            try self.drop(.{ .dyn = first });
            const n = try self.materialize(try self.expr(inst, next_e), inst.node);
            try self.storeSlot(slot, n);
            try f.br(join);
            try f.block(join);
            v = .{ .dyn = try self.loadSlot(slot, if (n.shape == v.dyn.shape) n.shape else .any) };
        }
        return v;
    }

    fn andValues(self: *Gen, inst: *Inst, a_: SVal, b: SVal) Error!SVal {
        if (a_ == .bool and a_.bool) return b;
        if (a_ == .bool and !a_.bool) {
            try self.drop(b);
            return a_;
        }
        const x = (try self.truth(a_, inst.node));
        const y = (try self.truth(b, inst.node));
        const xa = switch (x) {
            .known => |k| if (k) "true" else "false",
            .dyn => |d| d,
        };
        const yb = switch (y) {
            .known => |k| if (k) "true" else "false",
            .dyn => |d| d,
        };
        const r = try self.f.value("and i1 {s}, {s}", .{ xa, yb });
        return dyn("1", try self.f.value("zext i1 {s} to i64", .{r}), .bool);
    }

    /// `x if cond else y` with a run-time cond.
    fn branchValue(self: *Gen, inst: *Inst, cond: []const u8, then_e: *const front.Expr, else_e: *const front.Expr) Error!SVal {
        const f = &self.f;
        const slot = try f.alloca("{ i64, i64 }");
        const yes = try f.label("ifexp_then");
        const no = try f.label("ifexp_else");
        const join = try f.label("ifexp_end");
        try f.condBr(cond, yes, no);
        inst.dyn_depth += 1;
        try f.block(yes);
        const a_ = try self.materialize(try self.expr(inst, then_e), inst.node);
        try self.storeSlot(slot, a_);
        try f.br(join);
        try f.block(no);
        const b = try self.materialize(try self.expr(inst, else_e), inst.node);
        try self.storeSlot(slot, b);
        try f.br(join);
        inst.dyn_depth -= 1;
        try f.block(join);
        return .{ .dyn = try self.loadSlot(slot, if (a_.shape == b.shape) a_.shape else .any) };
    }

    fn listComp(self: *Gen, inst: *Inst, comp: front.Comp, pos: front.Pos) Error!SVal {
        // Known iterables: the list built now (its items may be run-time)
        const out = try self.a().create(SList);
        out.* = .{};
        try self.compLevel(inst, comp, 0, out, pos);
        return .{ .list = out };
    }

    fn compLevel(self: *Gen, inst: *Inst, comp: front.Comp, level: usize, out: *SList, pos: front.Pos) Error!void {
        if (level == comp.generators.len) {
            try out.items.append(self.a(), try self.expr(inst, comp.elt));
            return;
        }
        const g = comp.generators[level];
        const it = try self.expr(inst, g.iter);
        const items: []const SVal = switch (it) {
            .list => |l| l.items.items,
            .tuple => |t| t,
            else => return self.c.unsupportedAt(inst.func, pos, "a comprehension over a value only known at run time isn't compiled yet", .{}),
        };
        for (items) |item| {
            if (item == .dyn) try self.increfDyn(item.dyn);
            try self.assign(inst, g.target, item, pos);
            var keep = true;
            for (g.ifs) |cond_e| {
                switch (try self.truth(try self.expr(inst, cond_e), inst.node)) {
                    .known => |b| if (!b) {
                        keep = false;
                        break;
                    },
                    .dyn => return self.c.unsupportedAt(inst.func, pos, "a comprehension filtering on a run-time value isn't compiled yet", .{}),
                }
            }
            if (keep) try self.compLevel(inst, comp, level + 1, out, pos);
        }
    }
};

/// The locals a statement list assigns (or mutates through), anywhere in it.
fn collectAssigned(body: []const front.Stmt, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    for (body) |s| switch (s.kind) {
        .assign => |x| for (x.targets) |t| try collectTarget(t, set, a),
        .aug => |x| try collectTarget(x.target, set, a),
        .for_ => |x| {
            try collectTarget(x.target, set, a);
            try collectAssigned(x.body, set, a);
            try collectAssigned(x.else_, set, a);
        },
        .if_ => |x| {
            try collectAssigned(x.body, set, a);
            try collectAssigned(x.else_, set, a);
        },
        .while_ => |x| {
            try collectAssigned(x.body, set, a);
            try collectAssigned(x.else_, set, a);
        },
        .expr => |e| try collectMutated(e, set, a),
        else => {},
    };
}

fn collectTarget(t: front.Target, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    switch (t) {
        .local => |slot| try set.put(a, slot, {}),
        .tuple => |ts| for (ts) |x| try collectTarget(x, set, a),
        .attr => |x| if (x.obj.kind == .local) try set.put(a, x.obj.kind.local, {}),
        .index => |x| if (x.obj.kind == .local) try set.put(a, x.obj.kind.local, {}),
    }
}

/// `x.append(...)`: x is mutated (a list's or dict's mutating methods).
fn collectMutated(e: *const front.Expr, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    if (e.kind != .call) return;
    const func = e.kind.call.func;
    if (func.kind != .attr or func.kind.attr.obj.kind != .local) return;
    const mutating = [_][]const u8{ "append", "extend", "insert", "pop", "remove", "clear", "update", "setdefault", "sort", "reverse", "popitem" };
    for (mutating) |m| {
        if (std.mem.eql(u8, func.kind.attr.name, m)) {
            try set.put(a, func.kind.attr.obj.kind.local, {});
            return;
        }
    }
}
