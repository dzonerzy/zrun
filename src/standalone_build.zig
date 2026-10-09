//! A standalone build's compiling (lang.build_native()): a strict program
//! compiled ahead of time, all of it, into objects linked with zrun's
//! runtime (rt.zig) and no Python. What the JIT compiles as the program
//! runs is compiled here first: the thunks of nodes evaluated by code that
//! only knows them at run time, for every node they can be; the Python
//! functions held as values (a language's library), for every way the code
//! calls one it only knows at run time. No tiers (the code is the
//! optimized code). What the code refers to by address is in the image
//! (image.zig): the module made last, with the tables the runtime finds
//! the code by and `main` (standalone.zig's Desc).

const std = @import("std");
const image = @import("image.zig");
const ir = @import("ir.zig");
const llvm = @import("jit.zig");
const value = @import("value.zig");
const set_mod = @import("set.zig");
const front = @import("front.zig");
const gc = @import("gc.zig");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;

const L = llvm.f;
const c = llvm.c;
const Allocator = std.mem.Allocator;
const Value = value.Value;
const Tag = value.Tag;

/// What compiling notes for a standalone build (Compiler.aot)
pub const Aot = struct {
    /// What each address the code takes is (ir.Module.notes)
    notes: std.AutoHashMapUnmanaged(usize, image.Note) = .empty,
    /// Python functions called where they're known, by the code made for
    /// the call (a helper of the module calling it): its name
    called: std.AutoHashMapUnmanaged(CalledKey, [:0]const u8) = .empty,
    /// How code calls functions it only knows at run time: the arguments'
    /// count, which are rt values
    shapes: std.AutoArrayHashMapUnmanaged(Shape, void) = .empty,
    /// Where code evaluates nodes it only knows at run time: eval or exec,
    /// the scope it runs in
    run_sites: std.AutoArrayHashMapUnmanaged(RunSite, void) = .empty,

    pub const CalledKey = struct { func: usize, nargs: usize };
    pub const Shape = struct { nargs: usize, rt_mask: u64 };
    pub const RunSite = struct { which: u32, owner: u32 };

    pub fn noteShape(self: *Aot, a: Allocator, nargs: usize, rt_mask: u64) !void {
        try self.shapes.put(a, .{ .nargs = nargs, .rt_mask = rt_mask }, {});
    }

    pub fn noteRunSite(self: *Aot, a: Allocator, which: u32, owner: u32) !void {
        try self.run_sites.put(a, .{ .which = which, .owner = owner }, {});
    }
};

/// The Python objects held in the values the code refers to (a library's
/// functions in a table), each once, after those already in `out`
pub fn heldObjects(a: Allocator, notes: *const std.AutoHashMapUnmanaged(usize, image.Note), out: *std.ArrayListUnmanaged(*PyObject)) !void {
    var seen: std.AutoHashMapUnmanaged(usize, void) = .empty;
    defer seen.deinit(a);
    for (out.items) |o| try seen.put(a, @intFromPtr(o), {});
    var visited: std.AutoHashMapUnmanaged(usize, void) = .empty;
    defer visited.deinit(a);
    var it = notes.iterator();
    while (it.next()) |e| switch (e.value_ptr.*) {
        .value => |tag| try heldIn(a, .{ .tag = tag, .bits = e.key_ptr.* }, &visited, &seen, out),
        else => {},
    };
}

fn heldIn(a: Allocator, v: Value, visited: *std.AutoHashMapUnmanaged(usize, void), seen: *std.AutoHashMapUnmanaged(usize, void), out: *std.ArrayListUnmanaged(*PyObject)) !void {
    switch (v.kind()) {
        .host => {
            if ((try seen.getOrPut(a, v.bits)).found_existing) return;
            try out.append(a, @ptrFromInt(v.bits));
        },
        .list, .tuple, .dict, .set, .record => {
            if ((try visited.getOrPut(a, v.bits)).found_existing) return;
            switch (v.kind()) {
                .list => for (@as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice()) |x| try heldIn(a, x, visited, seen, out),
                .tuple => for (@as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice()) |x| try heldIn(a, x, visited, seen, out),
                .record => for (@as(*value.Record, @ptrCast(@alignCast(v.ptr()))).fields()) |x| try heldIn(a, x, visited, seen, out),
                .dict => {
                    const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
                    if (d.entries) |entries| for (entries[0..d.used]) |en| {
                        if (en.key.tag == value.DELETED) continue;
                        try heldIn(a, en.key, visited, seen, out);
                        try heldIn(a, en.value, visited, seen, out);
                    };
                },
                else => {
                    var items = set_mod.iterate(@ptrCast(@alignCast(v.ptr())));
                    while (items.next()) |x| try heldIn(a, x, visited, seen, out);
                },
            }
        },
        else => {},
    }
}

/// A thunk compiled ahead: for a node's eval or exec in a scope's frames
pub const ThunkMade = struct { node: u32, which: u32, owner: u32, name: [:0]const u8 };
/// A Python function's code compiled ahead, for calls of its shape
pub const CalledMade = struct { func: usize, nargs: usize, rt_mask: u64, name: [:0]const u8 };

pub const Error = error{ OutOfMemory, Compile, Unknown };

/// The objects of a build, as they're made
pub const Build = struct {
    arena: std.heap.ArenaAllocator,
    view: *const llvm.LlvmView,
    opt: u32,
    /// The objects compiled (the image's last)
    objects: std.ArrayListUnmanaged([]u8) = .empty,
    /// Every module's names for addresses
    syms: std.ArrayListUnmanaged(ir.Module.Sym) = .empty,
    thunks: std.ArrayListUnmanaged(ThunkMade) = .empty,
    called: std.ArrayListUnmanaged(CalledMade) = .empty,
    /// Why it failed (Error.Compile, Unknown)
    why: std.ArrayListUnmanaged(u8) = .empty,

    pub fn init(view: *const llvm.LlvmView, opt: u32) Build {
        return .{ .arena = std.heap.ArenaAllocator.init(std.heap.c_allocator), .view = view, .opt = opt };
    }

    pub fn deinit(self: *Build) void {
        self.arena.deinit();
    }

    fn a(self: *Build) Allocator {
        return self.arena.allocator();
    }

    /// A module compiled to an object (the module taken), its names kept.
    pub fn emit(self: *Build, m: *ir.Module) Error!void {
        for (m.syms.items) |s| try self.syms.append(self.a(), .{ .name = try self.a().dupeZ(u8, s.name), .addr = s.addr });
        var err: [2048]u8 = @splat(0);
        const bytes = llvm.emitObject(self.view, m.take(), self.opt, &err) catch {
            try self.why.print(self.a(), "LLVM rejected the compiled program (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
            return error.Compile;
        };
        defer llvm.freeBytes(self.view, bytes);
        try self.objects.append(self.a(), try self.a().dupe(u8, bytes));
    }

    fn fail(self: *Build, comptime fmt: []const u8, args: anytype) Error {
        self.why.clearRetainingCapacity();
        try self.why.print(self.a(), fmt, args);
        return error.Unknown;
    }

    // ------------------------------------------------------------------
    // The image's items
    // ------------------------------------------------------------------

    const Items = struct {
        img: image.Image,
        notes: *const std.AutoHashMapUnmanaged(usize, image.Note),
        /// Built items, in the order they're built (what one refers to
        /// before it, but for a cycle)
        order: std.ArrayListUnmanaged(u32) = .empty,
    };

    fn item(self: *Build, it: *Items, addr: usize, note: image.Note) Error!u32 {
        if (it.img.indexOf(addr)) |i| return i;
        const i = try it.img.reserve(addr);
        switch (note) {
            .str => it.img.set(i, .{ .static = try image.strBytes(self.a(), @ptrFromInt(addr)) }),
            .big => it.img.set(i, .{ .static = try image.bigBytes(self.a(), @ptrFromInt(addr)) }),
            .value => |tag| {
                var w = image.Writer{ .a = self.a() };
                const size = try self.describeObject(it, &w, .{ .tag = tag, .bits = addr });
                it.img.set(i, .{ .built = .{ .kind = .value_graph, .size = size, .desc = w.out.items } });
                try it.order.append(self.a(), i);
            },
            .rtype => {
                const t: *const value.RecordType = @ptrFromInt(addr);
                const base: i32 = if (t.base) |b| @intCast(try self.item(it, @intFromPtr(b), .rtype)) else -1;
                var w = image.Writer{ .a = self.a() };
                try w.bytes(t.name);
                try w.bytes(t.unset_name);
                try w.int(u32, @intCast(t.fields.len));
                for (t.fields) |f| try w.bytes(f);
                try w.int(u8, @as(u8, @intFromBool(t.value_eq)) | @as(u8, @intFromBool(t.slots)) << 1 | @as(u8, @intFromBool(t.frozen)) << 2);
                try w.int(i32, base);
                it.img.set(i, .{ .built = .{ .kind = .rtype, .size = @sizeOf(value.RecordType), .desc = w.out.items } });
                try it.order.append(self.a(), i);
            },
            .function => {
                const f: *const front.Function = @ptrFromInt(addr);
                var w = image.Writer{ .a = self.a() };
                try w.bytes(f.name);
                try w.bytes(f.qualname);
                try w.int(u32, f.own);
                try w.int(u32, f.param_count);
                try w.int(u32, @intCast(f.locals.len));
                for (f.locals) |l| try w.bytes(l);
                it.img.set(i, .{ .built = .{ .kind = .function, .size = @sizeOf(front.Function), .desc = w.out.items } });
                try it.order.append(self.a(), i);
            },
            .words => |t| {
                const src: [*]const u64 = @ptrFromInt(addr);
                const words = try self.a().dupe(u64, src[0..t.len]);
                const refs = try self.a().alloc(?u32, t.len);
                for (words, refs) |x, *r| r.* = if (t.strs) try self.item(it, x, .str) else null;
                it.img.set(i, .{ .words = .{ .words = words, .refs = refs } });
            },
            .code => |name| it.img.set(i, .{ .code = name }),
            .pyobj => {
                // (a stand-in: immortal, its name (standalone.Standin))
                const name = pyName(@ptrFromInt(addr));
                const name_item: u32 = @intCast(it.img.entries.items.len);
                try it.img.entries.append(self.a(), .{ .static = try self.a().dupe(u8, name) });
                const words = try self.a().dupe(u64, &.{ 1 << 60, 0, 0, name.len });
                const refs = try self.a().dupe(?u32, &.{ null, null, name_item, null });
                it.img.set(i, .{ .words = .{ .words = words, .refs = refs } });
            },
        }
        return i;
    }

    fn pyName(o: *PyObject) []const u8 {
        inline for (.{ "__qualname__", "__name__" }) |attr| {
            if (ph.attr(o, attr)) |n| {
                defer py.Py_DecRef(n);
                if (py.PyUnicode_Check(n)) if (ph.utf8(n, "name")) |s| return s;
            }
            py.c.PyErr_Clear();
        }
        return "object";
    }

    /// A value's object described, as zr_image_build builds it: its room
    fn describeObject(self: *Build, it: *Items, w: *image.Writer, v: Value) Error!usize {
        switch (v.kind()) {
            .list => {
                const l: *value.List = @ptrCast(@alignCast(v.ptr()));
                try w.tag(.list);
                try w.int(u64, l.len);
                for (l.slice()) |x| try self.child(it, w, x);
                return value.list_block;
            },
            .tuple => {
                const t: *value.Tuple = @ptrCast(@alignCast(v.ptr()));
                try w.tag(.tuple);
                try w.int(u64, t.len);
                for (t.slice()) |x| try self.child(it, w, x);
                return @sizeOf(value.Tuple) + t.len * @sizeOf(Value);
            },
            .dict => {
                const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
                try w.tag(.dict);
                try w.int(u64, d.len);
                if (d.entries) |entries| for (entries[0..d.used]) |e| {
                    if (e.key.tag == value.DELETED) continue;
                    try self.child(it, w, e.key);
                    try self.child(it, w, e.value);
                };
                return @sizeOf(value.Dict);
            },
            .set => {
                const s: *set_mod.Set = @ptrCast(@alignCast(v.ptr()));
                var n: u64 = 0;
                var count = set_mod.iterate(s);
                while (count.next()) |_| n += 1;
                try w.tag(.set);
                try w.int(u64, n);
                var items = set_mod.iterate(s);
                while (items.next()) |x| try self.child(it, w, x);
                return @sizeOf(set_mod.Set);
            },
            .record => {
                const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
                const t = try self.item(it, @intFromPtr(r.rtype), .rtype);
                try w.tag(.record);
                try w.int(u32, t);
                try w.int(u64, r.rtype.fields.len);
                for (r.fields()) |x| try self.child(it, w, x);
                return @sizeOf(value.Record) + r.rtype.fields.len * @sizeOf(Value);
            },
            .bytes => {
                const b: *value.Bytes = @ptrCast(@alignCast(v.ptr()));
                try w.tag(.bytes);
                try w.bytes(b.slice());
                return @sizeOf(value.Bytes);
            },
            else => return self.fail("a {s} as a constant of the program can't be in a standalone program", .{value.typeName(v)}),
        }
    }

    /// A value inside an object described: itself, or (an object) a
    /// reference to its item
    fn child(self: *Build, it: *Items, w: *image.Writer, v: Value) Error!void {
        if (v.tag == value.PINT_TAG) {
            try w.tag(.pint);
            return w.int(i64, @bitCast(v.bits));
        }
        if (v.tag == value.UNSET_TAG) return w.tag(.unset);
        const note: image.Note = switch (v.kind()) {
            .none => return w.tag(.none),
            .bool => return w.tag(if (v.bits != 0) .true_ else .false_),
            .int => {
                try w.tag(.int);
                return w.int(i64, @bitCast(v.bits));
            },
            .float => {
                try w.tag(.float);
                return w.int(u64, v.bits);
            },
            .node => {
                try w.tag(.node);
                return w.int(u32, @intCast(v.bits));
            },
            .str => .str,
            .big => .big,
            .host => .pyobj,
            .list, .tuple, .dict, .set, .record, .bytes => .{ .value = v.tag },
            else => return self.fail("a {s} as a constant of the program can't be in a standalone program", .{value.typeName(v)}),
        };
        const i = try self.item(it, v.bits, note);
        try w.tag(.ref);
        try w.int(u64, v.tag);
        try w.int(u32, i);
    }

    // ------------------------------------------------------------------
    // The image's module
    // ------------------------------------------------------------------

    /// What the image says of the program besides its items
    pub const Program = struct {
        main_name: [:0]const u8,
        globals: u64,
        max_depth: u64,
        objects: []const *PyObject,
        path: []const u8,
        source: []const u8,
        /// The tree (tree.FlatNode each), each node's parent, the grammar
        /// described (describeGrammar)
        nodes: []const u8,
        nodes_len: usize,
        parents: []const u32,
        grammar: []const u8,
        /// By node: its symbol, its owner; the symbols (standalone.Desc)
        sym_of: []const u32,
        owners: []const u32,
        syms: []const @import("standalone.zig").SymInfo,
    };

    /// The grammar, as the standalone runtime reads nodes with it
    /// (standalone.makeData reads it)
    pub fn describeGrammar(self: *Build, g: *const @import("grammar.zig").Grammar) Error![]const u8 {
        var w = image.Writer{ .a = self.a() };
        try w.int(u32, @intCast(g.rule_names.len));
        for (g.rule_names, 0..) |name, i| {
            try w.bytes(name);
            try w.bytes(if (i < g.kind_names.len) g.kind_names[i] else name);
            try w.int(u8, if (i < g.actions.len) @intFromEnum(g.actions[i]) else 0);
            const labels = if (i < g.labels.len) g.labels[i] else &.{};
            try w.int(u32, @intCast(labels.len));
            for (labels) |l| {
                try w.int(u8, l.field);
                try w.int(u8, @intFromBool(l.many));
            }
        }
        try w.int(u32, @intCast(g.field_names.len));
        for (g.field_names) |f| try w.bytes(f);
        return w.out.items;
    }

    /// The image (its items, the tables, `main`) compiled: the last object.
    pub fn emitImage(self: *Build, notes: *const std.AutoHashMapUnmanaged(usize, image.Note), p: Program) Error!void {
        var it = Items{ .img = .{ .a = self.a() }, .notes = notes };
        // The items: what the code names, the objects it holds
        const sym_items = try self.a().alloc(u32, self.syms.items.len);
        for (self.syms.items, sym_items) |s, *slot| {
            const note = notes.get(s.addr) orelse return self.fail("the compiled code refers to something a standalone program can't have (0x{x}, {s})", .{ s.addr, s.name });
            slot.* = try self.item(&it, s.addr, note);
        }
        const object_items = try self.a().alloc(u32, p.objects.len);
        for (p.objects, object_items) |o, *slot| slot.* = try self.item(&it, @intFromPtr(o), .pyobj);
        const called_items = try self.a().alloc(u32, self.called.items.len);
        for (self.called.items, called_items) |e, *slot| slot.* = try self.item(&it, e.func, .pyobj);

        var m = ir.Module.init(self.a(), "zr_img");
        defer m.deinit();
        const t = m.t;
        const ctx = m.ctx;
        const entries = it.img.entries.items;
        const head = gc.head_size;

        // (every item's global first: items refer to each other)
        const globals = try self.a().alloc(ir.Value, entries.len);
        const descs = try self.a().alloc(ir.Value, entries.len);
        var fns: std.StringHashMapUnmanaged(ir.Value) = .empty;
        const void_fn = L("LLVMFunctionType")(t.void, null, 0, 0);
        for (entries, globals, 0..) |e, *g, i| {
            var name_buf: [32]u8 = undefined;
            const name = std.fmt.bufPrintZ(&name_buf, "zr_img_{d}", .{i}) catch unreachable;
            const ty = switch (e) {
                .static => |s| L("LLVMArrayType2")(t.i8, head + s.len),
                .words => |w| L("LLVMArrayType2")(t.i64, head / 8 + w.words.len),
                .code => L("LLVMArrayType2")(t.i64, head / 8 + 1),
                .built => |b| L("LLVMArrayType2")(t.i8, head + b.size),
            };
            g.* = L("LLVMAddGlobal")(m.mod, ty, name.ptr);
            L("LLVMSetLinkage")(g.*, c.LLVMInternalLinkage);
            L("LLVMSetAlignment")(g.*, 16);
        }
        for (entries, globals, descs) |e, g, *d| {
            d.* = null;
            switch (e) {
                .static => |s| {
                    const bytes = try self.a().alloc(u8, head + s.len);
                    @memset(bytes[0..head], 0);
                    @memcpy(bytes[head..], s);
                    L("LLVMSetInitializer")(g, L("LLVMConstStringInContext2")(ctx, bytes.ptr, bytes.len, 1));
                },
                .words => |w| {
                    const vals = try self.a().alloc(ir.Value, head / 8 + w.words.len);
                    for (vals[0 .. head / 8]) |*x| x.* = ir.Module.kInt(t.i64, 0);
                    for (w.words, w.refs, vals[head / 8 ..]) |x, r, *slot| {
                        slot.* = if (r) |ri| L("LLVMConstPtrToInt")(at(&m, globals[ri]), t.i64) else ir.Module.kInt(t.i64, x);
                    }
                    L("LLVMSetInitializer")(g, L("LLVMConstArray2")(t.i64, vals.ptr, vals.len));
                },
                .code => |name| {
                    const f = fns.get(name) orelse blk: {
                        const fv = L("LLVMAddFunction")(m.mod, name.ptr, void_fn);
                        try fns.put(self.a(), name, fv);
                        break :blk fv;
                    };
                    var vals = [_]ir.Value{ ir.Module.kInt(t.i64, 0), ir.Module.kInt(t.i64, 0), L("LLVMConstPtrToInt")(f, t.i64) };
                    L("LLVMSetInitializer")(g, L("LLVMConstArray2")(t.i64, &vals, vals.len));
                },
                .built => |b| {
                    L("LLVMSetInitializer")(g, L("LLVMConstNull")(L("LLVMArrayType2")(t.i8, head + b.size)));
                    d.* = try self.constBytes(&m, b.desc);
                },
            }
        }
        // The code's names for them: aliases of their objects (past the
        // header)
        const add_alias: *const fn (c.LLVMModuleRef, ir.Type, c_uint, ir.Value, [*:0]const u8) callconv(.c) ir.Value = @ptrCast(llvm.optional("LLVMAddAlias2") orelse return self.fail("this zgram's LLVM can't make standalone programs (no LLVMAddAlias2): a newer zgram can", .{}));
        for (self.syms.items, sym_items) |s, i| _ = add_alias(m.mod, t.i8, 0, at(&m, globals[i]), s.name.ptr);

        // Every item's address, by index (zr_image_build's table)
        const table = try self.ptrTable(&m, "zr_img_table", globals, null);
        // Building the items, as the program starts
        const init_fn = L("LLVMAddFunction")(m.mod, "zr_img_init", void_fn);
        L("LLVMSetLinkage")(init_fn, c.LLVMInternalLinkage);
        {
            var build_params = [_]ir.Type{ t.i32, t.ptr, t.ptr, t.i64, t.ptr };
            const build_ty = L("LLVMFunctionType")(t.void, &build_params, build_params.len, 0);
            const build_fn = L("LLVMAddFunction")(m.mod, "zr_image_build", build_ty);
            const b = L("LLVMCreateBuilderInContext")(ctx);
            defer L("LLVMDisposeBuilder")(b);
            L("LLVMPositionBuilderAtEnd")(b, L("LLVMAppendBasicBlockInContext")(ctx, init_fn, "entry"));
            for (it.order.items) |i| {
                const e = entries[i].built;
                var args = [_]ir.Value{ ir.Module.kInt(t.i32, @intFromEnum(e.kind)), at(&m, globals[i]), descs[i], ir.Module.kInt(t.i64, e.desc.len), table };
                _ = L("LLVMBuildCall2")(b, build_ty, build_fn, &args, args.len, "");
            }
            _ = L("LLVMBuildRetVoid")(b);
        }

        // The tables
        const objects = try self.ptrTable(&m, "zr_img_objects", globals, object_items);
        const thunk_vals = try self.a().alloc(ir.Value, self.thunks.items.len);
        for (self.thunks.items, thunk_vals) |e, *slot| {
            var fields = [_]ir.Value{ ir.Module.kInt(t.i64, e.node), ir.Module.kInt(t.i64, e.which), ir.Module.kInt(t.i64, e.owner), try self.codeRef(&m, &fns, e.name, void_fn) };
            slot.* = L("LLVMConstStructInContext")(ctx, &fields, fields.len, 0);
        }
        const called_vals = try self.a().alloc(ir.Value, self.called.items.len);
        for (self.called.items, called_items, called_vals) |e, fi, *slot| {
            var fields = [_]ir.Value{ at(&m, globals[fi]), ir.Module.kInt(t.i64, e.nargs), ir.Module.kInt(t.i64, e.rt_mask), try self.codeRef(&m, &fns, e.name, void_fn) };
            slot.* = L("LLVMConstStructInContext")(ctx, &fields, fields.len, 0);
        }
        var entry_fields = [_]ir.Type{ t.i64, t.i64, t.i64, t.ptr };
        const entry_ty = L("LLVMStructTypeInContext")(ctx, &entry_fields, entry_fields.len, 0);
        var called_fields = [_]ir.Type{ t.ptr, t.i64, t.i64, t.ptr };
        const called_ty = L("LLVMStructTypeInContext")(ctx, &called_fields, called_fields.len, 0);
        const thunks = try self.constGlobal(&m, "zr_img_thunks", L("LLVMConstArray2")(entry_ty, thunk_vals.ptr, thunk_vals.len));
        const called = try self.constGlobal(&m, "zr_img_called", L("LLVMConstArray2")(called_ty, called_vals.ptr, called_vals.len));

        // The program, and main
        const main_fn = try self.codeRef(&m, &fns, p.main_name, void_fn);
        var desc_fields = [_]ir.Value{
            main_fn,                                      ir.Module.kInt(t.i64, p.globals),
            ir.Module.kInt(t.i64, p.max_depth),           objects,
            ir.Module.kInt(t.i64, p.objects.len),         thunks,
            ir.Module.kInt(t.i64, self.thunks.items.len), called,
            ir.Module.kInt(t.i64, self.called.items.len), try self.constBytes(&m, p.path),
            ir.Module.kInt(t.i64, p.path.len),            try self.constBytes(&m, p.source),
            ir.Module.kInt(t.i64, p.source.len),          try self.constBytes(&m, p.nodes),
            ir.Module.kInt(t.i64, p.nodes_len),           try self.constBytes(&m, std.mem.sliceAsBytes(p.parents)),
            try self.constBytes(&m, p.grammar),           ir.Module.kInt(t.i64, p.grammar.len),
            try self.constBytes(&m, std.mem.sliceAsBytes(p.sym_of)), try self.constBytes(&m, std.mem.sliceAsBytes(p.owners)),
            try self.constBytes(&m, std.mem.sliceAsBytes(p.syms)), ir.Module.kInt(t.i64, p.syms.len),
            init_fn,
        };
        const desc = try self.constGlobal(&m, "zr_img_program", L("LLVMConstStructInContext")(ctx, &desc_fields, desc_fields.len, 0));
        {
            var rt_params = [_]ir.Type{ t.ptr, t.i32, t.ptr };
            const rt_ty = L("LLVMFunctionType")(t.i32, &rt_params, rt_params.len, 0);
            const rt_main = L("LLVMAddFunction")(m.mod, "zr_rt_main", rt_ty);
            var main_params = [_]ir.Type{ t.i32, t.ptr };
            const main_ty = L("LLVMFunctionType")(t.i32, &main_params, main_params.len, 0);
            const main = L("LLVMAddFunction")(m.mod, "main", main_ty);
            const b = L("LLVMCreateBuilderInContext")(ctx);
            defer L("LLVMDisposeBuilder")(b);
            L("LLVMPositionBuilderAtEnd")(b, L("LLVMAppendBasicBlockInContext")(ctx, main, "entry"));
            var args = [_]ir.Value{ desc, L("LLVMGetParam")(main, 0), L("LLVMGetParam")(main, 1) };
            _ = L("LLVMBuildRet")(b, L("LLVMBuildCall2")(b, rt_ty, rt_main, &args, args.len, ""));
        }
        try self.emit(&m);
    }

    /// An item's object: its address past the header
    fn at(m: *ir.Module, g: ir.Value) ir.Value {
        var idx = [_]ir.Value{ir.Module.kInt(m.t.i64, gc.head_size)};
        return L("LLVMConstInBoundsGEP2")(m.t.i8, g, &idx, 1);
    }

    /// A function of another object, by name
    fn codeRef(self: *Build, m: *ir.Module, fns: *std.StringHashMapUnmanaged(ir.Value), name: [:0]const u8, ty: ir.Type) Error!ir.Value {
        if (fns.get(name)) |f| return f;
        const f = L("LLVMAddFunction")(m.mod, name.ptr, ty);
        try fns.put(self.a(), name, f);
        return f;
    }

    fn constGlobal(self: *Build, m: *ir.Module, name: [:0]const u8, initial: ir.Value) Error!ir.Value {
        _ = self;
        const g = L("LLVMAddGlobal")(m.mod, L("LLVMTypeOf")(initial), name.ptr);
        L("LLVMSetInitializer")(g, initial);
        L("LLVMSetGlobalConstant")(g, 1);
        L("LLVMSetLinkage")(g, c.LLVMInternalLinkage);
        L("LLVMSetAlignment")(g, 16);
        return g;
    }

    fn constBytes(self: *Build, m: *ir.Module, bytes: []const u8) Error!ir.Value {
        return self.constGlobal(m, "zr_img_bytes", L("LLVMConstStringInContext2")(m.ctx, bytes.ptr, bytes.len, 1));
    }

    /// A table of items' addresses (`which`: those items; null: all)
    fn ptrTable(self: *Build, m: *ir.Module, name: [:0]const u8, globals: []const ir.Value, which: ?[]const u32) Error!ir.Value {
        const n = if (which) |w| w.len else globals.len;
        const vals = try self.a().alloc(ir.Value, n);
        for (vals, 0..) |*slot, i| slot.* = at(m, globals[if (which) |w| w[i] else i]);
        return self.constGlobal(m, name, L("LLVMConstArray2")(m.t.ptr, vals.ptr, vals.len));
    }
};
