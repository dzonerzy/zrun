//! Compiled programs: compile.zig's IR compiled by zgram's LLVM, run with
//! an execution context.
//!
//! A semantic the compiler can't compile is run as Python instead (the
//! compiler says which, the program is compiled again); the code a Python
//! semantic runs through its rt (a node's eval or exec) is compiled when
//! first asked for, as a thunk, in a module of its own.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const compile_mod = @import("compile.zig");
const helpers = @import("helpers.zig");
const value = @import("value.zig");
// (not named llvm.zig: Zig names functions after their file, and LLVM
// reserves every name starting with "llvm.")
const llvm = @import("jit.zig");
const jit_f = llvm.f;
const cache = @import("cache.zig");
const program_mod = @import("program.zig");
const grammar_mod = @import("grammar.zig");

const allocator = std.heap.c_allocator;

pub const Main = *const fn (ctx: *helpers.Ctx, globals: *value.Frame) callconv(.c) bool;

/// A thunk's code: a node's eval or exec in the frames of `frame`
pub const Thunk = *const fn (ctx: *helpers.Ctx, frame: *value.Frame, out: *value.Value) callconv(.c) i32;

/// A helper's code out of line (compile.zig's genHelper): a status as a
/// thunk's
pub const Helper = *const fn (ctx: *helpers.Ctx, frame: ?*value.Frame, args: [*]const value.Value, at: u32, owner: u32, receiver: ?*const value.Value, varargs: ?*const value.Value, out: *value.Value) callconv(.c) i32;

/// The semantics run as Python, by their function (a language's set): why
pub const PythonSet = compile_mod.PythonSet;

/// Mark a semantic run as Python, with the compiler's reason.
fn markPython(set: *PythonSet, s: *PyObject, reason: []const u8) bool {
    const r = allocator.dupe(u8, reason) catch return false;
    set.put(allocator, s, r) catch {
        allocator.free(r);
        return false;
    };
    return true;
}

pub const Compiled = struct {
    /// The compiler's memory, and the compiler (kept: thunks are compiled
    /// while the program runs)
    arena: std.heap.ArenaAllocator,
    compiler: compile_mod.Compiler,
    failure: compile_mod.Failure = .{},
    python: *PythonSet,
    view: *const llvm.LlvmView,
    modules: std.ArrayListUnmanaged(llvm.Module) = .empty,
    main: Main,
    /// The number of the top level's variables
    globals: usize,
    /// The program's number (its functions' Function.program), never used
    /// again by another
    id: u64 = 0,
    /// Its code's names' start (prefixFor)
    prefix: []const u8 = "",
    /// LLVM's optimization level for its code (0: compiled fast, for a
    /// run that can't wait; 2: the code programs run)
    opt: u32 = 2,
    /// Its main code in the JIT: modules compiled after are compiled as it
    /// runs (tiers: compiled fast)
    installed: bool = false,
    /// Thunks compiled, by node, eval or exec, and the frame's owner
    thunks: std.AutoHashMapUnmanaged(ThunkKey, Thunk) = .empty,
    /// What running a node as a value does, the last time (by node and
    /// eval or exec: bridge.runNode's, looked up without hashing)
    runs: []Run = &.{},
    /// Specialized helpers' addresses, by name (specialize)
    special: std.StringHashMapUnmanaged(usize) = .empty,
    /// Modules loaded from the cache, compiled and kept in it (report())
    cache_loaded: u64 = 0,
    cache_kept: u64 = 0,
    /// The keys of the modules compiled (cache.zig), in order: what a
    /// compiled module of the program has (aot.zig); and their objects when
    /// there's no cache to have them from
    keys: std.ArrayListUnmanaged(cache.Key) = .empty,
    kept_objects: std.AutoHashMapUnmanaged(cache.Key, []u8) = .empty,
    /// Node attributes by name (attrOf), nodes' texts (textStr)
    attrs: std.AutoHashMapUnmanaged(*const value.Str, Attr) = .empty,
    texts: []?*value.Str = &.{},
    /// Kind and rule names as values (node.kind of a node known only at
    /// run time), looked up once (value.literal: immortal)
    names: std.AutoHashMapUnmanaged(u32, *value.Str) = .empty,
    /// The compiled code of Python functions the code calls (null: Python
    /// runs it)
    called: std.AutoHashMapUnmanaged(CalledKey, ?Helper) = .empty,

    const ThunkKey = struct { node: u32, which: compile_mod.Which, owner: u32 };

    /// A node run: its thunk for frames of `owner`'s, or its semantic run
    /// as Python (`python`: so for good, the semantics run as Python only
    /// grow); neither yet
    pub const Run = struct { thunk: ?Thunk = null, owner: u32 = 0, python: bool = false };

    /// Where a node run is remembered (null: out of memory).
    pub fn runOf(self: *Compiled, node: u32, which: compile_mod.Which, nodes: usize) ?*Run {
        if (self.runs.len == 0) {
            self.runs = allocator.alloc(Run, nodes * 2) catch return null;
            @memset(self.runs, .{});
        }
        return &self.runs[@as(usize, node) * 2 + @intFromEnum(which)];
    }
    const CalledKey = struct { func: *PyObject, nargs: usize, rt_mask: u64 };

    /// A kind's (or rule's) name as a str value (borrowed: immortal).
    pub fn nameStr(self: *Compiled, text: []const u8, rid: u32, is_kind: bool) ?*value.Str {
        const key = rid | (@as(u32, @intFromBool(is_kind)) << 31);
        if (self.names.get(key)) |s| return s;
        // (the process's: a kind kept in module state outlives the program)
        const s = value.literal(text) orelse return null;
        self.names.put(allocator, key, s) catch return null;
        return s;
    }

    /// What a node's attribute is (bridge.nodeAttr): a labelled field, or
    /// one every node has
    pub const Attr = union(enum) { field: u8, kind, rule, text, start, end, line, column, index, parent, children, other };

    /// The attribute a name (a literal's str: one per name, immortal) is,
    /// worked out once (null: out of memory).
    pub fn attrOf(self: *Compiled, name: *const value.Str, g: *const grammar_mod.Grammar) ?Attr {
        if (self.attrs.get(name)) |a| return a;
        const s = name.bytes();
        // (a label first: a grammar may have one named `text`...)
        const attr: Attr = if (g.field_ids.get(s)) |f| .{ .field = f } else blk: {
            const t = std.meta.stringToEnum(std.meta.Tag(Attr), s) orelse break :blk .other;
            break :blk switch (t) {
                .field, .other => .other,
                inline else => |tt| @unionInit(Attr, @tagName(tt), {}),
            };
        };
        self.attrs.put(allocator, name, attr) catch return null;
        return attr;
    }

    /// A node's text as a str (borrowed: kept for the program, a reference
    /// of its own: one kept after the program lives on).
    pub fn textStr(self: *Compiled, idx: u32, d: *const program_mod.Data) ?*value.Str {
        if (self.texts.len == 0) {
            self.texts = allocator.alloc(?*value.Str, d.nodes.len) catch return null;
            @memset(self.texts, null);
        }
        if (self.texts[idx]) |s| return s;
        const s = value.newStr(d.text(idx)) orelse return null;
        self.texts[idx] = s;
        return s;
    }

    /// The Python objects the code refers to (owned), by index
    pub fn objects(self: *const Compiled) []*PyObject {
        return self.compiler.objects.items;
    }

    /// The object of a module compiled for the program (owned by the
    /// caller): a compiled module's, the cache's, or kept here.
    pub fn objectOf(self: *const Compiled, key: cache.Key) ?[]u8 {
        if (self.kept_objects.get(key)) |b| return allocator.dupe(u8, b) catch null;
        return cache.objectOf(allocator, key);
    }

    pub fn destroy(self: *Compiled) void {
        for (self.modules.items) |*m| m.release();
        self.modules.deinit(allocator);
        self.keys.deinit(allocator);
        var it = self.kept_objects.valueIterator();
        while (it.next()) |b| allocator.free(b.*);
        self.kept_objects.deinit(allocator);
        self.thunks.deinit(allocator);
        if (self.runs.len > 0) allocator.free(self.runs);
        self.attrs.deinit(allocator);
        self.special.deinit(allocator);
        for (self.texts) |t| if (t) |s| value.decref(value.Value.obj(.str, &s.head));
        if (self.texts.len > 0) allocator.free(self.texts);
        self.called.deinit(allocator);
        self.names.deinit(allocator);
        for (self.compiler.objects.items) |o| py.Py_DecRef(o);
        self.compiler.m.deinit();
        self.compiler.deinit(allocator);
        self.arena.deinit();
        allocator.destroy(self);
    }

    /// The thunk of a node's eval or exec, compiled the first time: null
    /// with a Python exception.
    pub fn thunk(self: *Compiled, node: u32, which: compile_mod.Which, owner: u32) ?Thunk {
        const key = ThunkKey{ .node = node, .which = which, .owner = owner };
        if (self.thunks.get(key)) |t| return t;
        const c = &self.compiler;
        while (true) {
            c.failed_semantic = null;
            c.need_retry = false;
            const name = c.compileThunk(node, which, owner) catch |e| switch (blk: {
                c.forgetModule();
                break :blk e;
            }) {
                error.Unsupported => {
                    // (a literal now made at run time: compiled again)
                    if (c.need_retry) continue;
                    // (a semantic it reaches can't be compiled: as Python)
                    if (c.failed_semantic) |s| {
                        if (!self.python.contains(s)) {
                            if (!markPython(self.python, s, self.failure.message.items)) return oomT();
                            continue;
                        }
                    }
                    ph.raise(types().CompileError, "{s}", .{self.failure.message.items});
                    return null;
                },
                error.OutOfMemory => return oomT(),
                error.Python => return null,
            };
            const addr = self.add(name) orelse return null;
            const t: Thunk = @ptrFromInt(addr);
            self.thunks.put(allocator, key, t) catch return oomT();
            return t;
        }
    }

    /// A hot call site's helper compiled for what the site knows (bridge's
    /// zr_specialize): the site's code from now on. If it can't be, the
    /// site keeps calling the generic code (and isn't counted any more).
    pub fn specialize(self: *Compiled, site: usize) void {
        const c = &self.compiler;
        const hot = c.sites.items[site].hot;
        hot.count = std.math.minInt(i64);
        if (c.specializedBefore(site)) |name| {
            if (self.special.get(name)) |addr| hot.code = addr;
            return;
        }
        if (c.specialized >= compile_mod.max_specialized) return;
        c.failed_semantic = null;
        c.need_retry = false;
        const name = c.compileSpecialized(site) catch {
            c.forgetModule();
            // (a Python error on the way: not the program's)
            py.c.PyErr_Clear();
            return;
        };
        const addr = self.add(name) orelse {
            // (what it made isn't in the JIT)
            c.forgetModule();
            py.c.PyErr_Clear();
            return;
        };
        self.special.put(allocator, name, addr) catch return;
        hot.code = addr;
    }

    /// A hot language function compiled again for the kinds of values its
    /// arguments have always been (bridge's zr_speculate): its typed entry,
    /// which its generic entry calls for arguments of those kinds (as code
    /// compiled from now on does directly). Arguments of more than one kind,
    /// or of none an entry takes, or code that can't be compiled: nothing
    /// (it isn't counted any more).
    pub fn speculate(self: *Compiled, fnode: u32) void {
        const c = &self.compiler;
        const rec = c.speculations.get(fnode) orelse return;
        rec.count = std.math.minInt(i64);
        const shapes = c.a.alloc(compile_mod.Shape, rec.nparams) catch return;
        for (shapes, 0..) |*s, i| {
            const bits = (rec.seen >> @intCast(16 * i)) & 0xFFFF;
            s.* = if (bits == 1 << @intFromEnum(value.Tag.int))
                .int
            else if (bits == 1 << @intFromEnum(value.Tag.float))
                .float
            else if (bits == 1 << @intFromEnum(value.Tag.bool))
                .bool
            else
                return;
        }
        const key = fnode | compile_mod.Compiler.TYPED;
        c.typed_params.put(c.a, fnode, shapes) catch return;
        const name = std.fmt.allocPrintSentinel(c.a, "{s}_t{d}", .{ c.m.prefix, fnode }, 0) catch return;
        const addr = while (true) {
            c.failed_semantic = null;
            c.need_retry = false;
            c.newModule() catch return;
            _ = c.functionCode(key) catch return;
            c.drainQueues() catch |e| {
                c.forgetModule();
                if (e == error.Unsupported and c.need_retry) continue;
                // (it stays generic: as it was compiled)
                c.typed_params.put(c.a, fnode, null) catch {};
                py.c.PyErr_Clear();
                return;
            };
            break self.add(name) orelse {
                c.typed_params.put(c.a, fnode, null) catch {};
                py.c.PyErr_Clear();
                return;
            };
        };
        rec.want = rec.seen;
        rec.code = addr;
    }

    /// The compiled code of a Python function compiled code calls (for
    /// `nargs` arguments, those of rt_mask rt values): its address, made the
    /// first time; null if it can't be compiled (Python runs it, then).
    pub fn calledCode(self: *Compiled, o: *PyObject, nargs: usize, rt_mask: u64, closure: bool) ?Helper {
        // (a closure's code serves all its closures: its variables are read
        // from the one called)
        const code_obj = py.c.PyObject_GetAttrString(o, "__code__") orelse {
            py.c.PyErr_Clear();
            return null;
        };
        py.Py_DecRef(code_obj);
        const key = CalledKey{ .func = if (closure) code_obj else o, .nargs = nargs, .rt_mask = rt_mask };
        if (self.called.get(key)) |code| return code;
        const c = &self.compiler;
        var code: ?Helper = null;
        while (true) {
            c.failed_semantic = null;
            c.need_retry = false;
            const name = c.compileCalled(o, nargs, rt_mask, closure) catch |e| {
                c.forgetModule();
                // (a literal made at run time: again; anything else: Python)
                if (e == error.Unsupported and c.need_retry) continue;
                py.c.PyErr_Clear();
                break;
            };
            const addr = self.add(name) orelse {
                py.c.PyErr_Clear();
                break;
            };
            code = @ptrFromInt(addr);
            break;
        }
        // (the key kept alive: a code object compiled for)
        if (closure) _ = c.objectIndex(code_obj) catch {};
        self.called.put(allocator, key, code) catch {};
        return code;
    }

    /// The code of a language function (compiled now if it wasn't): its
    /// address, or null with a Python exception.
    pub fn functionAddr(self: *Compiled, fnode: u32) ?usize {
        const c = &self.compiler;
        const name = std.fmt.allocPrintSentinel(c.a, "{s}_f{d}", .{ c.m.prefix, fnode }, 0) catch {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        if (c.compiled_fns.contains(fnode)) {
            const addr = llvm.lookup(self.view, name);
            if (addr != 0) return addr;
        }
        attempt: while (true) {
            c.failed_semantic = null;
            c.need_retry = false;
            c.newModule() catch return oomA();
            _ = c.functionCode(fnode) catch return oomA();
            c.drainQueues() catch |e| {
                c.forgetModule();
                switch (e) {
                    error.Unsupported => {
                        // (its code needs its frame: compiled for the first
                        // time, it can have one; again)
                        if (c.need_frames and !(c.layoutOf(fnode) catch return oomA()).heap) {
                            c.need_frames = false;
                            c.heapFunction(fnode) catch return oomA();
                            continue :attempt;
                        }
                        // (as a thunk's: a literal made at run time, or a
                        // semantic run as Python, and compiled again)
                        if (c.need_retry) continue :attempt;
                        if (c.failed_semantic) |s| {
                            if (!self.python.contains(s)) {
                                if (!markPython(self.python, s, self.failure.message.items)) return oomA();
                                continue :attempt;
                            }
                        }
                        ph.raise(types().CompileError, "{s}", .{self.failure.message.items});
                        return null;
                    },
                    error.OutOfMemory => return oomA(),
                    error.Python => return null,
                }
            };
            return self.add(name);
        }
    }

    /// The compiler's module as code in the JIT: its object file from the
    /// cache (cache.zig) if it's there, else compiled (and kept there).
    /// Null with the error in `err`.
    fn compileModule(self: *Compiled, err: []u8) ?llvm.Module {
        const text = jit_f("LLVMPrintModuleToString")(self.compiler.m.mod);
        defer jit_f("LLVMDisposeMessage")(text);
        const key = cache.keyOf(std.mem.span(text), self.opt);
        if (self.atHand(key, err)) |module| {
            // (kept: the program saved as a compiled module has them all)
            self.keys.append(allocator, key) catch {};
            return module;
        }
        // Code compiled as the program runs, optimized code's (tiers): not
        // optimized now (seconds, for some) but compiled fast, and
        // optimized in the background for the cache (the next process's;
        // this one's names are taken by the code compiled fast). Its key
        // the optimized object's, in the cache once it's made: what
        // program.save() takes (waitJobs() first)
        self.keys.append(allocator, key) catch {};
        const fast = self.opt != 0 and tiers and self.installed and self.optimizeLater(key);
        const kept = if (fast) cache.keyOf(std.mem.span(text), 0) else key;
        if (fast) if (self.atHand(kept, err)) |module| return module;
        const bytes = llvm.emitObject(self.view, self.compiler.m.take(), if (fast) 0 else self.opt, err) catch return null;
        defer llvm.freeBytes(self.view, bytes);
        self.keep(kept, bytes);
        return llvm.loadObject(self.view, bytes, err) catch null;
    }

    /// The compiler's module optimized in the background (a copy: the
    /// module's compiled fast here), its object kept in the cache; false if
    /// it can't be (no cache: it's compiled optimized now).
    fn optimizeLater(self: *Compiled, key: cache.Key) bool {
        const path = cache.pathFor(allocator, key) orelse return false;
        const copy = llvm.copyOf(self.compiler.m.mod) orelse {
            allocator.free(path);
            return false;
        };
        _ = inBackground(self.view, copy, self.opt, path, false);
        return true;
    }

    /// The key of the compiler's module (its IR, at the program's level)
    fn moduleKey(self: *Compiled) cache.Key {
        const text = jit_f("LLVMPrintModuleToString")(self.compiler.m.mod);
        defer jit_f("LLVMDisposeMessage")(text);
        return cache.keyOf(std.mem.span(text), self.opt);
    }

    /// The object of a key in the JIT if it's at hand (a compiled module
    /// loaded's, the cache's: the compiler's module left to the compiler,
    /// which frees it), or null.
    fn atHand(self: *Compiled, key: cache.Key, err: []u8) ?llvm.Module {
        if (cache.givenObject(key)) |bytes| {
            if (llvm.loadObject(self.view, bytes, err)) |module| {
                self.cache_loaded += 1;
                return module;
            } else |_| {}
        }
        const path = cache.pathFor(allocator, key) orelse return null;
        defer allocator.free(path);
        const bytes = cache.read(allocator, path) orelse return null;
        defer allocator.free(bytes);
        if (llvm.loadObject(self.view, bytes, err)) |module| {
            self.cache_loaded += 1;
            return module;
        } else |_| {}
        // (one that won't load: compiled again)
        return null;
    }

    /// An object compiled: kept in the cache, or here when there's none
    /// (for saving).
    fn keep(self: *Compiled, key: cache.Key, bytes: []const u8) void {
        if (cache.pathFor(allocator, key)) |path| {
            defer allocator.free(path);
            cache.write(path, bytes);
            self.cache_kept += 1;
        } else self.keepHere(key, bytes);
    }

    fn keepHere(self: *Compiled, key: cache.Key, bytes: []const u8) void {
        const copy = allocator.dupe(u8, bytes) catch return;
        self.kept_objects.put(allocator, key, copy) catch allocator.free(copy);
    }

    /// zrun.configure(perf_map=True): the module's functions named for perf
    /// (/tmp/perf-<pid>.map: each one's address, its size taken as up to
    /// the next one's).
    fn perfMap(self: *Compiled) void {
        // (Linux's perf reads it: nothing elsewhere)
        if (@import("builtin").os.tag != .linux) return;
        var it = self.compiler.m.fns.keyIterator();
        while (it.next()) |k| {
            if (!std.mem.startsWith(u8, k.*, self.compiler.m.prefix)) continue;
            const z = allocator.dupeZ(u8, k.*) catch return;
            const addr = llvm.lookup(self.view, z);
            if (addr == 0) {
                allocator.free(z);
                continue;
            }
            perf_syms.put(allocator, addr, z) catch return;
        }
        const addrs = allocator.dupe(usize, perf_syms.keys()) catch return;
        defer allocator.free(addrs);
        std.mem.sort(usize, addrs, {}, std.sort.asc(usize));
        var path: [64]u8 = undefined;
        const p = std.fmt.bufPrintZ(&path, "/tmp/perf-{d}.map", .{std.c.getpid()}) catch return;
        const fd = std.c.open(p.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return;
        defer _ = std.c.close(fd);
        for (addrs, 0..) |a, i| {
            const size = if (i + 1 < addrs.len) @min(addrs[i + 1] - a, 1 << 20) else 4096;
            var line: [512]u8 = undefined;
            const s = std.fmt.bufPrint(&line, "{x} {x} {s}\n", .{ a, size, perf_syms.get(a).? }) catch continue;
            _ = std.c.write(fd, s.ptr, s.len);
        }
    }

    /// The compiler's module into the JIT: the address of `name` in it.
    fn add(self: *Compiled, name: [:0]const u8) ?usize {
        var err: [2048]u8 = undefined;
        @memset(&err, 0);
        if (!self.defineNames()) return null;
        const module = self.compileModule(&err) orelse {
            ph.raise(py.PyExc_RuntimeError(), "zrun: LLVM rejected the compiled program (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
            return null;
        };
        return self.install(module, name);
    }

    /// The addresses the compiler's module's code names (ir.Module.ptrConst)
    /// defined in the JIT (before its code is loaded); false with the
    /// exception.
    fn defineNames(self: *Compiled) bool {
        const m = &self.compiler.m;
        if (m.syms.items.len == 0) return true;
        var err: [2048]u8 = undefined;
        @memset(&err, 0);
        const names = allocator.alloc([*:0]const u8, m.syms.items.len) catch return oomA() != null;
        defer allocator.free(names);
        const addrs = allocator.alloc(u64, m.syms.items.len) catch return oomA() != null;
        defer allocator.free(addrs);
        for (m.syms.items, names, addrs) |s, *n, *a| {
            n.* = s.name.ptr;
            a.* = s.addr;
        }
        llvm.define(self.view, names, addrs, &err) catch {
            ph.raise(py.PyExc_RuntimeError(), "zrun: the JIT refused the compiled code's names (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
            return false;
        };
        return true;
    }

    /// A module's code in the JIT, the program's from now: the address of
    /// `name` in it (null with the exception).
    fn install(self: *Compiled, module: llvm.Module, name: [:0]const u8) ?usize {
        self.modules.append(allocator, module) catch {
            var code = module;
            code.release();
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        if (perf_map) self.perfMap();
        const addr = llvm.lookup(self.view, name);
        if (addr == 0) {
            ph.raise(py.PyExc_RuntimeError(), "zrun: compiled code without its entry point {s}", .{name});
            return null;
        }
        return addr;
    }
};

fn types() type {
    return @import("types.zig");
}

var helpers_defined = false;
var next_id: u64 = 0;

/// The names of a program's compiled code start with: what it is (`seed`:
/// its language's definition and its source), so its IR, and its objects
/// in the cache (by their IR), are the same in every process, whatever was
/// compiled before it; a counter after, if the process had it already (the
/// JIT's names are the process's, for good: the names of the addresses the
/// code refers to stay defined after its modules go).
var prefixes: std.StringHashMapUnmanaged(void) = .empty;

fn prefixFor(a: std.mem.Allocator, seed: *const [32]u8, opt: u32) ?[]const u8 {
    const hex = std.fmt.bytesToHex(seed[0..6].*, .lower);
    // (the code compiled fast: names of its own, its objects apart)
    const tag: []const u8 = if (opt == 0) "f" else "";
    var n: usize = 0;
    while (true) : (n += 1) {
        const p = (if (n == 0) std.fmt.allocPrint(a, "zr{s}{s}", .{ hex, tag }) else std.fmt.allocPrint(a, "zr{s}{s}x{d}", .{ hex, tag, n })) catch return null;
        const slot = prefixes.getOrPut(allocator, p) catch return null;
        if (slot.found_existing) continue;
        // (the set's own copy: the program's arena goes with it)
        slot.key_ptr.* = allocator.dupe(u8, p) catch {
            _ = prefixes.remove(p);
            return null;
        };
        return p;
    }
}
/// zrun.configure(perf_map=True): compiled functions named for perf
pub var perf_map = false;
/// zrun.configure(tiers=False): a program run compiled is compiled
/// optimized first (not compiled fast, then optimized in the background)
pub var tiers = true;
/// Compiled functions by address (perf_map)
var perf_syms: std.AutoArrayHashMapUnmanaged(usize, [:0]u8) = .empty;

/// Give zgram's JIT the runtime helpers (once per process).
fn defineHelpers(view: *const llvm.LlvmView) bool {
    if (helpers_defined) return true;
    const syms = helpers.symbols() ++ @import("bridge.zig").symbols();
    var names: [syms.len][*:0]const u8 = undefined;
    var addrs: [syms.len]u64 = undefined;
    var name_bufs: [syms.len][64:0]u8 = undefined;
    for (syms, 0..) |s, i| {
        @memcpy(name_bufs[i][0..s[0].len], s[0]);
        name_bufs[i][s[0].len] = 0;
        names[i] = &name_bufs[i];
        addrs[i] = s[1];
    }
    var err: [512]u8 = undefined;
    if (view.define(&names, &addrs, syms.len, &err, err.len) != 0) {
        ph.raise(py.PyExc_RuntimeError(), "zrun: the JIT refused the runtime: {s}", .{std.mem.sliceTo(&err, 0)});
        return false;
    }
    helpers_defined = true;
    return true;
}

/// Compile a program into `out` (its compiler set up): a semantic that
/// can't be compiled is run as Python, and the program compiled again.
/// False with an exception (zrun.CompileError when it can't be compiled
/// even so).
fn build(out: *Compiled, data: *program_mod.Data, lang: compile_mod.LangView, prefix: []const u8, compile_error: *PyObject) bool {
    var force_heap = false;
    // (functions whose code needs their frame: theirs on the heap)
    var heap_fns: std.ArrayListUnmanaged(u32) = .empty;
    defer heap_fns.deinit(allocator);
    while (true) {
        out.failure = .{};
        out.compiler = compile_mod.Compiler.init(out.arena.allocator(), data, lang, prefix, &out.failure);
        out.compiler.force_heap = force_heap;
        out.compiler.heap_fns = heap_fns.items;
        out.compiler.compileProgram() catch |e| {
            const failed = out.compiler.failed_semantic;
            const need_frames = out.compiler.need_frames;
            const need_frames_of = out.compiler.need_frames_of;
            const need_retry = out.compiler.need_retry;
            // (what this attempt kept)
            for (out.compiler.objects.items) |o| py.Py_DecRef(o);
            out.compiler.m.deinit();
            out.compiler.deinit(allocator);
            switch (e) {
                error.Unsupported => {
                    // (compiled again with the function's variables in a
                    // frame; every function's if that wasn't enough)
                    if (need_frames and need_frames_of != program_mod.NONE and std.mem.indexOfScalar(u32, heap_fns.items, need_frames_of) == null) {
                        heap_fns.append(allocator, need_frames_of) catch {
                            _ = py.c.PyErr_NoMemory();
                            return false;
                        };
                        continue;
                    }
                    if (need_frames and !force_heap) {
                        force_heap = true;
                        continue;
                    }
                    // (compiled again with a literal made at run time)
                    if (need_retry) continue;
                    if (failed) |s| {
                        if (!out.python.contains(s)) {
                            if (!markPython(out.python, s, out.failure.message.items)) {
                                _ = py.c.PyErr_NoMemory();
                                return false;
                            }
                            continue;
                        }
                    }
                    ph.raise(compile_error, "{s}", .{out.failure.message.items});
                },
                error.OutOfMemory => _ = py.c.PyErr_NoMemory(),
                error.Python => {},
            }
            return false;
        };
        return true;
    }
}

/// A program's IR made (its compiler's module, not compiled yet), for
/// LLVM's level `opt` (0: compiled fast, 2: optimized): null with an
/// exception (zrun.CompileError when the semantics can't be compiled).
fn prepare(data: *program_mod.Data, lang: compile_mod.LangView, python: *PythonSet, compile_error: *PyObject, seed: *const [32]u8, opt: u32) ?*Compiled {
    const view = llvm.get() orelse return null;
    if (!defineHelpers(view)) return null;

    const out = allocator.create(Compiled) catch return oom();
    out.* = .{ .arena = std.heap.ArenaAllocator.init(allocator), .compiler = undefined, .python = python, .view = view, .main = undefined, .globals = 0, .opt = opt };
    next_id += 1;
    out.id = next_id;
    const prefix = prefixFor(out.arena.allocator(), seed, opt) orelse {
        out.arena.deinit();
        allocator.destroy(out);
        return oom();
    };
    out.prefix = prefix;
    if (!build(out, data, lang, prefix, compile_error)) {
        out.arena.deinit();
        allocator.destroy(out);
        return null;
    }
    out.globals = if (out.compiler.layouts.get(program_mod.NONE)) |l| l.syms.items.len else 0;
    return out;
}

fn mainName(c: *Compiled) ?[:0]const u8 {
    return std.fmt.allocPrintSentinel(c.arena.allocator(), "{s}_main", .{c.prefix}, 0) catch null;
}

/// Compile a program at LLVM's level `opt` (0: fast, 2: optimized), now:
/// null with an exception (zrun.CompileError when the semantics can't be
/// compiled).
pub fn compileProgram(data: *program_mod.Data, lang: compile_mod.LangView, python: *PythonSet, compile_error: *PyObject, seed: *const [32]u8, opt: u32) ?*Compiled {
    const out = prepare(data, lang, python, compile_error, seed, opt) orelse return null;
    const main_name = mainName(out) orelse {
        out.destroy();
        return oom();
    };
    const addr = out.add(main_name) orelse {
        out.destroy();
        return null;
    };
    out.main = @ptrFromInt(addr);
    out.installed = true;
    return out;
}

/// LLVM's work in the background: modules made into objects on worker
/// threads (nothing of Python's: each module and its context its own, the
/// JIT not touched), as many as the CPUs but one, in the order given. Waited
/// for when Python finalizes (LLVM's own state goes as the process exits:
/// no thread in it then), so the objects are in the cache for the next
/// process.
const Job = struct {
    view: *const llvm.LlvmView,
    module: llvm.c.LLVMModuleRef,
    opt: u32,
    /// Where the object is kept (owned), or null
    path: ?[]u8,
    /// Someone waits for it (`done`, then `bytes`: null with LLVM's error
    /// in `err`); else it's freed when done
    waited: bool,
    bytes: ?[]u8 = null,
    err: [2048]u8 = @splat(0),
    done: std.atomic.Value(bool) = .init(false),

    fn work(self: *Job) void {
        if (llvm.emitObject(self.view, self.module, self.opt, &self.err)) |bytes| {
            if (self.path) |p| cache.write(p, bytes);
            if (self.waited) self.bytes = bytes else llvm.freeBytes(self.view, bytes);
        } else |_| {}
        if (self.path) |p| allocator.free(p);
        self.path = null;
        if (self.waited) self.done.store(true, .release) else allocator.destroy(self);
    }
};

var jobs: std.ArrayListUnmanaged(*Job) = .empty;
var jobs_lock: std.atomic.Mutex = .unlocked;
var workers: usize = 0;
/// Jobs given and not done
var outstanding: usize = 0;
var exit_hook = false;

fn lockJobs() void {
    while (!jobs_lock.tryLock()) std.atomic.spinLoopHint();
}

/// A job to a worker (one started if there's room); false if none can
/// take it (the caller does it).
fn submit(job: *Job) bool {
    if (!exit_hook) {
        if (py.c.Py_AtExit(&waitJobs) != 0) return false;
        exit_hook = true;
    }
    lockJobs();
    defer jobs_lock.unlock();
    jobs.append(allocator, job) catch return false;
    outstanding += 1;
    if (workers < (std.Thread.getCpuCount() catch 2) -| 1 or workers == 0) {
        if (std.Thread.spawn(.{}, worker, .{})) |t| {
            t.detach();
            workers += 1;
        } else |_| if (workers == 0) {
            _ = jobs.pop();
            outstanding -= 1;
            return false;
        }
    }
    return true;
}

fn worker() void {
    while (true) {
        lockJobs();
        if (jobs.items.len == 0) {
            workers -= 1;
            jobs_lock.unlock();
            return;
        }
        const job = jobs.orderedRemove(0);
        jobs_lock.unlock();
        job.work();
        lockJobs();
        outstanding -= 1;
        jobs_lock.unlock();
    }
}

/// Every job given done (program.save()'s objects in the cache; Python
/// finalizing).
pub fn waitJobs() callconv(.c) void {
    while (true) {
        lockJobs();
        const n = outstanding;
        jobs_lock.unlock();
        if (n == 0) return;
        std.Io.sleep(cache.io(), .fromMilliseconds(2), .awake) catch {};
    }
}

/// A module made into an object in the background (`waited`: the job's
/// the caller's to free once done), kept at `path` (taken); null if it
/// can't be (LLVM can't read its copy; out of memory).
fn inBackground(view: *const llvm.LlvmView, module: llvm.c.LLVMModuleRef, opt: u32, path: ?[]u8, waited: bool) ?*Job {
    const job = allocator.create(Job) catch {
        if (path) |p| allocator.free(p);
        return null;
    };
    job.* = .{ .view = view, .module = module, .opt = opt, .path = path, .waited = waited };
    if (!submit(job)) {
        // (done here, then)
        if (!waited) {
            job.work();
            return null;
        }
        job.work();
    }
    return job;
}

/// A program being compiled optimized while it runs otherwise: its IR made
/// and its names defined here (with the GIL: the compiler is Python's
/// too), its main module made into an object in the background (LLVM's
/// work, nearly all of it), the code put in the JIT here again (finish).
/// Its object kept in the cache by the worker: the next process has it
/// even if this one never finishes it.
pub const Pending = struct {
    compiled: *Compiled,
    main_name: [:0]const u8,
    /// The main module's code if it was at hand (no job)
    module: ?llvm.Module = null,
    job: ?*Job = null,
    /// The object's key, and whether the job keeps it in the cache (else
    /// the Compiled does)
    key: cache.Key = undefined,
    cached: bool = false,

    /// Whether finish() won't wait.
    pub fn ready(self: *Pending) bool {
        const job = self.job orelse return true;
        return job.done.load(.acquire);
    }

    /// The program compiled (waiting for its job if it's still at work),
    /// its code in the JIT; null with an exception. The Pending is gone
    /// either way.
    pub fn finish(self: *Pending) ?*Compiled {
        while (!self.ready()) std.Io.sleep(cache.io(), .fromMilliseconds(1), .awake) catch {};
        const c = self.compiled;
        defer allocator.destroy(self);
        const module = self.module orelse blk: {
            const job = self.job.?;
            defer allocator.destroy(job);
            const bytes = job.bytes orelse {
                ph.raise(py.PyExc_RuntimeError(), "zrun: LLVM rejected the compiled program (a zrun bug): {s}", .{std.mem.sliceTo(&job.err, 0)});
                c.destroy();
                return null;
            };
            defer llvm.freeBytes(c.view, bytes);
            if (self.cached) c.cache_kept += 1 else c.keepHere(self.key, bytes);
            var err: [2048]u8 = @splat(0);
            break :blk llvm.loadObject(c.view, bytes, &err) catch {
                ph.raise(py.PyExc_RuntimeError(), "zrun: the JIT refused the compiled program (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
                c.destroy();
                return null;
            };
        };
        const addr = c.install(module, self.main_name) orelse {
            c.destroy();
            return null;
        };
        c.main = @ptrFromInt(addr);
        c.installed = true;
        return c;
    }

    /// Given up (the program gone): freed now if its thread is done, else
    /// when it is (LLVM can't be stopped: the object it makes is kept in
    /// the cache all the same).
    pub fn abandon(self: *Pending) void {
        if (self.ready()) return self.drop();
        abandoned.append(allocator, self) catch self.drop();
    }

    /// Freed (its job done)
    fn drop(self: *Pending) void {
        if (self.module) |m| {
            var code = m;
            code.release();
        }
        if (self.job) |job| {
            if (job.bytes) |b| llvm.freeBytes(self.compiled.view, b);
            allocator.destroy(job);
        }
        self.compiled.destroy();
        allocator.destroy(self);
    }
};

/// Pendings given up, freed once their jobs are done
var abandoned: std.ArrayListUnmanaged(*Pending) = .empty;

fn freeAbandoned() void {
    var i: usize = 0;
    while (i < abandoned.items.len) {
        const p = abandoned.items[i];
        if (p.ready()) {
            _ = abandoned.swapRemove(i);
            p.drop();
        } else i += 1;
    }
}

/// Compile a program optimized (LLVM's level 2), the work in the background
/// when its code isn't at hand: null with an exception.
pub fn compileInBackground(data: *program_mod.Data, lang: compile_mod.LangView, python: *PythonSet, compile_error: *PyObject, seed: *const [32]u8) ?*Pending {
    freeAbandoned();
    const c = prepare(data, lang, python, compile_error, seed, 2) orelse return null;
    const p = allocator.create(Pending) catch {
        c.destroy();
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    p.* = .{ .compiled = c, .main_name = mainName(c) orelse {
        allocator.destroy(p);
        c.destroy();
        _ = py.c.PyErr_NoMemory();
        return null;
    } };
    if (!c.defineNames()) {
        p.drop();
        return null;
    }
    const key = c.moduleKey();
    c.keys.append(allocator, key) catch {};
    var err: [2048]u8 = @splat(0);
    if (c.atHand(key, &err)) |module| {
        p.module = module;
        return p;
    }
    p.key = key;
    // (where, worked out here: the cache's directory is found once)
    const path = cache.pathFor(allocator, key);
    p.cached = path != null;
    p.job = inBackground(c.view, c.compiler.m.take(), 2, path, true) orelse {
        p.drop();
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    return p;
}

/// The LLVM IR a program compiles to, as text (before optimization), for
/// debugging: a new str, or null with an exception.
pub fn irText(data: *program_mod.Data, lang: compile_mod.LangView, python: *PythonSet, compile_error: *PyObject) ?*PyObject {
    const view = llvm.get() orelse return null;
    const out = allocator.create(Compiled) catch return null;
    out.* = .{ .arena = std.heap.ArenaAllocator.init(allocator), .compiler = undefined, .python = python, .view = view, .main = undefined, .globals = 0 };
    if (!build(out, data, lang, "zr_ir", compile_error)) {
        out.arena.deinit();
        allocator.destroy(out);
        return null;
    }
    defer out.destroy();
    const text = out.compiler.m.text() catch {
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    return ph.newString(text);
}

fn oom() ?*Compiled {
    _ = py.c.PyErr_NoMemory();
    return null;
}

fn oomA() ?usize {
    _ = py.c.PyErr_NoMemory();
    return null;
}

fn oomT() ?Thunk {
    _ = py.c.PyErr_NoMemory();
    return null;
}
