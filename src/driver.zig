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
        const m = &self.compiler.m;
        const key = blk: {
            const text = jit_f("LLVMPrintModuleToString")(m.mod);
            defer jit_f("LLVMDisposeMessage")(text);
            break :blk cache.keyOf(std.mem.span(text));
        };
        // (kept: the program saved as a compiled module has them all)
        self.keys.append(allocator, key) catch {};
        // A compiled module's, loaded
        if (cache.givenObject(key)) |bytes| {
            // (the module isn't the JIT's: the compiler frees it)
            if (llvm.loadObject(self.view, bytes, err)) |module| {
                self.cache_loaded += 1;
                return module;
            } else |_| {}
        }
        const path = cache.pathFor(allocator, key) orelse {
            // (no cache: the object kept here, for saving)
            const bytes = llvm.emitObject(self.view, m.take(), 2, err) catch return null;
            defer llvm.freeBytes(self.view, bytes);
            if (allocator.dupe(u8, bytes)) |copy| {
                self.kept_objects.put(allocator, key, copy) catch allocator.free(copy);
            } else |_| {}
            return llvm.loadObject(self.view, bytes, err) catch null;
        };
        defer allocator.free(path);
        if (cache.read(allocator, path)) |bytes| {
            defer allocator.free(bytes);
            // (the module isn't the JIT's: the compiler frees it)
            if (llvm.loadObject(self.view, bytes, err)) |module| {
                self.cache_loaded += 1;
                return module;
            } else |_| {}
            // (one that won't load: compiled again)
        }
        const bytes = llvm.emitObject(self.view, m.take(), 2, err) catch return null;
        defer llvm.freeBytes(self.view, bytes);
        cache.write(path, bytes);
        self.cache_kept += 1;
        return llvm.loadObject(self.view, bytes, err) catch null;
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
        // The addresses the code names (ir.Module.ptrConst): defined first
        const m = &self.compiler.m;
        if (m.syms.items.len > 0) {
            const names = allocator.alloc([*:0]const u8, m.syms.items.len) catch return oomA();
            defer allocator.free(names);
            const addrs = allocator.alloc(u64, m.syms.items.len) catch return oomA();
            defer allocator.free(addrs);
            for (m.syms.items, names, addrs) |s, *n, *a| {
                n.* = s.name.ptr;
                a.* = s.addr;
            }
            llvm.define(self.view, names, addrs, &err) catch {
                ph.raise(py.PyExc_RuntimeError(), "zrun: the JIT refused the compiled code's names (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
                return null;
            };
        }
        const module = self.compileModule(&err) orelse {
            ph.raise(py.PyExc_RuntimeError(), "zrun: LLVM rejected the compiled program (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
            return null;
        };
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
/// zrun.configure(perf_map=True): compiled functions named for perf
pub var perf_map = false;
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

/// Compile a program: null with an exception (zrun.CompileError when the
/// semantics can't be compiled).
pub fn compileProgram(data: *program_mod.Data, lang: compile_mod.LangView, python: *PythonSet, compile_error: *PyObject) ?*Compiled {
    const view = llvm.get() orelse return null;
    if (!defineHelpers(view)) return null;

    const out = allocator.create(Compiled) catch return oom();
    out.* = .{ .arena = std.heap.ArenaAllocator.init(allocator), .compiler = undefined, .python = python, .view = view, .main = undefined, .globals = 0 };
    next_id += 1;
    out.id = next_id;
    const prefix = std.fmt.allocPrint(out.arena.allocator(), "zr{d}", .{next_id}) catch return oom();
    if (!build(out, data, lang, prefix, compile_error)) {
        out.arena.deinit();
        allocator.destroy(out);
        return null;
    }
    const main_name = std.fmt.allocPrintSentinel(out.arena.allocator(), "{s}_main", .{prefix}, 0) catch return oom();
    const addr = out.add(main_name) orelse {
        out.destroy();
        return null;
    };
    out.main = @ptrFromInt(addr);
    out.globals = if (out.compiler.layouts.get(program_mod.NONE)) |l| l.syms.items.len else 0;
    return out;
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
