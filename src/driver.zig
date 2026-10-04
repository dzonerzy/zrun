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
const program_mod = @import("program.zig");

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
    /// Thunks compiled, by node, eval or exec, and the frame's owner
    thunks: std.AutoHashMapUnmanaged(ThunkKey, Thunk) = .empty,
    /// Kind and rule names as values (node.kind of a node known only at
    /// run time), made once (immortal: freed with the program)
    names: std.AutoHashMapUnmanaged(u32, *value.Str) = .empty,
    /// The compiled code of Python functions the code calls (null: Python
    /// runs it)
    called: std.AutoHashMapUnmanaged(CalledKey, ?Helper) = .empty,

    const ThunkKey = struct { node: u32, which: compile_mod.Which, owner: u32 };
    const CalledKey = struct { func: *PyObject, nargs: usize, rt_mask: u64 };

    /// A kind's (or rule's) name as a str value (borrowed: immortal).
    pub fn nameStr(self: *Compiled, text: []const u8, rid: u32, is_kind: bool) ?*value.Str {
        const key = rid | (@as(u32, @intFromBool(is_kind)) << 31);
        if (self.names.get(key)) |s| return s;
        const s = value.newStr(text) orelse return null;
        s.head.rc = value.IMMORTAL;
        self.names.put(allocator, key, s) catch {
            s.head.rc = 1;
            value.decref(value.Value.obj(.str, &s.head));
            return null;
        };
        return s;
    }

    /// The Python objects the code refers to (owned), by index
    pub fn objects(self: *const Compiled) []*PyObject {
        return self.compiler.objects.items;
    }

    pub fn destroy(self: *Compiled) void {
        for (self.modules.items) |*m| m.release();
        self.modules.deinit(allocator);
        self.thunks.deinit(allocator);
        self.called.deinit(allocator);
        var it = self.names.valueIterator();
        while (it.next()) |s| {
            s.*.head.rc = 1;
            value.decref(value.Value.obj(.str, &s.*.head));
        }
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
                            if (std.c.getenv("ZRUN_STATS") != null) std.debug.print("thunk {d}: as Python: {s}\n", .{ node, self.failure.message.items });
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
            const t0 = nowUs();
            const addr = self.add(name) orelse return null;
            if (std.c.getenv("ZRUN_STATS") != null) std.debug.print("thunk node={d} which={s} owner={d} llvm={d}us\n", .{ node, @tagName(which), owner, nowUs() - t0 });
            const t: Thunk = @ptrFromInt(addr);
            self.thunks.put(allocator, key, t) catch return oomT();
            return t;
        }
    }

    /// The compiled code of a Python function compiled code calls (for
    /// `nargs` arguments, those of rt_mask rt values): its address, made the
    /// first time; null if it can't be compiled (Python runs it, then).
    pub fn calledCode(self: *Compiled, o: *PyObject, nargs: usize, rt_mask: u64) ?Helper {
        const key = CalledKey{ .func = o, .nargs = nargs, .rt_mask = rt_mask };
        if (self.called.get(key)) |code| return code;
        const c = &self.compiler;
        var code: ?Helper = null;
        while (true) {
            c.failed_semantic = null;
            c.need_retry = false;
            const name = c.compileCalled(o, nargs, rt_mask) catch |e| {
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
                                if (std.c.getenv("ZRUN_STATS") != null) std.debug.print("function {d}: as Python: {s}\n", .{ fnode, self.failure.message.items });
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

    /// The compiler's module into the JIT: the address of `name` in it.
    fn add(self: *Compiled, name: [:0]const u8) ?usize {
        var err: [2048]u8 = undefined;
        @memset(&err, 0);
        if (std.c.getenv("ZRUN_STATS") != null) {
            std.debug.print("module {s}: {d} bodies inlined, {d} helpers out of line\n", .{ name, self.compiler.inlined, self.compiler.helper_fns.items.len });
            self.compiler.inlined = 0;
        }
        const module = llvm.compile(self.view, self.compiler.m.take(), 2, &err) catch {
            ph.raise(py.PyExc_RuntimeError(), "zrun: LLVM rejected the compiled program (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
            return null;
        };
        self.modules.append(allocator, module) catch {
            var m = module;
            m.release();
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        const addr = llvm.lookup(self.view, name);
        if (addr == 0) {
            ph.raise(py.PyExc_RuntimeError(), "zrun: compiled code without its entry point {s}", .{name});
            return null;
        }
        return addr;
    }
};

fn nowUs() i64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i64, ts.sec) * 1_000_000 + @divTrunc(@as(i64, ts.nsec), 1000);
}

fn types() type {
    return @import("types.zig");
}

var helpers_defined = false;
var next_id: u64 = 0;

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
    while (true) {
        out.failure = .{};
        out.compiler = compile_mod.Compiler.init(out.arena.allocator(), data, lang, prefix, &out.failure);
        out.compiler.force_heap = force_heap;
        out.compiler.compileProgram() catch |e| {
            const failed = out.compiler.failed_semantic;
            const need_frames = out.compiler.need_frames;
            const need_retry = out.compiler.need_retry;
            if (std.c.getenv("ZRUN_STATS") != null) std.debug.print("build attempt: frames={} retry={} heap={} python={}: {s}\n", .{ need_frames, need_retry, force_heap, failed != null, out.failure.message.items });
            // (what this attempt kept)
            for (out.compiler.objects.items) |o| py.Py_DecRef(o);
            out.compiler.m.deinit();
            out.compiler.deinit(allocator);
            switch (e) {
                error.Unsupported => {
                    // (compiled again with its variables in frames)
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
