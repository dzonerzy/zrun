//! Compiled programs: compile.zig's IR compiled by zgram's LLVM, run with
//! an execution context.

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

pub const Compiled = struct {
    module: llvm.Module,
    main: Main,
    /// The Python objects the code refers to (owned)
    objects: []*PyObject,
    /// The number of the top level's variables
    globals: usize,
    /// The record types its values use
    record_types: []*value.RecordType,

    pub fn destroy(self: *Compiled) void {
        self.module.release();
        for (self.record_types) |t| freeRecordType(t);
        allocator.free(self.record_types);
        for (self.objects) |o| py.Py_DecRef(o);
        allocator.free(self.objects);
        allocator.destroy(self);
    }
};

var helpers_defined = false;
var next_id: u64 = 0;

/// Give zgram's JIT the runtime helpers (once per process).
fn defineHelpers(view: *const llvm.LlvmView) bool {
    if (helpers_defined) return true;
    const syms = helpers.symbols() ++ helpers.moreSymbols() ++ helpers.formatSymbols();
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

/// Build a program's module (in the compiler `c`); false with an exception
/// (zrun.CompileError when the semantics can't be compiled).
fn build(c: *compile_mod.Compiler, failure: *compile_mod.Failure, compile_error: *PyObject) bool {
    c.compileProgram() catch |e| switch (e) {
        error.Unsupported => {
            ph.raise(compile_error, "{s}", .{failure.message.items});
            return false;
        },
        error.OutOfMemory => {
            _ = py.c.PyErr_NoMemory();
            return false;
        },
        error.Python => return false,
    };
    return true;
}

/// Compile a program: null with an exception (zrun.CompileError when the
/// semantics can't be compiled).
pub fn compileProgram(data: *program_mod.Data, lang: compile_mod.LangView, compile_error: *PyObject) ?*Compiled {
    const view = llvm.get() orelse return null;
    if (!defineHelpers(view)) return null;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    next_id += 1;
    const prefix = std.fmt.allocPrint(a, "zr{d}", .{next_id}) catch return oom();
    var failure = compile_mod.Failure{};
    var c = compile_mod.Compiler.init(a, data, lang, prefix, &failure);
    defer c.deinit(allocator);
    defer c.m.deinit();
    if (!build(&c, &failure, compile_error)) return null;

    var err: [2048]u8 = undefined;
    @memset(&err, 0);
    const module = llvm.compile(view, c.m.take(), 2, &err) catch {
        ph.raise(py.PyExc_RuntimeError(), "zrun: LLVM rejected the compiled program (a zrun bug): {s}", .{std.mem.sliceTo(&err, 0)});
        return null;
    };
    const main_name = std.fmt.allocPrintSentinel(a, "{s}_main", .{prefix}, 0) catch return oom();
    const addr = llvm.lookup(view, main_name);
    if (addr == 0) {
        var m = module;
        m.release();
        ph.raise(py.PyExc_RuntimeError(), "zrun: the compiled program has no entry point", .{});
        return null;
    }
    const out = allocator.create(Compiled) catch return oom();
    const objs = allocator.dupe(*PyObject, c.objects.items) catch return oom();
    const globals = if (c.layouts.get(program_mod.NONE)) |l| l.syms.items.len else 0;
    const records = allocator.dupe(*value.RecordType, c.record_list.items) catch return oom();
    out.* = .{ .module = module, .main = @ptrFromInt(addr), .objects = objs, .globals = globals, .record_types = records };
    return out;
}

/// The LLVM IR a program compiles to, as text (before optimization), for
/// debugging: a new str, or null with an exception.
pub fn irText(data: *program_mod.Data, lang: compile_mod.LangView, compile_error: *PyObject) ?*PyObject {
    _ = llvm.get() orelse return null;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var failure = compile_mod.Failure{};
    var c = compile_mod.Compiler.init(a, data, lang, "zr_ir", &failure);
    defer c.deinit(allocator);
    defer c.m.deinit();
    // (objects the IR refers to are kept by the compiler: released here)
    defer for (c.objects.items) |o| py.Py_DecRef(o);
    defer for (c.record_list.items) |t| freeRecordType(t);
    if (!build(&c, &failure, compile_error)) return null;
    const text = c.m.text() catch {
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    return ph.newString(text);
}

fn freeRecordType(t: *value.RecordType) void {
    for (t.fields) |f| allocator.free(f);
    allocator.free(t.fields);
    allocator.free(t.name);
    allocator.destroy(t);
}

fn oom() ?*Compiled {
    _ = py.c.PyErr_NoMemory();
    return null;
}
