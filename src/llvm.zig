//! zgram's LLVM, through its `zgram.llvm.v1` capsule (zgram's
//! src/llvm_capsule.zig is the definition; this is the consumer's copy):
//! LLVM IR text in, JIT-compiled code or object files out.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;

pub const LLVM_ABI: u32 = 1;
pub const CAPSULE_NAME = "zgram.llvm.v1";

pub const LlvmView = extern struct {
    abi: u32,
    llvm_version: [*:0]const u8,
    compile: *const fn (ir: [*]const u8, len: usize, opt_level: u32, err: [*]u8, err_cap: usize) callconv(.c) ?*anyopaque,
    lookup: *const fn (name: [*:0]const u8) callconv(.c) u64,
    define: *const fn (names: [*]const [*:0]const u8, addrs: [*]const u64, n: usize, err: [*]u8, err_cap: usize) callconv(.c) i32,
    release: *const fn (handle: ?*anyopaque) callconv(.c) void,
    emit_object: *const fn (ir: [*]const u8, len: usize, opt_level: u32, triple: ?[*:0]const u8, cpu: ?[*:0]const u8, features: ?[*:0]const u8, out_len: *usize, err: [*]u8, err_cap: usize) callconv(.c) ?[*]u8,
    free_bytes: *const fn (bytes: ?[*]u8) callconv(.c) void,
    triple: *const fn () callconv(.c) ?[*:0]const u8,
    data_layout: *const fn () callconv(.c) ?[*:0]const u8,
};

/// The view, once loaded (the capsule is kept for the process's life)
var loaded: ?*const LlvmView = null;
var capsule_ref: ?*PyObject = null;

/// zgram's LLVM (needs the GIL the first time): null with ImportError if
/// this zgram has none or another version of it.
pub fn get() ?*const LlvmView {
    if (loaded) |v| return v;
    const zgram = py.c.PyImport_ImportModule("zgram") orelse return null;
    defer py.Py_DecRef(zgram);
    const capsule = py.c.PyObject_CallMethod(zgram, "llvm_capsule", null) orelse {
        py.c.PyErr_Clear();
        ph.raise(py.PyExc_ImportError(), "compiling needs zgram 0.3.6 or later (zgram.llvm_capsule())", .{});
        return null;
    };
    const raw = py.c.PyCapsule_GetPointer(capsule, CAPSULE_NAME) orelse {
        py.Py_DecRef(capsule);
        return null;
    };
    const view: *const LlvmView = @ptrCast(@alignCast(raw));
    if (view.abi != LLVM_ABI) {
        py.Py_DecRef(capsule);
        ph.raise(py.PyExc_ImportError(), "zrun uses zgram's LLVM with ABI {d}, this zgram has {d}", .{ LLVM_ABI, view.abi });
        return null;
    }
    capsule_ref = capsule;
    loaded = view;
    return view;
}

/// A compiled module: its code stays until release().
pub const Module = struct {
    handle: ?*anyopaque,

    pub fn release(self: *Module) void {
        if (loaded) |v| v.release(self.handle);
        self.handle = null;
    }
};

/// Compile IR text; error.Compile with the message in `err` (cut).
pub fn compile(view: *const LlvmView, ir: []const u8, opt_level: u32, err: []u8) error{Compile}!Module {
    const handle = view.compile(ir.ptr, ir.len, opt_level, err.ptr, err.len) orelse return error.Compile;
    return .{ .handle = handle };
}

pub fn lookup(view: *const LlvmView, name: [:0]const u8) usize {
    return @intCast(view.lookup(name.ptr));
}
