//! `zrun.Bytes`: data a program reads, not copied (bytes, bytearray,
//! memoryview, mmap): a read-only view of their memory. Slices share it.
//! The same in every mode: the language sees a zrun.Bytes, in Python as
//! this type, in compiled code as a native value (value.zig's Bytes) over
//! the same memory. `rt.u8(data, i)`, `rt.u32le(data, i)`... read it,
//! bounds checked (read()).
//!
//! The buffer protocol (PyObject_GetBuffer) is in the Stable ABI from 3.11;
//! its functions and Py_buffer's layout are the same in 3.10, declared here.
//! (On Windows the import library of 3.10's Stable ABI hasn't them: they're
//! looked up in Python's DLL as the module's made.)

const std = @import("std");
const builtin = @import("builtin");
const ph = @import("pyhelp.zig");
const py = ph.py;

const PyObject = py.PyObject;

const Py_buffer = extern struct {
    buf: ?*anyopaque = null,
    obj: ?*PyObject = null,
    len: isize = 0,
    itemsize: isize = 0,
    readonly: c_int = 0,
    ndim: c_int = 0,
    format: ?[*:0]u8 = null,
    shape: ?*isize = null,
    strides: ?*isize = null,
    suboffsets: ?*isize = null,
    internal: ?*anyopaque = null,
};
const Buffers = if (builtin.os.tag == .windows) struct {
    const HMODULE = *opaque {};
    extern "kernel32" fn GetProcAddress(module: HMODULE, name: [*:0]const u8) callconv(.winapi) ?*const anyopaque;

    var get: ?*const fn (o: *PyObject, view: *Py_buffer, flags: c_int) callconv(.c) c_int = null;
    var release: ?*const fn (view: *Py_buffer) callconv(.c) void = null;

    /// Looked up in Python's DLL (python3X.dll), `sys.dllhandle` (the
    /// address of an imported function is this module's import thunk, and
    /// python3.dll only forwards the Stable ABI of its version)
    fn init() bool {
        const handle = py.c.PySys_GetObject("dllhandle") orelse return false;
        const module: HMODULE = @ptrCast(py.c.PyLong_AsVoidPtr(handle) orelse {
            py.c.PyErr_Clear();
            return false;
        });
        get = @ptrCast(GetProcAddress(module, "PyObject_GetBuffer") orelse return false);
        release = @ptrCast(GetProcAddress(module, "PyBuffer_Release") orelse return false);
        return true;
    }
} else struct {
    const C = struct {
        extern "c" fn PyObject_GetBuffer(o: *PyObject, view: *Py_buffer, flags: c_int) c_int;
        extern "c" fn PyBuffer_Release(view: *Py_buffer) void;
    };
    const get: ?*const fn (o: *PyObject, view: *Py_buffer, flags: c_int) callconv(.c) c_int = &C.PyObject_GetBuffer;
    const release: ?*const fn (view: *Py_buffer) callconv(.c) void = &C.PyBuffer_Release;

    fn init() bool {
        return true;
    }
};

fn PyObject_GetBuffer(o: *PyObject, view: *Py_buffer, flags: c_int) c_int {
    return Buffers.get.?(o, view, flags);
}

fn PyBuffer_Release(view: *Py_buffer) void {
    Buffers.release.?(view);
}
/// PyBUF_SIMPLE: contiguous bytes, read only is enough
const buf_simple = 0;

pub const BytesObject = extern struct {
    ob_base: py.c.PyObject,
    /// The view holding the memory, for a slice (null: this one holds it)
    root: ?*PyObject,
    view: Py_buffer,
    ptr: [*]const u8,
    len: usize,

    pub fn bytes(self: *const BytesObject) []const u8 {
        return self.ptr[0..self.len];
    }
};

pub var BytesType: *PyObject = undefined;

fn ref(o: *PyObject) *PyObject {
    py.Py_IncRef(o);
    return o;
}

fn typeObj(o: *PyObject) *PyObject {
    return @ptrCast(@alignCast(ph.typeOf(o)));
}

pub fn init(module: *PyObject) !void {
    if (!Buffers.init()) {
        ph.raise(py.PyExc_ImportError(), "zrun: Python's buffer protocol (PyObject_GetBuffer) wasn't found", .{});
        return error.Python;
    }
    BytesType = py.c.PyType_FromSpec(&spec) orelse return error.Python;
    if (py.c.PyModule_AddObject(module, "Bytes", BytesType) != 0) return error.Python;
    py.Py_IncRef(BytesType);
}

pub fn isBytes(o: *PyObject) bool {
    return typeObj(o) == BytesType;
}

pub fn as(o: *PyObject) *BytesObject {
    return @ptrCast(@alignCast(o));
}

/// Whether an object is data a call hands over as a zrun.Bytes: bytes,
/// bytearray, memoryview, mmap.
pub fn isData(o: *PyObject) bool {
    const t = typeObj(o);
    if (t == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PyBytes_Type"))))) return true;
    if (t == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PyByteArray_Type"))))) return true;
    if (t == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PyMemoryView_Type"))))) return true;
    const name = ph.attr(t, "__name__") orelse {
        py.c.PyErr_Clear();
        return false;
    };
    defer py.Py_DecRef(name);
    return std.mem.eql(u8, ph.utf8(name, "name") orelse "", "mmap");
}

fn alloc() ?*BytesObject {
    const allocf: py.c.allocfunc = @ptrCast(py.c.PyType_GetSlot(@ptrCast(BytesType), py.c.Py_tp_alloc));
    const o = allocf.?(@ptrCast(BytesType), 0) orelse return null;
    const b = as(o);
    b.root = null;
    b.view = .{};
    b.len = 0;
    b.ptr = undefined;
    return b;
}

/// A zrun.Bytes of an object with the buffer protocol (one already: itself),
/// as a new reference; null with an exception.
pub fn of(o: *PyObject) ?*PyObject {
    if (isBytes(o)) {
        py.Py_IncRef(o);
        return o;
    }
    const b = alloc() orelse return null;
    if (PyObject_GetBuffer(o, &b.view, buf_simple) != 0) {
        // (no view: nothing to release)
        b.view = .{};
        py.Py_DecRef(@ptrCast(b));
        return null;
    }
    b.ptr = @ptrCast(b.view.buf orelse @as(*anyopaque, @ptrFromInt(1)));
    b.len = @intCast(b.view.len);
    return @ptrCast(b);
}

/// A zrun.Bytes of part of one's memory (shared), as a new reference.
pub fn slice(of_: *PyObject, ptr: [*]const u8, len: usize) ?*PyObject {
    const src = as(of_);
    const b = alloc() orelse return null;
    const root = src.root orelse of_;
    py.Py_IncRef(root);
    b.root = root;
    b.ptr = ptr;
    b.len = len;
    return @ptrCast(b);
}

fn dealloc(o: ?*PyObject) callconv(.c) void {
    const b = as(o.?);
    if (b.root) |r| py.Py_DecRef(r) else if (b.view.obj != null) PyBuffer_Release(&b.view);
    const t = o.?.ob_type;
    const free: py.c.freefunc = @ptrCast(py.c.PyType_GetSlot(t, py.c.Py_tp_free));
    free.?(o);
    py.Py_DecRef(@ptrCast(@alignCast(t)));
}

fn new(_: ?*py.c.PyTypeObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    if (kwargs != null and py.c.PyDict_Size(kwargs) != 0) {
        ph.raise(py.PyExc_TypeError(), "Bytes() takes no keyword arguments", .{});
        return null;
    }
    if (py.c.PyTuple_Size(args) != 1) {
        ph.raise(py.PyExc_TypeError(), "Bytes(data) takes one argument (bytes, bytearray, memoryview, mmap...)", .{});
        return null;
    }
    return of(py.c.PyTuple_GetItem(args, 0).?);
}

fn length(o: ?*PyObject) callconv(.c) isize {
    return @intCast(as(o.?).len);
}

/// The index of item `i` (negative: from the end), or null: out of range.
pub fn index(i: i64, len: usize) ?usize {
    const n: i64 = @intCast(len);
    const j = if (i < 0) i + n else i;
    if (j < 0 or j >= n) return null;
    return @intCast(j);
}

fn item(o: ?*PyObject, i: isize) callconv(.c) ?*PyObject {
    const b = as(o.?);
    const j = index(i, b.len) orelse {
        ph.raise(py.PyExc_IndexError(), "index out of range", .{});
        return null;
    };
    return py.c.PyLong_FromLong(b.ptr[j]);
}

fn subscript(o: ?*PyObject, key: ?*PyObject) callconv(.c) ?*PyObject {
    const b = as(o.?);
    if (typeObj(key.?) == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PySlice_Type"))))) {
        var start: isize = 0;
        var stop: isize = 0;
        var step: isize = 0;
        if (py.c.PySlice_Unpack(key.?, &start, &stop, &step) < 0) return null;
        const n = py.c.PySlice_AdjustIndices(@intCast(b.len), &start, &stop, step);
        if (step == 1) return slice(o.?, b.ptr + @as(usize, @intCast(start)), @intCast(n));
        // (with a step: its bytes, copied, as a zrun.Bytes of them)
        const copy = py.c.PyBytes_FromStringAndSize(null, n) orelse return null;
        defer py.Py_DecRef(copy);
        const dst: [*]u8 = @ptrCast(py.c.PyBytes_AsString(copy) orelse return null);
        var at = start;
        for (0..@intCast(n)) |k| {
            dst[k] = b.ptr[@intCast(at)];
            at += step;
        }
        return of(copy);
    }
    const i = py.c.PyNumber_AsSsize_t(key.?, py.c.PyExc_IndexError);
    if (i == -1 and py.c.PyErr_Occurred() != null) return null;
    return item(o, i);
}

/// The memory of anything with the buffer protocol, for a moment: `f`
/// given it.
fn withData(o: *PyObject, comptime T: type, f: anytype, extra: anytype) ?T {
    if (isBytes(o)) return f(as(o).bytes(), extra);
    var view: Py_buffer = .{};
    if (PyObject_GetBuffer(o, &view, buf_simple) != 0) return null;
    defer PyBuffer_Release(&view);
    const p: [*]const u8 = @ptrCast(view.buf orelse @as(*anyopaque, @ptrFromInt(1)));
    return f(p[0..@intCast(view.len)], extra);
}

fn compare(o: ?*PyObject, other: ?*PyObject, op: c_int) callconv(.c) ?*PyObject {
    if (op != py.c.Py_EQ and op != py.c.Py_NE) return ref(py.c.Py_NotImplemented());
    // (with data of any kind: their bytes; anything else isn't equal)
    if (!isBytes(other.?) and !isData(other.?)) return ref(if (op == py.c.Py_EQ) py.Py_False() else py.Py_True());
    const mine = as(o.?).bytes();
    const same = withData(other.?, bool, struct {
        fn f(theirs: []const u8, a: []const u8) bool {
            return std.mem.eql(u8, a, theirs);
        }
    }.f, mine) orelse return null;
    return ref(if (same == (op == py.c.Py_EQ)) py.Py_True() else py.Py_False());
}

/// Its bytes, copied (a new bytes object).
pub fn toBytes(b: *const BytesObject) ?*PyObject {
    return py.c.PyBytes_FromStringAndSize(@ptrCast(b.ptr), @intCast(b.len));
}

fn hash(o: ?*PyObject) callconv(.c) isize {
    // (as its bytes': equal to them, the same hash)
    const copy = toBytes(as(o.?)) orelse return -1;
    defer py.Py_DecRef(copy);
    return py.c.PyObject_Hash(copy);
}

fn repr(o: ?*PyObject) callconv(.c) ?*PyObject {
    const b = as(o.?);
    const shown = py.c.PyBytes_FromStringAndSize(@ptrCast(b.ptr), @intCast(@min(b.len, 32))) orelse return null;
    defer py.Py_DecRef(shown);
    const r = py.c.PyObject_Repr(shown) orelse return null;
    defer py.Py_DecRef(r);
    const text = ph.utf8(r, "repr") orelse return null;
    var buf: [256]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "zrun.Bytes({s}{s})", .{ text, if (b.len > 32) "..." else "" }) catch return null;
    return ph.newString(s);
}

fn bytesMethod(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    return toBytes(as(o.?));
}

var methods = [_]py.c.PyMethodDef{
    .{ .ml_name = "__bytes__", .ml_meth = @ptrCast(@constCast(&bytesMethod)), .ml_flags = py.c.METH_NOARGS, .ml_doc = "Its bytes, copied." },
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

var slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_new, .pfunc = @ptrCast(@constCast(&new)) },
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&dealloc)) },
    .{ .slot = py.c.Py_mp_length, .pfunc = @ptrCast(@constCast(&length)) },
    .{ .slot = py.c.Py_sq_length, .pfunc = @ptrCast(@constCast(&length)) },
    .{ .slot = py.c.Py_sq_item, .pfunc = @ptrCast(@constCast(&item)) },
    .{ .slot = py.c.Py_mp_subscript, .pfunc = @ptrCast(@constCast(&subscript)) },
    .{ .slot = py.c.Py_tp_richcompare, .pfunc = @ptrCast(@constCast(&compare)) },
    .{ .slot = py.c.Py_tp_hash, .pfunc = @ptrCast(@constCast(&hash)) },
    .{ .slot = py.c.Py_tp_repr, .pfunc = @ptrCast(@constCast(&repr)) },
    .{ .slot = py.c.Py_tp_methods, .pfunc = @ptrCast(@constCast(&methods)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("Bytes(data): a read-only view of data's memory (bytes, bytearray, memoryview, mmap), not copied; slices share it.")) },
    .{ .slot = 0, .pfunc = null },
};

var spec = py.c.PyType_Spec{
    .name = "zrun.Bytes",
    .basicsize = @sizeOf(BytesObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &slots,
};

// ----------------------------------------------------------------------
// rt.u8(data, i) and the others
// ----------------------------------------------------------------------

/// What rt reads: its width, sign, byte order.
pub const Read = enum(u32) {
    u8,
    i8,
    u16le,
    u16be,
    i16le,
    i16be,
    u32le,
    u32be,
    i32le,
    i32be,
    u64le,
    u64be,
    i64le,
    i64be,

    pub fn width(r: Read) usize {
        return switch (r) {
            .u8, .i8 => 1,
            .u16le, .u16be, .i16le, .i16be => 2,
            .u32le, .u32be, .i32le, .i32be => 4,
            else => 8,
        };
    }

    pub fn signed(r: Read) bool {
        return switch (r) {
            .i8, .i16le, .i16be, .i32le, .i32be, .i64le, .i64be => true,
            else => false,
        };
    }

    pub fn little(r: Read) bool {
        return switch (r) {
            .u16le, .i16le, .u32le, .i32le, .u64le, .i64le, .u8, .i8 => true,
            else => false,
        };
    }
};

/// A value read: a u64 beyond 63 bits, its bits as an i64 (`big` set)
pub const Got = struct { value: i64, big: bool };

/// The value at offset `at` of some bytes, or null: past their end.
pub fn readAt(r: Read, data: []const u8, at: i64) ?Got {
    const w = r.width();
    if (at < 0 or @as(u64, @intCast(at)) > data.len or data.len - @as(usize, @intCast(at)) < w) return null;
    const p = data[@intCast(at)..][0..w];
    var x: u64 = 0;
    if (r.little()) {
        var k: usize = w;
        while (k > 0) {
            k -= 1;
            x = (x << 8) | p[k];
        }
    } else for (p) |c| {
        x = (x << 8) | c;
    }
    if (r.signed() and w < 8) {
        const shift: u6 = @intCast(64 - 8 * w);
        return .{ .value = @as(i64, @bitCast(x << shift)) >> shift, .big = false };
    }
    return .{ .value = @bitCast(x), .big = !r.signed() and w == 8 and x >> 63 != 0 };
}

/// rt.u8(data, i) and the others of Python objects: an int; IndexError past
/// the end, TypeError for what isn't data.
pub fn read(r: Read, data: *PyObject, at_obj: *PyObject) ?*PyObject {
    const at = py.c.PyLong_AsLongLong(at_obj);
    if (at == -1 and py.c.PyErr_Occurred() != null) return null;
    if (!isBytes(data) and !isData(data)) {
        const name = ph.attr(typeObj(data), "__name__") orelse return null;
        defer py.Py_DecRef(name);
        ph.raise(py.PyExc_TypeError(), "rt.{s}() reads data (zrun.Bytes, bytes...), not '{s}'", .{ @tagName(r), ph.utf8(name, "name") orelse "?" });
        return null;
    }
    const Where = struct { r: Read, at: i64 };
    const got = withData(data, ?Got, struct {
        fn f(bytes_: []const u8, a: Where) ?Got {
            return readAt(a.r, bytes_, a.at);
        }
    }.f, Where{ .r = r, .at = at }) orelse return null;
    const v = got orelse {
        ph.raise(py.PyExc_IndexError(), "rt.{s}(): offset {d} out of range", .{ @tagName(r), at });
        return null;
    };
    if (v.big) return py.c.PyLong_FromUnsignedLongLong(@bitCast(v.value));
    return py.c.PyLong_FromLongLong(v.value);
}
