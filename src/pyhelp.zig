//! Small helpers over the Python C API. Everything here needs the GIL.

const std = @import("std");
const pyoz = @import("PyOZ");
pub const py = pyoz.py;
pub const PyObject = pyoz.PyObject;

/// The running Python's minor version (3.x), read as the module's made:
/// compiled code words errors as this Python does.
pub var minor: u32 = 10;

pub fn initVersion() error{Python}!void {
    const info = py.c.PySys_GetObject("version_info") orelse return error.Python;
    const m = py.c.PySequence_GetItem(info, 1) orelse return error.Python;
    defer py.Py_DecRef(m);
    minor = @intCast((try toInt(m)) orelse 10);
}

/// An object's type (Py_TYPE).
pub inline fn typeOf(o: *PyObject) *py.PyTypeObject {
    return py.Py_TYPE(o).?;
}

/// An object's reference count (Py_REFCNT: the first word of an object;
/// immortal ones read as huge).
pub inline fn refcnt(o: *PyObject) isize {
    return @as(*const isize, @ptrCast(@alignCast(o))).*;
}

/// Raise `exc` with a formatted message.
pub fn raise(exc: *PyObject, comptime fmt: []const u8, args: anytype) void {
    var buf: [512]u8 = undefined;
    const msg = std.fmt.bufPrintZ(&buf, fmt, args) catch blk: {
        buf[buf.len - 1] = 0;
        break :blk buf[0 .. buf.len - 1 :0];
    };
    py.PyErr_SetString(exc, msg.ptr);
}

/// UTF-8 of a str (borrowed from the object), or null with TypeError set.
pub fn utf8(obj: *PyObject, what: []const u8) ?[]const u8 {
    if (!py.PyUnicode_Check(obj)) {
        raise(py.PyExc_TypeError(), "{s} must be a str", .{what});
        return null;
    }
    var len: py.Py_ssize_t = 0;
    const ptr = py.c.PyUnicode_AsUTF8AndSize(obj, &len) orelse return null;
    return ptr[0..@intCast(len)];
}

/// `obj.name` (a new reference), or null with the error set.
pub fn attr(obj: *PyObject, name: [*:0]const u8) ?*PyObject {
    return py.c.PyObject_GetAttrString(obj, name);
}

/// `obj.name` as a str copied into `arena`, or null (None, or an error set).
pub fn attrString(arena: std.mem.Allocator, obj: *PyObject, name: [*:0]const u8) error{ Python, OutOfMemory }!?[]const u8 {
    const v = attr(obj, name) orelse return error.Python;
    defer py.Py_DecRef(v);
    if (v == py.Py_None()) return null;
    const s = utf8(v, std.mem.span(name)) orelse return error.Python;
    return try arena.dupe(u8, s);
}

/// An int object's value, or null for None.
pub fn toInt(v: *PyObject) error{Python}!?i64 {
    if (v == py.Py_None()) return null;
    const n = py.c.PyLong_AsLongLong(v);
    if (n == -1 and py.c.PyErr_Occurred() != null) return error.Python;
    return n;
}

pub fn attrInt(obj: *PyObject, name: [*:0]const u8) error{Python}!?i64 {
    const v = attr(obj, name) orelse return error.Python;
    defer py.Py_DecRef(v);
    return toInt(v);
}

pub fn attrBool(obj: *PyObject, name: [*:0]const u8) error{Python}!bool {
    const v = attr(obj, name) orelse return error.Python;
    defer py.Py_DecRef(v);
    const t = py.c.PyObject_IsTrue(v);
    if (t < 0) return error.Python;
    return t == 1;
}

/// A `(start, end)` pair, or null for None.
pub fn toSpan(v: *PyObject) error{Python}!?[2]u32 {
    if (v == py.Py_None()) return null;
    var out: [2]u32 = undefined;
    for (0..2) |i| {
        const item = py.c.PySequence_GetItem(v, @intCast(i)) orelse return error.Python;
        defer py.Py_DecRef(item);
        const n = (try toInt(item)) orelse return error.Python;
        out[i] = @intCast(std.math.clamp(n, 0, std.math.maxInt(u32)));
    }
    return out;
}

/// The message of the Python exception being raised, cleared, copied into
/// `buf` ("ValueError: ..."), for logging instead of raising.
pub fn takeError(buf: []u8) []const u8 {
    var t: ?*PyObject = null;
    var v: ?*PyObject = null;
    var tb: ?*PyObject = null;
    py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    // (as raised: a KeyError of a dict's is its key in a tuple until then)
    py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    defer inline for (.{ t, v, tb }) |o| {
        if (o) |obj| py.Py_DecRef(obj);
    };
    var len: usize = 0;
    if (t) |typ| {
        if (attr(typ, "__name__")) |name| {
            defer py.Py_DecRef(name);
            if (utf8(name, "name")) |s| len += copy(buf[len..], s) else py.c.PyErr_Clear();
        } else py.c.PyErr_Clear();
    }
    if (v) |val| {
        if (py.c.PyObject_Str(val)) |s| {
            defer py.Py_DecRef(s);
            if (utf8(s, "message")) |text| {
                len += copy(buf[len..], ": ");
                len += copy(buf[len..], text);
            } else py.c.PyErr_Clear();
        } else py.c.PyErr_Clear();
    }
    return buf[0..len];
}

fn copy(dst: []u8, src: []const u8) usize {
    const n = @min(dst.len, src.len);
    @memcpy(dst[0..n], src[0..n]);
    return n;
}

/// A new str from UTF-8 (invalid bytes replaced).
pub fn newString(s: []const u8) ?*PyObject {
    return py.c.PyUnicode_DecodeUTF8(s.ptr, @intCast(s.len), "replace");
}
