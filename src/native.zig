//! Native host functions: `lang.native_host(name, capsule)` registers a
//! function of a native library (a PyCapsule named "zrun.native.v1" over a
//! NativeV1) as the host function `name`. Compiled code calls it directly
//! (no Python, no GIL) when its arguments are of the kinds it takes;
//! anything else, and the reference mode, call it through this module's
//! callable (`zrun.NativeHost`), which converts Python values the same way
//! and raises the same errors.
//!
//! The C side (a header would say):
//!
//!     typedef struct { const uint8_t *ptr; uint64_t len; } zrun_bytes;
//!     typedef union { int64_t i; double f; zrun_bytes b; } zrun_arg;
//!     typedef struct {
//!         uint32_t abi;               /* 1 */
//!         const char *signature;      /* "bi:i": kinds of the arguments, ':', the result's */
//!         int (*call)(void *state, const zrun_arg *args, uint64_t nargs, zrun_arg *result);
//!         void *state;
//!         const char *(*error)(void *state, int code);   /* code != 0: why */
//!     } zrun_native_v1;
//!
//! Kinds: i an int (64 bits), f a float, ? a bool, b data (zrun.Bytes:
//! its bytes, read only, valid during the call); a result also n (None).

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const bytes_mod = @import("bytes.zig");

const PyObject = py.PyObject;

pub const capsule_name = "zrun.native.v1";

pub const Arg = extern union {
    i: i64,
    f: f64,
    b: extern struct { ptr: [*]const u8, len: u64 },
};

pub const NativeV1 = extern struct {
    abi: u32,
    signature: [*:0]const u8,
    call: *const fn (state: ?*anyopaque, args: [*]const Arg, nargs: u64, result: *Arg) callconv(.c) c_int,
    state: ?*anyopaque,
    @"error": ?*const fn (state: ?*anyopaque, code: c_int) callconv(.c) ?[*:0]const u8,
};

pub const Kind = enum(u8) { int = 'i', float = 'f', bool = '?', bytes = 'b', none = 'n' };

pub const max_args = 8;

pub const NativeObject = extern struct {
    ob_base: py.c.PyObject,
    /// The capsule (kept: the library's struct lives as long as it)
    capsule: ?*PyObject,
    native: *const NativeV1,
    name: ?*PyObject,
    nargs: u32,
    kinds: [max_args]Kind,
    result: Kind,

    pub fn args(self: *const NativeObject) []const Kind {
        return self.kinds[0..self.nargs];
    }

    pub fn nameText(self: *const NativeObject) []const u8 {
        return ph.utf8(self.name.?, "name") orelse "native";
    }
};

pub var NativeType: *PyObject = undefined;

fn ref(o: *PyObject) *PyObject {
    py.Py_IncRef(o);
    return o;
}

pub fn init(module: *PyObject) !void {
    NativeType = py.c.PyType_FromSpec(&spec) orelse return error.Python;
    if (py.c.PyModule_AddObject(module, "NativeHost", NativeType) != 0) return error.Python;
    py.Py_IncRef(NativeType);
}

pub fn isNative(o: *PyObject) bool {
    return @as(*PyObject, @ptrCast(@alignCast(ph.typeOf(o)))) == NativeType;
}

pub fn as(o: *PyObject) *NativeObject {
    return @ptrCast(@alignCast(o));
}

/// A native host function of a capsule, named (a new reference), or null
/// with an exception.
pub fn make(name: *PyObject, capsule: *PyObject) ?*PyObject {
    const ptr = py.c.PyCapsule_GetPointer(capsule, capsule_name) orelse return null;
    const native: *const NativeV1 = @ptrCast(@alignCast(ptr));
    if (native.abi != 1) {
        ph.raise(py.PyExc_ValueError(), "a native host function of ABI {d}: zrun knows ABI 1", .{native.abi});
        return null;
    }
    const sig = std.mem.span(native.signature);
    const colon = std.mem.indexOfScalar(u8, sig, ':') orelse sig.len;
    if (colon > max_args or colon + 2 != sig.len) {
        ph.raise(py.PyExc_ValueError(), "a native host function's signature is its arguments' kinds, ':', its result's (\"bi:i\"; at most {d} arguments), not '{s}'", .{ max_args, sig });
        return null;
    }
    const allocf: py.c.allocfunc = @ptrCast(py.c.PyType_GetSlot(@ptrCast(NativeType), py.c.Py_tp_alloc));
    const o = allocf.?(@ptrCast(NativeType), 0) orelse return null;
    const n = as(o);
    n.capsule = null;
    n.name = null;
    n.native = native;
    n.nargs = @intCast(colon);
    for (sig[0..colon], 0..) |ch, i| {
        n.kinds[i] = kindOf(ch, false) orelse {
            py.Py_DecRef(o);
            ph.raise(py.PyExc_ValueError(), "'{c}' isn't a kind of argument (i, f, ?, b)", .{ch});
            return null;
        };
    }
    n.result = kindOf(sig[colon + 1], true) orelse {
        py.Py_DecRef(o);
        ph.raise(py.PyExc_ValueError(), "'{c}' isn't a kind of result (i, f, ?, n)", .{sig[colon + 1]});
        return null;
    };
    py.Py_IncRef(capsule);
    n.capsule = capsule;
    py.Py_IncRef(name);
    n.name = name;
    return o;
}

fn kindOf(ch: u8, result: bool) ?Kind {
    return switch (ch) {
        'i' => .int,
        'f' => .float,
        '?' => .bool,
        'b' => if (result) null else .bytes,
        'n' => if (result) .none else null,
        else => null,
    };
}

/// Why a call failed (its message), as the reference mode raises it:
/// RuntimeError(message).
pub fn errorText(n: *const NativeObject, code: c_int) []const u8 {
    if (n.native.@"error") |e| if (e(n.native.state, code)) |m| return std.mem.span(m);
    return "failed";
}

fn dealloc(o: ?*PyObject) callconv(.c) void {
    const n = as(o.?);
    if (n.capsule) |c| py.Py_DecRef(c);
    if (n.name) |x| py.Py_DecRef(x);
    const t = o.?.ob_type;
    const free: py.c.freefunc = @ptrCast(py.c.PyType_GetSlot(t, py.c.Py_tp_free));
    free.?(o);
    py.Py_DecRef(@ptrCast(@alignCast(t)));
}

/// Called from Python (the reference mode; compiled code with arguments of
/// other kinds): the arguments converted to its kinds (TypeError if one
/// isn't), the result to Python; a failure, RuntimeError(its message).
fn call(o: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const n = as(o.?);
    if (kwargs != null and py.c.PyDict_Size(kwargs) != 0) {
        ph.raise(py.PyExc_TypeError(), "{s}() takes no keyword arguments", .{n.nameText()});
        return null;
    }
    const given: usize = @intCast(py.c.PyTuple_Size(args));
    if (given != n.nargs) {
        ph.raise(py.PyExc_TypeError(), "{s}() takes {d} arguments, {d} given", .{ n.nameText(), n.nargs, given });
        return null;
    }
    var vals: [max_args]Arg = undefined;
    // (data as zrun.Bytes, kept during the call)
    var keep: [max_args]?*PyObject = .{null} ** max_args;
    defer for (keep) |k| if (k) |x| py.Py_DecRef(x);
    for (n.args(), 0..) |kind, i| {
        const x = py.c.PyTuple_GetItem(args, @intCast(i)).?;
        switch (kind) {
            .int => {
                if (!isInt(x)) return wrongKind(n, i, "an int");
                var overflow: c_int = 0;
                const v = py.c.PyLong_AsLongLongAndOverflow(x, &overflow);
                if (overflow != 0) return wrongKind(n, i, "an int of 64 bits");
                vals[i] = .{ .i = v };
            },
            .float => {
                if (@as(*PyObject, @ptrCast(@alignCast(ph.typeOf(x)))) != @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PyFloat_Type"))))) return wrongKind(n, i, "a float");
                vals[i] = .{ .f = py.c.PyFloat_AsDouble(x) };
            },
            .bool => {
                if (x != py.Py_True() and x != py.Py_False()) return wrongKind(n, i, "a bool");
                vals[i] = .{ .i = @intFromBool(x == py.Py_True()) };
            },
            .bytes => {
                if (!bytes_mod.isBytes(x) and !bytes_mod.isData(x)) return wrongKind(n, i, "data (zrun.Bytes, bytes...)");
                const b = bytes_mod.of(x) orelse return null;
                keep[i] = b;
                const view = bytes_mod.as(b);
                vals[i] = .{ .b = .{ .ptr = view.ptr, .len = view.len } };
            },
            .none => unreachable,
        }
    }
    var out: Arg = .{ .i = 0 };
    const code = n.native.call(n.native.state, &vals, n.nargs, &out);
    if (code != 0) {
        ph.raise(py.PyExc_RuntimeError(), "{s}", .{errorText(n, code)});
        return null;
    }
    return switch (n.result) {
        .int => py.c.PyLong_FromLongLong(out.i),
        .float => py.c.PyFloat_FromDouble(out.f),
        .bool => ref(if (out.i != 0) py.Py_True() else py.Py_False()),
        else => ref(py.Py_None()),
    };
}

fn isInt(x: *PyObject) bool {
    const t = ph.typeOf(x);
    const long_type: *py.c.PyTypeObject = @ptrCast(@alignCast(py.types.typeObject("PyLong_Type")));
    const bool_type: *py.c.PyTypeObject = @ptrCast(@alignCast(py.types.typeObject("PyBool_Type")));
    return t != bool_type and py.c.PyType_IsSubtype(t, long_type) != 0;
}

fn wrongKind(n: *const NativeObject, i: usize, want: []const u8) ?*PyObject {
    ph.raise(py.PyExc_TypeError(), "{s}() takes {s} as argument {d}", .{ n.nameText(), want, i + 1 });
    return null;
}

fn getName(o: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    return ref(as(o.?).name.?);
}

var getset = [_]py.c.PyGetSetDef{
    .{ .name = "__name__", .get = @ptrCast(@constCast(&getName)), .set = null, .doc = "The host function's name.", .closure = null },
    .{ .name = null, .get = null, .set = null, .doc = null, .closure = null },
};

var slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_call, .pfunc = @ptrCast(@constCast(&call)) },
    .{ .slot = py.c.Py_tp_getset, .pfunc = @ptrCast(@constCast(&getset)) },
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&dealloc)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("A native library's function, a host function of a language (Language.native_host): called directly by compiled code.")) },
    .{ .slot = 0, .pfunc = null },
};

var spec = py.c.PyType_Spec{
    .name = "zrun.NativeHost",
    .basicsize = @sizeOf(NativeObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &slots,
};
