//! The Python types zrun makes when the module loads: the checked 64-bit
//! integer (`zrun.I64`) and the exceptions (`zrun.Error`, `LoadError`,
//! `IntegerOverflow`, and the control flow `Return`, `Break`, `Continue`).
//!
//! Integers are 64-bit in every mode. Semantics run as Python get them as
//! `I64`: an `int` whose arithmetic raises `IntegerOverflow` when a result
//! leaves the 64-bit range, so a program overflows at the same operation
//! whether its semantics run as Python or compiled.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;

pub const INT_MIN: i64 = std.math.minInt(i64);
pub const INT_MAX: i64 = std.math.maxInt(i64);

/// The types, set by init() (borrowed: the module holds them)
pub var I64: *PyObject = undefined;
pub var Error: *PyObject = undefined;
pub var LoadError: *PyObject = undefined;
pub var IntegerOverflow: *PyObject = undefined;
pub var Return: *PyObject = undefined;
pub var Break: *PyObject = undefined;
pub var Continue: *PyObject = undefined;
/// int's tp_new, to make I64 instances (calling int.__new__ is refused
/// for a subclass with its own __new__)
var int_new: py.c.newfunc = null;

/// Make the types and add them to the module: 0, or -1 with an exception.
pub fn init(module: *PyObject) callconv(.c) c_int {
    initTypes(module) catch return -1;
    return 0;
}

fn initTypes(module: *PyObject) !void {
    const int_type: *PyObject = @ptrCast(py.types.typeObject("PyLong_Type"));
    int_new = @ptrCast(py.c.PyType_GetSlot(@ptrCast(int_type), py.c.Py_tp_new) orelse return error.Python);

    const bases = py.c.PyTuple_Pack(1, int_type) orelse return error.Python;
    defer py.Py_DecRef(bases);
    I64 = py.c.PyType_FromSpecWithBases(&i64_spec, bases) orelse return error.Python;
    try add(module, "I64", I64);

    IntegerOverflow = try newException(module, "IntegerOverflow", py.PyExc_ArithmeticError(), "An integer result outside the 64-bit range.");
    Error = try newException(module, "Error", py.PyExc_Exception(), "A runtime error of the program. `diagnostic` is where (a zgram.Diagnostic at the failing node), `stack` the language's calls that led there (innermost first, (function, Diagnostic) pairs); str() renders it with the source.");
    LoadError = try newException(module, "LoadError", py.PyExc_Exception(), "The program has errors found before it runs (syntax errors, the rules' errors): `diagnostics` lists them, warnings included.");
    Return = try newException(module, "Return", py.PyExc_Exception(), "raise rt.Return(value): return from the function being run.");
    Break = try newException(module, "Break", py.PyExc_Exception(), "raise rt.Break(): leave the loop rt.loop() is running.");
    Continue = try newException(module, "Continue", py.PyExc_Exception(), "raise rt.Continue(): go on with the loop's next iteration.");
}

fn newException(module: *PyObject, comptime name: [:0]const u8, base: *PyObject, comptime doc: [:0]const u8) !*PyObject {
    const t = py.c.PyErr_NewExceptionWithDoc("zrun." ++ name, doc, base, null) orelse return error.Python;
    try add(module, name, t);
    return t;
}

fn add(module: *PyObject, name: [*:0]const u8, obj: *PyObject) !void {
    // (the module gets its own reference; ours stays for the globals)
    if (py.c.PyModule_AddObjectRef(module, name, obj) != 0) return error.Python;
}

// ----------------------------------------------------------------------
// I64
// ----------------------------------------------------------------------

/// An int as an I64 (checked), as a new reference; null with an exception
/// (IntegerOverflow) if it doesn't fit. Anything else is returned as it is.
pub fn wrap(value: *PyObject) ?*PyObject {
    if (value.ob_type != @as(*py.c.PyTypeObject, @ptrCast(py.types.typeObject("PyLong_Type")))) {
        py.Py_IncRef(value);
        return value;
    }
    var overflow: c_int = 0;
    _ = py.c.PyLong_AsLongLongAndOverflow(value, &overflow);
    if (overflow != 0) {
        py.c.PyErr_SetString(IntegerOverflow, "integer overflow");
        return null;
    }
    const args = py.c.PyTuple_Pack(1, value) orelse return null;
    defer py.Py_DecRef(args);
    return int_new.?(@ptrCast(I64), args, null);
}

/// Wrap and consume `value` (a new reference, or null passed through).
pub fn wrapOwned(value: ?*PyObject) ?*PyObject {
    const v = value orelse return null;
    defer py.Py_DecRef(v);
    return wrap(v);
}

/// An I64 from a Zig integer.
pub fn fromInt(v: i64) ?*PyObject {
    const value = py.c.PyLong_FromLongLong(v) orelse return null;
    return wrapOwned(value);
}

/// An operand as the plain int behind an I64 (a new reference); others as
/// they are.
fn plain(obj: ?*PyObject) ?*PyObject {
    const o = obj orelse return null;
    if (o.ob_type == @as(*py.c.PyTypeObject, @ptrCast(I64))) return py.c.PyNumber_Long(o);
    py.Py_IncRef(o);
    return o;
}

fn binary(comptime op: anytype) fn (?*PyObject, ?*PyObject) callconv(.c) ?*PyObject {
    return struct {
        fn f(a: ?*PyObject, b: ?*PyObject) callconv(.c) ?*PyObject {
            const x = plain(a) orelse return null;
            defer py.Py_DecRef(x);
            const y = plain(b) orelse return null;
            defer py.Py_DecRef(y);
            return checked(op(x, y));
        }
    }.f;
}

fn unary(comptime op: anytype) fn (?*PyObject) callconv(.c) ?*PyObject {
    return struct {
        fn f(a: ?*PyObject) callconv(.c) ?*PyObject {
            const x = plain(a) orelse return null;
            defer py.Py_DecRef(x);
            return checked(op(x));
        }
    }.f;
}

fn power(a: ?*PyObject, b: ?*PyObject, m: ?*PyObject) callconv(.c) ?*PyObject {
    const x = plain(a) orelse return null;
    defer py.Py_DecRef(x);
    const y = plain(b) orelse return null;
    defer py.Py_DecRef(y);
    return checked(py.c.PyNumber_Power(x, y, m));
}

fn divmod(a: ?*PyObject, b: ?*PyObject) callconv(.c) ?*PyObject {
    const x = plain(a) orelse return null;
    defer py.Py_DecRef(x);
    const y = plain(b) orelse return null;
    defer py.Py_DecRef(y);
    const r = py.c.PyNumber_Divmod(x, y) orelse return null;
    defer py.Py_DecRef(r);
    if (py.c.PyTuple_Size(r) != 2) {
        py.Py_IncRef(r);
        return r;
    }
    const q = wrap(py.c.PyTuple_GetItem(r, 0).?) orelse return null;
    const rem = wrap(py.c.PyTuple_GetItem(r, 1).?) orelse {
        py.Py_DecRef(q);
        return null;
    };
    const out = py.c.PyTuple_Pack(2, q, rem);
    py.Py_DecRef(q);
    py.Py_DecRef(rem);
    return out;
}

/// A result: an int is checked and made an I64; the rest as it is.
fn checked(result: ?*PyObject) ?*PyObject {
    return wrapOwned(result);
}

/// I64(x): int(x), checked.
fn i64New(_: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const int_type: *PyObject = @ptrCast(py.types.typeObject("PyLong_Type"));
    return wrapOwned(py.c.PyObject_Call(int_type, args, kwargs));
}

var i64_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_new, .pfunc = @ptrCast(@constCast(&i64New)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("A checked 64-bit integer: an int whose arithmetic raises zrun.IntegerOverflow outside the 64-bit range.")) },
    .{ .slot = py.c.Py_nb_add, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Add))) },
    .{ .slot = py.c.Py_nb_subtract, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Subtract))) },
    .{ .slot = py.c.Py_nb_multiply, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Multiply))) },
    .{ .slot = py.c.Py_nb_floor_divide, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_FloorDivide))) },
    .{ .slot = py.c.Py_nb_true_divide, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_TrueDivide))) },
    .{ .slot = py.c.Py_nb_remainder, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Remainder))) },
    .{ .slot = py.c.Py_nb_divmod, .pfunc = @ptrCast(@constCast(&divmod)) },
    .{ .slot = py.c.Py_nb_power, .pfunc = @ptrCast(@constCast(&power)) },
    .{ .slot = py.c.Py_nb_lshift, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Lshift))) },
    .{ .slot = py.c.Py_nb_rshift, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Rshift))) },
    .{ .slot = py.c.Py_nb_and, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_And))) },
    .{ .slot = py.c.Py_nb_or, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Or))) },
    .{ .slot = py.c.Py_nb_xor, .pfunc = @ptrCast(@constCast(&binary(py.c.PyNumber_Xor))) },
    .{ .slot = py.c.Py_nb_negative, .pfunc = @ptrCast(@constCast(&unary(py.c.PyNumber_Negative))) },
    .{ .slot = py.c.Py_nb_positive, .pfunc = @ptrCast(@constCast(&unary(py.c.PyNumber_Positive))) },
    .{ .slot = py.c.Py_nb_absolute, .pfunc = @ptrCast(@constCast(&unary(py.c.PyNumber_Absolute))) },
    .{ .slot = py.c.Py_nb_invert, .pfunc = @ptrCast(@constCast(&unary(py.c.PyNumber_Invert))) },
    .{ .slot = 0, .pfunc = null },
};

var i64_spec = py.c.PyType_Spec{
    .name = "zrun.I64",
    .basicsize = 0,
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &i64_slots,
};

// ----------------------------------------------------------------------
// Control flow
// ----------------------------------------------------------------------

pub const Control = enum { none, ret, brk, cont };

/// Which control flow exception is being raised, if one is.
pub fn pendingControl() Control {
    if (py.c.PyErr_Occurred() == null) return .none;
    if (py.c.PyErr_ExceptionMatches(Return) != 0) return .ret;
    if (py.c.PyErr_ExceptionMatches(Break) != 0) return .brk;
    if (py.c.PyErr_ExceptionMatches(Continue) != 0) return .cont;
    return .none;
}

/// Take the Return being raised: its value (a new reference; None without one).
pub fn takeReturn() ?*PyObject {
    var t: ?*PyObject = null;
    var v: ?*PyObject = null;
    var tb: ?*PyObject = null;
    py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    defer inline for (.{ t, tb }) |o| {
        if (o) |obj| py.Py_DecRef(obj);
    };
    const exc = v orelse {
        py.Py_IncRef(py.Py_None());
        return py.Py_None();
    };
    defer py.Py_DecRef(exc);
    const args = py.c.PyObject_GetAttrString(exc, "args") orelse return null;
    defer py.Py_DecRef(args);
    if (py.c.PyTuple_Size(args) < 1) {
        py.Py_IncRef(py.Py_None());
        return py.Py_None();
    }
    return wrap(py.c.PyTuple_GetItem(args, 0).?);
}
