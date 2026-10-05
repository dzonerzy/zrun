//! `rt.wrapping_add(a, b)`, `rt.wrapping_sub`, `rt.wrapping_mul`: 64-bit
//! two's complement arithmetic, as languages with fixed-size ints do it
//! (Lua's): the same in every mode. Ints of 64 bits only (an I64 or a plain
//! int); the result a plain int.

const ph = @import("pyhelp.zig");
const py = ph.py;

const PyObject = py.PyObject;

pub const Op = enum(u32) { add, sub, mul };

pub fn apply(op: Op, x: i64, y: i64) i64 {
    return switch (op) {
        .add => x +% y,
        .sub => x -% y,
        .mul => x *% y,
    };
}

/// Of Python ints: a new int, or null with a TypeError.
pub fn ofPython(op: Op, a: *PyObject, b: *PyObject) ?*PyObject {
    const x = int64(op, a) orelse return null;
    const y = int64(op, b) orelse return null;
    return py.c.PyLong_FromLongLong(apply(op, x, y));
}

fn int64(op: Op, o: *PyObject) ?i64 {
    const t = ph.typeOf(o);
    const long_type: *py.c.PyTypeObject = @ptrCast(@alignCast(py.types.typeObject("PyLong_Type")));
    const bool_type: *py.c.PyTypeObject = @ptrCast(@alignCast(py.types.typeObject("PyBool_Type")));
    if (t != bool_type and py.c.PyType_IsSubtype(t, long_type) != 0) {
        var overflow: c_int = 0;
        const n = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
        if (overflow == 0 and !(n == -1 and py.c.PyErr_Occurred() != null)) return n;
        py.c.PyErr_Clear();
    }
    ph.raise(py.PyExc_TypeError(), "rt.wrapping_{s}() takes ints of 64 bits", .{@tagName(op)});
    return null;
}
