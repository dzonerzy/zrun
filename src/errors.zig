//! Errors natively: the kinds of exception compiled code raises and
//! catches (Python's builtin ones, in CPython's hierarchy, and zrun's own),
//! with no Python object. The Python class of a kind is only needed where an
//! error leaves compiled code (pyClass), or for an exception of a class not
//! among them (a Python exception then, as before).

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const types = @import("types.zig");

pub const Kind = enum(u8) {
    BaseException,
    Exception,
    ArithmeticError,
    ZeroDivisionError,
    OverflowError,
    FloatingPointError,
    LookupError,
    IndexError,
    KeyError,
    ValueError,
    UnicodeError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    TypeError,
    AttributeError,
    NameError,
    UnboundLocalError,
    RuntimeError,
    RecursionError,
    NotImplementedError,
    AssertionError,
    StopIteration,
    BufferError,
    /// zrun.IntegerOverflow (an ArithmeticError)
    IntegerOverflow,
    /// zrun.Error: an error of the program (what compiled code's own errors
    /// are to Python code)
    ZrunError,
    /// rt.Throw: an error of the language carrying a value
    Throw,
    /// dataclasses.FrozenInstanceError (an AttributeError)
    FrozenInstanceError,
    /// struct.error (an Exception)
    StructError,

    /// The kind it's a subclass of (null: BaseException)
    pub fn parent(k: Kind) ?Kind {
        return switch (k) {
            .BaseException => null,
            .Exception => .BaseException,
            .ArithmeticError, .LookupError, .ValueError, .TypeError, .AttributeError, .NameError, .RuntimeError, .AssertionError, .StopIteration, .BufferError, .ZrunError, .Throw, .StructError => .Exception,
            .ZeroDivisionError, .OverflowError, .FloatingPointError, .IntegerOverflow => .ArithmeticError,
            .IndexError, .KeyError => .LookupError,
            .UnicodeError => .ValueError,
            .UnicodeDecodeError, .UnicodeEncodeError => .UnicodeError,
            .UnboundLocalError => .NameError,
            .RecursionError, .NotImplementedError => .RuntimeError,
            .FrozenInstanceError => .AttributeError,
        };
    }

    /// Whether an error of kind k is one of `of` (`except of:` catches it)
    pub fn isA(k: Kind, of: Kind) bool {
        var x: ?Kind = k;
        while (x) |y| : (x = y.parent()) if (y == of) return true;
        return false;
    }

    /// Its Python class (borrowed); null: not found (with no exception)
    pub fn pyClass(k: Kind) ?*PyObject {
        return switch (k) {
            .IntegerOverflow => types.IntegerOverflow,
            .ZrunError => types.Error,
            .Throw => types.Throw,
            .FrozenInstanceError => moduleClass("dataclasses", "FrozenInstanceError"),
            .StructError => moduleClass("struct", "error"),
            inline else => |t| builtinClass(@tagName(t)),
        };
    }

    /// Its class's name (Python's)
    pub fn name(k: Kind) []const u8 {
        return switch (k) {
            .ZrunError => "Error",
            .StructError => "error",
            inline else => |t| @tagName(t),
        };
    }
};

var module_classes: [@typeInfo(Kind).@"enum".fields.len]?*PyObject = @splat(null);

fn builtinClass(comptime n: [:0]const u8) ?*PyObject {
    const builtins = py.c.PyEval_GetBuiltins() orelse return null;
    return py.c.PyDict_GetItemString(builtins, n);
}

/// A module's class, found once (kept)
fn moduleClass(comptime module: [:0]const u8, comptime n: [:0]const u8) ?*PyObject {
    const slot = &module_classes[@intFromEnum(if (comptime std.mem.eql(u8, module, "struct")) Kind.StructError else Kind.FrozenInstanceError)];
    if (slot.*) |c| return c;
    const m = py.c.PyImport_ImportModule(module) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(m);
    slot.* = py.c.PyObject_GetAttrString(m, n) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    return slot.*;
}

/// In an except's mask of kinds (bits by Kind): a class that isn't one
pub const kind_unknown: u64 = 1 << 63;

/// A kind's bit in a mask
pub fn bit(k: Kind) u64 {
    return @as(u64, 1) << @intCast(@intFromEnum(k));
}

/// Whether an error of kind k is caught by an except of these kinds (a
/// class that isn't a kind, a subclass of one included, never is one of
/// them: an error of a kind is of exactly that class)
pub fn matches(k: Kind, mask: u64) bool {
    var x: ?Kind = k;
    while (x) |y| : (x = y.parent()) if (mask & bit(y) != 0) return true;
    return false;
}

/// The kind a Python class is, exactly (null: another class, a subclass of
/// one of them included: its own Python code may matter)
pub fn kindOfClass(o: *PyObject) ?Kind {
    inline for (@typeInfo(Kind).@"enum".fields) |f| {
        const k: Kind = @enumFromInt(f.value);
        if (k.pyClass()) |c| if (c == o) return k;
    }
    return null;
}
