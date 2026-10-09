//! The Python C API in a process without Python (standalone builds, rt.zig):
//! what the runtime's code names of it, defined so it links. Its objects
//! (None, True, False, the types, the exception classes) are inert: no
//! value is ever one of them. Its functions report no error pending and
//! keep no counts; any other stops the process saying which: a strict
//! language's compiled code never gets there (a path needing Python, which
//! compiling it refused).

const std = @import("std");

/// An object's room: a refcount (immortal), its type, what a type object
/// would hold
const Inert = extern struct {
    refcnt: isize = 1 << 60,
    type: ?*anyopaque = null,
    rest: [62]usize = @splat(0),
};

const data_objects = .{
    "PyBaseObject_Type", "PyBool_Type",    "PyByteArray_Type", "PyBytes_Type",   "PyComplex_Type", "PyDict_Type",
    "PyFloat_Type",      "PyFrozenSet_Type", "PyList_Type",    "PyLong_Type",    "PyMemoryView_Type", "PySet_Type",
    "PyTuple_Type",      "PyType_Type",    "PyUnicode_Type",   "_Py_FalseStruct", "_Py_NoneStruct", "_Py_TrueStruct",
};

const exception_classes = .{ "PyExc_BaseException", "PyExc_IndexError", "PyExc_RuntimeError", "PyExc_TypeError" };

const stopping = .{
    "PyBool_FromLong",           "PyBytes_AsString",            "PyBytes_FromStringAndSize", "PyBytes_Size",
    "PyDict_GetItemString",      "PyDict_GetItemWithError",     "PyDict_New",                "PyDict_SetItem",
    "PyDict_SetItemString",      "PyDict_Size",                 "PyErr_ExceptionMatches",    "PyErr_NormalizeException",
    "PyErr_SetNone",             "PyErr_SetObject",             "PyErr_SetString",           "PyEval_EvalCode",
    "PyEval_GetBuiltins",        "PyException_SetCause",        "PyFloat_AsDouble",          "PyFloat_FromDouble",
    "PyImport_ImportModule",     "PyList_GetItem",              "PyList_New",                "PyList_SetItem",
    "PyList_SetSlice",           "PyList_Size",                 "PyLong_AsLongLong",         "PyLong_AsLongLongAndOverflow",
    "PyLong_AsSsize_t",          "PyLong_FromLongLong",         "PyLong_FromString",         "PyLong_FromUnsignedLongLong",
    "PyNumber_Add",              "PyNumber_And",                "PyNumber_FloorDivide",      "PyNumber_InPlaceAdd",
    "PyNumber_InPlaceAnd",       "PyNumber_InPlaceFloorDivide", "PyNumber_InPlaceLshift",    "PyNumber_InPlaceMultiply",
    "PyNumber_InPlaceOr",        "PyNumber_InPlacePower",       "PyNumber_InPlaceRemainder", "PyNumber_InPlaceRshift",
    "PyNumber_InPlaceSubtract",  "PyNumber_InPlaceTrueDivide",  "PyNumber_InPlaceXor",       "PyNumber_Index",
    "PyNumber_Invert",           "PyNumber_Lshift",             "PyNumber_Multiply",         "PyNumber_Negative",
    "PyNumber_Or",               "PyNumber_Positive",           "PyNumber_Power",            "PyNumber_Remainder",
    "PyNumber_Rshift",           "PyNumber_Subtract",           "PyNumber_TrueDivide",       "PyNumber_Xor",
    "PyObject_ASCII",            "PyObject_Call",               "PyObject_CallFunctionObjArgs", "PyObject_CallNoArgs",
    "PyObject_CallObject",       "PyObject_DelItem",            "PyObject_Format",           "PyObject_GetAttr",
    "PyObject_GetAttrString",    "PyObject_GetItem",            "PyObject_HasAttrString",    "PyObject_Hash",
    "PyObject_HashNotImplemented", "PyObject_IsInstance",       "PyObject_IsSubclass",       "PyObject_IsTrue",
    "PyObject_Repr",             "PyObject_RichCompareBool",    "PyObject_SetAttr",          "PyObject_SetAttrString",
    "PyObject_SetItem",          "PyObject_Size",               "PyObject_Str",              "PySequence_Contains",
    "PySequence_List",           "PySet_Add",                   "PySet_New",                 "PySlice_New",
    "PyTuple_GetItem",           "PyTuple_New",                 "PyTuple_Pack",              "PyTuple_SetItem",
    "PyTuple_Size",              "PyType_GetSlot",              "PyType_IsSubtype",          "PyUnicode_AsUTF8AndSize",
    "PyUnicode_DecodeUTF8",      "PyUnicode_FromStringAndSize", "Py_CompileString",          "_PyObject_CallFunction_SizeT",
    "_PyObject_CallMethod_SizeT",
};

/// An object of its own for each name
fn object(comptime name: []const u8) *Inert {
    _ = name;
    return &struct {
        var o: Inert = .{};
    }.o;
}

/// An exception class's variable (a PyObject *), pointing at an object
fn classVar(comptime name: []const u8) *?*anyopaque {
    return &struct {
        var p: ?*anyopaque = object(name ++ " object");
    }.p;
}

fn stop(comptime name: []const u8) fn () callconv(.c) noreturn {
    return struct {
        fn f() callconv(.c) noreturn {
            const msg = "zrun: " ++ name ++ "(): a Python C API function, called in a runtime without Python (a path of the compiled code a strict language's never takes)\n";
            _ = std.c.write(2, msg, msg.len);
            std.c.abort();
        }
    }.f;
}

fn errOccurred() callconv(.c) ?*anyopaque {
    return null;
}

fn nothing() callconv(.c) void {}

fn counted(_: ?*anyopaque) callconv(.c) void {}

fn fetch(t: ?*?*anyopaque, v: ?*?*anyopaque, tb: ?*?*anyopaque) callconv(.c) void {
    inline for (.{ t, v, tb }) |p| if (p) |q| {
        q.* = null;
    };
}

fn restore(_: ?*anyopaque, _: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {}

fn noMemory() callconv(.c) ?*anyopaque {
    return null;
}

fn gilEnsure() callconv(.c) c_int {
    return 0;
}

comptime {
    for (data_objects) |name| @export(object(name), .{ .name = name });
    for (exception_classes) |name| @export(classVar(name), .{ .name = name });
    for (stopping) |name| @export(&stop(name), .{ .name = name });
    @export(&errOccurred, .{ .name = "PyErr_Occurred" });
    @export(&nothing, .{ .name = "PyErr_Clear" });
    @export(&counted, .{ .name = "Py_IncRef" });
    @export(&counted, .{ .name = "Py_DecRef" });
    @export(&fetch, .{ .name = "PyErr_Fetch" });
    @export(&restore, .{ .name = "PyErr_Restore" });
    @export(&noMemory, .{ .name = "PyErr_NoMemory" });
    @export(&gilEnsure, .{ .name = "PyGILState_Ensure" });
}
