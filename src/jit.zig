//! zgram's LLVM, through its `zgram.llvm.v1` capsule (zgram's
//! src/llvm_capsule.zig is the definition; this is the consumer's copy):
//! LLVM's C API functions, looked up by name once, to build modules in
//! memory (ir.zig); zgram's JIT to compile them.
//!
//! The prototypes come from LLVM's C headers (vendor/llvm/include, the
//! version zgram carries); the functions themselves are zgram's.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;

pub const c = @cImport({
    @cInclude("llvm-c/Core.h");
});

pub const LLVM_ABI: u32 = 2;
pub const CAPSULE_NAME = "zgram.llvm.v1";
pub const LLVM_VERSION = "21.1.8";

pub const LlvmView = extern struct {
    abi: u32,
    llvm_version: [*:0]const u8,
    function: *const fn (name: [*:0]const u8) callconv(.c) ?*const anyopaque,
    compile: *const fn (module: ?*anyopaque, opt_level: u32, err: [*]u8, err_cap: usize) callconv(.c) ?*anyopaque,
    lookup: *const fn (name: [*:0]const u8) callconv(.c) u64,
    define: *const fn (names: [*]const [*:0]const u8, addrs: [*]const u64, n: usize, err: [*]u8, err_cap: usize) callconv(.c) i32,
    release: *const fn (handle: ?*anyopaque) callconv(.c) void,
    emit_object: *const fn (module: ?*anyopaque, opt_level: u32, triple: ?[*:0]const u8, cpu: ?[*:0]const u8, features: ?[*:0]const u8, out_len: *usize, err: [*]u8, err_cap: usize) callconv(.c) ?[*]u8,
    free_bytes: *const fn (bytes: ?[*]u8) callconv(.c) void,
    triple: *const fn () callconv(.c) ?[*:0]const u8,
    data_layout: *const fn () callconv(.c) ?[*:0]const u8,
    load_object: *const fn (bytes: [*]const u8, len: usize, err: [*]u8, err_cap: usize) callconv(.c) ?*anyopaque,
};

// The C types of the prototypes below
const Context = c.LLVMContextRef;
const ModuleRef = c.LLVMModuleRef;
const Type = c.LLVMTypeRef;
const Value = c.LLVMValueRef;
const BasicBlock = c.LLVMBasicBlockRef;
const Builder = c.LLVMBuilderRef;
const Attribute = c.LLVMAttributeRef;
const Bool = c.LLVMBool;
const Str = [*c]const u8;

/// The C API functions zrun uses, each named as in LLVM without its
/// `LLVM` prefix and typed as its prototype in LLVM's headers (written
/// here rather than taken from the headers' declarations: zrun links no
/// LLVM, and Zig would import the functions it takes the types of). get()
/// fills them from the capsule, by name.
pub const Api = struct {
    ContextCreate: *const fn () callconv(.c) Context,
    ContextDispose: *const fn (Context) callconv(.c) void,
    ModuleCreateWithNameInContext: *const fn (Str, Context) callconv(.c) ModuleRef,
    DisposeModule: *const fn (ModuleRef) callconv(.c) void,
    PrintModuleToString: *const fn (ModuleRef) callconv(.c) [*c]u8,
    GetHostCPUName: *const fn () callconv(.c) [*c]u8,
    GetHostCPUFeatures: *const fn () callconv(.c) [*c]u8,
    DisposeMessage: *const fn ([*c]u8) callconv(.c) void,
    Int1TypeInContext: *const fn (Context) callconv(.c) Type,
    Int8TypeInContext: *const fn (Context) callconv(.c) Type,
    IntTypeInContext: *const fn (Context, c_uint) callconv(.c) Type,
    Int32TypeInContext: *const fn (Context) callconv(.c) Type,
    Int64TypeInContext: *const fn (Context) callconv(.c) Type,
    DoubleTypeInContext: *const fn (Context) callconv(.c) Type,
    VoidTypeInContext: *const fn (Context) callconv(.c) Type,
    PointerTypeInContext: *const fn (Context, c_uint) callconv(.c) Type,
    StructTypeInContext: *const fn (Context, [*c]Type, c_uint, Bool) callconv(.c) Type,
    ArrayType2: *const fn (Type, u64) callconv(.c) Type,
    FunctionType: *const fn (Type, [*c]Type, c_uint, Bool) callconv(.c) Type,
    GlobalGetValueType: *const fn (Value) callconv(.c) Type,
    TypeOf: *const fn (Value) callconv(.c) Type,
    ConstInt: *const fn (Type, c_ulonglong, Bool) callconv(.c) Value,
    ConstNull: *const fn (Type) callconv(.c) Value,
    ConstStringInContext2: *const fn (Context, Str, usize, Bool) callconv(.c) Value,
    ConstStructInContext: *const fn (Context, [*c]Value, c_uint, Bool) callconv(.c) Value,
    ConstIntToPtr: *const fn (Value, Type) callconv(.c) Value,
    AddGlobal: *const fn (ModuleRef, Type, Str) callconv(.c) Value,
    ConstPtrToInt: *const fn (Value, Type) callconv(.c) Value,
    SetInitializer: *const fn (Value, Value) callconv(.c) void,
    SetGlobalConstant: *const fn (Value, Bool) callconv(.c) void,
    SetLinkage: *const fn (Value, c.LLVMLinkage) callconv(.c) void,
    SetUnnamedAddress: *const fn (Value, c.LLVMUnnamedAddr) callconv(.c) void,
    SetAlignment: *const fn (Value, c_uint) callconv(.c) void,
    AddFunction: *const fn (ModuleRef, Str, Type) callconv(.c) Value,
    GetParam: *const fn (Value, c_uint) callconv(.c) Value,
    GetEnumAttributeKindForName: *const fn (Str, usize) callconv(.c) c_uint,
    CreateEnumAttribute: *const fn (Context, c_uint, u64) callconv(.c) Attribute,
    AddAttributeAtIndex: *const fn (Value, c.LLVMAttributeIndex, Attribute) callconv(.c) void,
    LookupIntrinsicID: *const fn (Str, usize) callconv(.c) c_uint,
    GetIntrinsicDeclaration: *const fn (ModuleRef, c_uint, [*c]Type, usize) callconv(.c) Value,
    AppendBasicBlockInContext: *const fn (Context, Value, Str) callconv(.c) BasicBlock,
    GetBasicBlockTerminator: *const fn (BasicBlock) callconv(.c) Value,
    CreateBuilderInContext: *const fn (Context) callconv(.c) Builder,
    DisposeBuilder: *const fn (Builder) callconv(.c) void,
    PositionBuilderAtEnd: *const fn (Builder, BasicBlock) callconv(.c) void,
    BuildRet: *const fn (Builder, Value) callconv(.c) Value,
    BuildRetVoid: *const fn (Builder) callconv(.c) Value,
    BuildBr: *const fn (Builder, BasicBlock) callconv(.c) Value,
    BuildCondBr: *const fn (Builder, Value, BasicBlock, BasicBlock) callconv(.c) Value,
    BuildUnreachable: *const fn (Builder) callconv(.c) Value,
    BuildAdd: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildSub: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildMul: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildAnd: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildOr: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildXor: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildShl: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildLShr: *const fn (Builder, Value, Value, Str) callconv(.c) Value,
    BuildICmp: *const fn (Builder, c.LLVMIntPredicate, Value, Value, Str) callconv(.c) Value,
    BuildFCmp: *const fn (Builder, c.LLVMRealPredicate, Value, Value, Str) callconv(.c) Value,
    BuildAlloca: *const fn (Builder, Type, Str) callconv(.c) Value,
    BuildLoad2: *const fn (Builder, Type, Value, Str) callconv(.c) Value,
    BuildStore: *const fn (Builder, Value, Value) callconv(.c) Value,
    BuildInBoundsGEP2: *const fn (Builder, Type, Value, [*c]Value, c_uint, Str) callconv(.c) Value,
    BuildZExt: *const fn (Builder, Value, Type, Str) callconv(.c) Value,
    BuildBitCast: *const fn (Builder, Value, Type, Str) callconv(.c) Value,
    BuildPtrToInt: *const fn (Builder, Value, Type, Str) callconv(.c) Value,
    BuildIntToPtr: *const fn (Builder, Value, Type, Str) callconv(.c) Value,
    BuildPhi: *const fn (Builder, Type, Str) callconv(.c) Value,
    AddIncoming: *const fn (Value, [*c]Value, [*c]BasicBlock, c_uint) callconv(.c) void,
    BuildCall2: *const fn (Builder, Type, Value, [*c]Value, c_uint, Str) callconv(.c) Value,
    BuildSelect: *const fn (Builder, Value, Value, Value, Str) callconv(.c) Value,
    BuildExtractValue: *const fn (Builder, Value, c_uint, Str) callconv(.c) Value,
};

/// The functions (filled by get())
var api: Api = undefined;

/// A C API function by its LLVM name (`f("LLVMBuildAdd")(builder, a, b,
/// "")`).
pub inline fn f(comptime name: []const u8) @FieldType(Api, name["LLVM".len..]) {
    return @field(api, name["LLVM".len..]);
}

/// The view, once loaded (the capsule is kept for the process's life)
var loaded: ?*const LlvmView = null;
var capsule_ref: ?*PyObject = null;

/// zgram's LLVM (needs the GIL the first time): null with ImportError if
/// this zgram has none, or another version of it.
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
    if (view.abi != LLVM_ABI or !std.mem.eql(u8, std.mem.span(view.llvm_version), LLVM_VERSION)) {
        py.Py_DecRef(capsule);
        ph.raise(py.PyExc_ImportError(), "zrun uses zgram's LLVM {s} (capsule ABI {d}); this zgram has LLVM {s} (ABI {d})", .{ LLVM_VERSION, LLVM_ABI, std.mem.span(view.llvm_version), view.abi });
        return null;
    }
    inline for (@typeInfo(Api).@"struct".fields) |field| {
        const name = "LLVM" ++ field.name;
        const p = view.function(name) orelse {
            py.Py_DecRef(capsule);
            ph.raise(py.PyExc_ImportError(), "zgram's LLVM capsule has no {s}: zrun needs a newer zgram", .{name});
            return null;
        };
        @field(api, field.name) = @ptrCast(@alignCast(p));
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

/// Compile a module (taking it and its context); error.Compile with the
/// message in `err` (cut).
pub fn compile(view: *const LlvmView, module: c.LLVMModuleRef, opt_level: u32, err: []u8) error{Compile}!Module {
    const handle = view.compile(@ptrCast(module), opt_level, err.ptr, err.len) orelse return error.Compile;
    return .{ .handle = handle };
}

pub fn lookup(view: *const LlvmView, name: [:0]const u8) usize {
    return @intCast(view.lookup(name.ptr));
}

/// A module compiled to an object file for this process (taking it and
/// its context): the bytes (free them with freeBytes).
pub fn emitObject(view: *const LlvmView, module: c.LLVMModuleRef, opt_level: u32, err: []u8) error{Compile}![]u8 {
    var n: usize = 0;
    const p = view.emit_object(@ptrCast(module), opt_level, null, null, null, &n, err.ptr, err.len) orelse return error.Compile;
    return p[0..n];
}

pub fn freeBytes(view: *const LlvmView, bytes: []u8) void {
    view.free_bytes(bytes.ptr);
}

/// An object file emitObject made into the JIT.
pub fn loadObject(view: *const LlvmView, bytes: []const u8, err: []u8) error{Compile}!Module {
    const handle = view.load_object(bytes.ptr, bytes.len, err.ptr, err.len) orelse return error.Compile;
    return .{ .handle = handle };
}

/// Names compiled code refers to, at these addresses.
pub fn define(view: *const LlvmView, names: []const [*:0]const u8, addrs: []const u64, err: []u8) error{Compile}!void {
    if (view.define(names.ptr, addrs.ptr, names.len, err.ptr, err.len) != 0) return error.Compile;
}
