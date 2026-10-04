//! Building LLVM IR in memory through LLVM's C API (zgram's, through
//! jit.zig): a module, its functions, their blocks and instructions,
//! string constants. The compiler (compile.zig) builds through this;
//! zgram's JIT compiles the module.

const std = @import("std");
const value = @import("value.zig");
const jit = @import("jit.zig");
const c = jit.c;
const L = jit.f;
const Allocator = std.mem.Allocator;

pub const Value = c.LLVMValueRef;
pub const Block = c.LLVMBasicBlockRef;
pub const Type = c.LLVMTypeRef;

/// A function to call: its value and type
pub const Fn = struct { v: Value, ty: Type };

/// The types the compiler uses
pub const Types = struct {
    i1: Type,
    i8: Type,
    i32: Type,
    i64: Type,
    f64: Type,
    ptr: Type,
    void: Type,
    /// A runtime value: { tag, bits }
    val: Type,
    /// A checked arithmetic result: { i64, i1 }
    ovf: Type,
};

pub const Module = struct {
    gpa: Allocator,
    ctx: c.LLVMContextRef,
    mod: c.LLVMModuleRef,
    t: Types,
    /// A prefix making the module's symbols unique in the process
    prefix: []const u8,
    /// String constants by content
    strings: std.StringHashMapUnmanaged(Value) = .empty,
    /// Functions by name (helpers declared, the module's own)
    fns: std.StringHashMapUnmanaged(Fn) = .empty,
    /// Given to the JIT: no longer ours to free
    taken: bool = false,

    pub fn init(gpa: Allocator, prefix: []const u8) Module {
        const ctx = L("LLVMContextCreate")();
        const mod = L("LLVMModuleCreateWithNameInContext")("zrun", ctx);
        const i64t = L("LLVMInt64TypeInContext")(ctx);
        const i1t = L("LLVMInt1TypeInContext")(ctx);
        var val_fields = [_]Type{ i64t, i64t };
        var ovf_fields = [_]Type{ i64t, i1t };
        return .{
            .gpa = gpa,
            .ctx = ctx,
            .mod = mod,
            .prefix = prefix,
            .t = .{
                .i1 = i1t,
                .i8 = L("LLVMInt8TypeInContext")(ctx),
                .i32 = L("LLVMInt32TypeInContext")(ctx),
                .i64 = i64t,
                .f64 = L("LLVMDoubleTypeInContext")(ctx),
                .ptr = L("LLVMPointerTypeInContext")(ctx, 0),
                .void = L("LLVMVoidTypeInContext")(ctx),
                .val = L("LLVMStructTypeInContext")(ctx, &val_fields, 2, 0),
                .ovf = L("LLVMStructTypeInContext")(ctx, &ovf_fields, 2, 0),
            },
        };
    }

    pub fn deinit(self: *Module) void {
        if (!self.taken) {
            L("LLVMDisposeModule")(self.mod);
            L("LLVMContextDispose")(self.ctx);
        }
        self.strings.deinit(self.gpa);
        self.fns.deinit(self.gpa);
    }

    /// The module for the JIT (it takes the module and its context).
    pub fn take(self: *Module) c.LLVMModuleRef {
        self.taken = true;
        return self.mod;
    }

    /// The module as text (for debugging: Program.compiled_ir()), owned by
    /// the caller.
    pub fn text(self: *Module) ![]u8 {
        const s = L("LLVMPrintModuleToString")(self.mod);
        defer L("LLVMDisposeMessage")(s);
        return self.gpa.dupe(u8, std.mem.span(s));
    }

    fn z(self: *Module, s: []const u8) ![:0]u8 {
        return self.gpa.dupeZ(u8, s);
    }

    pub fn fnType(self: *Module, ret: Type, params: []const Type) Type {
        _ = self;
        return L("LLVMFunctionType")(ret, @constCast(params.ptr), @intCast(params.len), 0);
    }

    /// Declare an external function (a runtime helper) once.
    pub fn declare(self: *Module, name: []const u8, ret: Type, params: []const Type) !void {
        if (self.fns.contains(name)) return;
        const ty = self.fnType(ret, params);
        const v = L("LLVMAddFunction")(self.mod, try self.z(name), ty);
        try self.fns.put(self.gpa, name, .{ .v = v, .ty = ty });
    }

    /// A function of the module by name: made (with this type) the first
    /// time; internal unless `external`.
    pub fn function(self: *Module, name: []const u8, ret: Type, params: []const Type, external: bool) !Fn {
        if (self.fns.get(name)) |f| return f;
        const ty = self.fnType(ret, params);
        const v = L("LLVMAddFunction")(self.mod, try self.z(name), ty);
        if (!external) L("LLVMSetLinkage")(v, c.LLVMInternalLinkage);
        const f = Fn{ .v = v, .ty = ty };
        try self.fns.put(self.gpa, name, f);
        return f;
    }

    pub fn get(self: *Module, name: []const u8) Fn {
        return self.fns.get(name) orelse std.debug.panic("zrun: {s} isn't declared", .{name});
    }

    /// An LLVM intrinsic ("llvm.sadd.with.overflow") for some types.
    pub fn intrinsic(self: *Module, name: []const u8, types: []const Type) Fn {
        const id = L("LLVMLookupIntrinsicID")(name.ptr, name.len);
        const v = L("LLVMGetIntrinsicDeclaration")(self.mod, id, @constCast(types.ptr), types.len);
        return .{ .v = v, .ty = L("LLVMGlobalGetValueType")(v) };
    }

    /// Mark a function always inlined.
    pub fn alwaysInline(self: *Module, f: Fn) void {
        const kind = L("LLVMGetEnumAttributeKindForName")("alwaysinline", "alwaysinline".len);
        const attr = L("LLVMCreateEnumAttribute")(self.ctx, kind, 0);
        L("LLVMAddAttributeAtIndex")(f.v, std.math.maxInt(c_uint), attr); // (LLVMAttributeFunctionIndex)
    }

    pub fn k64(self: *Module, n: i64) Value {
        return L("LLVMConstInt")(self.t.i64, @bitCast(n), 1);
    }

    pub fn k32(self: *Module, n: u32) Value {
        return L("LLVMConstInt")(self.t.i32, n, 0);
    }

    pub fn k1(self: *Module, b: bool) Value {
        return L("LLVMConstInt")(self.t.i1, @intFromBool(b), 0);
    }

    pub fn nullPtr(self: *Module) Value {
        return L("LLVMConstNull")(self.t.ptr);
    }

    /// A pointer known when compiling (an object of the compiler's).
    pub fn ptrConst(self: *Module, addr: usize) Value {
        return L("LLVMConstIntToPtr")(L("LLVMConstInt")(self.t.i64, addr, 0), self.t.ptr);
    }

    /// An immortal string object for a literal: a constant global laid out
    /// as the runtime's Str (header, lengths, its hash, the bytes).
    pub fn string(self: *Module, bytes: []const u8) !Value {
        if (self.strings.get(bytes)) |g| return g;
        const t = self.t;
        const chars = std.unicode.utf8CountCodepoints(bytes) catch bytes.len;
        const arr = L("LLVMConstStringInContext2")(self.ctx, bytes.ptr, bytes.len, 1);
        var fields = [_]Value{
            self.k64(@bitCast(value.IMMORTAL)),
            self.k32(4),
            self.k32(0),
            self.k64(@intCast(bytes.len)),
            self.k64(@intCast(chars)),
            self.k64(@bitCast(value.strHash(bytes))),
            arr,
        };
        const init_v = L("LLVMConstStructInContext")(self.ctx, &fields, fields.len, 0);
        const name = try std.fmt.allocPrintSentinel(self.gpa, "{s}_s{d}", .{ self.prefix, self.strings.count() }, 0);
        _ = t;
        const g = L("LLVMAddGlobal")(self.mod, L("LLVMTypeOf")(init_v), name);
        L("LLVMSetInitializer")(g, init_v);
        L("LLVMSetGlobalConstant")(g, 1);
        L("LLVMSetLinkage")(g, c.LLVMPrivateLinkage);
        L("LLVMSetUnnamedAddress")(g, c.LLVMGlobalUnnamedAddr);
        L("LLVMSetAlignment")(g, 8);
        try self.strings.put(self.gpa, try self.gpa.dupe(u8, bytes), g);
        return g;
    }
};

/// One function being built
pub const Function = struct {
    m: *Module,
    fv: Value,
    /// The entry block: allocas and their first values (then a jump to the
    /// code)
    entry: Block,
    eb: c.LLVMBuilderRef,
    /// The code's builder, and the block it's in
    b: c.LLVMBuilderRef,
    current: Block,
    start: Block,
    next_label: u32 = 0,
    /// Every block made (finish() closes those left open)
    blocks: std.ArrayListUnmanaged(Block) = .empty,

    pub fn init(m: *Module, fv: Value) Function {
        const entry = L("LLVMAppendBasicBlockInContext")(m.ctx, fv, "entry");
        const start = L("LLVMAppendBasicBlockInContext")(m.ctx, fv, "start");
        const eb = L("LLVMCreateBuilderInContext")(m.ctx);
        L("LLVMPositionBuilderAtEnd")(eb, entry);
        const b = L("LLVMCreateBuilderInContext")(m.ctx);
        L("LLVMPositionBuilderAtEnd")(b, start);
        return .{ .m = m, .fv = fv, .entry = entry, .eb = eb, .b = b, .current = start, .start = start };
    }

    /// Close the function: the entry jumps to the code; a block left open
    /// (one made but never used, or the end of dead code) is unreachable.
    pub fn finish(self: *Function) void {
        _ = L("LLVMBuildBr")(self.eb, self.start);
        if (!self.terminated()) _ = L("LLVMBuildUnreachable")(self.b);
        for (self.blocks.items) |blk| {
            if (L("LLVMGetBasicBlockTerminator")(blk) != null) continue;
            L("LLVMPositionBuilderAtEnd")(self.b, blk);
            _ = L("LLVMBuildUnreachable")(self.b);
        }
        self.blocks.deinit(self.m.gpa);
        L("LLVMDisposeBuilder")(self.eb);
        L("LLVMDisposeBuilder")(self.b);
    }

    pub fn param(self: *Function, i: u32) Value {
        return L("LLVMGetParam")(self.fv, i);
    }

    /// A fresh block.
    pub fn label(self: *Function, prefix: []const u8) !Block {
        self.next_label += 1;
        var buf: [64]u8 = undefined;
        const name = std.fmt.bufPrintZ(&buf, "{s}{d}", .{ prefix[0..@min(prefix.len, 40)], self.next_label }) catch "b";
        const blk = L("LLVMAppendBasicBlockInContext")(self.m.ctx, self.fv, name);
        try self.blocks.append(self.m.gpa, blk);
        return blk;
    }

    pub fn terminated(self: *const Function) bool {
        return L("LLVMGetBasicBlockTerminator")(self.current) != null;
    }

    /// Code after a jump is dead: it goes in a block of its own (nothing
    /// jumps to it).
    fn open(self: *Function) void {
        if (!self.terminated()) return;
        const dead = L("LLVMAppendBasicBlockInContext")(self.m.ctx, self.fv, "dead");
        // (closed by finish(); out of memory here only leaves it unclosed,
        // which the verifier reports)
        self.blocks.append(self.m.gpa, dead) catch {};
        L("LLVMPositionBuilderAtEnd")(self.b, dead);
        self.current = dead;
    }

    // (blocks, jumps and slots return error unions, as the rest of the
    // compiler's building does: `try f.block(b)`)

    /// Continue in a block (the current one, still open, jumps to it).
    pub fn block(self: *Function, blk: Block) error{OutOfMemory}!void {
        if (!self.terminated()) _ = L("LLVMBuildBr")(self.b, blk);
        L("LLVMPositionBuilderAtEnd")(self.b, blk);
        self.current = blk;
    }

    pub fn br(self: *Function, blk: Block) error{OutOfMemory}!void {
        self.open();
        _ = L("LLVMBuildBr")(self.b, blk);
    }

    pub fn condBr(self: *Function, cond: Value, yes: Block, no: Block) error{OutOfMemory}!void {
        self.open();
        _ = L("LLVMBuildCondBr")(self.b, cond, yes, no);
    }

    pub fn ret(self: *Function, v: Value) error{OutOfMemory}!void {
        self.open();
        _ = L("LLVMBuildRet")(self.b, v);
    }

    pub fn retVoid(self: *Function) error{OutOfMemory}!void {
        self.open();
        _ = L("LLVMBuildRetVoid")(self.b);
    }

    /// A stack slot (in the entry block).
    pub fn alloca(self: *Function, ty: Type) error{OutOfMemory}!Value {
        const a = L("LLVMBuildAlloca")(self.eb, ty, "");
        L("LLVMSetAlignment")(a, 8);
        return a;
    }

    /// A store done once, in the entry block (a slot's first value).
    pub fn entryStore(self: *Function, v: Value, ptr: Value) void {
        const s = L("LLVMBuildStore")(self.eb, v, ptr);
        L("LLVMSetAlignment")(s, 8);
    }

    pub fn add(self: *Function, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildAdd")(self.b, a, b, "");
    }

    pub fn sub(self: *Function, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildSub")(self.b, a, b, "");
    }

    pub fn and_(self: *Function, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildAnd")(self.b, a, b, "");
    }

    pub fn or_(self: *Function, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildOr")(self.b, a, b, "");
    }

    pub fn xor(self: *Function, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildXor")(self.b, a, b, "");
    }

    pub fn icmp(self: *Function, pred: c.LLVMIntPredicate, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildICmp")(self.b, pred, a, b, "");
    }

    pub fn fcmp(self: *Function, pred: c.LLVMRealPredicate, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildFCmp")(self.b, pred, a, b, "");
    }

    /// An i1 as an i64 (0 or 1).
    pub fn zext64(self: *Function, v: Value) Value {
        self.open();
        return L("LLVMBuildZExt")(self.b, v, self.m.t.i64, "");
    }

    pub fn bitcast(self: *Function, v: Value, ty: Type) Value {
        self.open();
        return L("LLVMBuildBitCast")(self.b, v, ty, "");
    }

    pub fn ptrToInt(self: *Function, v: Value) Value {
        self.open();
        return L("LLVMBuildPtrToInt")(self.b, v, self.m.t.i64, "");
    }

    pub fn intToPtr(self: *Function, v: Value) Value {
        self.open();
        return L("LLVMBuildIntToPtr")(self.b, v, self.m.t.ptr, "");
    }

    pub fn load(self: *Function, ty: Type, ptr: Value) Value {
        self.open();
        const v = L("LLVMBuildLoad2")(self.b, ty, ptr, "");
        L("LLVMSetAlignment")(v, 8);
        return v;
    }

    pub fn store(self: *Function, v: Value, ptr: Value) void {
        self.open();
        const s = L("LLVMBuildStore")(self.b, v, ptr);
        L("LLVMSetAlignment")(s, 8);
    }

    /// &p[i] for elements of a type.
    pub fn at(self: *Function, ty: Type, ptr: Value, i: Value) Value {
        self.open();
        var idx = [_]Value{i};
        return L("LLVMBuildInBoundsGEP2")(self.b, ty, ptr, &idx, 1, "");
    }

    /// &p.field_i of a struct.
    pub fn field(self: *Function, ty: Type, ptr: Value, i: u32) Value {
        self.open();
        var idx = [_]Value{ self.m.k32(0), self.m.k32(i) };
        return L("LLVMBuildInBoundsGEP2")(self.b, ty, ptr, &idx, 2, "");
    }

    /// p + n bytes.
    pub fn offset(self: *Function, ptr: Value, n: i64) Value {
        return self.at(self.m.t.i8, ptr, self.m.k64(n));
    }

    pub fn extract(self: *Function, v: Value, i: u32) Value {
        self.open();
        return L("LLVMBuildExtractValue")(self.b, v, i, "");
    }

    pub fn select(self: *Function, cond: Value, a: Value, b: Value) Value {
        self.open();
        return L("LLVMBuildSelect")(self.b, cond, a, b, "");
    }

    /// A value from two predecessors.
    pub fn phi(self: *Function, ty: Type, a: Value, from_a: Block, b: Value, from_b: Block) Value {
        self.open();
        const p = L("LLVMBuildPhi")(self.b, ty, "");
        var vals = [_]Value{ a, b };
        var blocks = [_]Block{ from_a, from_b };
        L("LLVMAddIncoming")(p, &vals, &blocks, 2);
        return p;
    }

    pub fn call(self: *Function, f: Fn, args: []const Value) Value {
        self.open();
        return L("LLVMBuildCall2")(self.b, f.ty, f.v, @constCast(args.ptr), @intCast(args.len), "");
    }

    /// A call of a declared function by name.
    pub fn callName(self: *Function, name: []const u8, args: []const Value) Value {
        return self.call(self.m.get(name), args);
    }
};
