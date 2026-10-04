//! What compiled code calls: the execution context and the runtime
//! helpers (callconv(.c), given to zgram's JIT by name).
//!
//! The common cases run here natively (int and float arithmetic, string
//! concatenation, comparisons, truth). Anything else, and every error, is
//! done by Python on the equivalent objects: the result, or the exception
//! made a runtime error the way the reference mode makes it, is then the
//! same on every Python version, message included.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const types = @import("types.zig");
const value = @import("value.zig");

const Value = value.Value;
const Tag = value.Tag;
const allocator = std.heap.c_allocator;

/// A frame of the language's call stack: the function called, where
pub const CallEntry = struct { name: *value.Str, node: u32 };

/// The execution context of a run (or a call) of compiled code
pub const Ctx = struct {
    /// Makes a Node object for a node index (the program's), for host
    /// functions given nodes
    node_maker: NodeMaker,
    /// The Python objects the code refers to (host functions, record
    /// classes...), by index (borrowed: the compiled program holds them)
    objects: []const *PyObject,
    /// The language's calls being run, outermost first
    calls: std.ArrayListUnmanaged(CallEntry) = .empty,
    max_depth: u32,
    /// The error, once one happened: its node, its message
    failed: bool = false,
    err_node: u32 = 0,
    err_msg: std.ArrayListUnmanaged(u8) = .empty,
    /// The call stack when it happened (innermost first)
    err_stack: std.ArrayListUnmanaged(CallEntry) = .empty,

    pub fn deinit(self: *Ctx) void {
        self.calls.deinit(allocator);
        self.err_msg.deinit(allocator);
        for (self.err_stack.items) |e| value.decref(Value.obj(.str, &e.name.head));
        self.err_stack.deinit(allocator);
    }
};

pub const NodeMaker = struct {
    ctx: *anyopaque,
    make_fn: *const fn (ctx: *anyopaque, idx: u32) ?*PyObject,

    pub fn make(self: NodeMaker, idx: u32) ?*PyObject {
        return self.make_fn(self.ctx, idx);
    }
};

/// Record an error at a node (the first one wins); false.
pub fn fail(ctx: *Ctx, node: u32, comptime fmt: []const u8, args: anytype) bool {
    if (ctx.failed) return false;
    ctx.failed = true;
    ctx.err_node = node;
    ctx.err_msg.clearRetainingCapacity();
    ctx.err_msg.print(allocator, fmt, args) catch {};
    // The stack, innermost first
    var i = ctx.calls.items.len;
    while (i > 0) {
        i -= 1;
        const e = ctx.calls.items[i];
        value.incref(Value.obj(.str, &e.name.head));
        ctx.err_stack.append(allocator, e) catch {};
    }
    return false;
}

/// The Python exception being raised as the run's error (as the reference
/// mode words it); false.
fn failPython(ctx: *Ctx, node: u32) bool {
    const msg = pythonMessage() orelse {
        py.c.PyErr_Clear();
        return fail(ctx, node, "error", .{});
    };
    defer py.Py_DecRef(msg);
    const text = ph.utf8(msg, "message") orelse {
        py.c.PyErr_Clear();
        return fail(ctx, node, "error", .{});
    };
    return fail(ctx, node, "{s}", .{text});
}

/// The reference mode's wording of a Python exception (lib.zig's
/// pythonMessage, shared): taken and cleared.
pub var pythonMessage: *const fn () ?*PyObject = undefined;

// ======================================================================
// Reference counts
// ======================================================================

export fn zr_incref(tag: u64, bits: u64) callconv(.c) void {
    value.incref(.{ .tag = tag, .bits = bits });
}

export fn zr_decref(tag: u64, bits: u64) callconv(.c) void {
    value.decref(.{ .tag = tag, .bits = bits });
}

// ======================================================================
// Errors
// ======================================================================

/// An error with a message given by the compiled code (a literal Str).
export fn zr_fail(ctx: *Ctx, node: u32, msg: *const value.Str) callconv(.c) bool {
    return fail(ctx, node, "{s}", .{msg.bytes()});
}

/// "'name' has no value yet"
export fn zr_unset(ctx: *Ctx, node: u32, name: *const value.Str) callconv(.c) bool {
    return fail(ctx, node, "'{s}' has no value yet", .{name.bytes()});
}

export fn zr_overflow(ctx: *Ctx, node: u32) callconv(.c) bool {
    return fail(ctx, node, "integer overflow", .{});
}

// ======================================================================
// Through Python: the slow and the error paths
// ======================================================================

fn objects(ctx: *Ctx, vals: []const Value, out: []*PyObject) bool {
    for (vals, 0..) |v, i| {
        out[i] = value.toPython(v, ctx.node_maker) orelse {
            for (out[0..i]) |o| py.Py_DecRef(o);
            return false;
        };
    }
    return true;
}

/// The result of a Python call as a value in `out`; or the error.
fn fromResult(ctx: *Ctx, node: u32, r: ?*PyObject, out: *Value) bool {
    const o = r orelse return failPython(ctx, node);
    defer py.Py_DecRef(o);
    out.* = value.fromPython(o) orelse return failPython(ctx, node);
    return true;
}

const Op = enum(u32) { add, sub, mul, div, floordiv, mod, pow, lshift, rshift, bitor, bitxor, bitand };

fn pythonBinary(ctx: *Ctx, node: u32, op: Op, a: Value, b: Value, out: *Value) bool {
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, &.{ a, b }, &objs)) return failPython(ctx, node);
    defer for (objs) |o| py.Py_DecRef(o);
    const x = objs[0];
    const y = objs[1];
    const r = switch (op) {
        .add => py.c.PyNumber_Add(x, y),
        .sub => py.c.PyNumber_Subtract(x, y),
        .mul => py.c.PyNumber_Multiply(x, y),
        .div => py.c.PyNumber_TrueDivide(x, y),
        .floordiv => py.c.PyNumber_FloorDivide(x, y),
        .mod => py.c.PyNumber_Remainder(x, y),
        .pow => py.c.PyNumber_Power(x, y, py.Py_None()),
        .lshift => py.c.PyNumber_Lshift(x, y),
        .rshift => py.c.PyNumber_Rshift(x, y),
        .bitor => py.c.PyNumber_Or(x, y),
        .bitxor => py.c.PyNumber_Xor(x, y),
        .bitand => py.c.PyNumber_And(x, y),
    };
    return fromResult(ctx, node, r, out);
}

// ======================================================================
// Arithmetic
// ======================================================================

fn isInt(v: Value) bool {
    return v.kind() == .int or v.kind() == .bool;
}

fn floorDiv(a: i64, b: i64) i64 {
    const q = @divTrunc(a, b);
    return if (@rem(a, b) != 0 and ((a < 0) != (b < 0))) q - 1 else q;
}

fn floorMod(a: i64, b: i64) i64 {
    const r = @rem(a, b);
    return if (r != 0 and ((r < 0) != (b < 0))) r + b else r;
}

fn floatMod(a: f64, b: f64) f64 {
    var r = @rem(a, b);
    if (r != 0 and ((r < 0) != (b < 0))) r += b;
    return r;
}

/// a <op> b, Python's way, in `out`.
export fn zr_binary(ctx: *Ctx, node: u32, op_code: u32, ta: u64, ba: u64, tb: u64, bb: u64, out: *Value) callconv(.c) bool {
    const a = Value{ .tag = ta, .bits = ba };
    const b = Value{ .tag = tb, .bits = bb };
    const op: Op = @enumFromInt(op_code);
    if (isInt(a) and isInt(b)) {
        const x = a.asInt();
        const y = b.asInt();
        switch (op) {
            .add => {
                const r = @addWithOverflow(x, y);
                if (r[1] != 0) return zr_overflow(ctx, node);
                out.* = Value.int(r[0]);
                return true;
            },
            .sub => {
                const r = @subWithOverflow(x, y);
                if (r[1] != 0) return zr_overflow(ctx, node);
                out.* = Value.int(r[0]);
                return true;
            },
            .mul => {
                const r = @mulWithOverflow(x, y);
                if (r[1] != 0) return zr_overflow(ctx, node);
                out.* = Value.int(r[0]);
                return true;
            },
            .floordiv, .mod => {
                if (y == 0) return fail(ctx, node, "division by zero", .{});
                if (x == std.math.minInt(i64) and y == -1) {
                    if (op == .mod) {
                        out.* = Value.int(0);
                        return true;
                    }
                    return zr_overflow(ctx, node);
                }
                out.* = Value.int(if (op == .floordiv) floorDiv(x, y) else floorMod(x, y));
                return true;
            },
            .div => {
                if (y == 0) return fail(ctx, node, "division by zero", .{});
                out.* = Value.float(@as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(y)));
                return true;
            },
            .bitand, .bitor, .bitxor => {
                const r = switch (op) {
                    .bitand => x & y,
                    .bitor => x | y,
                    else => x ^ y,
                };
                // (bool & bool is a bool)
                out.* = if (a.kind() == .bool and b.kind() == .bool) Value.boolean(r != 0) else Value.int(r);
                return true;
            },
            else => {},
        }
    }
    const fa: ?f64 = switch (a.kind()) {
        .int, .bool => @floatFromInt(a.asInt()),
        .float => a.asFloat(),
        else => null,
    };
    const fb: ?f64 = switch (b.kind()) {
        .int, .bool => @floatFromInt(b.asInt()),
        .float => b.asFloat(),
        else => null,
    };
    if (fa != null and fb != null and (a.kind() == .float or b.kind() == .float)) {
        const x = fa.?;
        const y = fb.?;
        switch (op) {
            .add => {
                out.* = Value.float(x + y);
                return true;
            },
            .sub => {
                out.* = Value.float(x - y);
                return true;
            },
            .mul => {
                out.* = Value.float(x * y);
                return true;
            },
            .div => {
                if (y == 0) return fail(ctx, node, "division by zero", .{});
                out.* = Value.float(x / y);
                return true;
            },
            .floordiv => {
                if (y == 0) return fail(ctx, node, "division by zero", .{});
                out.* = Value.float(@floor(x / y));
                return true;
            },
            .mod => {
                if (y == 0) return fail(ctx, node, "division by zero", .{});
                out.* = Value.float(floatMod(x, y));
                return true;
            },
            else => {},
        }
    }
    if (op == .add and a.kind() == .str and b.kind() == .str) {
        const x: *value.Str = @ptrCast(a.ptr());
        const y: *value.Str = @ptrCast(b.ptr());
        var buf = allocator.alloc(u8, x.len + y.len) catch return fail(ctx, node, "out of memory", .{});
        defer allocator.free(buf);
        @memcpy(buf[0..x.len], x.bytes());
        @memcpy(buf[x.len..], y.bytes());
        const r = value.newStr(buf) orelse return fail(ctx, node, "out of memory", .{});
        out.* = Value.obj(.str, &r.head);
        return true;
    }
    // Anything else: Python's result (or error) for the same objects
    return pythonBinary(ctx, node, op, a, b, out);
}

const Cmp = enum(u32) { eq, ne, lt, le, gt, ge, is, is_not, in, not_in };

/// a <cmp> b, Python's way: a bool in `out` (`in` too).
export fn zr_compare(ctx: *Ctx, node: u32, cmp_code: u32, ta: u64, ba: u64, tb: u64, bb: u64, out: *Value) callconv(.c) bool {
    const a = Value{ .tag = ta, .bits = ba };
    const b = Value{ .tag = tb, .bits = bb };
    const cmp: Cmp = @enumFromInt(cmp_code);
    switch (cmp) {
        .eq, .ne => {
            if (a.kind() != .host and b.kind() != .host) {
                const r = value.equal(a, b);
                out.* = Value.boolean(if (cmp == .eq) r else !r);
                return true;
            }
        },
        .is, .is_not => {
            const same = a.tag == b.tag and (a.bits == b.bits or (a.kind() == .none));
            out.* = Value.boolean(if (cmp == .is) same else !same);
            return true;
        },
        .lt, .le, .gt, .ge => {
            const order: ?std.math.Order = blk: {
                if (isInt(a) and isInt(b)) break :blk std.math.order(a.asInt(), b.asInt());
                const fa: ?f64 = switch (a.kind()) {
                    .int, .bool => @floatFromInt(a.asInt()),
                    .float => a.asFloat(),
                    else => null,
                };
                const fb: ?f64 = switch (b.kind()) {
                    .int, .bool => @floatFromInt(b.asInt()),
                    .float => b.asFloat(),
                    else => null,
                };
                if (fa != null and fb != null) {
                    if (std.math.isNan(fa.?) or std.math.isNan(fb.?)) {
                        out.* = Value.boolean(false);
                        return true;
                    }
                    break :blk std.math.order(fa.?, fb.?);
                }
                if (a.kind() == .str and b.kind() == .str) {
                    break :blk std.mem.order(u8, @as(*value.Str, @ptrCast(a.ptr())).bytes(), @as(*value.Str, @ptrCast(b.ptr())).bytes());
                }
                break :blk null;
            };
            if (order) |o| {
                const r = switch (cmp) {
                    .lt => o == .lt,
                    .le => o != .gt,
                    .gt => o == .gt,
                    else => o != .lt,
                };
                out.* = Value.boolean(r);
                return true;
            }
        },
        .in, .not_in => {},
    }
    // Through Python
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, &.{ a, b }, &objs)) return failPython(ctx, node);
    defer for (objs) |o| py.Py_DecRef(o);
    const r: c_int = switch (cmp) {
        .in, .not_in => py.c.PySequence_Contains(objs[1], objs[0]),
        else => blk: {
            const op: c_int = switch (cmp) {
                .eq => py.c.Py_EQ,
                .ne => py.c.Py_NE,
                .lt => py.c.Py_LT,
                .le => py.c.Py_LE,
                .gt => py.c.Py_GT,
                else => py.c.Py_GE,
            };
            break :blk py.c.PyObject_RichCompareBool(objs[0], objs[1], op);
        },
    };
    if (r < 0) return failPython(ctx, node);
    out.* = Value.boolean(if (cmp == .not_in) r == 0 else r == 1);
    return true;
}

const Unary = enum(u32) { neg, pos, not_, invert };

export fn zr_unary(ctx: *Ctx, node: u32, op_code: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const a = Value{ .tag = t, .bits = bits };
    const op: Unary = @enumFromInt(op_code);
    switch (op) {
        .not_ => {
            out.* = Value.boolean(!value.truthy(a));
            return true;
        },
        .neg => if (isInt(a)) {
            if (a.asInt() == std.math.minInt(i64)) return zr_overflow(ctx, node);
            out.* = Value.int(-a.asInt());
            return true;
        } else if (a.kind() == .float) {
            out.* = Value.float(-a.asFloat());
            return true;
        },
        .pos => if (isInt(a)) {
            out.* = Value.int(a.asInt());
            return true;
        } else if (a.kind() == .float) {
            out.* = a;
            return true;
        },
        .invert => if (isInt(a)) {
            out.* = Value.int(~a.asInt());
            return true;
        },
    }
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, &.{a}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const r = switch (op) {
        .neg => py.c.PyNumber_Negative(objs[0]),
        .pos => py.c.PyNumber_Positive(objs[0]),
        else => py.c.PyNumber_Invert(objs[0]),
    };
    return fromResult(ctx, node, r, out);
}

export fn zr_truthy(t: u64, bits: u64) callconv(.c) bool {
    return value.truthy(.{ .tag = t, .bits = bits });
}

// ======================================================================
// Functions and calls
// ======================================================================

/// The code of a compiled function: (ctx, env, args, nargs, receiver,
/// result) -> ok
pub const Code = *const fn (ctx: *Ctx, env: ?*value.Frame, args: [*]const Value, nargs: u64, receiver: ?*const Value, result: *Value) callconv(.c) bool;

/// A function value: its code, the frame it's made in, its node, its name
/// and parameter count, in `out`.
export fn zr_function(ctx: *Ctx, code: Code, env: ?*value.Frame, node: u32, name: *value.Str, nparams: u64, out: *Value) callconv(.c) bool {
    const f = allocator.create(value.Function) catch return fail(ctx, node, "out of memory", .{});
    if (env) |e| value.increfObj(&e.head);
    value.increfObj(&name.head);
    f.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.function), .flags = @intCast(nparams) }, .code = @ptrCast(code), .env = env, .node = node, .name = name };
    out.* = Value.obj(.function, &f.head);
    return true;
}

/// Call a function value (the program's, or a host one) with arguments
/// (borrowed); the result in `out`. `node`: the node calling (errors, the
/// stack).
export fn zr_call(ctx: *Ctx, node: u32, ft: u64, fb: u64, args: [*]const Value, nargs: u64, receiver: ?*const Value, out: *Value) callconv(.c) bool {
    const f = Value{ .tag = ft, .bits = fb };
    switch (f.kind()) {
        .function => {
            const fo: *value.Function = @ptrCast(@alignCast(f.ptr()));
            const nparams: u64 = fo.head.flags;
            if (nparams != nargs) {
                return fail(ctx, node, "{s}() takes {d} argument{s}, {d} given", .{ fo.name.bytes(), nparams, if (nparams == 1) "" else "s", nargs });
            }
            if (ctx.calls.items.len >= ctx.max_depth) return fail(ctx, node, "call stack too deep (more than {d} calls)", .{ctx.max_depth});
            ctx.calls.append(allocator, .{ .name = fo.name, .node = node }) catch return fail(ctx, node, "out of memory", .{});
            defer _ = ctx.calls.pop();
            const code: Code = @ptrCast(@alignCast(fo.code.?));
            return code(ctx, fo.env, args, nargs, receiver, out);
        },
        .host => {
            const callee: *PyObject = @ptrFromInt(f.bits);
            const n = nargs + @intFromBool(receiver != null);
            const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
            defer py.Py_DecRef(tuple);
            var k: usize = 0;
            if (receiver) |r| {
                const o = value.toPython(r.*, ctx.node_maker) orelse return failPython(ctx, node);
                _ = py.c.PyTuple_SetItem(tuple, 0, o);
                k = 1;
            }
            for (args[0..nargs], 0..) |a, i| {
                const o = value.toPython(a, ctx.node_maker) orelse return failPython(ctx, node);
                _ = py.c.PyTuple_SetItem(tuple, @intCast(k + i), o);
            }
            const r = py.c.PyObject_CallObject(callee, tuple) orelse return hostFailed(ctx, node, callee);
            defer py.Py_DecRef(r);
            out.* = value.fromPython(r) orelse return failPython(ctx, node);
            return true;
        },
        else => return fail(ctx, node, "'{s}' value is not callable", .{value.typeName(f)}),
    }
}

/// "name: Error: message", as the reference mode words a host function's
/// failure.
fn hostFailed(ctx: *Ctx, node: u32, f: *PyObject) bool {
    var buf: [512]u8 = undefined;
    const text = ph.takeError(&buf);
    var name_buf: [128]u8 = undefined;
    var name: []const u8 = "host function";
    if (ph.attr(f, "__name__")) |n| {
        defer py.Py_DecRef(n);
        if (ph.utf8(n, "name")) |s| {
            const k = @min(s.len, name_buf.len);
            @memcpy(name_buf[0..k], s[0..k]);
            name = name_buf[0..k];
        } else py.c.PyErr_Clear();
    } else py.c.PyErr_Clear();
    return fail(ctx, node, "{s}: {s}", .{ name, text });
}

/// A host object of the program (a host function...) as a value.
export fn zr_object(ctx: *Ctx, idx: u64, out: *Value) callconv(.c) void {
    const o = ctx.objects[idx];
    py.Py_IncRef(o);
    out.* = .{ .tag = @intFromEnum(Tag.host), .bits = @intFromPtr(o) };
}

/// A heap frame for a run of a function (its slots unset).
pub export fn zr_frame_new(parent: ?*value.Frame, n: u64) callconv(.c) ?*value.Frame {
    const f = value.newFrame(parent, n) orelse return null;
    for (f.slots()) |*s| s.tag = UNSET;
    return f;
}

export fn zr_frame_release(f: *value.Frame) callconv(.c) void {
    value.decrefFrame(f);
}

/// The tag of a variable without a value yet
pub const UNSET: u64 = 0xFFFF_0000;

// ======================================================================
// Containers and records
// ======================================================================

fn oomFail(ctx: *Ctx, node: u32) bool {
    return fail(ctx, node, "out of memory", .{});
}

/// A new list of n items (taking the references).
export fn zr_list(ctx: *Ctx, node: u32, items: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const l = value.newList(n) orelse return oomFail(ctx, node);
    for (items[0..n]) |v| _ = value.listPush(l, v);
    out.* = Value.obj(.list, &l.head);
    return true;
}

/// A new tuple of n items (taking the references).
export fn zr_tuple(ctx: *Ctx, node: u32, items: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const t = value.newTuple(n) orelse return oomFail(ctx, node);
    @memcpy(t.slice(), items[0..n]);
    out.* = Value.obj(.tuple, &t.head);
    return true;
}

/// A new dict from n keys and values (borrowing them), in order.
export fn zr_dict(ctx: *Ctx, node: u32, keys: [*]const Value, vals: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const d = value.newDict() orelse return oomFail(ctx, node);
    for (0..n) |i| {
        if (!value.hashable(keys[i])) {
            value.decref(Value.obj(.dict, &d.head));
            return fail(ctx, node, "unhashable type: '{s}'", .{value.typeName(keys[i])});
        }
        if (!value.dictSet(d, keys[i], vals[i])) return oomFail(ctx, node);
    }
    out.* = Value.obj(.dict, &d.head);
    return true;
}

/// A new record of a type with its fields (taking the references).
export fn zr_record(ctx: *Ctx, node: u32, rtype: *const value.RecordType, fields: [*]const Value, out: *Value) callconv(.c) bool {
    const r = value.newRecord(rtype) orelse return oomFail(ctx, node);
    @memcpy(r.fields(), fields[0..rtype.fields.len]);
    out.* = Value.obj(.record, &r.head);
    return true;
}

/// Is `v` a record of `rtype`?
export fn zr_is_record(t: u64, bits: u64, rtype: *const value.RecordType) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() != .record) return false;
    const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
    return r.rtype == rtype;
}

/// v.name
export fn zr_getattr(ctx: *Ctx, node: u32, t: u64, bits: u64, name: *const value.Str, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() == .record) {
        const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
        for (r.rtype.fields, 0..) |f, i| {
            if (std.mem.eql(u8, f, name.bytes())) {
                const x = r.fields()[i];
                value.incref(x);
                out.* = x;
                return true;
            }
        }
    }
    // Anything else (and the errors): Python's getattr
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, &.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const key = ph.newString(name.bytes()) orelse return failPython(ctx, node);
    defer py.Py_DecRef(key);
    return fromResult(ctx, node, py.c.PyObject_GetAttr(objs[0], key), out);
}

/// v.name = x (records; anything else as Python does it, which for a
/// copied value changes nothing visible: an error is raised instead)
export fn zr_setattr(ctx: *Ctx, node: u32, t: u64, bits: u64, name: *const value.Str, xt: u64, xb: u64) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const x = Value{ .tag = xt, .bits = xb };
    if (v.kind() == .record) {
        const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
        for (r.rtype.fields, 0..) |f, i| {
            if (std.mem.eql(u8, f, name.bytes())) {
                value.incref(x);
                value.decref(r.fields()[i]);
                r.fields()[i] = x;
                return true;
            }
        }
    }
    if (v.kind() == .host) {
        const o: *PyObject = @ptrFromInt(v.bits);
        const val = value.toPython(x, ctx.node_maker) orelse return failPython(ctx, node);
        defer py.Py_DecRef(val);
        const key = ph.newString(name.bytes()) orelse return failPython(ctx, node);
        defer py.Py_DecRef(key);
        if (py.c.PyObject_SetAttr(o, key, val) != 0) return failPython(ctx, node);
        return true;
    }
    return fail(ctx, node, "'{s}' object has no attribute '{s}'", .{ value.typeName(v), name.bytes() });
}

/// Normalize an index (negative from the end); null if out of range.
fn index(i: i64, len: u64) ?usize {
    const n: i64 = @intCast(len);
    const k = if (i < 0) i + n else i;
    if (k < 0 or k >= n) return null;
    return @intCast(k);
}

/// v[k]
export fn zr_getitem(ctx: *Ctx, node: u32, t: u64, bits: u64, kt: u64, kb: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const k = Value{ .tag = kt, .bits = kb };
    switch (v.kind()) {
        .list, .tuple => if (isInt(k)) {
            const items = if (v.kind() == .list) @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
            const i = index(k.asInt(), items.len) orelse return fail(ctx, node, "{s} index out of range", .{@tagName(v.kind())});
            value.incref(items[i]);
            out.* = items[i];
            return true;
        },
        .dict => {
            const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
            if (value.hashable(k)) {
                if (value.dictGet(d, k)) |x| {
                    value.incref(x);
                    out.* = x;
                    return true;
                }
            }
        },
        else => {},
    }
    // Strings, slices of everything, the errors: Python's
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, &.{ v, k }, &objs)) return failPython(ctx, node);
    defer for (objs) |o| py.Py_DecRef(o);
    return fromResult(ctx, node, py.c.PyObject_GetItem(objs[0], objs[1]), out);
}

/// v[k] = x
export fn zr_setitem(ctx: *Ctx, node: u32, t: u64, bits: u64, kt: u64, kb: u64, xt: u64, xb: u64) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const k = Value{ .tag = kt, .bits = kb };
    const x = Value{ .tag = xt, .bits = xb };
    switch (v.kind()) {
        .list => if (isInt(k)) {
            const l: *value.List = @ptrCast(@alignCast(v.ptr()));
            const i = index(k.asInt(), l.len) orelse return fail(ctx, node, "list assignment index out of range", .{});
            value.incref(x);
            value.decref(l.items.?[i]);
            l.items.?[i] = x;
            return true;
        } else return fail(ctx, node, "list indices must be integers or slices, not {s}", .{value.typeName(k)}),
        .dict => {
            const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
            if (!value.hashable(k)) return fail(ctx, node, "unhashable type: '{s}'", .{value.typeName(k)});
            if (!value.dictSet(d, k, x)) return oomFail(ctx, node);
            return true;
        },
        else => return fail(ctx, node, "'{s}' object does not support item assignment", .{value.typeName(v)}),
    }
}

/// The items of an iterable as a new list (lists, tuples, a dict's keys, a
/// string's characters, anything Python iterates).
export fn zr_items(ctx: *Ctx, node: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    switch (v.kind()) {
        .list, .tuple => {
            const items = if (v.kind() == .list) @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
            const l = value.newList(items.len) orelse return oomFail(ctx, node);
            for (items) |x| {
                value.incref(x);
                _ = value.listPush(l, x);
            }
            out.* = Value.obj(.list, &l.head);
            return true;
        },
        .dict => {
            const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
            const l = value.newList(d.len) orelse return oomFail(ctx, node);
            for (value.dictEntries(d)) |e| {
                if (value.isDeleted(e)) continue;
                value.incref(e.key);
                _ = value.listPush(l, e.key);
            }
            out.* = Value.obj(.list, &l.head);
            return true;
        },
        else => {},
    }
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, &.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const seq = py.c.PySequence_List(objs[0]);
    return fromResult(ctx, node, seq, out);
}

/// Python functions unpacking into n names (`a0, a1 = x`), by n: what
/// Python does (and raises) for anything not a list or tuple of n items.
var unpackers: ?*PyObject = null;

fn unpacker(n: u64) ?*PyObject {
    if (unpackers == null) unpackers = py.c.PyDict_New() orelse return null;
    const key = py.c.PyLong_FromUnsignedLongLong(n) orelse return null;
    defer py.Py_DecRef(key);
    if (py.c.PyDict_GetItemWithError(unpackers.?, key)) |f| return f;
    if (py.c.PyErr_Occurred() != null) return null;
    var src: std.ArrayListUnmanaged(u8) = .empty;
    defer src.deinit(allocator);
    src.appendSlice(allocator, "def unpack(x):\n    ") catch return null;
    for (0..n) |i| src.print(allocator, "a{d}, ", .{i}) catch return null;
    src.appendSlice(allocator, "= x\n    return (") catch return null;
    for (0..n) |i| src.print(allocator, "a{d}, ", .{i}) catch return null;
    src.appendSlice(allocator, ")\n") catch return null;
    const code = ph.newString(src.items) orelse return null;
    defer py.Py_DecRef(code);
    const ns = py.c.PyDict_New() orelse return null;
    defer py.Py_DecRef(ns);
    const builtins = py.c.PyEval_GetBuiltins() orelse return null;
    const exec = py.c.PyDict_GetItemString(builtins, "exec") orelse return null;
    const r = py.c.PyObject_CallFunctionObjArgs(exec, code, ns, @as(?*PyObject, null)) orelse return null;
    py.Py_DecRef(r);
    const f = py.c.PyDict_GetItemString(ns, "unpack") orelse return null;
    if (py.c.PyDict_SetItem(unpackers.?, key, f) != 0) return null;
    return f;
}

/// `a0, ..., an-1 = v`: the n items in out (new references).
export fn zr_unpack(ctx: *Ctx, node: u32, t: u64, bits: u64, n: u64, out: [*]Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    switch (v.kind()) {
        .list, .tuple => {
            const items = if (v.kind() == .list) @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
            if (items.len == n) {
                for (items, 0..) |x, i| {
                    value.incref(x);
                    out[i] = x;
                }
                return true;
            }
        },
        else => {},
    }
    const f = unpacker(n) orelse return failPython(ctx, node);
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, &.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const r = py.c.PyObject_CallFunctionObjArgs(f, objs[0], @as(?*PyObject, null)) orelse return failPython(ctx, node);
    defer py.Py_DecRef(r);
    for (0..n) |i| {
        const x = py.c.PyTuple_GetItem(r, @intCast(i)) orelse return failPython(ctx, node);
        out[i] = value.fromPython(x) orelse {
            for (out[0..i]) |y| value.decref(y);
            return failPython(ctx, node);
        };
    }
    return true;
}

/// The length of a list (its items' count), for loops over it.
export fn zr_list_len(t: u64, bits: u64) callconv(.c) u64 {
    const v = Value{ .tag = t, .bits = bits };
    return @as(*value.List, @ptrCast(@alignCast(v.ptr()))).len;
}

/// A list's i-th item (a new reference), for loops over it.
export fn zr_list_at(t: u64, bits: u64, i: u64, out: *Value) callconv(.c) void {
    const v = Value{ .tag = t, .bits = bits };
    const x = @as(*value.List, @ptrCast(@alignCast(v.ptr()))).items.?[i];
    value.incref(x);
    out.* = x;
}

/// list.append(x) (borrowing x)
export fn zr_append(ctx: *Ctx, node: u32, t: u64, bits: u64, xt: u64, xb: u64) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const x = Value{ .tag = xt, .bits = xb };
    if (v.kind() != .list) return fail(ctx, node, "'{s}' object has no attribute 'append'", .{value.typeName(v)});
    value.incref(x);
    if (!value.listPush(@ptrCast(@alignCast(v.ptr())), x)) return oomFail(ctx, node);
    return true;
}

/// A method of a value called with arguments, as Python does it (for
/// methods that don't change the value: str's, a dict's get...).
export fn zr_call_method(ctx: *Ctx, node: u32, t: u64, bits: u64, name: *const value.Str, args: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    // A dict's get(): natively (Python would see a copy)
    if (v.kind() == .dict and std.mem.eql(u8, name.bytes(), "get") and (n == 1 or n == 2)) {
        const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
        if (value.hashable(args[0])) {
            const x = value.dictGet(d, args[0]) orelse (if (n == 2) args[1] else Value.none_v);
            value.incref(x);
            out.* = x;
            return true;
        }
    }
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, &.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const key = ph.newString(name.bytes()) orelse return failPython(ctx, node);
    defer py.Py_DecRef(key);
    const method = py.c.PyObject_GetAttr(objs[0], key) orelse return failPython(ctx, node);
    defer py.Py_DecRef(method);
    const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
    defer py.Py_DecRef(tuple);
    for (args[0..n], 0..) |a, i| {
        const o = value.toPython(a, ctx.node_maker) orelse return failPython(ctx, node);
        _ = py.c.PyTuple_SetItem(tuple, @intCast(i), o);
    }
    return fromResult(ctx, node, py.c.PyObject_CallObject(method, tuple), out);
}

/// A builtin called with arguments, as Python does it (int(), float(),
/// len(), zip()... of run-time values): a host object of the program.
export fn zr_call_python(ctx: *Ctx, node: u32, callee_index: u64, args: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const callee = ctx.objects[callee_index];
    const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
    defer py.Py_DecRef(tuple);
    for (args[0..n], 0..) |a, i| {
        const o = value.toPython(a, ctx.node_maker) orelse return failPython(ctx, node);
        _ = py.c.PyTuple_SetItem(tuple, @intCast(i), o);
    }
    return fromResult(ctx, node, py.c.PyObject_CallObject(callee, tuple), out);
}

/// isinstance(v, <a builtin type>): by tag (int, float, str, bool, list,
/// tuple, dict); `code` says which.
export fn zr_is_type(t: u64, bits: u64, code: u32) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    return switch (code) {
        0 => v.kind() == .int or v.kind() == .bool, // int (bool is an int)
        1 => v.kind() == .float,
        2 => v.kind() == .str,
        3 => v.kind() == .bool,
        4 => v.kind() == .list,
        5 => v.kind() == .tuple,
        6 => v.kind() == .dict,
        7 => v.kind() == .none,
        else => false,
    };
}

/// A value formatted for an f-string ({v!conversion:spec}), as Python does.
export fn zr_format(ctx: *Ctx, node: u32, t: u64, bits: u64, conversion: u32, spec: *const value.Str, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, &.{v}, &objs)) return failPython(ctx, node);
    var o = objs[0];
    defer py.Py_DecRef(o);
    if (conversion != 0) {
        const conv = switch (conversion) {
            'r' => py.c.PyObject_Repr(o),
            'a' => py.c.PyObject_ASCII(o),
            else => py.c.PyObject_Str(o),
        } orelse return failPython(ctx, node);
        py.Py_DecRef(o);
        o = conv;
    }
    const s = ph.newString(spec.bytes()) orelse return failPython(ctx, node);
    defer py.Py_DecRef(s);
    return fromResult(ctx, node, py.c.PyObject_Format(o, s), out);
}

/// Strings joined (an f-string's pieces): n strs, borrowed.
export fn zr_concat(ctx: *Ctx, node: u32, items: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    var total: usize = 0;
    for (items[0..n]) |v| total += @as(*value.Str, @ptrCast(v.ptr())).len;
    const buf = allocator.alloc(u8, total) catch return oomFail(ctx, node);
    defer allocator.free(buf);
    var at: usize = 0;
    for (items[0..n]) |v| {
        const b = @as(*value.Str, @ptrCast(v.ptr())).bytes();
        @memcpy(buf[at..][0..b.len], b);
        at += b.len;
    }
    const s = value.newStr(buf) orelse return oomFail(ctx, node);
    out.* = Value.obj(.str, &s.head);
    return true;
}

/// The other helpers, by name
pub fn moreSymbols() [16]struct { []const u8, usize } {
    return .{
        .{ "zr_list", @intFromPtr(&zr_list) },
        .{ "zr_tuple", @intFromPtr(&zr_tuple) },
        .{ "zr_dict", @intFromPtr(&zr_dict) },
        .{ "zr_record", @intFromPtr(&zr_record) },
        .{ "zr_is_record", @intFromPtr(&zr_is_record) },
        .{ "zr_getattr", @intFromPtr(&zr_getattr) },
        .{ "zr_setattr", @intFromPtr(&zr_setattr) },
        .{ "zr_getitem", @intFromPtr(&zr_getitem) },
        .{ "zr_setitem", @intFromPtr(&zr_setitem) },
        .{ "zr_items", @intFromPtr(&zr_items) },
        .{ "zr_list_len", @intFromPtr(&zr_list_len) },
        .{ "zr_list_at", @intFromPtr(&zr_list_at) },
        .{ "zr_append", @intFromPtr(&zr_append) },
        .{ "zr_call_method", @intFromPtr(&zr_call_method) },
        .{ "zr_call_python", @intFromPtr(&zr_call_python) },
        .{ "zr_is_type", @intFromPtr(&zr_is_type) },
    };
}

pub fn formatSymbols() [3]struct { []const u8, usize } {
    return .{
        .{ "zr_format", @intFromPtr(&zr_format) },
        .{ "zr_concat", @intFromPtr(&zr_concat) },
        .{ "zr_unpack", @intFromPtr(&zr_unpack) },
    };
}

/// The names compiled code calls them by, and their addresses
pub fn symbols() [14]struct { []const u8, usize } {
    return .{
        .{ "zr_incref", @intFromPtr(&zr_incref) },
        .{ "zr_decref", @intFromPtr(&zr_decref) },
        .{ "zr_fail", @intFromPtr(&zr_fail) },
        .{ "zr_unset", @intFromPtr(&zr_unset) },
        .{ "zr_overflow", @intFromPtr(&zr_overflow) },
        .{ "zr_binary", @intFromPtr(&zr_binary) },
        .{ "zr_compare", @intFromPtr(&zr_compare) },
        .{ "zr_unary", @intFromPtr(&zr_unary) },
        .{ "zr_truthy", @intFromPtr(&zr_truthy) },
        .{ "zr_function", @intFromPtr(&zr_function) },
        .{ "zr_call", @intFromPtr(&zr_call) },
        .{ "zr_object", @intFromPtr(&zr_object) },
        .{ "zr_frame_new", @intFromPtr(&zr_frame_new) },
        .{ "zr_frame_release", @intFromPtr(&zr_frame_release) },
    };
}
