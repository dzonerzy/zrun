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
    /// (the compiler's list itself: compiling more while the program runs
    /// adds to it, and may move it)
    objects: *const std.ArrayListUnmanaged(*PyObject),
    /// The language's calls being run, outermost first
    calls: std.ArrayListUnmanaged(CallEntry) = .empty,
    max_depth: u32,
    /// The error, once one happened: its node, its message
    failed: bool = false,
    err_node: u32 = 0,
    err_msg: std.ArrayListUnmanaged(u8) = .empty,
    /// The call stack when it happened (innermost first)
    err_stack: std.ArrayListUnmanaged(CallEntry) = .empty,
    /// The program running, for semantics run as Python (bridge.zig's
    /// Link; set by lib.zig)
    link: ?*anyopaque = null,
    /// A Python exception going up through the compiled code (a rt.Throw,
    /// a zrun.Error raised by Python), the error then: raised again where
    /// Python code is back in charge (owned)
    pending: ?*PyObject = null,
    /// The Python exception behind the error, when Python code called by
    /// the compiled code raised it (a ValueError...): what `except` matches
    /// and Python code above gets (owned)
    exc: ?*PyObject = null,
    /// An error of the native code the reference mode raises as a Python
    /// exception: its class and Python's message for it (the exception
    /// made only when something needs the object: exceptionOf)
    exc_class: ?*PyObject = null,
    exc_msg: std.ArrayListUnmanaged(u8) = .empty,

    /// The object at an index.
    pub fn object(self: *const Ctx, i: u64) *PyObject {
        return self.objects.items[i];
    }

    pub fn deinit(self: *Ctx) void {
        self.calls.deinit(allocator);
        self.clearError();
        self.exc_msg.deinit(allocator);
        self.err_msg.deinit(allocator);
        self.err_stack.deinit(allocator);
    }

    /// Forget the error (one caught by Python code).
    pub fn clearError(self: *Ctx) void {
        self.failed = false;
        self.err_msg.clearRetainingCapacity();
        for (self.err_stack.items) |e| value.decref(Value.obj(.str, &e.name.head));
        self.err_stack.clearRetainingCapacity();
        if (self.pending) |p| py.Py_DecRef(p);
        self.pending = null;
        if (self.exc) |e| py.Py_DecRef(e);
        self.exc = null;
        self.exc_class = null;
        self.exc_msg.clearRetainingCapacity();
    }

    /// The Python exception behind the error, if it has one: the one kept,
    /// or the one its class and message make (a new reference; null
    /// without one, or with the exception of making it).
    pub fn exceptionOf(self: *Ctx) ?*PyObject {
        if (self.pending orelse self.exc) |e| {
            py.Py_IncRef(e);
            return e;
        }
        const cls = self.exc_class orelse return null;
        const s = py.c.PyUnicode_FromStringAndSize(self.exc_msg.items.ptr, @intCast(self.exc_msg.items.len)) orelse return null;
        defer py.Py_DecRef(s);
        return py.c.PyObject_CallFunctionObjArgs(cls, s, @as(?*PyObject, null));
    }
};

pub const NodeMaker = struct {
    ctx: *anyopaque,
    make_fn: *const fn (ctx: *anyopaque, idx: u32) ?*PyObject,
    /// The program object (borrowed): what a compiled function given to
    /// Python keeps alive (its code is the program's)
    owner: ?*PyObject = null,

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

/// An error the reference mode raises as a Python exception (a
/// ZeroDivisionError...): its message, and the exception (`exc`: its
/// class; Python's own wording of it, `py_msg`, null for the message), for
/// `except` and Python code above.
fn failAs(ctx: *Ctx, node: u32, exc: *PyObject, py_msg: ?[]const u8, comptime fmt: []const u8, args: anytype) bool {
    if (ctx.failed) return false;
    // (the class and message: the exception itself made when needed)
    ctx.exc_class = exc;
    ctx.exc_msg.clearRetainingCapacity();
    if (py_msg) |m| ctx.exc_msg.appendSlice(allocator, m) catch {} else ctx.exc_msg.print(allocator, fmt, args) catch {};
    return fail(ctx, node, fmt, args);
}

/// ZRUN_STATS: how many times compiled code went through Python, by what
/// (printed when a run ends).
var stats_on: ?bool = null;
var stats: std.StringHashMapUnmanaged(u64) = .empty;

pub fn stat(comptime fmt: []const u8, args: anytype) void {
    if (stats_on == null) stats_on = std.c.getenv("ZRUN_STATS") != null;
    if (!stats_on.?) return;
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, fmt, args) catch return;
    const e = stats.getOrPut(allocator, key) catch return;
    if (!e.found_existing) {
        e.key_ptr.* = allocator.dupe(u8, key) catch return;
        e.value_ptr.* = 0;
    }
    e.value_ptr.* += 1;
}

/// A value's type's name, for stats (a Python object's class's).
pub fn statType(v: Value, buf: []u8) []const u8 {
    if (v.kind() != .host) return value.typeName(v);
    const o: *PyObject = @ptrFromInt(v.bits);
    const t: *PyObject = @ptrCast(@alignCast(ph.typeOf(o)));
    const n = py.c.PyObject_GetAttrString(t, "__name__") orelse {
        py.c.PyErr_Clear();
        return "?";
    };
    defer py.Py_DecRef(n);
    const s = ph.utf8(n, "name") orelse {
        py.c.PyErr_Clear();
        return "?";
    };
    const k = @min(s.len, buf.len);
    @memcpy(buf[0..k], s[0..k]);
    return buf[0..k];
}

/// The counts so far, most first (ZRUN_STATS), then forgotten.
pub fn printStats() void {
    if (stats_on != true or stats.count() == 0) return;
    const Entry = struct { k: []const u8, v: u64 };
    var list: std.ArrayListUnmanaged(Entry) = .empty;
    defer list.deinit(allocator);
    var it = stats.iterator();
    while (it.next()) |e| list.append(allocator, .{ .k = e.key_ptr.*, .v = e.value_ptr.* }) catch return;
    std.mem.sort(Entry, list.items, {}, struct {
        fn lt(_: void, a: Entry, b: Entry) bool {
            return a.v > b.v;
        }
    }.lt);
    std.debug.print("through Python:\n", .{});
    for (list.items[0..@min(list.items.len, 25)]) |e| std.debug.print("  {d:>9}  {s}\n", .{ e.v, e.k });
    var it2 = stats.keyIterator();
    while (it2.next()) |k| allocator.free(k.*);
    stats.clearRetainingCapacity();
}

/// The Python exception being raised as the run's error (as the reference
/// mode words it); false.
fn failPython(ctx: *Ctx, node: u32) bool {
    // (a rt.Throw, a zrun.Error: carried up as itself, for Python code
    // above to catch, as the bridge does it)
    if (py.c.PyErr_Occurred() != null and (py.c.PyErr_ExceptionMatches(types.Throw) != 0 or py.c.PyErr_ExceptionMatches(types.Error) != 0))
        return @import("bridge.zig").pythonFailure(ctx, node) != 0;
    // (the exception kept, for `except` and Python code above; its message
    // the error's)
    if (!ctx.failed and py.c.PyErr_Occurred() != null) {
        var t: ?*PyObject = null;
        var v: ?*PyObject = null;
        var tb: ?*PyObject = null;
        py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        if (v) |e| {
            py.Py_IncRef(e);
            if (ctx.exc) |old| py.Py_DecRef(old);
            ctx.exc = e;
        }
        py.c.PyErr_Restore(t, v, tb);
    }
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

/// An object whose count the compiled code took to 0.
export fn zr_free(tag: u64, bits: u64) callconv(.c) void {
    value.free(@enumFromInt(tag), @ptrFromInt(bits));
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
    return failAs(ctx, node, types.IntegerOverflow, null, "integer overflow", .{});
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
    if (stats_on == true) {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        stat("binary {s} {s} {s}", .{ statType(a, &b1), @tagName(op), statType(b, &b2) });
    }
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
        // (an I64 among them: an I64, checked; else plain, Python's: past 64
        // bits a big int)
        const checked = a.tag == @intFromEnum(Tag.int) or b.tag == @intFromEnum(Tag.int);
        const mk = if (checked) &Value.int else &Value.pint;
        switch (op) {
            .add, .sub, .mul => {
                const r = switch (op) {
                    .add => @addWithOverflow(x, y),
                    .sub => @subWithOverflow(x, y),
                    else => @mulWithOverflow(x, y),
                };
                if (r[1] != 0) return if (checked) zr_overflow(ctx, node) else pythonBinary(ctx, node, op, a, b, out);
                out.* = mk(r[0]);
                return true;
            },
            .floordiv, .mod => {
                if (y == 0) return failAs(ctx, node, py.PyExc_ZeroDivisionError(), "integer division or modulo by zero", "division by zero", .{});
                if (x == std.math.minInt(i64) and y == -1) {
                    if (op == .mod) {
                        out.* = mk(0);
                        return true;
                    }
                    return if (checked) zr_overflow(ctx, node) else pythonBinary(ctx, node, op, a, b, out);
                }
                out.* = mk(if (op == .floordiv) floorDiv(x, y) else floorMod(x, y));
                return true;
            },
            .div => {
                if (y == 0) return failAs(ctx, node, py.PyExc_ZeroDivisionError(), null, "division by zero", .{});
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
                out.* = if (a.kind() == .bool and b.kind() == .bool) Value.boolean(r != 0) else mk(r);
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
                if (y == 0) return failAs(ctx, node, py.PyExc_ZeroDivisionError(), "float division by zero", "division by zero", .{});
                out.* = Value.float(x / y);
                return true;
            },
            .floordiv => {
                if (y == 0) return failAs(ctx, node, py.PyExc_ZeroDivisionError(), "float floor division by zero", "division by zero", .{});
                out.* = Value.float(@floor(x / y));
                return true;
            },
            .mod => {
                if (y == 0) return failAs(ctx, node, py.PyExc_ZeroDivisionError(), "float modulo", "division by zero", .{});
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
        .in, .not_in => if (contains(a, b)) |r| {
            out.* = Value.boolean(if (cmp == .in) r else !r);
            return true;
        },
    }
    // Through Python
    if (stats_on == true) {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        stat("compare {s} {s} {s}", .{ statType(a, &b1), @tagName(cmp), statType(b, &b2) });
    }
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

/// a in b, natively (lists, tuples, dicts, strings); null: Python decides.
fn contains(a: Value, b: Value) ?bool {
    if (a.kind() == .host) return null;
    switch (b.kind()) {
        .list, .tuple => {
            const items = if (b.kind() == .list) @as(*value.List, @ptrCast(@alignCast(b.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(b.ptr()))).slice();
            for (items) |x| {
                if (x.kind() == .host) return null;
                if (value.equal(a, x)) return true;
            }
            return false;
        },
        .dict => {
            if (!value.hashable(a)) return null;
            return value.dictGet(@ptrCast(@alignCast(b.ptr())), a) != null;
        },
        .str => {
            if (a.kind() != .str) return null;
            const x: *value.Str = @ptrCast(a.ptr());
            const y: *value.Str = @ptrCast(b.ptr());
            return std.mem.indexOf(u8, y.bytes(), x.bytes()) != null;
        },
        else => return null,
    }
}

const Unary = enum(u32) { neg, pos, not_, invert };

export fn zr_unary(ctx: *Ctx, node: u32, op_code: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const a = Value{ .tag = t, .bits = bits };
    const op: Unary = @enumFromInt(op_code);
    // (an I64 stays one, checked; a plain int or a bool gives a plain int)
    const checked = a.tag == @intFromEnum(Tag.int);
    const mk = if (checked) &Value.int else &Value.pint;
    switch (op) {
        .not_ => {
            out.* = Value.boolean(!value.truthy(a));
            return true;
        },
        .neg => if (isInt(a)) {
            if (a.asInt() == std.math.minInt(i64)) {
                if (checked) return zr_overflow(ctx, node);
            } else {
                out.* = mk(-a.asInt());
                return true;
            }
        } else if (a.kind() == .float) {
            out.* = Value.float(-a.asFloat());
            return true;
        },
        .pos => if (isInt(a)) {
            out.* = mk(a.asInt());
            return true;
        } else if (a.kind() == .float) {
            out.* = a;
            return true;
        },
        .invert => if (isInt(a)) {
            out.* = mk(~a.asInt());
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

/// A function value's flags (its Obj header): its parameter count, and
/// what a call with fewer or more arguments does (Language.function's
/// missing= and extra=)
pub const FunctionFlags = struct {
    nparams: u16,
    missing_none: bool,
    extra: enum(u2) { @"error", drop, keep },

    pub fn of(flags: u32) FunctionFlags {
        return .{ .nparams = @truncate(flags), .missing_none = flags & 0x10000 != 0, .extra = @enumFromInt(@as(u2, @truncate(flags >> 17))) };
    }

    pub fn word(self: FunctionFlags) u32 {
        return @as(u32, self.nparams) | (@as(u32, @intFromBool(self.missing_none)) << 16) | (@as(u32, @intFromEnum(self.extra)) << 17);
    }
};

/// The arguments beyond a function's parameters, as a tuple (their own
/// references), in `out` (extra="keep": rt.varargs).
export fn zr_varargs(ctx: *Ctx, node: u32, args: [*]const Value, nargs: u64, nparams: u64, out: *Value) callconv(.c) bool {
    const n = if (nargs > nparams) nargs - nparams else 0;
    const t = value.newTuple(n) orelse return oomFail(ctx, node);
    for (t.slice(), 0..) |*slot, i| {
        // (arguments as rt.call hands them over: ints I64s)
        const v = args[nparams + i].checked();
        value.incref(v);
        slot.* = v;
    }
    out.* = Value.obj(.tuple, &t.head);
    return true;
}

/// A function value: its code, the frame it's made in, its node, its name
/// and parameter count, in `out`.
pub export fn zr_function(ctx: *Ctx, code: Code, env: ?*value.Frame, node: u32, name: *value.Str, flags: u64, out: *Value) callconv(.c) bool {
    const f = allocator.create(value.Function) catch return fail(ctx, node, "out of memory", .{});
    if (env) |e| value.increfObj(&e.head);
    value.increfObj(&name.head);
    // (flags: FunctionFlags.word())
    f.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.function), .flags = @intCast(flags) }, .code = @ptrCast(code), .env = env, .node = node, .name = name };
    out.* = Value.obj(.function, &f.head);
    return true;
}

/// Call a function value (the program's, or a host one) with arguments
/// (borrowed); the result in `out`. `node`: the node calling (errors, the
/// stack).
pub export fn zr_call(ctx: *Ctx, node: u32, ft: u64, fb: u64, args: [*]const Value, nargs: u64, receiver: ?*const Value, out: *Value) callconv(.c) bool {
    const f = Value{ .tag = ft, .bits = fb };
    switch (f.kind()) {
        .function => {
            const fo: *value.Function = @ptrCast(@alignCast(f.ptr()));
            const policy = FunctionFlags.of(fo.head.flags);
            const nparams: u64 = policy.nparams;
            if ((nargs < nparams and !policy.missing_none) or (nargs > nparams and policy.extra == .@"error")) {
                return fail(ctx, node, "{s}() takes {d} argument{s}, {d} given", .{ fo.name.bytes(), nparams, if (nparams == 1) "" else "s", nargs });
            }
            if (ctx.calls.items.len >= ctx.max_depth) return fail(ctx, node, "call stack too deep (more than {d} calls)", .{ctx.max_depth});
            ctx.calls.append(allocator, .{ .name = fo.name, .node = node }) catch return fail(ctx, node, "out of memory", .{});
            defer _ = ctx.calls.pop();
            const code: Code = @ptrCast(@alignCast(fo.code.?));
            // (its result as rt.call gives it: an int an I64; its
            // parameters are made I64s by its code)
            defer out.* = out.*.checked();
            if (nargs >= nparams) return code(ctx, fo.env, args, nargs, receiver, out);
            // Fewer than its parameters: the rest None (the code reads one
            // argument per parameter)
            var buf: [16]Value = undefined;
            const padded = if (nparams <= buf.len) buf[0..nparams] else allocator.alloc(Value, nparams) catch return fail(ctx, node, "out of memory", .{});
            defer if (nparams > buf.len) allocator.free(padded);
            @memcpy(padded[0..nargs], args[0..nargs]);
            @memset(padded[nargs..], Value.none_v);
            return code(ctx, fo.env, padded.ptr, nparams, receiver, out);
        },
        .host => {
            const callee: *PyObject = @ptrFromInt(f.bits);
            if (stats_on == true) {
                var b: [64]u8 = undefined;
                stat("call host {s}", .{statType(f, &b)});
            }
            const n = nargs + @intFromBool(receiver != null);
            const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
            defer py.Py_DecRef(tuple);
            // (its arguments and result as rt.call hands them over: ints
            // I64s)
            var k: usize = 0;
            if (receiver) |r| {
                const o = value.toPython(r.*.checked(), ctx.node_maker) orelse return failPython(ctx, node);
                _ = py.c.PyTuple_SetItem(tuple, 0, o);
                k = 1;
            }
            for (args[0..nargs], 0..) |a, i| {
                const o = value.toPython(a.checked(), ctx.node_maker) orelse return failPython(ctx, node);
                _ = py.c.PyTuple_SetItem(tuple, @intCast(k + i), o);
            }
            const r = py.c.PyObject_CallObject(callee, tuple) orelse return hostFailed(ctx, node, callee);
            defer py.Py_DecRef(r);
            out.* = (value.fromPython(r) orelse return failPython(ctx, node)).checked();
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
    const o = ctx.object(idx);
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
pub const UNSET: u64 = value.UNSET_TAG;

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
            return failAs(ctx, node, py.PyExc_TypeError(), null, "unhashable type: '{s}'", .{value.typeName(keys[i])});
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

/// A new record of a class with __slots__, its fields unset (its __init__
/// sets them).
export fn zr_record_new(ctx: *Ctx, node: u32, rtype: *const value.RecordType, out: *Value) callconv(.c) bool {
    const r = value.newRecord(rtype) orelse return oomFail(ctx, node);
    out.* = Value.obj(.record, &r.head);
    return true;
}

/// isinstance(v, <the class of rtype>): a record of it or of a subclass,
/// or a Python object of the class (one Python made).
export fn zr_is_record(t: u64, bits: u64, rtype: *const value.RecordType) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() == .host) {
        const cls = rtype.py_class orelse return false;
        const r = py.c.PyObject_IsInstance(@ptrFromInt(v.bits), cls);
        if (r < 0) py.c.PyErr_Clear();
        return r == 1;
    }
    if (v.kind() != .record) return false;
    const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
    return r.rtype.isA(rtype);
}

/// isinstance(v, cls) for any class (objects[cls_index]), as Python does
/// it (a compiled value given as Python sees it).
export fn zr_isinstance(ctx: *Ctx, node: u32, t: u64, bits: u64, cls_index: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, &.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const r = py.c.PyObject_IsInstance(objs[0], ctx.object(cls_index));
    if (r < 0) return failPython(ctx, node);
    out.* = Value.boolean(r == 1);
    return true;
}

/// v.name
export fn zr_getattr(ctx: *Ctx, node: u32, t: u64, bits: u64, name: *const value.Str, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    // (a node's: from the program's tree)
    if (v.kind() == .node) if (@import("bridge.zig").nodeAttr(ctx, @intCast(v.bits), name.bytes(), out)) |ok| return ok;
    if (v.kind() == .record) {
        const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
        for (r.rtype.fields, 0..) |f, i| {
            if (std.mem.eql(u8, f, name.bytes())) {
                const x = r.fields()[i];
                // (a slot never assigned: as Python says it)
                if (x.tag == UNSET) return failAs(ctx, node, py.PyExc_AttributeError(), null, "'{s}' object has no attribute '{s}'", .{ r.rtype.name, f });
                value.incref(x);
                out.* = x;
                return true;
            }
        }
    }
    // Anything else (and the errors): Python's getattr
    if (stats_on == true) {
        var b: [64]u8 = undefined;
        stat("getattr {s}.{s}", .{ statType(v, &b), name.bytes() });
    }
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
                if (r.rtype.frozen) return fail(ctx, node, "cannot assign to field '{s}'", .{f});
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
    return failAs(ctx, node, py.PyExc_AttributeError(), null, "'{s}' object has no attribute '{s}'", .{ value.typeName(v), name.bytes() });
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
            const i = index(k.asInt(), items.len) orelse return failAs(ctx, node, py.PyExc_IndexError(), null, "{s} index out of range", .{@tagName(v.kind())});
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
    if (stats_on == true) {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        stat("getitem {s}[{s}]", .{ statType(v, &b1), statType(k, &b2) });
    }
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
            const i = index(k.asInt(), l.len) orelse return failAs(ctx, node, py.PyExc_IndexError(), null, "list assignment index out of range", .{});
            value.incref(x);
            value.decref(l.items.?[i]);
            l.items.?[i] = x;
            return true;
        } else return failAs(ctx, node, py.PyExc_TypeError(), null, "list indices must be integers or slices, not {s}", .{value.typeName(k)}),
        .dict => {
            const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
            if (!value.hashable(k)) return failAs(ctx, node, py.PyExc_TypeError(), null, "unhashable type: '{s}'", .{value.typeName(k)});
            if (!value.dictSet(d, k, x)) return oomFail(ctx, node);
            return true;
        },
        // (a Python object: itself changed, as Python does it)
        .host => {
            var objs: [3]*PyObject = undefined;
            if (!objects(ctx, &.{ v, k, x }, &objs)) return failPython(ctx, node);
            defer for (objs) |o| py.Py_DecRef(o);
            if (py.c.PyObject_SetItem(objs[0], objs[1], objs[2]) != 0) return failPython(ctx, node);
            return true;
        },
        else => return failAs(ctx, node, py.PyExc_TypeError(), null, "'{s}' object does not support item assignment", .{value.typeName(v)}),
    }
}

/// The items of an iterable as a new list (lists, tuples, a dict's keys, a
/// string's characters, anything Python iterates).
export fn zr_items(ctx: *Ctx, node: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    switch (v.kind()) {
        // (a list itself: a loop over it sees it change, as Python's does)
        .list => {
            value.incref(v);
            out.* = v;
            return true;
        },
        .tuple => {
            const items = @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
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
    if (stats_on == true) {
        var b: [64]u8 = undefined;
        stat("items {s}", .{statType(v, &b)});
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
    if (v.kind() != .list) {
        // (a Python object's append(), or Python's error)
        var objs: [2]*PyObject = undefined;
        if (!objects(ctx, &.{ v, x }, &objs)) return failPython(ctx, node);
        defer for (objs) |o| py.Py_DecRef(o);
        const r = py.c.PyObject_CallMethod(objs[0], "append", "(O)", objs[1]) orelse return failPython(ctx, node);
        py.Py_DecRef(r);
        return true;
    }
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
    // A list's extend() by a list or tuple, its pop(): natively
    if (v.kind() == .list) {
        const l: *value.List = @ptrCast(@alignCast(v.ptr()));
        if (std.mem.eql(u8, name.bytes(), "extend") and n == 1 and (args[0].kind() == .list or args[0].kind() == .tuple)) {
            // (a copy of the items first: a list extended by itself)
            const src = if (args[0].kind() == .list) @as(*value.List, @ptrCast(@alignCast(args[0].ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(args[0].ptr()))).slice();
            const count = src.len;
            var i: usize = 0;
            while (i < count) : (i += 1) {
                const x = (if (args[0].kind() == .list) @as(*value.List, @ptrCast(@alignCast(args[0].ptr()))).slice() else src)[i];
                value.incref(x);
                if (!value.listPush(l, x)) return oomFail(ctx, node);
            }
            out.* = Value.none_v;
            return true;
        }
        if (std.mem.eql(u8, name.bytes(), "pop") and (n == 0 or (n == 1 and isInt(args[0])))) {
            if (l.len == 0) return failAs(ctx, node, py.PyExc_IndexError(), null, "pop from empty list", .{});
            const at = if (n == 0) l.len - 1 else index(args[0].asInt(), l.len) orelse return failAs(ctx, node, py.PyExc_IndexError(), null, "pop index out of range", .{});
            const items = l.items.?;
            out.* = items[at];
            std.mem.copyForwards(Value, items[at .. l.len - 1], items[at + 1 .. l.len]);
            l.len -= 1;
            return true;
        }
    }
    if (stats_on == true) {
        var b: [64]u8 = undefined;
        stat("method {s}.{s}", .{ statType(v, &b), name.bytes() });
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
    const callee = ctx.object(callee_index);
    if (stats_on == true) {
        var b: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        const name = if (ph.attr(callee, "__name__")) |nm| blk: {
            defer py.Py_DecRef(nm);
            const s = ph.utf8(nm, "name") orelse "?";
            const k = @min(s.len, b.len);
            @memcpy(b[0..k], s[0..k]);
            break :blk b[0..k];
        } else "?";
        py.c.PyErr_Clear();
        stat("call_python {s}({s})", .{ name, if (n > 0) statType(args[0], &b2) else "" });
    }
    const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
    defer py.Py_DecRef(tuple);
    for (args[0..n], 0..) |a, i| {
        const o = value.toPython(a, ctx.node_maker) orelse return failPython(ctx, node);
        _ = py.c.PyTuple_SetItem(tuple, @intCast(i), o);
    }
    return fromResult(ctx, node, py.c.PyObject_CallObject(callee, tuple), out);
}

/// v[lo:hi:step] with bounds only known at run time: lists, tuples and
/// ASCII strings natively, the rest (and the errors) as Python does it.
export fn zr_slice(ctx: *Ctx, node: u32, t: u64, bits: u64, lt: u64, lb: u64, ht: u64, hb: u64, st: u64, sb: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const bounds = [3]Value{ .{ .tag = lt, .bits = lb }, .{ .tag = ht, .bits = hb }, .{ .tag = st, .bits = sb } };
    native: {
        // (ints or None: else Python's error)
        for (bounds) |b| if (b.kind() != .none and !isInt(b)) break :native;
        const step: i64 = if (bounds[2].kind() == .none) 1 else bounds[2].asInt();
        if (step == 0) return failAs(ctx, node, py.PyExc_ValueError(), null, "slice step cannot be zero", .{});
        const len: i64 = switch (v.kind()) {
            .list => @intCast(@as(*value.List, @ptrCast(@alignCast(v.ptr()))).len),
            .tuple => @intCast(@as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).len),
            .str => blk: {
                const s: *value.Str = @ptrCast(v.ptr());
                if (s.chars != s.len) break :native;
                break :blk @intCast(s.len);
            },
            else => break :native,
        };
        // (PySlice_AdjustIndices)
        var start: i64 = if (step > 0) 0 else len - 1;
        var stop: i64 = if (step > 0) len else -1;
        if (bounds[0].kind() != .none) start = adjust(bounds[0].asInt(), len, step);
        if (bounds[1].kind() != .none) stop = adjust(bounds[1].asInt(), len, step);
        var count: usize = 0;
        if (step > 0 and start < stop) count = @intCast(@divFloor(stop - start - 1, step) + 1);
        if (step < 0 and stop < start) count = @intCast(@divFloor(start - stop - 1, -step) + 1);
        switch (v.kind()) {
            .str => {
                const src = @as(*value.Str, @ptrCast(v.ptr())).bytes();
                const buf = allocator.alloc(u8, count) catch return oomFail(ctx, node);
                defer allocator.free(buf);
                var i = start;
                for (buf) |*c| {
                    c.* = src[@intCast(i)];
                    i += step;
                }
                const s = value.newStr(buf) orelse return oomFail(ctx, node);
                out.* = Value.obj(.str, &s.head);
            },
            .list => {
                const items = @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice();
                const l = value.newList(count) orelse return oomFail(ctx, node);
                var i = start;
                for (0..count) |_| {
                    value.incref(items[@intCast(i)]);
                    _ = value.listPush(l, items[@intCast(i)]);
                    i += step;
                }
                out.* = Value.obj(.list, &l.head);
            },
            else => {
                const items = @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
                const tu = value.newTuple(count) orelse return oomFail(ctx, node);
                var i = start;
                for (tu.slice()) |*slot| {
                    value.incref(items[@intCast(i)]);
                    slot.* = items[@intCast(i)];
                    i += step;
                }
                out.* = Value.obj(.tuple, &tu.head);
            },
        }
        return true;
    }
    var objs: [4]*PyObject = undefined;
    if (!objects(ctx, &.{ v, bounds[0], bounds[1], bounds[2] }, &objs)) return failPython(ctx, node);
    defer for (objs) |o| py.Py_DecRef(o);
    const sl = py.c.PySlice_New(objs[1], objs[2], objs[3]) orelse return failPython(ctx, node);
    defer py.Py_DecRef(sl);
    return fromResult(ctx, node, py.c.PyObject_GetItem(objs[0], sl), out);
}

/// A slice bound made an index of a sequence of `len` (Python's clamping).
fn adjust(i: i64, len: i64, step: i64) i64 {
    if (i < 0) {
        const j = i +| len;
        if (j < 0) return if (step < 0) -1 else 0;
        return j;
    }
    if (i >= len) return if (step < 0) len - 1 else len;
    return i;
}

/// rt.call(f, args) with the arguments a sequence only known at run time
/// (a list, a tuple, anything Python iterates), borrowed.
export fn zr_call_seq(ctx: *Ctx, node: u32, ft: u64, fb: u64, st: u64, sb: u64, receiver: ?*const Value, out: *Value) callconv(.c) bool {
    const s = Value{ .tag = st, .bits = sb };
    switch (s.kind()) {
        .list => {
            const items = @as(*value.List, @ptrCast(@alignCast(s.ptr()))).slice();
            return zr_call(ctx, node, ft, fb, items.ptr, items.len, receiver, out);
        },
        .tuple => {
            const items = @as(*value.Tuple, @ptrCast(@alignCast(s.ptr()))).slice();
            return zr_call(ctx, node, ft, fb, items.ptr, items.len, receiver, out);
        },
        else => {
            var l: Value = undefined;
            if (!zr_items(ctx, node, st, sb, &l)) return false;
            defer value.decref(l);
            const items = @as(*value.List, @ptrCast(@alignCast(l.ptr()))).slice();
            return zr_call(ctx, node, ft, fb, items.ptr, items.len, receiver, out);
        },
    }
}

/// range(args) for a loop counting at run time: start, stop, step into
/// `out`; Python's error for arguments range() refuses.
export fn zr_range(ctx: *Ctx, node: u32, args: [*]const Value, n: u64, out: *[3]i64) callconv(.c) bool {
    var vals = [3]i64{ 0, 0, 1 };
    for (args[0..n], 0..) |a, i| {
        vals[i] = if (isInt(a)) a.asInt() else if (a.kind() == .host) blk: {
            // (a Python object: its own __index__, as range() asks it)
            const o = py.c.PyNumber_Index(@ptrFromInt(a.bits)) orelse return failPython(ctx, node);
            defer py.Py_DecRef(o);
            var overflow: c_int = 0;
            const x = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
            if (overflow != 0) return failAs(ctx, node, py.PyExc_OverflowError(), null, "range() beyond 64-bit ints isn't supported in compiled code", .{});
            break :blk x;
        } else return failAs(ctx, node, py.PyExc_TypeError(), null, "'{s}' object cannot be interpreted as an integer", .{value.typeName(a)});
    }
    const b: [3]i64 = if (n == 1) .{ 0, vals[0], 1 } else .{ vals[0], vals[1], if (n == 3) vals[2] else 1 };
    if (b[2] == 0) return failAs(ctx, node, py.PyExc_ValueError(), null, "range() arg 3 must not be zero", .{});
    out.* = b;
    return true;
}

/// The builtins zr_builtin does natively, by code
pub const Builtin = enum(u32) { int, float, len, abs, str, bool };

/// A builtin of one argument (`code`), natively for native values, as
/// Python does it; anything else (and the errors) by the builtin itself
/// (objects[callee_index]).
export fn zr_builtin(ctx: *Ctx, node: u32, code: u32, callee_index: u64, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    switch (@as(Builtin, @enumFromInt(code))) {
        // (int() of an int: a plain one, as int(I64) gives)
        .int => switch (v.kind()) {
            .int, .bool => {
                out.* = Value.pint(v.asInt());
                return true;
            },
            .float => {
                const f = v.asFloat();
                if (f == f and @abs(f) < 9.2e18) {
                    out.* = Value.pint(@intFromFloat(@trunc(f)));
                    return true;
                }
            },
            else => {},
        },
        .float => switch (v.kind()) {
            .int, .bool => {
                out.* = Value.float(@floatFromInt(v.asInt()));
                return true;
            },
            .float => {
                out.* = v;
                return true;
            },
            else => {},
        },
        .len => switch (v.kind()) {
            .list => {
                out.* = Value.pint(@intCast(@as(*value.List, @ptrCast(@alignCast(v.ptr()))).len));
                return true;
            },
            .tuple => {
                out.* = Value.pint(@intCast(@as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).len));
                return true;
            },
            .dict => {
                out.* = Value.pint(@intCast(@as(*value.Dict, @ptrCast(@alignCast(v.ptr()))).len));
                return true;
            },
            .str => {
                out.* = Value.pint(@intCast(@as(*value.Str, @ptrCast(v.ptr())).chars));
                return true;
            },
            else => {},
        },
        .abs => switch (v.kind()) {
            .int, .bool => if (v.asInt() != std.math.minInt(i64)) {
                // (abs of an I64: an I64; of a plain int or a bool: plain)
                const r = if (v.asInt() < 0) -v.asInt() else v.asInt();
                out.* = if (v.tag == @intFromEnum(Tag.int)) Value.int(r) else Value.pint(r);
                return true;
            },
            .float => {
                out.* = Value.float(@abs(v.asFloat()));
                return true;
            },
            else => {},
        },
        .str => switch (v.kind()) {
            .str => {
                value.incref(v);
                out.* = v;
                return true;
            },
            .int => {
                var buf: [24]u8 = undefined;
                const s = std.fmt.bufPrint(&buf, "{d}", .{v.asInt()}) catch unreachable;
                const r = value.newStr(s) orelse return oomFail(ctx, node);
                out.* = Value.obj(.str, &r.head);
                return true;
            },
            else => {},
        },
        .bool => switch (v.kind()) {
            .none, .bool, .int, .float, .str, .list, .tuple, .dict => {
                out.* = Value.boolean(value.truthy(v));
                return true;
            },
            else => {},
        },
    }
    return zr_call_python(ctx, node, callee_index, @ptrCast(&v), 1, out);
}

/// type(v): its class, as the reference mode has it (an I64's zrun.I64, a
/// plain int's int, a record's its class...), without Python.
export fn zr_type(t: u64, bits: u64, out: *Value) callconv(.c) void {
    const v = Value{ .tag = t, .bits = bits };
    const cls: *PyObject = switch (v.kind()) {
        .none => @ptrCast(@alignCast(ph.typeOf(py.Py_None()))),
        .bool => @ptrCast(@alignCast(py.types.typeObject("PyBool_Type"))),
        .int => if (v.isPlain()) @ptrCast(@alignCast(py.types.typeObject("PyLong_Type"))) else types.I64,
        .float => @ptrCast(@alignCast(py.types.typeObject("PyFloat_Type"))),
        .str => @ptrCast(@alignCast(py.types.typeObject("PyUnicode_Type"))),
        .list => @ptrCast(@alignCast(py.types.typeObject("PyList_Type"))),
        .tuple => @ptrCast(@alignCast(py.types.typeObject("PyTuple_Type"))),
        .dict => @ptrCast(@alignCast(py.types.typeObject("PyDict_Type"))),
        .record => @as(*value.Record, @ptrCast(@alignCast(v.ptr()))).rtype.py_class orelse @import("proxies.zig").RecordType,
        .function => @import("objects.zig").FunctionType,
        .node => @import("objects.zig").NodeType,
        .host => @ptrCast(@alignCast(ph.typeOf(@ptrFromInt(v.bits)))),
        else => @ptrCast(@alignCast(ph.typeOf(py.Py_None()))),
    };
    py.Py_IncRef(cls);
    out.* = .{ .tag = @intFromEnum(Tag.host), .bits = @intFromPtr(cls) };
}

/// A module-level name some function assigns (`global`), read when the
/// code runs from the module's dict (objects[globals_index]), then the
/// builtins; Python's NameError if neither has it.
export fn zr_global(ctx: *Ctx, node: u32, globals_index: u64, name: *const value.Str, out: *Value) callconv(.c) bool {
    const g = ctx.object(globals_index);
    const key = ph.newString(name.bytes()) orelse return failPython(ctx, node);
    defer py.Py_DecRef(key);
    const v = py.c.PyDict_GetItemWithError(g, key) orelse blk: {
        if (py.c.PyErr_Occurred() != null) return failPython(ctx, node);
        const builtins = py.c.PyEval_GetBuiltins() orelse return failPython(ctx, node);
        break :blk py.c.PyDict_GetItemWithError(builtins, key) orelse {
            if (py.c.PyErr_Occurred() != null) return failPython(ctx, node);
            return failAs(ctx, node, py.PyExc_NameError(), null, "name '{s}' is not defined", .{name.bytes()});
        };
    };
    // (a reference of ours: the dict's is another, so a list stays itself)
    py.Py_IncRef(v);
    defer py.Py_DecRef(v);
    out.* = value.fromPython(v) orelse return failPython(ctx, node);
    return true;
}

/// isinstance(v, <a builtin type>): by tag (int, float, str, bool, list,
/// tuple, dict); `code` says which.
export fn zr_is_type(t: u64, bits: u64, code: u32) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    // (a Python object: of the type or a subclass of it)
    if (v.kind() == .host) {
        const o: *PyObject = @ptrFromInt(v.bits);
        return switch (code) {
            0 => py.PyLong_Check(o),
            1 => py.PyFloat_Check(o),
            2 => py.PyUnicode_Check(o),
            3 => py.PyBool_Check(o),
            4 => py.PyList_Check(o),
            5 => py.PyTuple_Check(o),
            6 => py.PyDict_Check(o),
            8 => @import("objects.zig").asFunction(o) != null,
            else => false,
        };
    }
    return switch (code) {
        0 => v.kind() == .int or v.kind() == .bool, // int (bool is an int)
        1 => v.kind() == .float,
        2 => v.kind() == .str,
        3 => v.kind() == .bool,
        4 => v.kind() == .list,
        5 => v.kind() == .tuple,
        6 => v.kind() == .dict,
        7 => v.kind() == .none,
        8 => v.kind() == .function,
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

/// The helpers compiled code calls, by name
const helper_names = [_][]const u8{
    "zr_incref",   "zr_decref",     "zr_fail",       "zr_unset",         "zr_overflow",
    "zr_binary",   "zr_compare",    "zr_unary",      "zr_truthy",        "zr_function",
    "zr_call",     "zr_object",     "zr_frame_new",  "zr_frame_release", "zr_free",
    "zr_list",     "zr_tuple",      "zr_dict",       "zr_record",        "zr_is_record",
    "zr_getattr",  "zr_setattr",    "zr_getitem",    "zr_setitem",       "zr_items",
    "zr_list_len", "zr_list_at",    "zr_append",     "zr_call_method",   "zr_call_python",
    "zr_is_type",  "zr_global",     "zr_format",     "zr_concat",        "zr_unpack",
    "zr_varargs",  "zr_record_new", "zr_isinstance", "zr_call_seq",      "zr_slice",
    "zr_type",     "zr_builtin",    "zr_range",
};

/// The names compiled code calls them by, and their addresses
pub fn symbols() [helper_names.len]struct { []const u8, usize } {
    var out: [helper_names.len]struct { []const u8, usize } = undefined;
    inline for (helper_names, 0..) |name, i| out[i] = .{ name, @intFromPtr(&@field(@This(), name)) };
    return out;
}
