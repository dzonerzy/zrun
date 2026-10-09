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
const gil = @import("gil.zig");
const errors = @import("errors.zig");
const gc = @import("gc.zig");

const Value = value.Value;
const Tag = value.Tag;
const allocator = std.heap.c_allocator;

/// A frame of the language's call stack: the function called, where
pub const CallEntry = extern struct { name: *value.Str, node: u32 };

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
    /// The compiled program running (driver.Compiled.id): its functions'
    /// code is the only code that runs here
    program: u64 = 0,
    /// The next call is a semantic's own (zr_call_plain): a Python object
    /// called as Python calls it; taken by that call
    plain_call: bool = false,
    /// The language's calls being run, outermost first: calls[0..depth]
    /// (room for max_depth of them, made at the first call; compiled code
    /// pushes and pops them inline too)
    calls: [*]CallEntry = undefined,
    calls_room: u64 = 0,
    depth: u64 = 0,
    max_depth: u32,
    /// Where the native stack ends, with room to spare (0: a stack made
    /// for the run, room enough): a call with less left is refused as too
    /// deep, not run over its end
    stack_low: u64 = 0,
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
    /// exception: its kind and Python's message for it (the exception
    /// made only when something needs the object: exceptionOf)
    exc_kind: ?errors.Kind = null,
    /// The exception being raised, an exception value of compiled code's
    /// (value.Exc: `raise ValueError(...)`, rt.Throw), what `except ... as`
    /// gets (owned; its kind exc_kind)
    exc_value: ?Value = null,
    exc_msg: std.ArrayListUnmanaged(u8) = .empty,
    /// Functions read without a reference (value.FN_BORROWED) whose top
    /// level variables were stored to while the code ran: their references
    /// kept (bury) till it's done (drainGraveyard), so no read of one
    /// outlives it
    graveyard: std.ArrayListUnmanaged(Value) = .empty,
    /// The call rt.tail_call left for the caller to make, its frame gone
    /// (the function's result tagged TAIL_TAG meanwhile): the function,
    /// its arguments (a tuple or a list), its receiver (owned; none: none)
    tail_f: Value = Value.none_v,
    tail_args: Value = Value.none_v,
    tail_recv: ?Value = null,

    /// The object at an index.
    pub fn object(self: *const Ctx, i: u64) *PyObject {
        return self.objects.items[i];
    }

    /// A top-level variable's old value, let go of (taking its reference):
    /// a function read without a reference kept till the code's done.
    pub fn bury(self: *Ctx, v: Value) void {
        if (v.tag == @intFromEnum(Tag.function) and v.ptr().flags & value.FN_BORROWED != 0) {
            self.graveyard.append(allocator, v) catch {
                // (no room to keep it: leaked, never freed under a read)
                return;
            };
            return;
        }
        if (v.tag != UNSET) value.decref(v);
    }

    /// The code is done (it returned to Python): what it buried let go of.
    pub fn drainGraveyard(self: *Ctx) void {
        while (self.graveyard.pop()) |v| value.decref(v);
    }

    pub fn deinit(self: *Ctx) void {
        self.drainGraveyard();
        self.graveyard.deinit(allocator);
        if (self.calls_room > 0) allocator.free(self.calls[0..self.calls_room]);
        self.calls_room = 0;
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
        // (an error's Python objects: what strict mode allows)
        if (self.pending != null or self.exc != null) {
            gil.allowBegin();
            defer gil.allowEnd();
            if (self.pending) |p| py.Py_DecRef(p);
            if (self.exc) |e| py.Py_DecRef(e);
        }
        self.pending = null;
        self.exc = null;
        self.exc_kind = null;
        self.exc_msg.clearRetainingCapacity();
        if (self.exc_value) |v| value.decref(v);
        self.exc_value = null;
    }

    /// The Python exception behind the error, if it has one: the one kept,
    /// or the one its class and message make (a new reference; null
    /// without one, or with the exception of making it).
    pub fn exceptionOf(self: *Ctx) ?*PyObject {
        if (self.pending orelse self.exc) |e| {
            py.Py_IncRef(e);
            return e;
        }
        if (self.exc_value) |v| return value.toPython(v, self.node_maker);
        const kind = self.exc_kind orelse return null;
        const cls = kind.pyClass() orelse {
            ph.raise(py.PyExc_RuntimeError(), "zrun: the class of {s} wasn't found", .{kind.name()});
            return null;
        };
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
    /// A zrun.Error at a node with a message (a native one given to Python:
    /// its diagnostic the program's); null: none here
    error_fn: ?*const fn (ctx: *anyopaque, idx: u32, message: []const u8) ?*PyObject = null,

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
    var i = ctx.depth;
    while (i > 0) {
        i -= 1;
        const e = ctx.calls[i];
        value.incref(Value.obj(.str, &e.name.head));
        ctx.err_stack.append(allocator, e) catch {};
    }
    return false;
}

/// An error the reference mode raises as a Python exception (a
/// ZeroDivisionError...): its message, and the exception (`kind`: its
/// class's; Python's own wording of it, `py_msg`, null for the message),
/// for `except` and Python code above.
fn failAs(ctx: *Ctx, node: u32, kind: errors.Kind, py_msg: ?[]const u8, comptime fmt: []const u8, args: anytype) bool {
    if (ctx.failed) return false;
    // (the kind and message: the exception itself made when needed)
    ctx.exc_kind = kind;
    ctx.exc_msg.clearRetainingCapacity();
    if (py_msg) |m| ctx.exc_msg.appendSlice(allocator, m) catch {} else ctx.exc_msg.print(allocator, fmt, args) catch {};
    return fail(ctx, node, fmt, args);
}

/// A value of type `name` used as a dict key, unhashable: as this Python
/// words it (from 3.14 saying where it was used)
fn unhashableKey(ctx: *Ctx, node: u32, name: []const u8) bool {
    if (ph.minor >= 14)
        return failAs(ctx, node, .TypeError, null, "cannot use '{s}' as a dict key (unhashable type: '{s}')", .{ name, name });
    return failAs(ctx, node, .TypeError, null, "unhashable type: '{s}'", .{name});
}

/// While a run is reported (Program.run(report=True)): how many times the
/// compiled code went through Python, by what (Program.report()).
pub var collecting: bool = false;
var stats: std.StringHashMapUnmanaged(u64) = .empty;

pub fn stat(comptime fmt: []const u8, args: anytype) void {
    if (!collecting) return;
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

/// The counts so far as {what: count} (a new reference; null with an
/// exception), then forgotten.
pub fn takeStats() ?*PyObject {
    defer {
        var it = stats.keyIterator();
        while (it.next()) |k| allocator.free(k.*);
        stats.clearRetainingCapacity();
    }
    const out = py.c.PyDict_New() orelse return null;
    var it = stats.iterator();
    while (it.next()) |e| {
        const k = ph.newString(e.key_ptr.*) orelse return null;
        defer py.Py_DecRef(k);
        const v = py.c.PyLong_FromUnsignedLongLong(e.value_ptr.*) orelse return null;
        defer py.Py_DecRef(v);
        if (py.c.PyDict_SetItem(out, k, v) != 0) return null;
    }
    return out;
}

/// The Python exception being raised as the run's error (as the reference
/// mode words it); false.
fn failPython(ctx: *Ctx, node: u32) bool {
    gil.allowBegin();
    defer gil.allowEnd();
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

/// A top-level variable's old function, stored over (Ctx.bury).
export fn zr_bury(ctx: *Ctx, tag: u64, bits: u64) callconv(.c) void {
    ctx.bury(.{ .tag = tag, .bits = bits });
}

// (the GIL taken by value.incref/decref for a Python object, theirs:
// native values' counts touch no Python, and checking costs a thread-local)
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
    return failAs(ctx, node, .IntegerOverflow, null, "integer overflow", .{});
}

// ======================================================================
// Through Python: the slow and the error paths
// ======================================================================

/// Values as Python objects (new references), for Python to do what the
/// code at `node` asks of them (strict mode's message: their kinds)
fn objects(ctx: *Ctx, node: u32, vals: []const Value, out: []*PyObject) bool {
    if (gil.strictOn()) {
        var buf: [96]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        for (vals, 0..) |v, i| w.print("{s}{s}", .{ if (i > 0) ", " else "", value.typeName(v) }) catch {};
        w.writeAll(" handed to Python") catch {};
        gil.ensurePhrase(@src(), node, w.buffered());
    } else gil.ensure(@src());
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

/// An int against a float (not NaN), exactly.
fn orderIntFloat(x: i128, f: f64) std.math.Order {
    if (std.math.isInf(f)) return if (f > 0) .lt else .gt;
    if (@abs(f) >= 1.7e38) return if (f > 0) .lt else .gt;
    const whole = @trunc(f);
    const i: i128 = @intFromFloat(whole);
    if (x != i) return std.math.order(x, i);
    // (equal whole parts: the float's fraction decides)
    return std.math.order(0, f - whole);
}

/// a op b on ints in 128 bits (Bigs, or a 64-bit result's overflow): an
/// I64 among them makes it an I64 (beyond 64 bits: the overflow error);
/// past 128 bits, or division and powers, Python's.
fn wideBinary(ctx: *Ctx, node: u32, op: Op, a: Value, b: Value, out: *Value) bool {
    const x = value.wide(a).?;
    const y = value.wide(b).?;
    const checked = a.tag == @intFromEnum(Tag.int) or b.tag == @intFromEnum(Tag.int);
    const r: ?i128 = switch (op) {
        .add => blk: {
            const s = @addWithOverflow(x, y);
            break :blk if (s[1] != 0) null else s[0];
        },
        .sub => blk: {
            const s = @subWithOverflow(x, y);
            break :blk if (s[1] != 0) null else s[0];
        },
        .mul => blk: {
            const s = @mulWithOverflow(x, y);
            break :blk if (s[1] != 0) null else s[0];
        },
        .floordiv, .mod => blk: {
            if (y == 0) return failAs(ctx, node, .ZeroDivisionError, "integer division or modulo by zero", "division by zero", .{});
            if (x == std.math.minInt(i128) and y == -1) break :blk null;
            break :blk if (op == .floordiv) @divFloor(x, y) else @mod(x, y);
        },
        .bitand => x & y,
        .bitor => x | y,
        .bitxor => x ^ y,
        .rshift => blk: {
            if (y < 0) break :blk null;
            break :blk if (y >= 127) (if (x < 0) -1 else 0) else x >> @intCast(y);
        },
        .lshift => blk: {
            if (y < 0 or y >= 127) break :blk null;
            const s = @shlWithOverflow(x, @as(u7, @intCast(y)));
            break :blk if (s[1] != 0) null else s[0];
        },
        // (true division correctly rounded, powers: Python's)
        .div, .pow => null,
    };
    const v = r orelse return pythonBinary(ctx, node, op, a, b, out);
    if (checked) {
        if (v < std.math.minInt(i64) or v > std.math.maxInt(i64)) return zr_overflow(ctx, node);
        out.* = Value.int(@intCast(v));
        return true;
    }
    out.* = value.intValue(v) orelse return oomFail(ctx, node);
    return true;
}

fn pythonBinary(ctx: *Ctx, node: u32, op: Op, a: Value, b: Value, out: *Value) bool {
    if (collecting) {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        stat("binary {s} {s} {s}", .{ statType(a, &b1), @tagName(op), statType(b, &b2) });
    }
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, node,&.{ a, b }, &objs)) return failPython(ctx, node);
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
    if (setBinary(ctx, node, op, a, b, out)) |ok| return ok;
    if (bytesBinary(ctx, node, op, a, b, out)) |ok| return ok;
    // (a Big among ints: in 128 bits)
    if ((a.kind() == .big or b.kind() == .big) and value.wide(a) != null and value.wide(b) != null) return wideBinary(ctx, node, op, a, b, out);
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
                if (r[1] != 0) return if (checked) zr_overflow(ctx, node) else wideBinary(ctx, node, op, a, b, out);
                out.* = mk(r[0]);
                return true;
            },
            .floordiv, .mod => {
                if (y == 0) return failAs(ctx, node, .ZeroDivisionError, "integer division or modulo by zero", "division by zero", .{});
                if (x == std.math.minInt(i64) and y == -1) {
                    if (op == .mod) {
                        out.* = mk(0);
                        return true;
                    }
                    return if (checked) zr_overflow(ctx, node) else wideBinary(ctx, node, op, a, b, out);
                }
                out.* = mk(if (op == .floordiv) floorDiv(x, y) else floorMod(x, y));
                return true;
            },
            .div => {
                if (y == 0) return failAs(ctx, node, .ZeroDivisionError, null, "division by zero", .{});
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
                if (y == 0) return failAs(ctx, node, .ZeroDivisionError, "float division by zero", "division by zero", .{});
                out.* = Value.float(x / y);
                return true;
            },
            .floordiv => {
                if (y == 0) return failAs(ctx, node, .ZeroDivisionError, "float floor division by zero", "division by zero", .{});
                out.* = Value.float(@floor(x / y));
                return true;
            },
            .mod => {
                if (y == 0) return failAs(ctx, node, .ZeroDivisionError, "float modulo", "division by zero", .{});
                out.* = Value.float(floatMod(x, y));
                return true;
            },
            else => {},
        }
    }
    // fmt % args: printf-style, natively (percentFormat); what it doesn't
    // do, and the errors, Python's
    if (op == .mod and a.kind() == .str) {
        if (percentFormat(ctx, node, @ptrCast(a.ptr()), b, out)) |ok| return ok;
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
    // A list + a list, a tuple + a tuple: a new one of both's items
    if (op == .add and a.kind() == b.kind() and (a.kind() == .list or a.kind() == .tuple)) {
        const xs = itemsOfSeq(a);
        const ys = itemsOfSeq(b);
        return sequenceOf(ctx, node, a.kind(), &.{ xs, ys }, 1, out);
    }
    // A list, tuple or str times an int (either side): repeated (none for
    // n <= 0); a huge one Python's (its MemoryError)
    if (op == .mul) {
        const seq, const times = if (isInt(b)) .{ a, b } else .{ b, a };
        if (isInt(times) and (seq.kind() == .list or seq.kind() == .tuple or seq.kind() == .str)) {
            const n: usize = @intCast(@max(times.asInt(), 0));
            const size: usize = if (seq.kind() == .str) @as(*value.Str, @ptrCast(seq.ptr())).len else itemsOfSeq(seq).len;
            if (size == 0 or n <= (1 << 31) / size) {
                if (seq.kind() != .str) return sequenceOf(ctx, node, seq.kind(), &.{itemsOfSeq(seq)}, n, out);
                const s: *value.Str = @ptrCast(seq.ptr());
                const buf = allocator.alloc(u8, s.len * n) catch return fail(ctx, node, "out of memory", .{});
                defer allocator.free(buf);
                for (0..n) |i| @memcpy(buf[i * s.len ..][0..s.len], s.bytes());
                const r = value.newStr(buf) orelse return fail(ctx, node, "out of memory", .{});
                out.* = Value.obj(.str, &r.head);
                return true;
            }
        }
    }
    // Anything else: Python's result (or error) for the same objects
    return pythonBinary(ctx, node, op, a, b, out);
}

extern "c" fn snprintf(buf: [*]u8, n: usize, fmt: [*:0]const u8, ...) c_int;

/// fmt % arg, printf-style, as Python's str formats it: %d %i %u (a float
/// truncated), %o %x %X (`#`: 0o, 0x, 0X), %c, %s, %r, %%, and on Linux %e
/// %E %f %F %g %G (C's printf there: correctly rounded, as CPython's own
/// conversion); flags `-+ 0#` (`0` with a precision too, as Python has
/// it), a width and a precision (or `*`). Null for what isn't done here
/// (%(key)s, values that aren't numbers or strs, an error: Python's then,
/// its words).
fn percentFormat(ctx: *Ctx, node: u32, fmt_s: *value.Str, arg: Value, out: *Value) ?bool {
    const f = fmt_s.bytes();
    const one = [1]Value{arg};
    const args: []const Value = if (arg.kind() == .tuple) itemsOfSeq(arg) else if (arg.kind() == .dict or arg.kind() == .host) return null else &one;
    var next: usize = 0;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);
    var i: usize = 0;
    while (i < f.len) {
        const pct = std.mem.indexOfScalarPos(u8, f, i, '%') orelse {
            buf.appendSlice(allocator, f[i..]) catch return oomFail(ctx, node);
            break;
        };
        buf.appendSlice(allocator, f[i..pct]) catch return oomFail(ctx, node);
        i = pct + 1;
        if (i >= f.len) return null;
        if (f[i] == '(') return null;
        // flags
        var left = false;
        var plus = false;
        var space = false;
        var alt = false;
        var zero = false;
        while (i < f.len) : (i += 1) switch (f[i]) {
            '-' => left = true,
            '+' => plus = true,
            ' ' => space = true,
            '#' => alt = true,
            '0' => zero = true,
            else => break,
        };
        // width, precision (or *: the next argument, an int)
        var width: usize = 0;
        var prec: ?usize = null;
        inline for (.{ false, true }) |is_prec| {
            if (!is_prec or (i < f.len and f[i] == '.')) {
                if (is_prec) i += 1;
                var n: usize = 0;
                if (i < f.len and f[i] == '*') {
                    if (next >= args.len or !isInt(args[next])) return null;
                    const w = args[next].asInt();
                    next += 1;
                    i += 1;
                    if (w < 0) {
                        if (is_prec) return null;
                        left = true;
                    }
                    n = @intCast(@abs(w));
                } else while (i < f.len and std.ascii.isDigit(f[i])) : (i += 1) {
                    n = n * 10 + (f[i] - '0');
                    if (n > 1 << 20) return null;
                }
                if (is_prec) prec = n else width = n;
            }
        }
        while (i < f.len and (f[i] == 'h' or f[i] == 'l' or f[i] == 'L')) i += 1;
        if (i >= f.len) return null;
        const conv = f[i];
        i += 1;
        if (conv == '%') {
            buf.append(allocator, '%') catch return oomFail(ctx, node);
            continue;
        }
        if (next >= args.len) return null;
        const v = args[next];
        next += 1;
        // The text, and for a number its sign (or prefix) apart: what `0`
        // pads after
        var tbuf: [512]u8 = undefined;
        var head: []const u8 = "";
        var text: []const u8 = undefined;
        var numeric = true;
        var hb: [4]u8 = undefined;
        switch (conv) {
            'd', 'i', 'u', 'o', 'x', 'X' => {
                var huge: ?f64 = null;
                const n: i128 = switch (v.kind()) {
                    .bool, .int, .big => value.wide(v) orelse @as(i128, v.asInt()),
                    .float => if (conv == 'd' or conv == 'i' or conv == 'u') blk: {
                        const x = v.asFloat();
                        if (!std.math.isFinite(x)) return null;
                        // (beyond 128 bits: its exact digits, C's printf's)
                        if (@abs(x) >= 1.7e38) {
                            if (@import("builtin").os.tag != .linux) return null;
                            huge = x;
                            break :blk if (x < 0) -1 else 1;
                        }
                        break :blk @intFromFloat(@trunc(x));
                    } else return null,
                    else => return null,
                };
                const mag: u128 = @abs(n);
                var w = std.Io.Writer.fixed(&tbuf);
                if (huge) |x| {
                    const got = snprintf(&tbuf, tbuf.len, "%.0f", @abs(x));
                    if (got < 0 or @as(usize, @intCast(got)) >= tbuf.len) return null;
                    w.end = @intCast(got);
                } else switch (conv) {
                    'o' => w.print("{o}", .{mag}) catch return null,
                    'x' => w.print("{x}", .{mag}) catch return null,
                    'X' => w.print("{X}", .{mag}) catch return null,
                    else => w.print("{d}", .{mag}) catch return null,
                }
                var digits = w.buffered();
                // (the precision: the least digits)
                if (prec) |p| if (digits.len < p) {
                    if (p > tbuf.len - 1) return null;
                    const pad = p - digits.len;
                    std.mem.copyBackwards(u8, tbuf[pad..p], digits);
                    @memset(tbuf[0..pad], '0');
                    digits = tbuf[0..p];
                };
                var hn: usize = 0;
                if (n < 0) {
                    hb[0] = '-';
                    hn = 1;
                } else if (plus) {
                    hb[0] = '+';
                    hn = 1;
                } else if (space) {
                    hb[0] = ' ';
                    hn = 1;
                }
                if (alt and (conv == 'o' or conv == 'x' or conv == 'X')) {
                    hb[hn] = '0';
                    hb[hn + 1] = if (conv == 'o') 'o' else conv;
                    hn += 2;
                }
                head = hb[0..hn];
                text = digits;
            },
            'e', 'E', 'f', 'F', 'g', 'G' => {
                if (@import("builtin").os.tag != .linux) return null;
                const x: f64 = switch (v.kind()) {
                    .float => v.asFloat(),
                    .bool, .int => @floatFromInt(v.asInt()),
                    else => return null,
                };
                // (C's printf: the digits, no sign (ours, Python's flags);
                // `#` its alternate form, the same as Python's)
                var cf: [16]u8 = undefined;
                const cfs = std.fmt.bufPrintZ(&cf, "%{s}.*{c}", .{ if (alt) "#" else "", conv }) catch return null;
                const p: c_int = @intCast(prec orelse 6);
                const got = snprintf(&tbuf, tbuf.len, cfs.ptr, p, @abs(x));
                if (got < 0 or @as(usize, @intCast(got)) >= tbuf.len) return null;
                text = tbuf[0..@intCast(got)];
                var hn: usize = 0;
                if (std.math.signbit(x) and !std.math.isNan(x)) {
                    hb[0] = '-';
                    hn = 1;
                } else if (plus) {
                    hb[0] = '+';
                    hn = 1;
                } else if (space) {
                    hb[0] = ' ';
                    hn = 1;
                }
                head = hb[0..hn];
            },
            'c' => {
                numeric = false;
                if (v.kind() == .str) {
                    const s: *value.Str = @ptrCast(v.ptr());
                    if (s.chars != 1) return null;
                    text = s.bytes();
                } else if (isInt(v)) {
                    const cp = v.asInt();
                    if (cp < 0 or cp > 0x10FFFF or (cp >= 0xD800 and cp <= 0xDFFF)) return null;
                    const n = std.unicode.utf8Encode(@intCast(cp), &tbuf) catch return null;
                    text = tbuf[0..n];
                } else return null;
            },
            's', 'r' => {
                numeric = false;
                if (v.kind() == .str) {
                    const s: *value.Str = @ptrCast(v.ptr());
                    if (conv == 's') {
                        text = s.bytes();
                    } else {
                        // (repr of an ASCII str only: the others' escapes
                        // are Unicode's to say)
                        if (s.chars != s.len) return null;
                        text = pyRepr(s.bytes(), &tbuf) orelse return null;
                    }
                } else if (v.kind() == .int or v.kind() == .big) {
                    var w = std.Io.Writer.fixed(&tbuf);
                    w.print("{d}", .{value.wide(v) orelse @as(i128, v.asInt())}) catch return null;
                    text = w.buffered();
                } else if (plainStr(v)) |ps| {
                    text = ps.of(tbuf[0..40]);
                } else return null;
                // (a precision: its first characters)
                if (prec) |p| {
                    var at: usize = 0;
                    var k: usize = 0;
                    while (at < text.len and k < p) : (k += 1) at += std.unicode.utf8ByteSequenceLength(text[at]) catch 1;
                    text = text[0..at];
                }
            },
            else => return null,
        }
        // The width: spaces before (after, `-`), or zeros after the sign
        // for a number (`0`)
        const len = head.len + (std.unicode.utf8CountCodepoints(text) catch text.len);
        const pad = if (width > len) width - len else 0;
        if (left) {
            buf.appendSlice(allocator, head) catch return oomFail(ctx, node);
            buf.appendSlice(allocator, text) catch return oomFail(ctx, node);
            buf.appendNTimes(allocator, ' ', pad) catch return oomFail(ctx, node);
        } else if (zero and numeric) {
            buf.appendSlice(allocator, head) catch return oomFail(ctx, node);
            buf.appendNTimes(allocator, '0', pad) catch return oomFail(ctx, node);
            buf.appendSlice(allocator, text) catch return oomFail(ctx, node);
        } else {
            buf.appendNTimes(allocator, ' ', pad) catch return oomFail(ctx, node);
            buf.appendSlice(allocator, head) catch return oomFail(ctx, node);
            buf.appendSlice(allocator, text) catch return oomFail(ctx, node);
        }
    }
    // (arguments left over: Python's TypeError)
    if (next != args.len) return null;
    const r = value.newStr(buf.items) orelse return oomFail(ctx, node);
    out.* = Value.obj(.str, &r.head);
    return true;
}

/// A list's or tuple's items
fn itemsOfSeq(v: Value) []const Value {
    return if (v.kind() == .list) @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
}

/// A new list or tuple (`kind`) of the parts' items, `times` over: in
/// `out`
fn sequenceOf(ctx: *Ctx, node: u32, kind: value.Tag, parts: []const []const Value, times: usize, out: *Value) bool {
    var n: usize = 0;
    for (parts) |p| n += p.len;
    n *= times;
    if (kind == .list) {
        const l = value.newList(n) orelse return oomFail(ctx, node);
        for (0..times) |_| for (parts) |p| for (p) |x| {
            value.incref(x);
            _ = value.listPush(l, x);
        };
        out.* = Value.obj(.list, &l.head);
        return true;
    }
    const t = value.newTuple(n) orelse return oomFail(ctx, node);
    var i: usize = 0;
    for (0..times) |_| for (parts) |p| for (p) |x| {
        value.incref(x);
        t.slice()[i] = x;
        i += 1;
    };
    out.* = Value.obj(.tuple, &t.head);
    return true;
}

const Cmp = enum(u32) { eq, ne, lt, le, gt, ge, is, is_not, in, not_in };

/// How two values order, as Python's <, <=, >, >= see them: numbers (ints
/// of any width and floats, exactly), strs; `unordered` with a NaN (every
/// comparison false); null for anything else (Python's to say).
const Order = enum { lt, eq, gt, unordered };

fn orderOf(a: Value, b: Value) ?Order {
    const from = struct {
        fn std_(o: std.math.Order) Order {
            return switch (o) {
                .lt => .lt,
                .eq => .eq,
                .gt => .gt,
            };
        }
    };
    if (isInt(a) and isInt(b)) return from.std_(std.math.order(a.asInt(), b.asInt()));
    // (ints of any width, and ints with floats: exactly, as Python
    // compares them)
    if (value.wide(a)) |x| {
        if (value.wide(b)) |y| return from.std_(std.math.order(x, y));
        if (b.kind() == .float) {
            if (std.math.isNan(b.asFloat())) return .unordered;
            return from.std_(orderIntFloat(x, b.asFloat()));
        }
    } else if (value.wide(b)) |y| if (a.kind() == .float) {
        if (std.math.isNan(a.asFloat())) return .unordered;
        return from.std_(orderIntFloat(y, a.asFloat()).invert());
    };
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
        if (std.math.isNan(fa.?) or std.math.isNan(fb.?)) return .unordered;
        return from.std_(std.math.order(fa.?, fb.?));
    }
    if (a.kind() == .str and b.kind() == .str)
        return from.std_(std.mem.order(u8, @as(*value.Str, @ptrCast(a.ptr())).bytes(), @as(*value.Str, @ptrCast(b.ptr())).bytes()));
    return null;
}

/// min(*args) / max(*args) of two or more values (objects[callee_index]:
/// the builtin): natively for numbers and strs, the first of the smallest
/// (largest) as Python picks it; anything else by the builtin.
export fn zr_min_max(ctx: *Ctx, node: u32, is_max: u32, callee_index: u64, args: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    var best = args[0];
    for (args[1..n]) |x| {
        const o = orderOf(x, best) orelse return zr_call_python(ctx, node, callee_index, args, n, out);
        // (Python's: x replaces the best one if x < best (max: x > best))
        if (if (is_max != 0) o == .gt else o == .lt) best = x;
    }
    value.incref(best);
    out.* = best;
    return true;
}

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
        .lt, .le, .gt, .ge => if (setOrder(cmp, a, b)) |r| {
            out.* = Value.boolean(r);
            return true;
        } else if (orderOf(a, b)) |order| {
            const r = switch (order) {
                // (NaN: every comparison false)
                .unordered => false,
                inline else => |o| switch (cmp) {
                    .lt => o == .lt,
                    .le => o != .gt,
                    .gt => o == .gt,
                    else => o != .lt,
                },
            };
            out.* = Value.boolean(r);
            return true;
        },
        .in, .not_in => {
            if (contains(a, b)) |r| {
                out.* = Value.boolean(if (cmp == .in) r else !r);
                return true;
            }
            // (a list, a dict... looked up in a dict: Python's error)
            if (b.kind() == .dict and a.kind() != .host and !value.hashable(a))
                return unhashableKey(ctx, node, value.typeName(a));
        },
    }
    // Through Python
    if (collecting) {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        stat("compare {s} {s} {s}", .{ statType(a, &b1), @tagName(cmp), statType(b, &b2) });
    }
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, node,&.{ a, b }, &objs)) return failPython(ctx, node);
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
        .set => {
            if (!value.hashable(a)) return null;
            return set_mod.contains(setOf(b), a);
        },
        .bytes => return bytesContains(a, b),
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
    // (a Big: in 128 bits)
    if (a.kind() == .big and op != .not_) {
        const x = value.wide(a).?;
        const r: ?i128 = switch (op) {
            .neg => if (x == std.math.minInt(i128)) null else -x,
            .pos => x,
            .invert => ~x,
            .not_ => unreachable,
        };
        if (r) |v| {
            out.* = value.intValue(v) orelse return oomFail(ctx, node);
            return true;
        }
    }
    switch (op) {
        .not_ => {
            out.* = Value.boolean(!value.truthy(a));
            return true;
        },
        .neg => if (isInt(a)) {
            if (a.asInt() == std.math.minInt(i64)) {
                if (checked) return zr_overflow(ctx, node);
                // (-(-2**63): a Big)
                out.* = value.intValue(-@as(i128, a.asInt())) orelse return oomFail(ctx, node);
                return true;
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
    if (!objects(ctx, node,&.{a}, &objs)) return failPython(ctx, node);
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

/// The empty tuple, one for all (made by init(): compiled code names it)
pub var empty_tuple: ?*value.Tuple = null;

/// The arguments beyond a function's parameters, as a tuple (their own
/// references), in `out` (extra="keep": rt.varargs).
export fn zr_varargs(ctx: *Ctx, node: u32, args: [*]const Value, nargs: u64, nparams: u64, out: *Value) callconv(.c) bool {
    const n = if (nargs > nparams) nargs - nparams else 0;
    // (none: the empty tuple, one for all, as Python's ())
    if (n == 0) {
        const e = empty_tuple orelse blk: {
            const t = value.newTuple(0) orelse return oomFail(ctx, node);
            t.head.rc = value.IMMORTAL;
            empty_tuple = t;
            break :blk t;
        };
        out.* = Value.obj(.tuple, &e.head);
        return true;
    }
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
    // (a container: tracked by the cycle collector; value.zig frees it)
    const f: *value.Function = @ptrCast(@alignCast(gc.alloc(@sizeOf(value.Function)) orelse return fail(ctx, node, "out of memory", .{})));
    if (env) |e| value.increfObj(&e.head);
    value.increfObj(&name.head);
    // (flags: FunctionFlags.word())
    f.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.function), .flags = @intCast(flags) }, .code = @ptrCast(code), .env = env, .node = node, .name = name, .program = ctx.program };
    out.* = Value.obj(.function, &f.head);
    return true;
}

/// A closure a semantic made (a lambda, a nested def as a value: zr_closure)
/// in `out`: its code (the function's generic code), its env.
export fn zr_closure(ctx: *Ctx, node: u32, func: u64, env: ?*value.Frame, code: *const anyopaque, out: *Value) callconv(.c) bool {
    const c = value.newClosure(@ptrFromInt(func), env, ctx.program) orelse return oomFail(ctx, node);
    c.code = code;
    out.* = Value.obj(.closure, &c.head);
    return true;
}

/// A closure called (its arguments borrowed): its code, given them and its
/// env; Python's TypeError for a count of arguments it doesn't take.
fn callClosure(ctx: *Ctx, node: u32, c: *value.Closure, args: []const Value, out: *Value) bool {
    const func: *const @import("front.zig").Function = @ptrCast(@alignCast(c.func));
    if (c.program != ctx.program) return fail(ctx, node, "a function of another program can't be called here", .{});
    const own = func.own;
    if (args.len > own) {
        return failAs(ctx, node, .TypeError, null, "{s}() takes {d} positional argument{s} but {d} {s} given", .{ func.qualname, own, if (own == 1) "" else "s", args.len, if (args.len == 1) "was" else "were" });
    }
    if (args.len < own) {
        const missing = own - args.len;
        var buf: [256]u8 = undefined;
        var w = std.Io.Writer.fixed(&buf);
        // (Python's words: 'a', 'a' and 'b', 'a', 'b', and 'c')
        for (func.locals[args.len..own], 0..) |name, i| {
            if (i > 0) w.writeAll(if (missing == 2) " and " else if (i + 1 == missing) ", and " else ", ") catch {};
            w.print("'{s}'", .{name}) catch {};
        }
        return failAs(ctx, node, .TypeError, null, "{s}() missing {d} required positional argument{s}: {s}", .{ func.qualname, missing, if (missing == 1) "" else "s", w.buffered() });
    }
    if (own >= 63) return fail(ctx, node, "a closure of more than 62 parameters can't be called", .{});
    // (a semantic's call, not the program's: no entry on its stack, the
    // native stack's room checked)
    if (@frameAddress() < ctx.stack_low) return fail(ctx, node, "the semantics' calls nest too deep", .{});
    var call_args: [64]Value = undefined;
    @memcpy(call_args[0..own], args);
    call_args[own] = .{ .tag = value.ENV_TAG, .bits = @intFromPtr(c.env) };
    const code: @import("driver.zig").Helper = @ptrCast(@alignCast(c.code));
    return switch (code(ctx, null, &call_args, node, 0, null, null, out)) {
        1 => true,
        0 => false,
        else => fail(ctx, node, "rt.Return, rt.Break or rt.Continue raised out of a function a semantic made", .{}),
    };
}

// ----------------------------------------------------------------------
// Sets (set.zig: CPython's table, its order)
// ----------------------------------------------------------------------

const set_mod = @import("set.zig");

fn setOf(v: Value) *set_mod.Set {
    return @ptrCast(@alignCast(v.ptr()));
}

fn unhashableItem(ctx: *Ctx, node: u32, v: Value) bool {
    // (a Python object: its class's name, Python's)
    if (v.kind() == .host) {
        gil.allowBegin();
        defer gil.allowEnd();
        const t: *PyObject = @ptrCast(@alignCast(ph.typeOf(@ptrFromInt(v.bits))));
        if (ph.attr(t, "__name__")) |n| {
            defer py.Py_DecRef(n);
            if (ph.utf8(n, "name")) |s| return failAs(ctx, node, .TypeError, null, "unhashable type: '{s}'", .{s});
        }
        py.c.PyErr_Clear();
    }
    return failAs(ctx, node, .TypeError, null, "unhashable type: '{s}'", .{value.typeName(v)});
}

/// `{a, b, ...}` (the items borrowed), added in turn; `folded`: as CPython
/// makes one of constants (a frozenset of them, copied)
export fn zr_set(ctx: *Ctx, node: u32, items: [*]const Value, n: u64, folded: u32, out: *Value) callconv(.c) bool {
    for (items[0..n]) |x| if (!value.hashable(x)) return unhashableItem(ctx, node, x);
    var s = set_mod.new() orelse return oomFail(ctx, node);
    for (items[0..n]) |x| if (!set_mod.add(s, x)) {
        value.decref(Value.obj(.set, &s.head));
        return oomFail(ctx, node);
    };
    if (folded != 0) {
        const c = set_mod.copy(s);
        value.decref(Value.obj(.set, &s.head));
        s = c orelse return oomFail(ctx, node);
    }
    out.* = Value.obj(.set, &s.head);
    return true;
}

/// v's items added to s, as set.update(v) adds them (a set's merged, a
/// dict's keys after one resize, anything else's in turn)
fn setUpdate(ctx: *Ctx, node: u32, s: *set_mod.Set, v: Value) bool {
    switch (v.kind()) {
        .set => return set_mod.merge(s, setOf(v)) or oomFail(ctx, node),
        .dict => {
            const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
            var keys: std.ArrayListUnmanaged(Value) = .empty;
            defer keys.deinit(allocator);
            for (value.dictEntries(d)) |e| if (!value.isDeleted(e)) keys.append(allocator, e.key) catch return oomFail(ctx, node);
            return set_mod.mergeKeys(s, keys.items) or oomFail(ctx, node);
        },
        else => {
            var l = Value.none_v;
            if (!zr_items(ctx, node, v.tag, v.bits, &l)) return false;
            defer value.decref(l);
            for (@as(*value.List, @ptrCast(@alignCast(l.ptr()))).slice()) |x| {
                if (!value.hashable(x)) return unhashableItem(ctx, node, x);
                if (!set_mod.add(s, x)) return oomFail(ctx, node);
            }
            return true;
        },
    }
}

/// A set of v's items (set(v), a set comprehension's list)
fn newSetOf(ctx: *Ctx, node: u32, v: Value) ?*set_mod.Set {
    const s = set_mod.new() orelse {
        _ = oomFail(ctx, node);
        return null;
    };
    if (!setUpdate(ctx, node, s, v)) {
        value.decref(Value.obj(.set, &s.head));
        return null;
    }
    return s;
}

export fn zr_to_set(ctx: *Ctx, node: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const s = newSetOf(ctx, node, .{ .tag = t, .bits = bits }) orelse return false;
    out.* = Value.obj(.set, &s.head);
    return true;
}

fn setResult(ctx: *Ctx, node: u32, s: ?*set_mod.Set, out: *Value) bool {
    const r = s orelse return oomFail(ctx, node);
    out.* = Value.obj(.set, &r.head);
    return true;
}

/// A set's method, natively (null: not one of those; Python's then). Its
/// errors that Python words (a missing item's KeyError) are Python's, on a
/// copy: nothing changed by then.
fn setMethod(ctx: *Ctx, node: u32, v: Value, name: []const u8, args: []const Value, out: *Value) ?bool {
    const s = setOf(v);
    const eq = std.mem.eql;
    const python = struct {
        fn call(c: *Ctx, n: u32, sv: Value, nm: []const u8, a: []const Value, o: *Value) bool {
            gil.allowBegin();
            defer gil.allowEnd();
            const str = value.newStr(nm) orelse return oomFail(c, n);
            defer value.decref(Value.obj(.str, &str.head));
            return callMethodPython(c, n, sv, str, a, o);
        }
    };
    out.* = Value.none_v;
    if (args.len == 1 and (eq(u8, name, "add") or eq(u8, name, "discard") or eq(u8, name, "remove"))) {
        const x = args[0];
        if (!value.hashable(x)) return unhashableItem(ctx, node, x);
        if (eq(u8, name, "add")) return set_mod.add(s, x) or oomFail(ctx, node);
        if (set_mod.discard(s, x) or eq(u8, name, "discard")) return true;
        return python.call(ctx, node, v, name, args, out);
    }
    if (args.len == 0) {
        if (eq(u8, name, "pop")) {
            out.* = set_mod.pop(s) orelse return python.call(ctx, node, v, name, args, out);
            return true;
        }
        if (eq(u8, name, "clear")) return set_mod.clear(s) or oomFail(ctx, node);
        if (eq(u8, name, "copy")) return setResult(ctx, node, set_mod.copy(s), out);
    }
    if (eq(u8, name, "update")) {
        for (args) |x| if (!setUpdate(ctx, node, s, x)) return false;
        return true;
    }
    if (eq(u8, name, "union")) {
        const r = set_mod.copy(s) orelse return oomFail(ctx, node);
        out.* = Value.obj(.set, &r.head);
        for (args) |x| if (!setUpdate(ctx, node, r, x)) return false;
        return true;
    }
    if (args.len != 1) return null;
    const other = args[0];
    // (another iterable's items, in its order, for what Python goes over
    // them for: intersection, difference)
    if (other.kind() != .set and (eq(u8, name, "intersection") or eq(u8, name, "difference") or eq(u8, name, "intersection_update") or eq(u8, name, "difference_update"))) {
        var l = Value.none_v;
        if (!zr_items(ctx, node, other.tag, other.bits, &l)) return false;
        defer value.decref(l);
        const items = @as(*value.List, @ptrCast(@alignCast(l.ptr()))).slice();
        for (items) |x| if (!value.hashable(x)) return unhashableItem(ctx, node, x);
        if (eq(u8, name, "intersection")) return setResult(ctx, node, set_mod.intersectionItems(s, items), out);
        if (eq(u8, name, "intersection_update")) return set_mod.intersectionUpdateItems(s, items) or oomFail(ctx, node);
        if (eq(u8, name, "difference_update")) return set_mod.differenceUpdateItems(s, items) or oomFail(ctx, node);
        // (a copy less them: Python's way with an iterable)
        const r = set_mod.copy(s) orelse return oomFail(ctx, node);
        out.* = Value.obj(.set, &r.head);
        return set_mod.differenceUpdateItems(r, items) or oomFail(ctx, node);
    }
    // (the other one a set: Python's way with sets; another iterable made
    // one first, where Python's result is the same)
    const os: *set_mod.Set = if (other.kind() == .set) setOf(other) else newSetOf(ctx, node, other) orelse return false;
    defer if (other.kind() != .set) value.decref(Value.obj(.set, &os.head));
    if (eq(u8, name, "intersection")) return setResult(ctx, node, set_mod.intersection(s, os), out);
    if (eq(u8, name, "difference")) return setResult(ctx, node, set_mod.difference(s, os), out);
    if (eq(u8, name, "symmetric_difference")) return setResult(ctx, node, set_mod.symmetricDifference(s, os), out);
    if (eq(u8, name, "intersection_update")) return set_mod.intersectionUpdate(s, os) or oomFail(ctx, node);
    if (eq(u8, name, "difference_update")) return set_mod.differenceUpdate(s, os) or oomFail(ctx, node);
    if (eq(u8, name, "symmetric_difference_update")) return set_mod.symmetricUpdate(s, os) or oomFail(ctx, node);
    if (eq(u8, name, "issubset")) {
        out.* = Value.boolean(set_mod.isSubset(s, os));
        return true;
    }
    if (eq(u8, name, "issuperset")) {
        out.* = Value.boolean(set_mod.isSubset(os, s));
        return true;
    }
    if (eq(u8, name, "isdisjoint")) {
        var it = set_mod.iterate(if (s.used <= os.used) s else os);
        const big = if (s.used <= os.used) os else s;
        while (it.next()) |x| if (set_mod.contains(big, x)) {
            out.* = Value.boolean(false);
            return true;
        };
        out.* = Value.boolean(true);
        return true;
    }
    return null;
}

/// a | b, a & b, a - b, a ^ b of two sets (null: not two sets); a <= b,
/// a < b, a >= b, a > b of them
fn setBinary(ctx: *Ctx, node: u32, op: Op, a: Value, b: Value, out: *Value) ?bool {
    if (a.kind() != .set or b.kind() != .set) return null;
    const x = setOf(a);
    const y = setOf(b);
    return switch (op) {
        .bitor => setResult(ctx, node, set_mod.union_(x, y), out),
        .bitand => setResult(ctx, node, set_mod.intersection(x, y), out),
        .sub => setResult(ctx, node, set_mod.difference(x, y), out),
        .bitxor => setResult(ctx, node, set_mod.symmetricDifference(x, y), out),
        else => null,
    };
}

fn setOrder(cmp: Cmp, a: Value, b: Value) ?bool {
    if (a.kind() != .set or b.kind() != .set) return null;
    const x = setOf(a);
    const y = setOf(b);
    return switch (cmp) {
        .le => set_mod.isSubset(x, y),
        .lt => x.used < y.used and set_mod.isSubset(x, y),
        .ge => set_mod.isSubset(y, x),
        .gt => y.used < x.used and set_mod.isSubset(y, x),
        else => null,
    };
}

/// `a op= b`: in place where Python's is (a list's +=: extend; a set's |=,
/// &=, -=, ^=; a dict's |=: update; a Python object's own); else a op b
export fn zr_inplace(ctx: *Ctx, node: u32, op_code: u32, ta: u64, ba: u64, tb: u64, bb: u64, out: *Value) callconv(.c) bool {
    const a = Value{ .tag = ta, .bits = ba };
    const b = Value{ .tag = tb, .bits = bb };
    const op: Op = @enumFromInt(op_code);
    const in_place: bool = switch (a.kind()) {
        .list => op == .add,
        .set => op == .bitor or op == .bitand or op == .sub or op == .bitxor,
        .dict => op == .bitor,
        else => false,
    };
    if (in_place) {
        switch (a.kind()) {
            .list => {
                const l: *value.List = @ptrCast(@alignCast(a.ptr()));
                var items = Value.none_v;
                if (!zr_items(ctx, node, b.tag, b.bits, &items)) return false;
                defer value.decref(items);
                // (a copy of the items first: `l += l` adds them once)
                const src = @as(*value.List, @ptrCast(@alignCast(items.ptr()))).slice();
                const copied = allocator.dupe(Value, src) catch return oomFail(ctx, node);
                defer allocator.free(copied);
                for (copied) |x| {
                    value.incref(x);
                    if (!value.listPush(l, x)) return oomFail(ctx, node);
                }
            },
            .set => {
                if (b.kind() != .set) {
                    const sym = switch (op) {
                        .bitor => "|=",
                        .bitand => "&=",
                        .sub => "-=",
                        else => "^=",
                    };
                    return failAs(ctx, node, .TypeError, null, "unsupported operand type(s) for {s}: 'set' and '{s}'", .{ sym, value.typeName(b) });
                }
                const s = setOf(a);
                const ok = switch (op) {
                    .bitor => set_mod.merge(s, setOf(b)),
                    .bitand => set_mod.intersectionUpdate(s, setOf(b)),
                    .sub => set_mod.differenceUpdate(s, setOf(b)),
                    else => set_mod.symmetricUpdate(s, setOf(b)),
                };
                if (!ok) return oomFail(ctx, node);
            },
            else => {
                var r = Value.none_v;
                const upd = value.literal("update") orelse return oomFail(ctx, node);
                if (!zr_call_method(ctx, node, a.tag, a.bits, upd, @ptrCast(&b), 1, &r)) return false;
                value.decref(r);
            },
        }
        value.incref(a);
        out.* = a;
        return true;
    }
    // (a Python object: its in-place operator, Python's)
    if (a.kind() == .host) {
        var objs: [2]*PyObject = undefined;
        if (!objects(ctx, node, &.{ a, b }, &objs)) return failPython(ctx, node);
        defer for (objs) |o| py.Py_DecRef(o);
        const r = switch (op) {
            .add => py.c.PyNumber_InPlaceAdd(objs[0], objs[1]),
            .sub => py.c.PyNumber_InPlaceSubtract(objs[0], objs[1]),
            .mul => py.c.PyNumber_InPlaceMultiply(objs[0], objs[1]),
            .div => py.c.PyNumber_InPlaceTrueDivide(objs[0], objs[1]),
            .floordiv => py.c.PyNumber_InPlaceFloorDivide(objs[0], objs[1]),
            .mod => py.c.PyNumber_InPlaceRemainder(objs[0], objs[1]),
            .pow => py.c.PyNumber_InPlacePower(objs[0], objs[1], py.Py_None()),
            .lshift => py.c.PyNumber_InPlaceLshift(objs[0], objs[1]),
            .rshift => py.c.PyNumber_InPlaceRshift(objs[0], objs[1]),
            .bitor => py.c.PyNumber_InPlaceOr(objs[0], objs[1]),
            .bitxor => py.c.PyNumber_InPlaceXor(objs[0], objs[1]),
            .bitand => py.c.PyNumber_InPlaceAnd(objs[0], objs[1]),
        };
        return fromResult(ctx, node, r, out);
    }
    return zr_binary(ctx, node, op_code, ta, ba, tb, bb, out);
}

// ----------------------------------------------------------------------
// Python bytes (value.Bytes, PY_BYTES), natively
// ----------------------------------------------------------------------

fn pyBytesOf(v: Value) ?[]const u8 {
    if (v.kind() != .bytes) return null;
    const b: *value.Bytes = @ptrCast(@alignCast(v.ptr()));
    if (!b.isPyBytes()) return null;
    return b.slice();
}

fn bytesResult(ctx: *Ctx, node: u32, b: ?*value.Bytes, out: *Value) bool {
    const r = b orelse return oomFail(ctx, node);
    out.* = Value.obj(.bytes, &r.head);
    return true;
}

/// a + b, a * n of Python bytes (null: not those)
fn bytesBinary(ctx: *Ctx, node: u32, op: Op, a: Value, b: Value, out: *Value) ?bool {
    const x = pyBytesOf(a) orelse {
        // (n * b)
        if (op == .mul and isInt(a)) if (pyBytesOf(b)) |y| return repeatBytes(ctx, node, y, a.asInt(), out);
        return null;
    };
    switch (op) {
        .add => {
            const y = pyBytesOf(b) orelse return null;
            const r = value.newOwnedBytes(x.len + y.len) orelse return oomFail(ctx, node);
            const m = value.bytesMemory(r);
            @memcpy(m[0..x.len], x);
            @memcpy(m[x.len..], y);
            out.* = Value.obj(.bytes, &r.head);
            return true;
        },
        .mul => if (isInt(b)) return repeatBytes(ctx, node, x, b.asInt(), out),
        else => {},
    }
    return null;
}

fn repeatBytes(ctx: *Ctx, node: u32, x: []const u8, n: i64, out: *Value) bool {
    const k: usize = if (n <= 0) 0 else @intCast(n);
    const total = std.math.mul(usize, x.len, k) catch return oomFail(ctx, node);
    const r = value.newOwnedBytes(total) orelse return oomFail(ctx, node);
    const m = value.bytesMemory(r);
    for (0..k) |i| @memcpy(m[i * x.len ..][0..x.len], x);
    out.* = Value.obj(.bytes, &r.head);
    return true;
}

/// `a in b`, b a Python bytes: a byte (an int) or a bytes in it (null: not
/// those; Python's then, its errors)
fn bytesContains(a: Value, b: Value) ?bool {
    const hay = pyBytesOf(b) orelse return null;
    if (isInt(a)) {
        const n = a.asInt();
        if (n < 0 or n > 255) return null;
        return std.mem.indexOfScalar(u8, hay, @intCast(n)) != null;
    }
    const needle = pyBytesOf(a) orelse return null;
    return std.mem.indexOf(u8, hay, needle) != null;
}

/// What a bytes method's argument searches for: a bytes' bytes, an int's
/// byte (null: neither)
fn byteNeedle(v: Value, buf: *[1]u8) ?[]const u8 {
    if (pyBytesOf(v)) |s| return s;
    if (isInt(v)) {
        const n = v.asInt();
        if (n < 0 or n > 255) return null;
        buf[0] = @intCast(n);
        return buf[0..1];
    }
    return null;
}

const bytes_space = " \t\n\r\x0b\x0c";

/// A Python bytes' method, natively (null: not one of these, or arguments
/// Python takes otherwise: Python's then, its errors on a copy)
fn bytesMethod(ctx: *Ctx, node: u32, v: Value, name: []const u8, args: []const Value, out: *Value) ?bool {
    const s = pyBytesOf(v) orelse return null;
    const eq = std.mem.eql;
    var nb: [1]u8 = undefined;
    // find, rfind, count, index, rindex (no start, end: Python's for those)
    inline for (.{ "find", "rfind", "count", "index", "rindex" }) |m| if (eq(u8, name, m) and args.len == 1) {
        const needle = byteNeedle(args[0], &nb) orelse return null;
        if (comptime eq(u8, m, "count")) {
            out.* = Value.pint(if (needle.len == 0) @intCast(s.len + 1) else @intCast(std.mem.count(u8, s, needle)));
            return true;
        }
        const at = if (comptime m[0] == 'r') std.mem.lastIndexOf(u8, s, needle) else std.mem.indexOf(u8, s, needle);
        if (at) |i| {
            out.* = Value.pint(@intCast(i));
            return true;
        }
        if (comptime eq(u8, m, "find") or eq(u8, m, "rfind")) {
            out.* = Value.pint(-1);
            return true;
        }
        return null;
    };
    if ((eq(u8, name, "startswith") or eq(u8, name, "endswith")) and args.len == 1) {
        const start = name[0] == 's';
        const one = struct {
            fn f(hay: []const u8, x: []const u8, st: bool) bool {
                return if (st) std.mem.startsWith(u8, hay, x) else std.mem.endsWith(u8, hay, x);
            }
        }.f;
        if (pyBytesOf(args[0])) |x| {
            out.* = Value.boolean(one(s, x, start));
            return true;
        }
        if (args[0].kind() == .tuple) {
            const items = @as(*value.Tuple, @ptrCast(@alignCast(args[0].ptr()))).slice();
            for (items) |it| if (pyBytesOf(it) == null) return null;
            for (items) |it| if (one(s, pyBytesOf(it).?, start)) {
                out.* = Value.boolean(true);
                return true;
            };
            out.* = Value.boolean(false);
            return true;
        }
        return null;
    }
    if (eq(u8, name, "decode") and args.len <= 1) {
        const enc = if (args.len == 0) "utf-8" else blk: {
            if (args[0].kind() != .str) return null;
            break :blk @as(*value.Str, @ptrCast(args[0].ptr())).bytes();
        };
        const lower = std.ascii.eqlIgnoreCase;
        if (lower(enc, "utf-8") or lower(enc, "utf8")) {
            if (!std.unicode.utf8ValidateSlice(s)) return null;
            const r = value.newStr(s) orelse return oomFail(ctx, node);
            out.* = Value.obj(.str, &r.head);
            return true;
        }
        if (lower(enc, "ascii")) {
            for (s) |c| if (c >= 0x80) return null;
            const r = value.newStr(s) orelse return oomFail(ctx, node);
            out.* = Value.obj(.str, &r.head);
            return true;
        }
        if (lower(enc, "latin-1") or lower(enc, "latin1") or lower(enc, "iso-8859-1")) {
            var buf: std.ArrayListUnmanaged(u8) = .empty;
            defer buf.deinit(allocator);
            for (s) |c| {
                var u: [2]u8 = undefined;
                const n = std.unicode.utf8Encode(c, &u) catch unreachable;
                buf.appendSlice(allocator, u[0..n]) catch return oomFail(ctx, node);
            }
            const r = value.newStr(buf.items) orelse return oomFail(ctx, node);
            out.* = Value.obj(.str, &r.head);
            return true;
        }
        return null;
    }
    if (eq(u8, name, "hex") and args.len == 0) {
        const m = allocator.alloc(u8, s.len * 2) catch return oomFail(ctx, node);
        defer allocator.free(m);
        const digits = "0123456789abcdef";
        for (s, 0..) |c, i| {
            m[2 * i] = digits[c >> 4];
            m[2 * i + 1] = digits[c & 15];
        }
        const r = value.newStr(m) orelse return oomFail(ctx, node);
        out.* = Value.obj(.str, &r.head);
        return true;
    }
    if ((eq(u8, name, "upper") or eq(u8, name, "lower")) and args.len == 0) {
        const r = value.newOwnedBytes(s.len) orelse return oomFail(ctx, node);
        const m = value.bytesMemory(r);
        for (s, m) |c, *d| d.* = if (name[0] == 'u') std.ascii.toUpper(c) else std.ascii.toLower(c);
        out.* = Value.obj(.bytes, &r.head);
        return true;
    }
    if ((eq(u8, name, "strip") or eq(u8, name, "lstrip") or eq(u8, name, "rstrip")) and args.len <= 1) {
        const chars = if (args.len == 0 or args[0].kind() == .none) bytes_space else pyBytesOf(args[0]) orelse return null;
        var lo: usize = 0;
        var hi: usize = s.len;
        if (name[0] != 'r') {
            while (lo < hi and std.mem.indexOfScalar(u8, chars, s[lo]) != null) lo += 1;
        }
        if (name[0] != 'l') {
            while (hi > lo and std.mem.indexOfScalar(u8, chars, s[hi - 1]) != null) hi -= 1;
        }
        return bytesResult(ctx, node, value.bytesSlice(@ptrCast(@alignCast(v.ptr())), lo, hi), out);
    }
    if (eq(u8, name, "split") and args.len == 1) {
        const sep = pyBytesOf(args[0]) orelse return null;
        if (sep.len == 0) return null;
        const l = value.newList(0) orelse return oomFail(ctx, node);
        out.* = Value.obj(.list, &l.head);
        var it = std.mem.splitSequence(u8, s, sep);
        while (it.next()) |part| {
            const from = @intFromPtr(part.ptr) - @intFromPtr(s.ptr);
            const p = value.bytesSlice(@ptrCast(@alignCast(v.ptr())), from, from + part.len) orelse return oomFail(ctx, node);
            if (!value.listPush(l, Value.obj(.bytes, &p.head))) return oomFail(ctx, node);
        }
        return true;
    }
    if (eq(u8, name, "replace") and args.len == 2) {
        const old = pyBytesOf(args[0]) orelse return null;
        const new = pyBytesOf(args[1]) orelse return null;
        if (old.len == 0) return null;
        const n = std.mem.replacementSize(u8, s, old, new);
        const r = value.newOwnedBytes(n) orelse return oomFail(ctx, node);
        _ = std.mem.replace(u8, s, old, new, value.bytesMemory(r));
        out.* = Value.obj(.bytes, &r.head);
        return true;
    }
    if (eq(u8, name, "join") and args.len == 1) {
        var l = Value.none_v;
        if (!zr_items(ctx, node, args[0].tag, args[0].bits, &l)) return false;
        defer value.decref(l);
        const items = @as(*value.List, @ptrCast(@alignCast(l.ptr()))).slice();
        var total: usize = 0;
        for (items, 0..) |x, i| {
            const part = pyBytesOf(x) orelse return null;
            total += part.len + (if (i > 0) s.len else 0);
        }
        const r = value.newOwnedBytes(total) orelse return oomFail(ctx, node);
        const m = value.bytesMemory(r);
        var at: usize = 0;
        for (items, 0..) |x, i| {
            if (i > 0) {
                @memcpy(m[at..][0..s.len], s);
                at += s.len;
            }
            const part = pyBytesOf(x).?;
            @memcpy(m[at..][0..part.len], part);
            at += part.len;
        }
        out.* = Value.obj(.bytes, &r.head);
        return true;
    }
    return null;
}

// ----------------------------------------------------------------------
// sorted(), list.sort()
// ----------------------------------------------------------------------

/// a < b as Python's sort compares them, natively; null: not natively (other
/// kinds, a NaN: Python decides then)
fn sortLess(a: Value, b: Value) ?bool {
    if (orderOf(a, b)) |o| return switch (o) {
        .lt => true,
        .eq, .gt => false,
        .unordered => null,
    };
    // (tuples, lists: at the first items that aren't equal, else by length)
    const seq = (a.kind() == .tuple and b.kind() == .tuple) or (a.kind() == .list and b.kind() == .list);
    if (seq) {
        const x = itemsOfSeq(a);
        const y = itemsOfSeq(b);
        for (x[0..@min(x.len, y.len)], y[0..@min(x.len, y.len)]) |p, q| {
            if (p.kind() == .host or q.kind() == .host) return null;
            if (!value.equal(p, q)) return sortLess(p, q);
        }
        return x.len < y.len;
    }
    if (pyBytesOf(a)) |x| if (pyBytesOf(b)) |y| return std.mem.order(u8, x, y) == .lt;
    return null;
}

/// sorted(v, key=key, reverse=reverse) (in_place: v.sort(...), v a list):
/// the items' keys (key called on each, in order, as Python calls it),
/// a stable sort of them natively; keys it can't compare natively ordered
/// by Python (sorted(range(n), key=keys.__getitem__): Python's comparisons
/// of the keys, its errors). key None: the items themselves.
export fn zr_sorted(ctx: *Ctx, node: u32, t: u64, bits: u64, kt: u64, kb: u64, rt: u64, rb: u64, in_place: u32, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const key = Value{ .tag = kt, .bits = kb };
    const reverse = value.truthy(.{ .tag = rt, .bits = rb });
    // (a Python list: sorted as sorted() does it, the result put back in it,
    // the key called natively)
    if (in_place != 0 and v.kind() == .host and ph.typeOf(@ptrFromInt(v.bits)) == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyList_Type"))))) {
        var r = Value.none_v;
        if (!zr_sorted(ctx, node, t, bits, kt, kb, rt, rb, 0, &r)) return false;
        defer value.decref(r);
        var objs: [1]*PyObject = undefined;
        if (!objects(ctx, node, &.{r}, &objs)) return failPython(ctx, node);
        defer py.Py_DecRef(objs[0]);
        const o: *PyObject = @ptrFromInt(v.bits);
        if (py.c.PyList_SetSlice(o, 0, py.c.PyList_Size(o), objs[0]) != 0) return failPython(ctx, node);
        out.* = Value.none_v;
        return true;
    }
    // (x.sort() of anything else: its own method, Python's)
    if (in_place != 0 and v.kind() != .list) {
        var objs: [3]*PyObject = undefined;
        if (!objects(ctx, node, &.{ v, key, .{ .tag = rt, .bits = rb } }, &objs)) return failPython(ctx, node);
        defer for (objs) |o| py.Py_DecRef(o);
        const meth = py.c.PyObject_GetAttrString(objs[0], "sort") orelse return failPython(ctx, node);
        defer py.Py_DecRef(meth);
        const none_args = py.c.PyTuple_New(0) orelse return failPython(ctx, node);
        defer py.Py_DecRef(none_args);
        const kw = py.c.PyDict_New() orelse return failPython(ctx, node);
        defer py.Py_DecRef(kw);
        if (py.c.PyDict_SetItemString(kw, "key", objs[1]) != 0 or py.c.PyDict_SetItemString(kw, "reverse", objs[2]) != 0) return failPython(ctx, node);
        return fromResult(ctx, node, py.c.PyObject_Call(meth, none_args, kw), out);
    }
    // (the items: the list itself, or a list of v's)
    var list_v = v;
    if (in_place == 0) {
        if (!zr_items(ctx, node, t, bits, &list_v)) return false;
    } else value.incref(v);
    defer value.decref(list_v);
    const l: *value.List = @ptrCast(@alignCast(list_v.ptr()));
    const n = l.len;
    const items = allocator.dupe(Value, l.slice()) catch return oomFail(ctx, node);
    defer allocator.free(items);
    for (items) |x| value.incref(x);
    defer for (items) |x| value.decref(x);
    // The keys
    const keys = allocator.alloc(Value, n) catch return oomFail(ctx, node);
    defer allocator.free(keys);
    var made: usize = 0;
    defer if (key.kind() != .none) for (keys[0..made]) |k| value.decref(k);
    for (items, keys) |x, *k| {
        if (key.kind() == .none) {
            k.* = x;
        } else {
            if (!zr_call(ctx, node, key.tag, key.bits, @ptrCast(&x), 1, null, k)) return false;
            made += 1;
        }
    }
    // The order: natively, or Python's
    const perm = allocator.alloc(usize, n) catch return oomFail(ctx, node);
    defer allocator.free(perm);
    for (perm, 0..) |*p, i| p.* = i;
    const Sorter = struct {
        keys: []const Value,
        reverse: bool,
        failed: bool = false,
        fn less(s: *@This(), i: usize, j: usize) bool {
            const r = if (s.reverse) sortLess(s.keys[j], s.keys[i]) else sortLess(s.keys[i], s.keys[j]);
            return r orelse blk: {
                s.failed = true;
                break :blk false;
            };
        }
    };
    var sorter = Sorter{ .keys = keys, .reverse = reverse };
    std.sort.block(usize, perm, &sorter, Sorter.less);
    if (sorter.failed and !pythonsOrder(ctx, node, keys, reverse, perm)) return false;
    // The result: the items in that order
    if (in_place != 0) {
        for (perm, l.slice()) |p, *slot| {
            value.incref(items[p]);
            value.decref(slot.*);
            slot.* = items[p];
        }
        out.* = Value.none_v;
        return true;
    }
    const r = value.newList(n) orelse return oomFail(ctx, node);
    for (perm) |p| {
        value.incref(items[p]);
        _ = value.listPush(r, items[p]);
    }
    out.* = Value.obj(.list, &r.head);
    return true;
}

/// The order Python's sort puts keys in (stable, its comparisons, its
/// errors): sorted(range(n), key=keys.__getitem__, reverse=reverse), in `perm`
fn pythonsOrder(ctx: *Ctx, node: u32, keys: []const Value, reverse: bool, perm: []usize) bool {
    gil.allowBegin();
    defer gil.allowEnd();
    const pk = py.c.PyList_New(@intCast(keys.len)) orelse return failPython(ctx, node);
    defer py.Py_DecRef(pk);
    for (keys, 0..) |k, i| {
        const o = value.toPython(k, ctx.node_maker) orelse return failPython(ctx, node);
        _ = py.c.PyList_SetItem(pk, @intCast(i), o);
    }
    const builtins = py.c.PyEval_GetBuiltins() orelse return failPython(ctx, node);
    const sorted_f = py.c.PyDict_GetItemString(builtins, "sorted") orelse return failPython(ctx, node);
    const range_f = py.c.PyDict_GetItemString(builtins, "range") orelse return failPython(ctx, node);
    const indices = py.c.PyObject_CallFunction(range_f, "n", @as(isize, @intCast(keys.len))) orelse return failPython(ctx, node);
    defer py.Py_DecRef(indices);
    const getter = py.c.PyObject_GetAttrString(pk, "__getitem__") orelse return failPython(ctx, node);
    defer py.Py_DecRef(getter);
    const args = py.c.PyTuple_Pack(1, indices) orelse return failPython(ctx, node);
    defer py.Py_DecRef(args);
    const kw = py.c.PyDict_New() orelse return failPython(ctx, node);
    defer py.Py_DecRef(kw);
    if (py.c.PyDict_SetItemString(kw, "key", getter) != 0) return failPython(ctx, node);
    if (py.c.PyDict_SetItemString(kw, "reverse", if (reverse) py.Py_True() else py.Py_False()) != 0) return failPython(ctx, node);
    const order = py.c.PyObject_Call(sorted_f, args, kw) orelse return failPython(ctx, node);
    defer py.Py_DecRef(order);
    for (perm, 0..) |*p, i| p.* = @intCast(py.c.PyLong_AsSsize_t(py.c.PyList_GetItem(order, @intCast(i)).?));
    return true;
}

/// The library functions done natively (zr_lib)
pub const Lib = enum(u32) { bytes_of, fromhex, from_bytes, unpack, unpack_from, pack, crc32, adler32, calcsize };

/// A library function of Python's (objects[callee_index]) natively, as
/// Python computes it; what it doesn't take natively (other kinds of
/// arguments, its errors) by the function itself. `signed`: int.from_bytes'
/// keyword (2: not given).
export fn zr_lib(ctx: *Ctx, node: u32, which: u32, callee_index: u64, args: [*]const Value, n: u64, signed: u32, out: *Value) callconv(.c) bool {
    const a = args[0..n];
    if (libNative(ctx, node, @enumFromInt(which), a, signed, out)) |ok| return ok;
    // (Python's: the function called, its keyword given back)
    gil.allowBegin();
    defer gil.allowEnd();
    if (signed == 2) return zr_call_python(ctx, node, callee_index, args, n, out);
    var objs: [8]*PyObject = undefined;
    if (n > objs.len) return fail(ctx, node, "too many arguments", .{});
    if (!objects(ctx, node, a, objs[0..n])) return failPython(ctx, node);
    defer for (objs[0..n]) |o| py.Py_DecRef(o);
    const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
    defer py.Py_DecRef(tuple);
    for (objs[0..n], 0..) |o, i| {
        py.Py_IncRef(o);
        _ = py.c.PyTuple_SetItem(tuple, @intCast(i), o);
    }
    const kw = py.c.PyDict_New() orelse return failPython(ctx, node);
    defer py.Py_DecRef(kw);
    if (py.c.PyDict_SetItemString(kw, "signed", if (signed == 1) py.Py_True() else py.Py_False()) != 0) return failPython(ctx, node);
    return fromResult(ctx, node, py.c.PyObject_Call(ctx.object(callee_index), tuple, kw), out);
}

fn byteOrder(v: Value) ?std.builtin.Endian {
    if (v.kind() != .str) return null;
    const s = @as(*value.Str, @ptrCast(v.ptr())).bytes();
    if (std.mem.eql(u8, s, "little")) return .little;
    if (std.mem.eql(u8, s, "big")) return .big;
    return null;
}

/// Bytes' memory, of a Python bytes or a zrun.Bytes (what iterating them
/// gives: int.from_bytes takes both)
fn anyBytes(v: Value) ?[]const u8 {
    if (v.kind() != .bytes) return null;
    return @as(*value.Bytes, @ptrCast(@alignCast(v.ptr()))).slice();
}

fn libNative(ctx: *Ctx, node: u32, which: Lib, a: []const Value, signed: u32, out: *Value) ?bool {
    switch (which) {
        // bytes(), bytes(n), bytes(items), bytes(b)
        .bytes_of => {
            if (a.len == 0) return bytesResult(ctx, node, value.newOwnedBytes(0), out);
            if (a.len != 1) return null;
            const x = a[0];
            if (pyBytesOf(x) != null) {
                value.incref(x);
                out.* = x;
                return true;
            }
            if (anyBytes(x)) |s| return bytesResult(ctx, node, value.bytesOf(s), out);
            if (isInt(x)) {
                if (x.asInt() < 0) return null;
                const r = value.newOwnedBytes(@intCast(x.asInt())) orelse return oomFail(ctx, node);
                @memset(value.bytesMemory(r), 0);
                out.* = Value.obj(.bytes, &r.head);
                return true;
            }
            if (x.kind() == .list or x.kind() == .tuple) {
                const items = itemsOfSeq(x);
                for (items) |it| if (!isInt(it) or it.asInt() < 0 or it.asInt() > 255) return null;
                const r = value.newOwnedBytes(items.len) orelse return oomFail(ctx, node);
                for (items, value.bytesMemory(r)) |it, *d| d.* = @intCast(it.asInt());
                out.* = Value.obj(.bytes, &r.head);
                return true;
            }
            return null;
        },
        // bytes.fromhex(s): pairs of hex digits, whitespace between them
        .fromhex => {
            if (a.len != 1 or a[0].kind() != .str) return null;
            const s = @as(*value.Str, @ptrCast(a[0].ptr())).bytes();
            var buf: std.ArrayListUnmanaged(u8) = .empty;
            defer buf.deinit(allocator);
            var i: usize = 0;
            while (i < s.len) {
                if (std.ascii.isWhitespace(s[i])) {
                    i += 1;
                    continue;
                }
                if (i + 1 >= s.len) return null;
                const hi = std.fmt.charToDigit(s[i], 16) catch return null;
                const lo = std.fmt.charToDigit(s[i + 1], 16) catch return null;
                buf.append(allocator, hi * 16 + lo) catch return oomFail(ctx, node);
                i += 2;
            }
            return bytesResult(ctx, node, value.bytesOf(buf.items), out);
        },
        // int.from_bytes(b, byteorder, signed=...)
        .from_bytes => {
            if (a.len < 1 or a.len > 2) return null;
            const s = anyBytes(a[0]) orelse return null;
            const order: std.builtin.Endian = if (a.len == 2) byteOrder(a[1]) orelse return null else if (ph.minor >= 11) .big else return null;
            if (s.len > 16) return null;
            var x: u128 = 0;
            for (0..s.len) |k| {
                const byte = if (order == .big) s[k] else s[s.len - 1 - k];
                x = (x << 8) | byte;
            }
            const is_signed = signed == 1;
            var r: i128 = undefined;
            if (is_signed and s.len > 0 and x >> @intCast(s.len * 8 - 1) & 1 != 0) {
                // (two's complement of its width)
                if (s.len == 16) r = @bitCast(x) else r = @as(i128, @intCast(x)) - (@as(i128, 1) << @intCast(s.len * 8));
            } else {
                if (x > std.math.maxInt(i128)) return null;
                r = @intCast(x);
            }
            out.* = value.intValue(r) orelse return oomFail(ctx, node);
            return true;
        },
        .crc32, .adler32 => {
            if (a.len < 1 or a.len > 2) return null;
            const s = anyBytes(a[0]) orelse return null;
            var start: u32 = if (which == .crc32) 0 else 1;
            if (a.len == 2) {
                const w = value.wide(a[1]) orelse return null;
                start = @truncate(@as(u128, @bitCast(w)));
            }
            const r: u32 = if (which == .crc32) blk: {
                var c = std.hash.Crc32{ .crc = ~start };
                c.update(s);
                break :blk c.final();
            } else blk: {
                var lo: u32 = start & 0xffff;
                var hi: u32 = start >> 16;
                for (s) |byte| {
                    lo = (lo + byte) % 65521;
                    hi = (hi + lo) % 65521;
                }
                break :blk (hi << 16) | lo;
            };
            out.* = Value.pint(r);
            return true;
        },
        .unpack, .unpack_from => {
            if (a.len < 2 or a[0].kind() != .str) return null;
            const fmt = @as(*value.Str, @ptrCast(a[0].ptr())).bytes();
            const data = anyBytes(a[1]) orelse return null;
            var at: usize = 0;
            if (which == .unpack_from) {
                if (a.len > 3) return null;
                if (a.len == 3) {
                    if (!isInt(a[2]) or a[2].asInt() < 0) return null;
                    at = @intCast(a[2].asInt());
                }
            } else if (a.len != 2) return null;
            const layout = structLayout(fmt) orelse return null;
            if (at > data.len or data.len - at < layout.size) return null;
            if (which == .unpack and data.len != layout.size) return null;
            return structUnpack(ctx, node, fmt, data[at..], out);
        },
        .pack => {
            if (a.len < 1 or a[0].kind() != .str) return null;
            const fmt = @as(*value.Str, @ptrCast(a[0].ptr())).bytes();
            return structPack(ctx, node, fmt, a[1..], out);
        },
        .calcsize => return null,
    }
}

// ----------------------------------------------------------------------
// struct (Modules/_struct.c's formats)
// ----------------------------------------------------------------------

const StructMode = struct { order: std.builtin.Endian, native: bool };

const native_long: usize = if (@import("builtin").os.tag == .windows) 4 else 8;

/// A format character's size (standard or native) and alignment, null for
/// one not done natively (e, n, N, P, p)
fn structSize(c: u8, native: bool) ?usize {
    return switch (c) {
        'x', 'c', 'b', 'B', '?', 's' => 1,
        'h', 'H' => 2,
        'i', 'I', 'f' => 4,
        'l', 'L' => if (native) native_long else 4,
        'q', 'Q', 'd' => 8,
        else => null,
    };
}

fn structMode(fmt: []const u8) struct { mode: StructMode, rest: []const u8 } {
    if (fmt.len > 0) switch (fmt[0]) {
        '<' => return .{ .mode = .{ .order = .little, .native = false }, .rest = fmt[1..] },
        '>', '!' => return .{ .mode = .{ .order = .big, .native = false }, .rest = fmt[1..] },
        '=' => return .{ .mode = .{ .order = @import("builtin").cpu.arch.endian(), .native = false }, .rest = fmt[1..] },
        '@' => return .{ .mode = .{ .order = @import("builtin").cpu.arch.endian(), .native = true }, .rest = fmt[1..] },
        else => {},
    };
    return .{ .mode = .{ .order = @import("builtin").cpu.arch.endian(), .native = true }, .rest = fmt };
}

const StructItem = struct { code: u8, count: usize, offset: usize };

/// A format's items (each code, its count, where it starts) and size; null
/// for one with codes not done natively, or malformed (Python's error then)
fn structItems(fmt: []const u8, out: ?*std.ArrayListUnmanaged(StructItem)) ?usize {
    const m = structMode(fmt);
    var size: usize = 0;
    var i: usize = 0;
    const r = m.rest;
    while (i < r.len) {
        if (std.ascii.isWhitespace(r[i])) {
            i += 1;
            continue;
        }
        var count: usize = 1;
        if (std.ascii.isDigit(r[i])) {
            const from = i;
            while (i < r.len and std.ascii.isDigit(r[i])) i += 1;
            count = std.fmt.parseInt(usize, r[from..i], 10) catch return null;
            if (i >= r.len) return null;
        }
        const c = r[i];
        i += 1;
        const sz = structSize(c, m.mode.native) orelse return null;
        // (native: each item aligned to its size)
        if (m.mode.native and c != 's' and c != 'x' and sz > 1) size = std.mem.alignForward(usize, size, sz);
        if (out) |o| o.append(allocator, .{ .code = c, .count = count, .offset = size }) catch return null;
        size = std.math.add(usize, size, std.math.mul(usize, sz, count) catch return null) catch return null;
    }
    return size;
}

fn structLayout(fmt: []const u8) ?struct { size: usize } {
    return .{ .size = structItems(fmt, null) orelse return null };
}

fn structUnpack(ctx: *Ctx, node: u32, fmt: []const u8, data: []const u8, out: *Value) ?bool {
    var items: std.ArrayListUnmanaged(StructItem) = .empty;
    defer items.deinit(allocator);
    _ = structItems(fmt, &items) orelse return null;
    const order = structMode(fmt).mode.order;
    var n: usize = 0;
    for (items.items) |it| n += switch (it.code) {
        'x' => 0,
        's' => 1,
        else => it.count,
    };
    const t = value.newTuple(n) orelse return oomFail(ctx, node);
    const slots = t.slice();
    for (slots) |*s| s.* = Value.none_v;
    out.* = Value.obj(.tuple, &t.head);
    var k: usize = 0;
    for (items.items) |it| {
        if (it.code == 'x') continue;
        if (it.code == 's') {
            slots[k] = Value.obj(.bytes, &(value.bytesOf(data[it.offset..][0..it.count]) orelse return oomFail(ctx, node)).head);
            k += 1;
            continue;
        }
        const sz = structSize(it.code, structMode(fmt).mode.native).?;
        for (0..it.count) |j| {
            const p = data[it.offset + j * sz ..][0..sz];
            slots[k] = switch (it.code) {
                'c' => Value.obj(.bytes, &(value.bytesOf(p) orelse return oomFail(ctx, node)).head),
                '?' => Value.boolean(p[0] != 0),
                'b' => Value.pint(@as(i8, @bitCast(p[0]))),
                'B' => Value.pint(p[0]),
                'h' => Value.pint(std.mem.readInt(i16, p[0..2], order)),
                'H' => Value.pint(std.mem.readInt(u16, p[0..2], order)),
                'i' => Value.pint(std.mem.readInt(i32, p[0..4], order)),
                'I' => Value.pint(std.mem.readInt(u32, p[0..4], order)),
                'l', 'q' => if (sz == 4) Value.pint(std.mem.readInt(i32, p[0..4], order)) else Value.pint(std.mem.readInt(i64, p[0..8], order)),
                'L', 'Q' => if (sz == 4) Value.pint(std.mem.readInt(u32, p[0..4], order)) else value.intValue(std.mem.readInt(u64, p[0..8], order)) orelse return oomFail(ctx, node),
                'f' => Value.float(@floatCast(@as(f32, @bitCast(std.mem.readInt(u32, p[0..4], order))))),
                'd' => Value.float(@bitCast(std.mem.readInt(u64, p[0..8], order))),
                else => unreachable,
            };
            k += 1;
        }
    }
    return true;
}

fn structPack(ctx: *Ctx, node: u32, fmt: []const u8, vals: []const Value, out: *Value) ?bool {
    var items: std.ArrayListUnmanaged(StructItem) = .empty;
    defer items.deinit(allocator);
    const size = structItems(fmt, &items) orelse return null;
    const m = structMode(fmt).mode;
    var want: usize = 0;
    for (items.items) |it| want += switch (it.code) {
        'x' => 0,
        's' => 1,
        else => it.count,
    };
    if (want != vals.len) return null;
    // (checked first: anything Python refuses is its error, nothing made)
    var k: usize = 0;
    for (items.items) |it| {
        if (it.code == 'x') continue;
        const n = if (it.code == 's') 1 else it.count;
        for (vals[k..][0..n]) |x| {
            const ok = switch (it.code) {
                's', 'c' => pyBytesOf(x) != null and (it.code == 's' or pyBytesOf(x).?.len == 1),
                '?' => x.kind() == .bool or isInt(x),
                'f', 'd' => x.kind() == .float or isInt(x),
                else => blk: {
                    if (!isInt(x)) break :blk false;
                    const v = x.asInt();
                    const sz = structSize(it.code, m.native).?;
                    const unsigned = std.ascii.isUpper(it.code);
                    if (unsigned) break :blk v >= 0 and (sz == 8 or v < (@as(i64, 1) << @intCast(sz * 8)));
                    break :blk sz == 8 or (v >= -(@as(i64, 1) << @intCast(sz * 8 - 1)) and v < (@as(i64, 1) << @intCast(sz * 8 - 1)));
                },
            };
            if (!ok) return null;
            // (a float too big for 4 bytes: Python's OverflowError)
            if (it.code == 'f') {
                const f: f64 = if (x.kind() == .float) x.asFloat() else @floatFromInt(x.asInt());
                if (!std.math.isInf(f) and !std.math.isNan(f) and @abs(f) > std.math.floatMax(f32)) return null;
            }
        }
        k += n;
    }
    const r = value.newOwnedBytes(size) orelse return oomFail(ctx, node);
    const mem = value.bytesMemory(r);
    @memset(mem, 0);
    k = 0;
    for (items.items) |it| {
        if (it.code == 'x') continue;
        if (it.code == 's') {
            const src = pyBytesOf(vals[k]).?;
            const len = @min(src.len, it.count);
            @memcpy(mem[it.offset..][0..len], src[0..len]);
            k += 1;
            continue;
        }
        const sz = structSize(it.code, m.native).?;
        for (0..it.count) |j| {
            const x = vals[k];
            k += 1;
            const p = mem[it.offset + j * sz ..][0..sz];
            switch (it.code) {
                'c' => p[0] = pyBytesOf(x).?[0],
                '?' => p[0] = @intFromBool(value.truthy(x)),
                'f' => std.mem.writeInt(u32, p[0..4], @bitCast(@as(f32, @floatCast(if (x.kind() == .float) x.asFloat() else @as(f64, @floatFromInt(x.asInt()))))), m.order),
                'd' => std.mem.writeInt(u64, p[0..8], @bitCast(if (x.kind() == .float) x.asFloat() else @as(f64, @floatFromInt(x.asInt()))), m.order),
                else => {
                    const v: u64 = @bitCast(x.asInt());
                    switch (sz) {
                        1 => p[0] = @truncate(v),
                        2 => std.mem.writeInt(u16, p[0..2], @truncate(v), m.order),
                        4 => std.mem.writeInt(u32, p[0..4], @truncate(v), m.order),
                        else => std.mem.writeInt(u64, p[0..8], v, m.order),
                    }
                },
            }
        }
    }
    out.* = Value.obj(.bytes, &r.head);
    return true;
}

/// n.to_bytes(length, byteorder[, signed]) natively (null: Python's way)
fn toBytes(ctx: *Ctx, node: u32, v: Value, args: []const Value, out: *Value) ?bool {
    const x = value.wide(v) orelse return null;
    if (args.len < 2 or args.len > 3) return null;
    if (!isInt(args[0]) or args[0].asInt() < 0 or args[0].asInt() > 16) return null;
    const len: usize = @intCast(args[0].asInt());
    const order = byteOrder(args[1]) orelse return null;
    const signed = args.len == 3 and value.truthy(args[2]);
    // (what doesn't fit: Python's OverflowError)
    if (!signed and x < 0) return null;
    if (len < 16) {
        const bits: u7 = @intCast(len * 8);
        if (signed) {
            if (len == 0) {
                if (x != 0) return null;
            } else if (x < -(@as(i128, 1) << (bits - 1)) or x >= (@as(i128, 1) << (bits - 1))) return null;
        } else if (x >= (@as(i128, 1) << bits)) return null;
    }
    const r = value.newOwnedBytes(len) orelse return oomFail(ctx, node);
    const m = value.bytesMemory(r);
    const u: u128 = @bitCast(x);
    for (0..len) |k| {
        const byte: u8 = @truncate(u >> @intCast(k * 8));
        if (order == .little) m[k] = byte else m[len - 1 - k] = byte;
    }
    out.* = Value.obj(.bytes, &r.head);
    return true;
}

/// A function's result standing for a call it left to its caller
/// (rt.tail_call: Ctx.tail_f...)
pub const TAIL_TAG: u64 = 0xFFFF0002;

/// Call a function value (the program's, or a host one) with arguments
/// (borrowed); the result in `out`. `node`: the node calling (errors, the
/// stack). A function ending with rt.tail_call: the call it left made
/// here, after its frame's gone, and so on (zr_tail_resolve).
/// A semantic's own call `f(args)` of a value: zr_call, a Python object
/// called as Python calls it (Gen.dynCall's `plain`)
export fn zr_call_plain(ctx: *Ctx, node: u32, ft: u64, fb: u64, args: [*]const Value, nargs: u64, receiver: ?*const Value, out: *Value) callconv(.c) bool {
    ctx.plain_call = true;
    return zr_call(ctx, node, ft, fb, args, nargs, receiver, out);
}

pub export fn zr_call(ctx: *Ctx, node: u32, ft: u64, fb: u64, args: [*]const Value, nargs: u64, receiver: ?*const Value, out: *Value) callconv(.c) bool {
    if (!callOnce(ctx, node, ft, fb, args, nargs, receiver, out)) return false;
    if (out.tag != TAIL_TAG) return true;
    return zr_tail_resolve(ctx, node, out);
}

/// rt.tail_call(f, args, receiver) in a function: the call kept for its
/// caller to make (references of its own taken).
export fn zr_tail_set(ctx: *Ctx, ft: u64, fb: u64, at: u64, ab: u64, recv: ?*const Value) callconv(.c) void {
    const f = Value{ .tag = ft, .bits = fb };
    const a = Value{ .tag = at, .bits = ab };
    value.incref(f);
    value.incref(a);
    ctx.tail_f = f;
    ctx.tail_args = a;
    ctx.tail_recv = if (recv) |r| blk: {
        value.incref(r.*);
        break :blk r.*;
    } else null;
}

/// The pending tail call moved to `keep` (3 values: the function, the
/// arguments, the receiver or TAIL_TAG for none) while code that may make
/// calls of its own runs (a finally on the way out).
export fn zr_tail_take(ctx: *Ctx, keep: [*]Value) callconv(.c) void {
    keep[0] = ctx.tail_f;
    keep[1] = ctx.tail_args;
    keep[2] = ctx.tail_recv orelse .{ .tag = TAIL_TAG, .bits = 0 };
    ctx.tail_f = Value.none_v;
    ctx.tail_args = Value.none_v;
    ctx.tail_recv = null;
}

/// The tail call zr_tail_take moved, pending again.
export fn zr_tail_put(ctx: *Ctx, keep: [*]const Value) callconv(.c) void {
    ctx.tail_f = keep[0];
    ctx.tail_args = keep[1];
    ctx.tail_recv = if (keep[2].tag == TAIL_TAG) null else keep[2];
}

/// The pending tail call given up (a handler of the semantics' caught it).
export fn zr_tail_clear(ctx: *Ctx) callconv(.c) void {
    value.decref(ctx.tail_f);
    value.decref(ctx.tail_args);
    if (ctx.tail_recv) |r| value.decref(r);
    ctx.tail_f = Value.none_v;
    ctx.tail_args = Value.none_v;
    ctx.tail_recv = null;
}

/// The call a function left with rt.tail_call (its result in `out` tagged
/// TAIL_TAG), made at the depth of the call it ended (`node`'s), and the
/// one that one leaves, until one returns: its result in `out`.
pub export fn zr_tail_resolve(ctx: *Ctx, node: u32, out: *Value) callconv(.c) bool {
    while (out.tag == TAIL_TAG) {
        const f = ctx.tail_f;
        const a = ctx.tail_args;
        const r = ctx.tail_recv;
        ctx.tail_f = Value.none_v;
        ctx.tail_args = Value.none_v;
        ctx.tail_recv = null;
        defer {
            value.decref(f);
            value.decref(a);
            if (r) |x| value.decref(x);
        }
        const items: []const Value = switch (a.kind()) {
            .list => @as(*value.List, @ptrCast(@alignCast(a.ptr()))).slice(),
            .tuple => @as(*value.Tuple, @ptrCast(@alignCast(a.ptr()))).slice(),
            else => return fail(ctx, node, "rt.tail_call's arguments must be a list or a tuple", .{}),
        };
        if (!callOnce(ctx, node, f.tag, f.bits, items.ptr, items.len, if (r) |*x| x else null, out)) return false;
    }
    return true;
}

fn callOnce(ctx: *Ctx, node: u32, ft: u64, fb: u64, args: [*]const Value, nargs: u64, receiver: ?*const Value, out: *Value) bool {
    const f = Value{ .tag = ft, .bits = fb };
    // (this call's only: what it calls makes its own)
    const plain = ctx.plain_call;
    ctx.plain_call = false;
    switch (f.kind()) {
        .function => {
            const fo: *value.Function = @ptrCast(@alignCast(f.ptr()));
            // (one a table both programs see has: as the reference mode)
            if (fo.program != ctx.program) return fail(ctx, node, "a function of another program can't be called here", .{});
            const policy = FunctionFlags.of(fo.head.flags);
            const nparams: u64 = policy.nparams;
            if ((nargs < nparams and !policy.missing_none) or (nargs > nparams and policy.extra == .@"error")) {
                return fail(ctx, node, "{s}() takes {d} argument{s}, {d} given", .{ fo.name.bytes(), nparams, if (nparams == 1) "" else "s", nargs });
            }
            if (ctx.depth >= ctx.max_depth or @frameAddress() < ctx.stack_low)
                return fail(ctx, node, "call stack too deep (more than {d} calls)", .{ctx.max_depth});
            // (room for the deepest stack made once: a call just stores)
            if (ctx.calls_room == 0) {
                ctx.calls = (allocator.alloc(CallEntry, ctx.max_depth) catch return fail(ctx, node, "out of memory", .{})).ptr;
                ctx.calls_room = ctx.max_depth;
            }
            ctx.calls[ctx.depth] = .{ .name = fo.name, .node = node };
            ctx.depth += 1;
            defer ctx.depth -= 1;
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
        .closure => return callClosure(ctx, node, @ptrCast(@alignCast(f.ptr())), args[0..nargs], out),
        .host => {
            const callee: *PyObject = @ptrFromInt(f.bits);
            // (a Python function: by its compiled code, if it can have one;
            // finding it the first time touches Python, as compiling does)
            if (receiver == null) if (@import("bridge.zig").compiledMethod(ctx, node, callee, args[0..nargs], !plain, out)) |ok| return ok;
            // (else called through Python)
            gil.ensureAt(@src(), node, callee);
            if (collecting) {
                var b: [64]u8 = undefined;
                stat("call host {s}", .{statType(f, &b)});
            }
            const n = nargs + @intFromBool(receiver != null);
            const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
            defer py.Py_DecRef(tuple);
            // (its arguments and result as rt.call hands them over: ints
            // I64s; a semantic's own call: as they are, its error its own)
            var k: usize = 0;
            if (receiver) |r| {
                const o = value.toPython(if (plain) r.* else r.*.checked(), ctx.node_maker) orelse return failPython(ctx, node);
                _ = py.c.PyTuple_SetItem(tuple, 0, o);
                k = 1;
            }
            for (args[0..nargs], 0..) |a, i| {
                const o = value.toPython(if (plain) a else a.checked(), ctx.node_maker) orelse return failPython(ctx, node);
                _ = py.c.PyTuple_SetItem(tuple, @intCast(k + i), o);
            }
            const r = py.c.PyObject_CallObject(callee, tuple) orelse return if (plain) failPython(ctx, node) else hostFailed(ctx, node, callee);
            defer py.Py_DecRef(r);
            const got = value.fromPython(r) orelse return failPython(ctx, node);
            out.* = if (plain) got else got.checked();
            return true;
        },
        else => return fail(ctx, node, "'{s}' value is not callable", .{value.typeName(f)}),
    }
}

/// A call site's host function (rt.call's, no receiver: the compiled code's
/// object `idx`, a Python function of the language's), the first time: its
/// compiled code kept at the site (`slot`), whose code calls it directly
/// from then on (zr_host_jump for what that code says); without one, as
/// zr_call.
pub export fn zr_call_site(ctx: *Ctx, node: u32, slot: *u64, idx: u64, args: [*]const Value, nargs: u64, out: *Value) callconv(.c) bool {
    const callee = ctx.object(idx);
    // (its code compiled now: what strict mode allows; without code, a
    // call through Python, as zr_call makes it)
    gil.allowBegin();
    const host_code = @import("bridge.zig").hostCode(ctx, callee, nargs);
    gil.allowEnd();
    if (host_code) |code| {
        @atomicStore(u64, slot, @intFromPtr(code), .release);
        // (its arguments made I64s, as rt.call hands them over)
        var checked: [64]Value = undefined;
        for (args[0..nargs], checked[0..nargs]) |a, *g| g.* = a.checked();
        const status = code(ctx, null, &checked, node, 0, null, null, out);
        if (status == 1) {
            out.* = out.*.checked();
            return true;
        }
        return if (status == 0) hostCodeFailed(ctx, callee) else zr_host_jump(ctx, node);
    }
    return zr_call(ctx, node, @intFromEnum(Tag.host), @intFromPtr(callee), args, nargs, null, out);
}

/// A host function's compiled code failed (the compiled code's object
/// `idx`): its error worded as the reference mode words a host function's
/// failure (hostFailed: "name: Type: message"). False.
pub export fn zr_host_failed(ctx: *Ctx, node: u32, idx: u64) callconv(.c) bool {
    _ = node;
    return hostCodeFailed(ctx, ctx.object(idx));
}

pub fn hostCodeFailed(ctx: *Ctx, f: *PyObject) bool {
    if (!ctx.failed) return false;
    gil.allowBegin();
    defer gil.allowEnd();
    var name_buf: [128]u8 = undefined;
    const name = pyName(f, &name_buf, "host function");
    // (the exception's type: one Python code raised, or the native error's
    // class)
    var type_buf: [128]u8 = undefined;
    const type_name = if (ctx.pending orelse ctx.exc) |e|
        pyName(@ptrCast(@alignCast(ph.typeOf(e))), &type_buf, "Error")
    else if (ctx.exc_kind) |k| k.name() else "Error";
    // (the message as Python's str() of the exception says it: a Python
    // exception's own, a native error's Python wording (exc_msg), else the
    // error's)
    var text_obj: ?*PyObject = null;
    defer if (text_obj) |o| py.Py_DecRef(o);
    if (ctx.pending orelse ctx.exc) |e| {
        text_obj = py.c.PyObject_Str(e);
        if (text_obj == null) py.c.PyErr_Clear();
    }
    const msg: []const u8 = if (text_obj) |o| (ph.utf8(o, "message") orelse "") else if (ctx.exc_kind != null) ctx.exc_msg.items else ctx.err_msg.items;
    const old = allocator.dupe(u8, msg) catch return false;
    defer allocator.free(old);
    ctx.err_msg.clearRetainingCapacity();
    ctx.err_msg.print(allocator, "{s}: {s}: {s}", .{ name, type_name, old }) catch {};
    // (a native error: a zrun.Error now, as rt.call raises a host
    // function's exception, what `except` matches; an rt.Throw goes up as
    // itself)
    if (ctx.exc_kind != null and ctx.exc_kind != .Throw and ctx.pending == null and ctx.exc == null) {
        ctx.exc_kind = .ZrunError;
        ctx.exc_msg.clearRetainingCapacity();
        ctx.exc_msg.appendSlice(allocator, ctx.err_msg.items) catch {};
        if (ctx.exc_value) |v| value.decref(v);
        ctx.exc_value = null;
    }
    return false;
}

/// An object's __name__ (copied into `buf`), or `default`.
fn pyName(o: *PyObject, buf: []u8, default: []const u8) []const u8 {
    const n = ph.attr(o, "__name__") orelse {
        py.c.PyErr_Clear();
        return default;
    };
    defer py.Py_DecRef(n);
    const s = ph.utf8(n, "name") orelse {
        py.c.PyErr_Clear();
        return default;
    };
    const k = @min(s.len, buf.len);
    @memcpy(buf[0..k], s[0..k]);
    return buf[0..k];
}

/// A host function called with rt.call let rt.Return, Break or Continue
/// out: an error, as the reference mode raises them out of a call.
pub export fn zr_host_jump(ctx: *Ctx, node: u32) callconv(.c) bool {
    return fail(ctx, node, "rt.Return, rt.Break or rt.Continue raised out of a function called with rt.call", .{});
}

/// "name: Error: message", as the reference mode words a host function's
/// failure.
fn hostFailed(ctx: *Ctx, node: u32, f: *PyObject) bool {
    gil.allowBegin();
    defer gil.allowEnd();
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
    // (a count, not Python running: strict mode looks at what's done with
    // it; compiling refused the objects it shouldn't hold)
    gil.allowBegin();
    defer gil.allowEnd();
    const o = ctx.object(idx);
    py.Py_IncRef(o);
    out.* = .{ .tag = @intFromEnum(Tag.host), .bits = @intFromPtr(o) };
}

/// A heap frame for a run of a function (its slots unset).
/// The frame of scope `home` seen from `frame` (of scope `owner`): up the
/// frames' parents as up the scopes' owners (`owners`: by node).
export fn zr_frame_of(frame: *value.Frame, owner: u32, home: u32, owners: [*]const u64) callconv(.c) *value.Frame {
    var f = frame;
    var at = owner;
    while (at != home and at != std.math.maxInt(u32)) : (at = @intCast(owners[at])) f = f.parent orelse break;
    return f;
}

pub export fn zr_frame_new(parent: ?*value.Frame, n: u64) callconv(.c) ?*value.Frame {
    return value.newFrame(parent, n, .{ .tag = UNSET, .bits = 0 });
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
    // (room for n made: the items written in place)
    for (0..n) |i| {
        l.items.?[i] = given(&items[i]);
        value.listStored(l, l.items.?[i]);
    }
    l.len = n;
    out.* = Value.obj(.list, &l.head);
    return true;
}

/// A value compiled code just stored (its tag and bits, 8 bytes each), read
/// as it was stored: a 16-byte read of them waits for the stores to
/// reach the cache.
fn given(p: *const Value) Value {
    const words: *const volatile [2]u64 = @ptrCast(p);
    return .{ .tag = words[0], .bits = words[1] };
}

/// A new tuple of n items (taking the references).
export fn zr_tuple(ctx: *Ctx, node: u32, items: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const t = value.newTuple(n) orelse return oomFail(ctx, node);
    for (t.slice(), 0..) |*slot, i| slot.* = given(&items[i]);
    out.* = Value.obj(.tuple, &t.head);
    return true;
}

/// A new dict from n keys and values (borrowing them), in order.
export fn zr_dict(ctx: *Ctx, node: u32, keys: [*]const Value, vals: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const d = value.newDict() orelse return oomFail(ctx, node);
    for (0..n) |i| {
        if (!value.hashable(keys[i])) {
            value.decref(Value.obj(.dict, &d.head));
            return unhashableKey(ctx, node, value.typeName(keys[i]));
        }
        if (!value.dictSet(d, keys[i], vals[i])) return oomFail(ctx, node);
    }
    out.* = Value.obj(.dict, &d.head);
    return true;
}

/// A new record of a type with its fields (taking the references).
export fn zr_record(ctx: *Ctx, node: u32, rtype: *const value.RecordType, fields: [*]const Value, out: *Value) callconv(.c) bool {
    const r = value.newRecord(rtype) orelse return oomFail(ctx, node);
    for (r.fields(), 0..) |*slot, i| slot.* = given(&fields[i]);
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
        gil.ensure(@src());
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
    gil.ensureAt(@src(), node, ctx.object(cls_index));
    const v = Value{ .tag = t, .bits = bits };
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node,&.{v}, &objs)) return failPython(ctx, node);
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
    if (v.kind() == .node) if (@import("bridge.zig").nodeAttr(ctx, @intCast(v.bits), name, out)) |ok| return ok;
    if (v.kind() == .exc and excAttr(v, name.bytes(), out)) return true;
    if (v.kind() == .record) {
        const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
        for (r.rtype.fields, 0..) |f, i| {
            if (std.mem.eql(u8, f, name.bytes())) {
                const x = r.fields()[i];
                // (a slot never assigned: as Python says it)
                if (x.tag == UNSET) return failAs(ctx, node, .AttributeError, null, "'{s}' object has no attribute '{s}'", .{ r.rtype.unset_name, f });
                value.incref(x);
                out.* = x;
                return true;
            }
        }
    }
    // Anything else (and the errors): Python's getattr
    if (collecting) {
        var b: [64]u8 = undefined;
        stat("getattr {s}.{s}", .{ statType(v, &b), name.bytes() });
    }
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node,&.{v}, &objs)) return failPython(ctx, node);
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
                if (r.rtype.frozen) return failAs(ctx, node, .FrozenInstanceError, null, "cannot assign to field '{s}'", .{f});
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
    // (from 3.13 Python says why it can't be added)
    if (ph.minor >= 13)
        return failAs(ctx, node, .AttributeError, null, "'{s}' object has no attribute '{s}' and no __dict__ for setting new attributes", .{ value.typeName(v), name.bytes() });
    return failAs(ctx, node, .AttributeError, null, "'{s}' object has no attribute '{s}'", .{ value.typeName(v), name.bytes() });
}

/// Normalize an index (negative from the end); null if out of range.
fn index(i: i64, len: u64) ?usize {
    const n: i64 = @intCast(len);
    const k = if (i < 0) i + n else i;
    if (k < 0 or k >= n) return null;
    return @intCast(k);
}

/// del v[k]: a dict's key, a list's item at an int, natively; one missing,
/// Python's error (its words: KeyError(k), IndexError); anything else as
/// Python deletes it
export fn zr_delitem(ctx: *Ctx, node: u32, t: u64, bits: u64, kt: u64, kb: u64) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const k = Value{ .tag = kt, .bits = kb };
    switch (v.kind()) {
        .dict => if (value.hashable(k)) {
            if (value.dictDelete(@ptrCast(@alignCast(v.ptr())), k)) return true;
        },
        .list => if (isInt(k)) {
            const l: *value.List = @ptrCast(@alignCast(v.ptr()));
            if (index(k.asInt(), l.len)) |i| {
                const items = l.items.?;
                const gone = items[i];
                std.mem.copyForwards(Value, items[i .. l.len - 1], items[i + 1 .. l.len]);
                l.len -= 1;
                value.decref(gone);
                return true;
            }
        },
        else => {},
    }
    // (missing, or not one of these: as Python does it, its error its own)
    if (v.kind() == .dict or v.kind() == .list) gil.allowBegin() else gil.ensureAt(@src(), node, null);
    defer if (v.kind() == .dict or v.kind() == .list) gil.allowEnd();
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, node, &.{ v, k }, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    defer py.Py_DecRef(objs[1]);
    if (py.c.PyObject_DelItem(objs[0], objs[1]) != 0) return failPython(ctx, node);
    return true;
}

/// v[k]
export fn zr_getitem(ctx: *Ctx, node: u32, t: u64, bits: u64, kt: u64, kb: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const k = Value{ .tag = kt, .bits = kb };
    switch (v.kind()) {
        .list, .tuple => if (isInt(k)) {
            const items = if (v.kind() == .list) @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
            const i = index(k.asInt(), items.len) orelse return failAs(ctx, node, .IndexError, null, "{s} index out of range", .{@tagName(v.kind())});
            value.incref(items[i]);
            out.* = items[i];
            return true;
        },
        // (bytes at an int: the byte, an int)
        .bytes => if (isInt(k)) {
            const b: *value.Bytes = @ptrCast(@alignCast(v.ptr()));
            const i = index(k.asInt(), b.len) orelse return failAs(ctx, node, .IndexError, null, "index out of range", .{});
            out.* = Value.pint(b.ptr[i]);
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
        // A str at an int: its character (one not ASCII found by its index:
        // value.charOffset)
        .str => if (isInt(k)) {
            const s: *value.Str = @ptrCast(v.ptr());
            const i = index(k.asInt(), s.chars) orelse return failAs(ctx, node, .IndexError, null, "string index out of range", .{});
            const b = s.bytes();
            const at = value.charOffset(s, i);
            const n = std.unicode.utf8ByteSequenceLength(b[at]) catch 1;
            const ch = charStr(b[at .. at + n]) orelse return oomFail(ctx, node);
            out.* = Value.obj(.str, &ch.head);
            return true;
        },
        // A Python list or tuple (data given by Python) at an int in it:
        // its item read and converted (no int made for the index, no call
        // of Python's); out of range, Python's error
        .host => if (isInt(k)) {
            const o: *PyObject = @ptrFromInt(v.bits);
            gil.ensureAt(@src(), node, null);
            const t_ = ph.typeOf(o);
            const is_list = t_ == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyList_Type"))));
            const is_tuple = !is_list and t_ == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyTuple_Type"))));
            if (is_list or is_tuple) {
                const n: usize = @intCast(if (is_list) py.c.PyList_Size(o) else py.c.PyTuple_Size(o));
                if (index(k.asInt(), n)) |i| {
                    const item = (if (is_list) py.c.PyList_GetItem(o, @intCast(i)) else py.c.PyTuple_GetItem(o, @intCast(i))) orelse return failPython(ctx, node);
                    // (an int of 64 bits, the common item: read here)
                    if (ph.typeOf(item) == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyLong_Type"))))) {
                        var overflow: c_int = 0;
                        const x = py.c.PyLong_AsLongLongAndOverflow(item, &overflow);
                        if (overflow == 0) {
                            out.* = Value.pint(x);
                            return true;
                        }
                    }
                    out.* = value.fromBorrowed(item) orelse return failPython(ctx, node);
                    return true;
                }
            }
        },
        else => {},
    }
    // Strings, slices of everything, the errors: Python's
    if (collecting) {
        var b1: [64]u8 = undefined;
        var b2: [64]u8 = undefined;
        stat("getitem {s}[{s}]", .{ statType(v, &b1), statType(k, &b2) });
    }
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, node,&.{ v, k }, &objs)) return failPython(ctx, node);
    defer for (objs) |o| py.Py_DecRef(o);
    return fromResult(ctx, node, py.c.PyObject_GetItem(objs[0], objs[1]), out);
}

/// rt.wrapping_add(a, b) and the others (wrapping.zig) of anything but two
/// ints of 64 bits (compiled code does those): as the reference mode does
/// it, its error the same.
export fn zr_wrapping(ctx: *Ctx, node: u32, op: u32, at: u64, ab: u64, bt: u64, bb: u64, out: *Value) callconv(.c) bool {
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, node,&.{ .{ .tag = at, .bits = ab }, .{ .tag = bt, .bits = bb } }, &objs)) return failPython(ctx, node);
    defer for (objs) |o| py.Py_DecRef(o);
    return fromResult(ctx, node, @import("wrapping.zig").ofPython(@enumFromInt(op), objs[0], objs[1]), out);
}

/// rt.u8(data, i) and the others (bytes.zig) where compiled code's own
/// read doesn't: as the reference mode does it (a big int, the errors).
export fn zr_read(ctx: *Ctx, node: u32, r: u32, dt: u64, db: u64, at: u64, ab: u64, out: *Value) callconv(.c) bool {
    const read: @import("bytes.zig").Read = @enumFromInt(r);
    var objs: [2]*PyObject = undefined;
    if (!objects(ctx, node,&.{ .{ .tag = dt, .bits = db }, .{ .tag = at, .bits = ab } }, &objs)) return failPython(ctx, node);
    defer for (objs) |o| py.Py_DecRef(o);
    return fromResult(ctx, node, @import("bytes.zig").read(read, objs[0], objs[1]), out);
}

/// A native host function's failure (native.zig, code its call returned):
/// the error a call through Python gives, RuntimeError(its message).
export fn zr_native_fail(ctx: *Ctx, node: u32, idx: u64, code: i32) callconv(.c) bool {
    gil.allowBegin();
    defer gil.allowEnd();
    const o = ctx.objects.items[idx];
    const n = @import("native.zig").as(o);
    ph.raise(py.PyExc_RuntimeError(), "{s}", .{@import("native.zig").errorText(n, code)});
    return hostFailed(ctx, node, o);
}

/// v[k] = x
export fn zr_setitem(ctx: *Ctx, node: u32, t: u64, bits: u64, kt: u64, kb: u64, xt: u64, xb: u64) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const k = Value{ .tag = kt, .bits = kb };
    const x = Value{ .tag = xt, .bits = xb };
    switch (v.kind()) {
        .list => if (isInt(k)) {
            const l: *value.List = @ptrCast(@alignCast(v.ptr()));
            const i = index(k.asInt(), l.len) orelse return failAs(ctx, node, .IndexError, null, "list assignment index out of range", .{});
            value.incref(x);
            value.decref(l.items.?[i]);
            l.items.?[i] = x;
            value.listStored(l, x);
            return true;
        } else return failAs(ctx, node, .TypeError, null, "list indices must be integers or slices, not {s}", .{value.typeName(k)}),
        .dict => {
            const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
            if (!value.hashable(k)) return unhashableKey(ctx, node, value.typeName(k));
            if (!value.dictSet(d, k, x)) return oomFail(ctx, node);
            return true;
        },
        // (a Python object: itself changed, as Python does it)
        .host => {
            var objs: [3]*PyObject = undefined;
            if (!objects(ctx, node,&.{ v, k, x }, &objs)) return failPython(ctx, node);
            defer for (objs) |o| py.Py_DecRef(o);
            if (py.c.PyObject_SetItem(objs[0], objs[1], objs[2]) != 0) return failPython(ctx, node);
            return true;
        },
        else => return failAs(ctx, node, .TypeError, null, "'{s}' object does not support item assignment", .{value.typeName(v)}),
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
        // (a str: its characters)
        .str => {
            const s: *value.Str = @ptrCast(v.ptr());
            const src = s.bytes();
            const l = value.newList(s.chars) orelse return oomFail(ctx, node);
            var at: usize = 0;
            while (at < src.len) {
                const n = std.unicode.utf8ByteSequenceLength(src[at]) catch 1;
                const ch = charStr(src[at .. at + n]) orelse {
                    value.decref(Value.obj(.list, &l.head));
                    return oomFail(ctx, node);
                };
                _ = value.listPush(l, Value.obj(.str, &ch.head));
                at += n;
            }
            out.* = Value.obj(.list, &l.head);
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
        // (a set: its items in its table's order, Python's)
        .set => {
            const s = setOf(v);
            const l = value.newList(s.used) orelse return oomFail(ctx, node);
            var it = set_mod.iterate(s);
            while (it.next()) |x| {
                value.incref(x);
                _ = value.listPush(l, x);
            }
            out.* = Value.obj(.list, &l.head);
            return true;
        },
        else => {},
    }
    if (collecting) {
        var b: [64]u8 = undefined;
        stat("items {s}", .{statType(v, &b)});
    }
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node,&.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const seq = py.c.PySequence_List(objs[0]);
    return fromResult(ctx, node, seq, out);
}

/// What `for ... in v.items()` (which 0), `v.keys()` (1), `v.values()` (2)
/// goes over: a native dict's, natively (a list of them, made as the loop
/// starts, in the dict's order); anything else's view (Python's), as a
/// list.
export fn zr_dict_view(ctx: *Ctx, node: u32, which: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const names = [3][:0]const u8{ "items", "keys", "values" };
    if (v.kind() == .dict) {
        const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
        const l = value.newList(d.len) orelse return oomFail(ctx, node);
        for (value.dictEntries(d)) |e| {
            if (value.isDeleted(e)) continue;
            const item = switch (which) {
                0 => blk: {
                    const pair = value.newTuple(2) orelse {
                        value.decref(Value.obj(.list, &l.head));
                        return oomFail(ctx, node);
                    };
                    value.incref(e.key);
                    value.incref(e.value);
                    pair.slice()[0] = e.key;
                    pair.slice()[1] = e.value;
                    break :blk Value.obj(.tuple, &pair.head);
                },
                1 => blk: {
                    value.incref(e.key);
                    break :blk e.key;
                },
                else => blk: {
                    value.incref(e.value);
                    break :blk e.value;
                },
            };
            if (!value.listPush(l, item)) return oomFail(ctx, node);
        }
        out.* = Value.obj(.list, &l.head);
        return true;
    }
    // (anything else: its method's result, Python's, as a list)
    if (gil.strictOn()) {
        var nb: [96]u8 = undefined;
        gil.ensureNamed(@src(), node, std.fmt.bufPrint(&nb, "{s}.{s}", .{ value.typeName(v), names[which] }) catch names[which]);
    } else gil.ensure(@src());
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node,&.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const view = py.c.PyObject_CallMethod(objs[0], names[which].ptr, null) orelse return failPython(ctx, node);
    defer py.Py_DecRef(view);
    return fromResult(ctx, node, py.c.PySequence_List(view), out);
}

/// The one-character strs of ASCII, made once (immortal)
var ascii_chars: [128]?*value.Str = .{null} ** 128;

/// A one-character str (a new reference; an ASCII one shared).
/// The values shared by every run, made once (before compiled code runs on
/// several threads): the empty tuple, the one-character ASCII strs.
pub fn init() !void {
    const t = value.newTuple(0) orelse return error.OutOfMemory;
    t.head.rc = value.IMMORTAL;
    empty_tuple = t;
    for (0..128) |c| {
        const s = value.newStr(&.{@intCast(c)}) orelse return error.OutOfMemory;
        s.head.rc = value.IMMORTAL;
        ascii_chars[c] = s;
    }
}

fn charStr(bytes: []const u8) ?*value.Str {
    if (bytes.len == 1 and bytes[0] < 128) {
        if (ascii_chars[bytes[0]]) |s| return s;
        const s = value.newStr(bytes) orelse return null;
        s.head.rc = value.IMMORTAL;
        ascii_chars[bytes[0]] = s;
        return s;
    }
    return value.newStr(bytes);
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
            // (Python's words for a list or tuple: from 3.14 with its length)
            if (items.len < n)
                return failAs(ctx, node, .ValueError, null, "not enough values to unpack (expected {d}, got {d})", .{ n, items.len });
            if (ph.minor >= 14)
                return failAs(ctx, node, .ValueError, null, "too many values to unpack (expected {d}, got {d})", .{ n, items.len });
            return failAs(ctx, node, .ValueError, null, "too many values to unpack (expected {d})", .{n});
        },
        else => {},
    }
    // (anything else: as Python unpacks it)
    gil.ensureAt(@src(), node, null);
    const f = unpacker(n) orelse return failPython(ctx, node);
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node,&.{v}, &objs)) return failPython(ctx, node);
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
        if (!objects(ctx, node,&.{ v, x }, &objs)) return failPython(ctx, node);
        defer for (objs) |o| py.Py_DecRef(o);
        const r = py.c.PyObject_CallMethod(objs[0], "append", "(O)", objs[1]) orelse return failPython(ctx, node);
        py.Py_DecRef(r);
        return true;
    }
    value.incref(x);
    if (!value.listPush(@ptrCast(@alignCast(v.ptr())), x)) return oomFail(ctx, node);
    return true;
}

/// v.extend(items) for items the code knows one by one (borrowed): a list's
/// natively, anything else's by its extend() with a list of them.
export fn zr_extend_items(ctx: *Ctx, node: u32, t: u64, bits: u64, items: [*]const Value, n: u64) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() == .list) {
        const l: *value.List = @ptrCast(@alignCast(v.ptr()));
        for (0..n) |i| {
            const x = given(&items[i]);
            value.incref(x);
            if (!value.listPush(l, x)) return oomFail(ctx, node);
        }
        return true;
    }
    const l = value.newList(n) orelse return oomFail(ctx, node);
    for (0..n) |i| {
        const x = given(&items[i]);
        value.incref(x);
        _ = value.listPush(l, x);
    }
    const lv = Value.obj(.list, &l.head);
    defer value.decref(lv);
    var r = Value.none_v;
    const ext = value.literal("extend") orelse return oomFail(ctx, node);
    if (!zr_call_method(ctx, node, t, bits, ext, @ptrCast(&lv), 1, &r)) return false;
    value.decref(r);
    return true;
}

/// A method of a value called with arguments, as Python does it (for
/// methods that don't change the value: str's, a dict's get...).
export fn zr_call_method(ctx: *Ctx, node: u32, t: u64, bits: u64, name: *const value.Str, args: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    // A set's methods: natively (Python would change a copy)
    if (v.kind() == .set) if (setMethod(ctx, node, v, name.bytes(), args[0..n], out)) |ok| return ok;
    if (v.kind() == .bytes) if (bytesMethod(ctx, node, v, name.bytes(), args[0..n], out)) |ok| return ok;
    // An int's to_bytes(length, byteorder[, signed]): natively; else Python's
    // (signed given back as the keyword it was)
    if (value.wide(v) != null and std.mem.eql(u8, name.bytes(), "to_bytes")) {
        if (toBytes(ctx, node, v, args[0..n], out)) |ok| return ok;
        if (n == 3) {
            gil.allowBegin();
            defer gil.allowEnd();
            var objs: [3]*PyObject = undefined;
            if (!objects(ctx, node, &.{ v, args[0], args[1] }, &objs)) return failPython(ctx, node);
            defer for (objs) |o| py.Py_DecRef(o);
            const meth = py.c.PyObject_GetAttrString(objs[0], "to_bytes") orelse return failPython(ctx, node);
            defer py.Py_DecRef(meth);
            const tuple = py.c.PyTuple_Pack(2, objs[1], objs[2]) orelse return failPython(ctx, node);
            defer py.Py_DecRef(tuple);
            const kw = py.c.PyDict_New() orelse return failPython(ctx, node);
            defer py.Py_DecRef(kw);
            if (py.c.PyDict_SetItemString(kw, "signed", if (value.truthy(args[2])) py.Py_True() else py.Py_False()) != 0) return failPython(ctx, node);
            return fromResult(ctx, node, py.c.PyObject_Call(meth, tuple, kw), out);
        }
    }
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
    // A record's field holding a function (`b.fn(args)`): the function
    // called, as Python finds the attribute (no self)
    if (v.kind() == .record) {
        const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
        for (r.rtype.fields, r.fields()) |f, x| {
            if (!std.mem.eql(u8, f, name.bytes())) continue;
            if (x.tag == value.UNSET_TAG) break;
            return zr_call(ctx, node, x.tag, x.bits, args, n, null, out);
        }
        // (a method of its class: its compiled code, the record self)
        if (@import("bridge.zig").recordMethod(ctx, node, v, name, args[0..n], out)) |ok| return ok;
    }
    // A list's or tuple's index(x), count(x): natively, by ==; x missing
    // from index(): Python's ValueError, its words (an error: what strict
    // mode allows)
    if ((v.kind() == .list or v.kind() == .tuple) and n == 1 and args[0].kind() != .host) {
        const is_index = std.mem.eql(u8, name.bytes(), "index");
        if (is_index or std.mem.eql(u8, name.bytes(), "count")) {
            var count: i64 = 0;
            for (itemsOfSeq(v), 0..) |x, i| {
                if (x.kind() == .host) break;
                if (!value.equal(x, args[0])) continue;
                if (is_index) {
                    out.* = Value.pint(@intCast(i));
                    return true;
                }
                count += 1;
            } else {
                if (!is_index) {
                    out.* = Value.pint(count);
                    return true;
                }
                gil.allowBegin();
                defer gil.allowEnd();
                return callMethodPython(ctx, node, v, name, args[0..n], out);
            }
        }
    }
    // A float's is_integer(), hex()
    if (v.kind() == .float and n == 0 and std.mem.eql(u8, name.bytes(), "is_integer")) {
        const x = v.asFloat();
        out.* = Value.boolean(std.math.isFinite(x) and @floor(x) == x);
        return true;
    }
    if (v.kind() == .float and n == 0 and std.mem.eql(u8, name.bytes(), "hex")) {
        var buf: [32]u8 = undefined;
        const r = value.newStr(floatHex(v.asFloat(), &buf)) orelse return oomFail(ctx, node);
        out.* = Value.obj(.str, &r.head);
        return true;
    }
    // rt.load(), rt.store() of an rt handed over: natively
    if (v.kind() == .rt) {
        if (@import("bridge.zig").rtValueMethod(ctx, node, v, name.bytes(), args[0..n], out)) |ok| return ok;
    }
    // A str's common methods, natively (an ASCII one: Unicode's case
    // rules are Python's)
    if (v.kind() == .str) {
        if (strMethod(ctx, node, @ptrCast(v.ptr()), name.bytes(), args[0..n], out)) |ok| return ok;
    }
    // A dict's pop(): natively
    if (v.kind() == .dict and std.mem.eql(u8, name.bytes(), "pop") and (n == 1 or n == 2) and value.hashable(args[0])) {
        const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
        if (value.dictGet(d, args[0])) |x| {
            value.incref(x);
            _ = value.dictDelete(d, args[0]);
            out.* = x;
            return true;
        }
        if (n == 2) {
            value.incref(args[1]);
            out.* = args[1];
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
            if (l.len == 0) return failAs(ctx, node, .IndexError, null, "pop from empty list", .{});
            const at = if (n == 0) l.len - 1 else index(args[0].asInt(), l.len) orelse return failAs(ctx, node, .IndexError, null, "pop index out of range", .{});
            const items = l.items.?;
            out.* = items[at];
            std.mem.copyForwards(Value, items[at .. l.len - 1], items[at + 1 .. l.len]);
            l.len -= 1;
            return true;
        }
    }
    // Anything else: the method as Python finds it (strict mode's message:
    // the value's kind and the method, `str.encode`)
    if (gil.strictOn()) {
        var nb: [96]u8 = undefined;
        const what = std.fmt.bufPrint(&nb, "{s}.{s}", .{ value.typeName(v), name.bytes() }) catch name.bytes();
        gil.ensureNamed(@src(), node, what);
    } else gil.ensure(@src());
    return callMethodPython(ctx, node, v, name, args[0..n], out);
}

/// v.name(args) as Python does it (its compiled code if it's a Python
/// function's), the GIL taken.
fn callMethodPython(ctx: *Ctx, node: u32, v: Value, name: *const value.Str, args: []const Value, out: *Value) bool {
    const n = args.len;
    if (collecting) {
        var b: [64]u8 = undefined;
        stat("method {s}.{s}", .{ statType(v, &b), name.bytes() });
    }
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node, &.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const key = ph.newString(name.bytes()) orelse return failPython(ctx, node);
    defer py.Py_DecRef(key);
    const method = py.c.PyObject_GetAttr(objs[0], key) orelse return failPython(ctx, node);
    defer py.Py_DecRef(method);
    // (a Python function, or one bound to its object: its compiled code)
    if (@import("bridge.zig").compiledMethod(ctx, node, method, args[0..n], false, out)) |ok| return ok;
    const tuple = py.c.PyTuple_New(@intCast(n)) orelse return failPython(ctx, node);
    defer py.Py_DecRef(tuple);
    for (args[0..n], 0..) |a, i| {
        const o = value.toPython(a, ctx.node_maker) orelse return failPython(ctx, node);
        _ = py.c.PyTuple_SetItem(tuple, @intCast(i), o);
    }
    return fromResult(ctx, node, py.c.PyObject_CallObject(method, tuple), out);
}

/// A str method natively (true / false: an error), or null for one (or
/// arguments, or a str) not done here.
fn strMethod(ctx: *Ctx, node: u32, s: *value.Str, name: []const u8, args: []const Value, out: *Value) ?bool {
    // encode() (UTF-8; ASCII and Latin-1 of strs they hold): a bytes
    if (std.mem.eql(u8, name, "encode") and args.len <= 1) {
        const enc = if (args.len == 0) "utf-8" else blk: {
            if (args[0].kind() != .str) return null;
            break :blk @as(*value.Str, @ptrCast(args[0].ptr())).bytes();
        };
        const lower = std.ascii.eqlIgnoreCase;
        const b = s.bytes();
        if (lower(enc, "utf-8") or lower(enc, "utf8")) return bytesResult(ctx, node, value.bytesOf(b), out);
        if (lower(enc, "ascii")) {
            if (s.chars != s.len) return null;
            return bytesResult(ctx, node, value.bytesOf(b), out);
        }
        if (lower(enc, "latin-1") or lower(enc, "latin1") or lower(enc, "iso-8859-1")) {
            const r = value.newOwnedBytes(s.chars) orelse return oomFail(ctx, node);
            const m = value.bytesMemory(r);
            var it = std.unicode.Utf8View.initUnchecked(b).iterator();
            var k: usize = 0;
            while (it.nextCodepoint()) |cp| : (k += 1) {
                if (cp > 0xff) {
                    value.decref(Value.obj(.bytes, &r.head));
                    return null;
                }
                m[k] = @intCast(cp);
            }
            out.* = Value.obj(.bytes, &r.head);
            return true;
        }
        return null;
    }
    if (strMethodAny(ctx, node, s, name, args, out)) |ok| return ok;
    return strMethodAscii(ctx, node, s, name, args, out);
}

/// Python's whitespace (str.isspace(), str.split(), str.strip()): these
/// code points exactly
fn isSpaceCp(cp: u21) bool {
    return switch (cp) {
        '\t', '\n', 0x0b, 0x0c, '\r', 0x1c, 0x1d, 0x1e, 0x1f, ' ', 0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        0x2000...0x200a => true,
        else => false,
    };
}

/// The code point at byte `at` of UTF-8 `b`, and its length
fn cpAt(b: []const u8, at: usize) struct { cp: u21, n: usize } {
    const n = std.unicode.utf8ByteSequenceLength(b[at]) catch 1;
    const cp = std.unicode.utf8Decode(b[at..@min(b.len, at + n)]) catch b[at];
    return .{ .cp = cp, .n = n };
}

/// The code point before byte `end` of UTF-8 `b`, and its length
fn cpBefore(b: []const u8, end: usize) struct { cp: u21, n: usize } {
    var start = end - 1;
    while (start > 0 and b[start] & 0xC0 == 0x80) start -= 1;
    return .{ .cp = std.unicode.utf8Decode(b[start..end]) catch b[end - 1], .n = end - start };
}

/// A str method of any str (UTF-8: substrings found by their bytes,
/// indexes in code points; whitespace Python's), as Python does it; null
/// for one (or arguments) not done here.
fn strMethodAny(ctx: *Ctx, node: u32, s: *value.Str, name: []const u8, args: []const Value, out: *Value) ?bool {
    const eq = std.mem.eql;
    const b = s.bytes();
    const strOf = struct {
        fn f(x: Value) ?[]const u8 {
            if (x.kind() != .str) return null;
            return @as(*value.Str, @ptrCast(x.ptr())).bytes();
        }
    }.f;
    const newStrValue = struct {
        fn f(c: *Ctx, at: u32, o: *Value, bytes: []const u8) bool {
            const r = value.newStr(bytes) orelse return oomFail(c, at);
            o.* = Value.obj(.str, &r.head);
            return true;
        }
    }.f;
    // (a char index of `s` given as start or end, as a slice's bound: its
    // byte offset; null for one not an int or None)
    const Bound = struct {
        fn of(str: *value.Str, x: ?Value, default: usize) ??usize {
            const v = x orelse return default;
            if (v.kind() == .none) return default;
            if (!isInt(v)) return @as(?usize, null);
            const n: i64 = @intCast(str.chars);
            var i = v.asInt();
            if (i < 0) i = @max(i + n, 0);
            if (i > n) return @as(?usize, str.len + 1);
            return value.charOffset(str, @intCast(i));
        }
    };
    const chars = struct {
        fn of(bytes: []const u8) i64 {
            return @intCast(std.unicode.utf8CountCodepoints(bytes) catch bytes.len);
        }
    }.of;

    // find, rfind, index, rindex, count (sub[, start[, end]])
    const finds = [_][]const u8{ "find", "rfind", "index", "rindex", "count" };
    for (finds) |m| if (eq(u8, name, m) and args.len >= 1 and args.len <= 3) {
        const sub = strOf(args[0]) orelse return null;
        const lo = (Bound.of(s, if (args.len > 1) args[1] else null, 0) orelse return null) orelse return null;
        const hi_raw = (Bound.of(s, if (args.len > 2) args[2] else null, s.len) orelse return null) orelse return null;
        const hi = @min(hi_raw, s.len);
        const is_count = eq(u8, m, "count");
        const is_index = eq(u8, m, "index") or eq(u8, m, "rindex");
        // (a start past the end: nothing there, not even "")
        if (lo > s.len or lo > hi) {
            if (is_count) {
                out.* = Value.pint(0);
                return true;
            }
            if (is_index) return failAs(ctx, node, .ValueError, null, "substring not found", .{});
            out.* = Value.pint(-1);
            return true;
        }
        const hay = b[lo..hi];
        if (is_count) {
            out.* = Value.pint(if (sub.len == 0) chars(hay) + 1 else @intCast(std.mem.count(u8, hay, sub)));
            return true;
        }
        const from_end = m[0] == 'r';
        const at = if (from_end) std.mem.lastIndexOf(u8, hay, sub) else std.mem.indexOf(u8, hay, sub);
        if (at) |i| {
            out.* = Value.pint(chars(b[0 .. lo + i]));
            return true;
        }
        if (is_index) return failAs(ctx, node, .ValueError, null, "substring not found", .{});
        out.* = Value.pint(-1);
        return true;
    };
    // startswith, endswith (a str or a tuple of them)
    if ((eq(u8, name, "startswith") or eq(u8, name, "endswith")) and args.len == 1) {
        const starts = eq(u8, name, "startswith");
        const one = [1]Value{args[0]};
        const subs: []const Value = if (args[0].kind() == .tuple) @as(*value.Tuple, @ptrCast(@alignCast(args[0].ptr()))).slice() else &one;
        for (subs) |x| {
            const sub = strOf(x) orelse return null;
            if (if (starts) std.mem.startsWith(u8, b, sub) else std.mem.endsWith(u8, b, sub)) {
                out.* = Value.boolean(true);
                return true;
            }
        }
        out.* = Value.boolean(false);
        return true;
    }
    // replace(old, new): every one; old "": new around every code point
    if (eq(u8, name, "replace") and args.len == 2) {
        const old = strOf(args[0]) orelse return null;
        const new = strOf(args[1]) orelse return null;
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(allocator);
        if (old.len == 0) {
            var at: usize = 0;
            buf.appendSlice(allocator, new) catch return oomFail(ctx, node);
            while (at < b.len) {
                const c = cpAt(b, at);
                buf.appendSlice(allocator, b[at .. at + c.n]) catch return oomFail(ctx, node);
                buf.appendSlice(allocator, new) catch return oomFail(ctx, node);
                at += c.n;
            }
        } else {
            var at: usize = 0;
            while (std.mem.indexOfPos(u8, b, at, old)) |i| {
                buf.appendSlice(allocator, b[at..i]) catch return oomFail(ctx, node);
                buf.appendSlice(allocator, new) catch return oomFail(ctx, node);
                at = i + old.len;
            }
            buf.appendSlice(allocator, b[at..]) catch return oomFail(ctx, node);
        }
        return newStrValue(ctx, node, out, buf.items);
    }
    // strip, lstrip, rstrip ([chars]): Python's whitespace, or the code
    // points of chars
    if ((eq(u8, name, "strip") or eq(u8, name, "lstrip") or eq(u8, name, "rstrip")) and args.len <= 1) {
        const set: ?[]const u8 = if (args.len == 1 and args[0].kind() != .none) (strOf(args[0]) orelse return null) else null;
        const inSet = struct {
            fn f(cp: u21, chars_: ?[]const u8) bool {
                const cs = chars_ orelse return isSpaceCp(cp);
                var it = std.unicode.Utf8View.initUnchecked(cs).iterator();
                while (it.nextCodepoint()) |c| if (c == cp) return true;
                return false;
            }
        }.f;
        var lo: usize = 0;
        var hi: usize = b.len;
        if (name[0] != 'r') while (lo < hi) {
            const c = cpAt(b, lo);
            if (!inSet(c.cp, set)) break;
            lo += c.n;
        };
        if (name[0] != 'l') while (hi > lo) {
            const c = cpBefore(b, hi);
            if (!inSet(c.cp, set)) break;
            hi -= c.n;
        };
        return newStrValue(ctx, node, out, b[lo..hi]);
    }
    // split(sep[, maxsplit]), split() / split(None[, maxsplit]): a list
    if (eq(u8, name, "split") and args.len <= 2) {
        const sep: ?[]const u8 = if (args.len >= 1 and args[0].kind() != .none) (strOf(args[0]) orelse return null) else null;
        var maxsplit: i64 = -1;
        if (args.len == 2) {
            if (!isInt(args[1])) return null;
            maxsplit = args[1].asInt();
        }
        if (sep) |sp| if (sp.len == 0) return failAs(ctx, node, .ValueError, null, "empty separator", .{});
        const l = value.newList(4) orelse return oomFail(ctx, node);
        const push = struct {
            fn f(c: *Ctx, at: u32, list: *value.List, bytes: []const u8) bool {
                const r = value.newStr(bytes) orelse return oomFail(c, at);
                if (!value.listPush(list, Value.obj(.str, &r.head))) return oomFail(c, at);
                return true;
            }
        }.f;
        var splits: i64 = 0;
        if (sep) |sp| {
            var at: usize = 0;
            while (maxsplit < 0 or splits < maxsplit) : (splits += 1) {
                const i = std.mem.indexOfPos(u8, b, at, sp) orelse break;
                if (!push(ctx, node, l, b[at..i])) return false;
                at = i + sp.len;
            }
            if (!push(ctx, node, l, b[at..])) return false;
        } else {
            // (runs of whitespace; none at the ends)
            var at: usize = 0;
            while (true) {
                while (at < b.len) {
                    const c = cpAt(b, at);
                    if (!isSpaceCp(c.cp)) break;
                    at += c.n;
                }
                if (at >= b.len) break;
                if (maxsplit >= 0 and splits >= maxsplit) {
                    // (the rest as it is, its trailing whitespace too)
                    if (!push(ctx, node, l, b[at..])) return false;
                    break;
                }
                const start = at;
                while (at < b.len) {
                    const c = cpAt(b, at);
                    if (isSpaceCp(c.cp)) break;
                    at += c.n;
                }
                if (!push(ctx, node, l, b[start..at])) return false;
                splits += 1;
            }
        }
        out.* = Value.obj(.list, &l.head);
        return true;
    }
    // join (any str between them)
    if (eq(u8, name, "join") and args.len == 1 and (args[0].kind() == .list or args[0].kind() == .tuple)) {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(allocator);
        for (itemsOfSeq(args[0]), 0..) |x, i| {
            // (anything not a str: Python's TypeError)
            const t = strOf(x) orelse return null;
            if (i > 0) buf.appendSlice(allocator, b) catch return oomFail(ctx, node);
            buf.appendSlice(allocator, t) catch return oomFail(ctx, node);
        }
        return newStrValue(ctx, node, out, buf.items);
    }
    return null;
}

/// A str method of an ASCII str (one byte one character, Python's case
/// rules plain); null for one not done here.
fn strMethodAscii(ctx: *Ctx, node: u32, s: *value.Str, name: []const u8, args: []const Value, out: *Value) ?bool {
    const eq = std.mem.eql;
    const b = s.bytes();
    if (s.chars != s.len) return null;
    const strArg = struct {
        fn f(x: Value) ?[]const u8 {
            if (x.kind() != .str) return null;
            return @as(*value.Str, @ptrCast(x.ptr())).bytes();
        }
    }.f;
    const result = struct {
        fn str(c: *Ctx, at: u32, o: *Value, bytes: []const u8) bool {
            const r = value.newStr(bytes) orelse return oomFail(c, at);
            o.* = Value.obj(.str, &r.head);
            return true;
        }
    }.str;
    if (args.len == 0) {
        if (eq(u8, name, "lower") or eq(u8, name, "upper")) {
            const buf = allocator.alloc(u8, b.len) catch return oomFail(ctx, node);
            defer allocator.free(buf);
            for (b, buf) |c, *d| d.* = if (eq(u8, name, "lower")) std.ascii.toLower(c) else std.ascii.toUpper(c);
            return result(ctx, node, out, buf);
        }
        if (eq(u8, name, "strip") or eq(u8, name, "lstrip") or eq(u8, name, "rstrip")) {
            const ws = " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f";
            const t = if (eq(u8, name, "strip")) std.mem.trim(u8, b, ws) else if (eq(u8, name, "lstrip")) std.mem.trimStart(u8, b, ws) else std.mem.trimEnd(u8, b, ws);
            return result(ctx, node, out, t);
        }
        const preds = .{ .{ "isdigit", std.ascii.isDigit }, .{ "isalpha", std.ascii.isAlphabetic }, .{ "isalnum", std.ascii.isAlphanumeric }, .{ "isspace", isPySpace } };
        inline for (preds) |p| if (eq(u8, name, p[0])) {
            var all = b.len > 0;
            for (b) |c| all = all and p[1](c);
            out.* = Value.boolean(all);
            return true;
        };
        if (eq(u8, name, "isupper") or eq(u8, name, "islower")) {
            // (cased characters all upper (lower), and at least one)
            var cased = false;
            var ok = true;
            for (b) |c| {
                if (std.ascii.isUpper(c)) {
                    cased = true;
                    if (eq(u8, name, "islower")) ok = false;
                } else if (std.ascii.isLower(c)) {
                    cased = true;
                    if (eq(u8, name, "isupper")) ok = false;
                }
            }
            out.* = Value.boolean(cased and ok);
            return true;
        }
        return null;
    }
    if (args.len == 1) {
        if (eq(u8, name, "join") and (args[0].kind() == .list or args[0].kind() == .tuple)) {
            const items = if (args[0].kind() == .list) @as(*value.List, @ptrCast(@alignCast(args[0].ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(args[0].ptr()))).slice();
            var buf: std.ArrayListUnmanaged(u8) = .empty;
            defer buf.deinit(allocator);
            for (items, 0..) |x, i| {
                // (anything not a str: Python's TypeError)
                const t = strArg(x) orelse return null;
                if (i > 0) buf.appendSlice(allocator, b) catch return oomFail(ctx, node);
                buf.appendSlice(allocator, t) catch return oomFail(ctx, node);
            }
            return result(ctx, node, out, buf.items);
        }
        const sub = strArg(args[0]) orelse return null;
        if (eq(u8, name, "startswith")) {
            out.* = Value.boolean(std.mem.startsWith(u8, b, sub));
            return true;
        }
        if (eq(u8, name, "endswith")) {
            out.* = Value.boolean(std.mem.endsWith(u8, b, sub));
            return true;
        }
        if (eq(u8, name, "find") or eq(u8, name, "count")) {
            // (the other str may be non-ASCII: then never found in ASCII)
            if (eq(u8, name, "find")) {
                out.* = Value.pint(if (std.mem.indexOf(u8, b, sub)) |i| @intCast(i) else -1);
            } else {
                out.* = Value.pint(@intCast(if (sub.len == 0) b.len + 1 else std.mem.count(u8, b, sub)));
            }
            return true;
        }
    }
    return null;
}

/// int(s) of an ASCII str as Python reads it in base 10 (whitespace
/// around, a sign, digits with single underscores between them), or null
/// (not one, or beyond 64 bits).
fn parseInt(s: []const u8) ?i64 {
    const t = std.mem.trim(u8, s, int_space);
    if (!looksLikeInt(s)) return null;
    var i: usize = 0;
    var neg = false;
    if (t[0] == '+' or t[0] == '-') {
        neg = t[0] == '-';
        i = 1;
    }
    var acc: i128 = 0;
    while (i < t.len) : (i += 1) {
        if (t[i] == '_') continue;
        acc = acc * 10 + (t[i] - '0');
        if (acc > std.math.maxInt(i64) + 1) return null;
    }
    if (neg) acc = -acc;
    if (acc > std.math.maxInt(i64)) return null;
    return @intCast(acc);
}

/// The whitespace int() takes around a number (not all isspace()'s: the
/// \x1c-\x1f separators aren't)
const int_space = " \t\n\r\x0b\x0c";

/// Whether int() reads an ASCII str as a base 10 int (of any size).
fn looksLikeInt(s: []const u8) bool {
    const t = std.mem.trim(u8, s, int_space);
    var i: usize = 0;
    if (t.len > 0 and (t[0] == '+' or t[0] == '-')) i = 1;
    if (i >= t.len) return false;
    var prev_digit = false;
    while (i < t.len) : (i += 1) {
        if (std.ascii.isDigit(t[i])) {
            prev_digit = true;
        } else if (t[i] == '_' and prev_digit and i + 1 < t.len and std.ascii.isDigit(t[i + 1])) {
            prev_digit = false;
        } else return false;
    }
    return true;
}

/// repr() of an ASCII str, as Python writes it (quotes, escapes), into
/// `buf`; null if it doesn't fit.
fn pyRepr(s: []const u8, buf: []u8) ?[]const u8 {
    const has_single = std.mem.indexOfScalar(u8, s, '\'') != null;
    const has_double = std.mem.indexOfScalar(u8, s, '"') != null;
    const q: u8 = if (has_single and !has_double) '"' else '\'';
    var w = std.Io.Writer.fixed(buf);
    w.writeByte(q) catch return null;
    for (s) |c| {
        switch (c) {
            '\\' => w.writeAll("\\\\") catch return null,
            '\n' => w.writeAll("\\n") catch return null,
            '\r' => w.writeAll("\\r") catch return null,
            '\t' => w.writeAll("\\t") catch return null,
            else => if (c == q) {
                w.writeByte('\\') catch return null;
                w.writeByte(c) catch return null;
            } else if (c < 0x20 or c == 0x7f) {
                w.print("\\x{x:0>2}", .{c}) catch return null;
            } else w.writeByte(c) catch return null,
        }
    }
    w.writeByte(q) catch return null;
    return w.buffered();
}

/// Python's str.isspace() for an ASCII character.
fn isPySpace(c: u8) bool {
    return switch (c) {
        ' ', '\t', '\n', '\r', 0x0b, 0x0c, 0x1c, 0x1d, 0x1e, 0x1f => true,
        else => false,
    };
}

/// A builtin called with arguments, as Python does it (int(), float(),
/// len(), zip()... of run-time values): a host object of the program.
export fn zr_call_python(ctx: *Ctx, node: u32, callee_index: u64, args: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    const callee = ctx.object(callee_index);
    gil.ensureAt(@src(), node, callee);
    if (collecting) {
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

/// An exception made (zr_call_python of an exception class: rt.Throw(v),
/// ValueError(msg)...): an error on its way, what strict mode allows.
export fn zr_new_exception(ctx: *Ctx, node: u32, callee_index: u64, kind: u32, args: [*]const Value, n: u64, out: *Value) callconv(.c) bool {
    // (a class compiled code knows (errors.Kind): natively, an exception
    // value; an rt.Throw's value and message its first two arguments)
    if (kind != no_kind) {
        const t = value.newTuple(n) orelse return oomFail(ctx, node);
        for (args[0..n], t.slice()) |x, *slot| {
            value.incref(x);
            slot.* = x;
        }
        const tv = Value.obj(.tuple, &t.head);
        defer value.decref(tv);
        const is_throw = kind == @intFromEnum(errors.Kind.Throw);
        const val = if (is_throw and n > 0) args[0] else Value.none_v;
        const msg = if (is_throw and n > 1) args[1] else Value.none_v;
        const e = value.newExc(kind, tv, val, msg, node) orelse return oomFail(ctx, node);
        out.* = Value.obj(.exc, &e.head);
        return true;
    }
    gil.allowBegin();
    defer gil.allowEnd();
    return zr_call_python(ctx, node, callee_index, args, n, out);
}

/// zr_new_exception's kind for a class that isn't one
pub const no_kind: u32 = 0xff;

/// str() of a value, natively where it can be (strs, numbers, bools,
/// None), else Python's
fn strInto(ctx: *Ctx, v: Value, buf: *std.ArrayListUnmanaged(u8)) bool {
    if (v.kind() == .str) {
        buf.appendSlice(allocator, @as(*value.Str, @ptrCast(v.ptr())).bytes()) catch return false;
        return true;
    }
    if (value.wide(v)) |x| if (v.kind() != .bool) {
        buf.print(allocator, "{d}", .{x}) catch return false;
        return true;
    };
    var small: [40]u8 = undefined;
    if (plainStr(v)) |p| {
        buf.appendSlice(allocator, p.of(&small)) catch return false;
        return true;
    }
    return anyStr(ctx, v, buf, false);
}

/// Running without Python (a standalone build's runtime): what Python
/// would do, natively
pub const standalone = @import("build_options").standalone;

// -- output: print(), sys.stdout.write(), sys.stderr.write() --

/// What a standalone program wrote to its standard output and hasn't
/// written out yet (as Python's sys.stdout buffers it: by line on a
/// terminal, else by block)
var out_buf: std.ArrayListUnmanaged(u8) = .empty;
var out_tty: ?bool = null;

/// The standard output written out (before the standard error is
/// written, and as the program ends).
pub fn flushOutput() void {
    writeAll(1, out_buf.items);
    out_buf.clearRetainingCapacity();
}

fn writeAll(fd: i32, bytes: []const u8) void {
    var rest = bytes;
    while (rest.len > 0) {
        const n = std.c.write(fd, rest.ptr, rest.len);
        if (n <= 0) return;
        rest = rest[@intCast(n)..];
    }
}

/// Text to the standard output (fd 1) or error (2): Python's sys.stdout
/// or sys.stderr (whatever they are when it's written, as Python's
/// print() finds them), or, without Python, the file.
fn emit(ctx: *Ctx, node: u32, fd: u32, text: []const u8) bool {
    if (standalone) {
        if (fd == 1) {
            out_buf.appendSlice(allocator, text) catch return oomFail(ctx, node);
            const tty = out_tty orelse blk: {
                const t = std.c.isatty(1) != 0;
                out_tty = t;
                break :blk t;
            };
            if (out_buf.items.len >= 8192 or (tty and std.mem.indexOfScalar(u8, text, '\n') != null)) flushOutput();
        } else {
            flushOutput();
            writeAll(2, text);
        }
        return true;
    }
    // (I/O: what strict mode allows)
    gil.allowBegin();
    defer gil.allowEnd();
    const file = py.c.PySys_GetObject(if (fd == 1) "stdout" else "stderr") orelse
        return failAs(ctx, node, .RuntimeError, null, "lost sys.{s}", .{if (fd == 1) "stdout" else "stderr"});
    const s = py.c.PyUnicode_DecodeUTF8(text.ptr, @intCast(text.len), null) orelse return failPython(ctx, node);
    defer py.Py_DecRef(s);
    const w = py.c.PyObject_GetAttrString(file, "write") orelse return failPython(ctx, node);
    defer py.Py_DecRef(w);
    const r = py.c.PyObject_CallFunctionObjArgs(w, s, @as(?*PyObject, null)) orelse return failPython(ctx, node);
    py.Py_DecRef(r);
    return true;
}

/// sys.stdout.write(s) (fd 1), sys.stderr.write(s) (2): the str written,
/// its length in characters the result.
export fn zr_write(ctx: *Ctx, node: u32, fd: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() != .str) return failAs(ctx, node, .TypeError, null, "write() argument must be str, not {s}", .{value.typeName(v)});
    const s: *value.Str = @ptrCast(v.ptr());
    if (!emit(ctx, node, fd, s.bytes())) return false;
    out.* = Value.pint(@intCast(s.chars));
    return true;
}

/// print(*args, *seq, sep=sep, end=end) to the standard output (fd 1) or
/// error (2): each item's str(), sep between them (None: a space), end
/// after (None: a newline). `seq`: the items of a list known only at run
/// time, after args (none: none).
export fn zr_print(ctx: *Ctx, node: u32, fd: u32, args: [*]const Value, n: u64, seq_t: u64, seq_bits: u64, sep_t: u64, sep_bits: u64, end_t: u64, end_bits: u64, out: *Value) callconv(.c) bool {
    const sep = Value{ .tag = sep_t, .bits = sep_bits };
    const end = Value{ .tag = end_t, .bits = end_bits };
    inline for (.{ .{ sep, "sep" }, .{ end, "end" } }) |p| {
        if (p[0].kind() != .none and p[0].kind() != .str) return failAs(ctx, node, .TypeError, null, p[1] ++ " must be None or a string, not {s}", .{value.typeName(p[0])});
    }
    const seq = Value{ .tag = seq_t, .bits = seq_bits };
    const more: []const Value = if (seq.kind() == .list) @as(*value.List, @ptrCast(@alignCast(seq.ptr()))).slice() else &.{};
    var text: std.ArrayListUnmanaged(u8) = .empty;
    defer text.deinit(allocator);
    var i: usize = 0;
    while (i < n + more.len) : (i += 1) {
        if (i > 0) {
            if (sep.kind() == .str) text.appendSlice(allocator, @as(*value.Str, @ptrCast(sep.ptr())).bytes()) catch return oomFail(ctx, node) else text.append(allocator, ' ') catch return oomFail(ctx, node);
        }
        const x = if (i < n) args[i] else more[i - n];
        if (!strInto(ctx, x, &text)) return if (ctx.failed) false else fail(ctx, node, "print(): str() of a {s} failed", .{value.typeName(x)});
    }
    if (end.kind() == .str) text.appendSlice(allocator, @as(*value.Str, @ptrCast(end.ptr())).bytes()) catch return oomFail(ctx, node) else text.append(allocator, '\n') catch return oomFail(ctx, node);
    if (!emit(ctx, node, fd, text.items)) return false;
    out.* = Value.none_v;
    return true;
}

/// str() (repr: repr()) of any value: natively, else Python's
fn anyStr(ctx: *Ctx, v: Value, buf: *std.ArrayListUnmanaged(u8), repr: bool) bool {
    const start = buf.items.len;
    var seen: std.ArrayListUnmanaged(usize) = .empty;
    defer seen.deinit(allocator);
    nativeStr(v, buf, repr, &seen) catch |e| switch (e) {
        error.OutOfMemory => return false,
        error.NotNative => {
            buf.shrinkRetainingCapacity(start);
            return pythonStr(ctx, v, buf, repr);
        },
    };
    return true;
}

/// str() (repr: repr()) of a value as Python writes it, natively: NotNative
/// when a part of it isn't written natively (a host object, a str repr()
/// would escape beyond ASCII, a record of a Python class...: Python's
/// then). Without Python, every value is (those Python alone knows by
/// their kind). `seen`: the containers being written (one inside itself
/// is `[...]`, as Python's).
fn nativeStr(v: Value, buf: *std.ArrayListUnmanaged(u8), repr: bool, seen: *std.ArrayListUnmanaged(usize)) error{ OutOfMemory, NotNative }!void {
    var small: [40]u8 = undefined;
    if (plainStr(v)) |p| return buf.appendSlice(allocator, p.of(&small));
    if (v.kind() != .bool) if (value.wide(v)) |x| return buf.print(allocator, "{d}", .{x});
    switch (v.kind()) {
        .str => {
            const s = @as(*value.Str, @ptrCast(v.ptr())).bytes();
            if (!repr) return buf.appendSlice(allocator, s);
            if (!standalone) for (s) |ch| if (ch >= 0x80) return error.NotNative;
            return quoted(s, buf, false);
        },
        .bytes => {
            const b: *value.Bytes = @ptrCast(@alignCast(v.ptr()));
            if (!standalone and !b.isPyBytes()) return error.NotNative;
            try buf.append(allocator, 'b');
            return quoted(b.slice(), buf, true);
        },
        .list, .dict, .set => {
            const addr = v.bits;
            const open: u8, const close: u8 = if (v.kind() == .list) .{ '[', ']' } else .{ '{', '}' };
            if (std.mem.indexOfScalar(usize, seen.items, addr) != null) return buf.print(allocator, "{c}...{c}", .{ open, close });
            try seen.append(allocator, addr);
            defer _ = seen.pop();
            switch (v.kind()) {
                .list => {
                    const l: *value.List = @ptrCast(@alignCast(v.ptr()));
                    try buf.append(allocator, '[');
                    // (by index: an item's repr can't change the list, but
                    // its length is read each time, as Python's)
                    var i: usize = 0;
                    while (i < l.len) : (i += 1) {
                        if (i > 0) try buf.appendSlice(allocator, ", ");
                        try nativeStr(l.slice()[i], buf, true, seen);
                    }
                    try buf.append(allocator, ']');
                },
                .dict => {
                    const d: *value.Dict = @ptrCast(@alignCast(v.ptr()));
                    try buf.append(allocator, '{');
                    var first = true;
                    if (d.entries) |entries| for (entries[0..d.used]) |e| {
                        if (e.key.tag == value.DELETED) continue;
                        if (!first) try buf.appendSlice(allocator, ", ");
                        first = false;
                        try nativeStr(e.key, buf, true, seen);
                        try buf.appendSlice(allocator, ": ");
                        try nativeStr(e.value, buf, true, seen);
                    };
                    try buf.append(allocator, '}');
                },
                else => {
                    const s: *@import("set.zig").Set = @ptrCast(@alignCast(v.ptr()));
                    var it = @import("set.zig").iterate(s);
                    var first = true;
                    while (it.next()) |x| {
                        try buf.appendSlice(allocator, if (first) "{" else ", ");
                        first = false;
                        try nativeStr(x, buf, true, seen);
                    }
                    try buf.appendSlice(allocator, if (first) "set()" else "}");
                },
            }
        },
        .tuple => {
            const items = @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
            try buf.append(allocator, '(');
            for (items, 0..) |x, i| {
                if (i > 0) try buf.appendSlice(allocator, ", ");
                try nativeStr(x, buf, true, seen);
            }
            if (items.len == 1) try buf.append(allocator, ',');
            try buf.append(allocator, ')');
        },
        .record => {
            // (a dataclass's repr: its class's, unless it's Python's to say)
            const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
            if (!standalone and r.rtype.py_class != null) return error.NotNative;
            if (std.mem.indexOfScalar(usize, seen.items, v.bits) != null) return buf.appendSlice(allocator, "...");
            try seen.append(allocator, v.bits);
            defer _ = seen.pop();
            try buf.print(allocator, "{s}(", .{r.rtype.name});
            for (r.fields(), r.rtype.fields, 0..) |x, name, i| {
                if (i > 0) try buf.appendSlice(allocator, ", ");
                try buf.print(allocator, "{s}=", .{name});
                try nativeStr(x, buf, true, seen);
            }
            try buf.append(allocator, ')');
        },
        .exc => {
            const e: *value.Exc = @ptrCast(@alignCast(v.ptr()));
            const args = @as(*value.Tuple, @ptrCast(@alignCast(e.args.ptr()))).slice();
            if (repr) {
                try buf.print(allocator, "{s}(", .{@as(errors.Kind, @enumFromInt(e.kind)).name()});
                for (args, 0..) |x, i| {
                    if (i > 0) try buf.appendSlice(allocator, ", ");
                    try nativeStr(x, buf, true, seen);
                }
                return buf.append(allocator, ')');
            }
            if (args.len == 0) return;
            if (args.len == 1) return nativeStr(args[0], buf, e.kind == @intFromEnum(errors.Kind.KeyError), seen);
            return nativeStr(e.args, buf, true, seen);
        },
        else => {
            if (!standalone) return error.NotNative;
            try buf.print(allocator, "<{s}>", .{@tagName(v.kind())});
        },
    }
}

/// A str's (bytes: a bytes') repr() between quotes, as Python writes it:
/// ' unless the text has ' and not ", escapes for what isn't printable
/// (beyond ASCII: a str's characters as they are; a bytes' \x escapes)
fn quoted(s: []const u8, buf: *std.ArrayListUnmanaged(u8), is_bytes: bool) !void {
    const has_single = std.mem.indexOfScalar(u8, s, '\'') != null;
    const has_double = std.mem.indexOfScalar(u8, s, '"') != null;
    const q: u8 = if (has_single and !has_double) '"' else '\'';
    try buf.append(allocator, q);
    for (s) |ch| switch (ch) {
        '\\' => try buf.appendSlice(allocator, "\\\\"),
        '\n' => try buf.appendSlice(allocator, "\\n"),
        '\r' => try buf.appendSlice(allocator, "\\r"),
        '\t' => try buf.appendSlice(allocator, "\\t"),
        else => if (ch == q) {
            try buf.append(allocator, '\\');
            try buf.append(allocator, ch);
        } else if (ch < 0x20 or ch == 0x7f or (is_bytes and ch >= 0x80)) {
            try buf.print(allocator, "\\x{x:0>2}", .{ch});
        } else try buf.append(allocator, ch),
    };
    try buf.append(allocator, q);
}

/// str() (repr: repr()) of a value, Python's (an error's way: what strict
/// mode allows)
fn pythonStr(ctx: *Ctx, v: Value, buf: *std.ArrayListUnmanaged(u8), repr: bool) bool {
    gil.allowBegin();
    defer gil.allowEnd();
    const o = value.toPython(v, ctx.node_maker) orelse {
        py.c.PyErr_Clear();
        return false;
    };
    defer py.Py_DecRef(o);
    const s = (if (repr) py.c.PyObject_Repr(o) else py.c.PyObject_Str(o)) orelse {
        py.c.PyErr_Clear();
        return false;
    };
    defer py.Py_DecRef(s);
    buf.appendSlice(allocator, ph.utf8(s, "str") orelse "") catch return false;
    return true;
}

/// repr() of a key (a KeyError's str): natively for strs and ints
fn reprInto(ctx: *Ctx, v: Value, buf: *std.ArrayListUnmanaged(u8)) bool {
    if (v.kind() == .str) {
        var tmp: [1024]u8 = undefined;
        if (pyRepr(@as(*value.Str, @ptrCast(v.ptr())).bytes(), &tmp)) |r| {
            buf.appendSlice(allocator, r) catch return false;
            return true;
        }
    }
    if (value.wide(v)) |x| if (v.kind() != .bool) {
        buf.print(allocator, "{d}", .{x}) catch return false;
        return true;
    };
    return anyStr(ctx, v, buf, true);
}

/// Python's str() of an exception value: its arguments' (none: "", one:
/// its str (a KeyError: its repr), more: the tuple's)
fn excStr(ctx: *Ctx, e: *value.Exc, buf: *std.ArrayListUnmanaged(u8)) bool {
    const args = @as(*value.Tuple, @ptrCast(@alignCast(e.args.ptr()))).slice();
    if (args.len == 0) return true;
    if (args.len == 1) {
        if (e.kind == @intFromEnum(errors.Kind.KeyError)) return reprInto(ctx, args[0], buf);
        return strInto(ctx, args[0], buf);
    }
    return anyStr(ctx, e.args, buf, false);
}

/// `raise e` of an exception value: the run's error, natively (what
/// except matches and gets: the value itself), worded as the reference
/// mode words the exception (an rt.Throw: its message, or its value's str)
pub fn raiseExc(ctx: *Ctx, at: u32, v: Value) bool {
    if (ctx.failed) return false;
    const e: *value.Exc = @ptrCast(@alignCast(v.ptr()));
    const kind: errors.Kind = @enumFromInt(e.kind);
    var str_buf: std.ArrayListUnmanaged(u8) = .empty;
    defer str_buf.deinit(allocator);
    var msg: std.ArrayListUnmanaged(u8) = .empty;
    defer msg.deinit(allocator);
    if (kind == .Throw) {
        _ = strInto(ctx, if (e.message.kind() != .none) e.message else e.value, &msg);
    } else if (kind == .ZrunError and e.message.kind() == .str) {
        msg.appendSlice(allocator, @as(*value.Str, @ptrCast(e.message.ptr())).bytes()) catch {};
    } else {
        _ = excStr(ctx, e, &str_buf);
        // (pythonMessage's wording: the run's error as the reference mode
        // gives it)
        switch (kind) {
            .IntegerOverflow => msg.appendSlice(allocator, "integer overflow") catch {},
            .ZeroDivisionError => msg.appendSlice(allocator, "division by zero") catch {},
            .RecursionError => msg.appendSlice(allocator, "call stack too deep") catch {},
            .KeyError => {
                msg.appendSlice(allocator, "key not found") catch {};
                if (str_buf.items.len > 0) msg.print(allocator, ": {s}", .{str_buf.items}) catch {};
            },
            else => if (kind.isA(.TypeError) or kind.isA(.IndexError) or kind.isA(.ValueError) or kind.isA(.AttributeError))
                msg.appendSlice(allocator, str_buf.items) catch {}
            else
                msg.print(allocator, "{s}: {s}", .{ kind.name(), str_buf.items }) catch {},
        }
    }
    value.incref(v);
    ctx.exc_value = v;
    ctx.exc_kind = kind;
    ctx.exc_msg.clearRetainingCapacity();
    ctx.exc_msg.appendSlice(allocator, str_buf.items) catch {};
    // (a zrun.Error raised again: at its own node, where it happened)
    return fail(ctx, if (kind == .ZrunError) e.node else at, "{s}", .{msg.items});
}

/// The error being raised, leaving a semantic: a zrun.Error from here (its
/// node and message the error's), as the reference mode makes an exception
/// leaving a semantic; an rt.Throw, a zrun.Error already: themselves.
pub export fn zr_exc_wrap(ctx: *Ctx) callconv(.c) void {
    if (!ctx.failed or ctx.pending != null or ctx.exc_kind == .Throw or ctx.exc_kind == .ZrunError) return;
    if (ctx.exc) |e| {
        gil.allowBegin();
        defer gil.allowEnd();
        py.Py_DecRef(e);
        ctx.exc = null;
    }
    if (ctx.exc_value) |v| value.decref(v);
    ctx.exc_value = null;
    ctx.exc_kind = null;
    ctx.exc_msg.clearRetainingCapacity();
}

/// The error being raised, caught as a value natively (`except ... as e`):
/// its exception value; one of the native code's, a value made of its
/// kind and message (a zrun.Error's: its message and node); null: a Python
/// exception's (Python's object then)
pub fn caughtValue(ctx: *Ctx, out: *Value) ?bool {
    if (ctx.pending != null or ctx.exc != null) return null;
    if (ctx.exc_value) |v| {
        ctx.exc_value = null;
        ctx.clearError();
        out.* = v;
        return true;
    }
    const kind = ctx.exc_kind orelse .ZrunError;
    const text = if (ctx.exc_kind != null) ctx.exc_msg.items else ctx.err_msg.items;
    const s = value.newStr(text) orelse return oomFail(ctx, ctx.err_node);
    const sv = Value.obj(.str, &s.head);
    defer value.decref(sv);
    const t = value.newTuple(1) orelse return oomFail(ctx, ctx.err_node);
    value.incref(sv);
    t.slice()[0] = sv;
    const tv = Value.obj(.tuple, &t.head);
    defer value.decref(tv);
    const e = value.newExc(@intFromEnum(kind), tv, Value.none_v, if (kind == .ZrunError) sv else Value.none_v, ctx.err_node) orelse return oomFail(ctx, ctx.err_node);
    ctx.clearError();
    out.* = Value.obj(.exc, &e.head);
    return true;
}

/// An exception value's attribute (args, an rt.Throw's value and message,
/// a zrun.Error's diagnostic (itself: its message what's read of it));
/// null: another (Python's, of the exception as Python makes it)
fn excAttr(v: Value, name: []const u8, out: *Value) bool {
    const e: *value.Exc = @ptrCast(@alignCast(v.ptr()));
    const kind: errors.Kind = @enumFromInt(e.kind);
    const eq = std.mem.eql;
    const r: Value = if (eq(u8, name, "args")) e.args else if (kind == .Throw and eq(u8, name, "value")) e.value else if ((kind == .Throw or kind == .ZrunError) and eq(u8, name, "message")) e.message else if (kind == .ZrunError and eq(u8, name, "diagnostic")) v else if (eq(u8, name, "__cause__")) e.cause else return false;
    value.incref(r);
    out.* = r;
    return true;
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
        if (step == 0) return failAs(ctx, node, .ValueError, null, "slice step cannot be zero", .{});
        const len: i64 = switch (v.kind()) {
            .list => @intCast(@as(*value.List, @ptrCast(@alignCast(v.ptr()))).len),
            .tuple => @intCast(@as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).len),
            .str => @intCast(@as(*value.Str, @ptrCast(v.ptr())).chars),
            .bytes => @intCast(@as(*value.Bytes, @ptrCast(@alignCast(v.ptr()))).len),
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
                const str: *value.Str = @ptrCast(v.ptr());
                const src = str.bytes();
                var buf: std.ArrayListUnmanaged(u8) = .empty;
                defer buf.deinit(allocator);
                if (str.chars == str.len) {
                    // (ASCII: a byte a character)
                    buf.ensureTotalCapacity(allocator, count) catch return oomFail(ctx, node);
                    var i = start;
                    for (0..count) |_| {
                        buf.appendAssumeCapacity(src[@intCast(i)]);
                        i += step;
                    }
                } else {
                    // (UTF-8: each character's bytes, by where they start)
                    const starts = allocator.alloc(usize, str.chars + 1) catch return oomFail(ctx, node);
                    defer allocator.free(starts);
                    var at: usize = 0;
                    for (starts[0..str.chars]) |*p| {
                        p.* = at;
                        at += std.unicode.utf8ByteSequenceLength(src[at]) catch 1;
                    }
                    starts[str.chars] = src.len;
                    var i = start;
                    for (0..count) |_| {
                        const k: usize = @intCast(i);
                        buf.appendSlice(allocator, src[starts[k]..starts[k + 1]]) catch return oomFail(ctx, node);
                        i += step;
                    }
                }
                const s = value.newStr(buf.items) orelse return oomFail(ctx, node);
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
            // (bytes: of their kind, a view of the same memory if the step's
            // 1; else their bytes, a bytes of them as a zrun.Bytes' step
            // slice is)
            .bytes => {
                const b: *value.Bytes = @ptrCast(@alignCast(v.ptr()));
                if (step == 1) {
                    const from: usize = if (count == 0) 0 else @intCast(start);
                    const s = value.bytesSlice(b, from, from + count) orelse return oomFail(ctx, node);
                    out.* = Value.obj(.bytes, &s.head);
                    return true;
                }
                const s = value.newOwnedBytes(count) orelse return oomFail(ctx, node);
                const dst = value.bytesMemory(s);
                var i = start;
                for (dst) |*d| {
                    d.* = b.ptr[@intCast(i)];
                    i += step;
                }
                out.* = Value.obj(.bytes, &s.head);
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
    if (!objects(ctx, node,&.{ v, bounds[0], bounds[1], bounds[2] }, &objs)) return failPython(ctx, node);
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
            gil.ensureAt(@src(), node, null);
            const o = py.c.PyNumber_Index(@ptrFromInt(a.bits)) orelse return failPython(ctx, node);
            defer py.Py_DecRef(o);
            var overflow: c_int = 0;
            const x = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
            if (overflow != 0) return failAs(ctx, node, .OverflowError, null, "range() beyond 64-bit ints isn't supported in compiled code", .{});
            break :blk x;
        } else return failAs(ctx, node, .TypeError, null, "'{s}' object cannot be interpreted as an integer", .{value.typeName(a)});
    }
    const b: [3]i64 = if (n == 1) .{ 0, vals[0], 1 } else .{ vals[0], vals[1], if (n == 3) vals[2] else 1 };
    if (b[2] == 0) return failAs(ctx, node, .ValueError, null, "range() arg 3 must not be zero", .{});
    out.* = b;
    return true;
}

/// The builtins zr_builtin does natively, by code
/// The builtins zr_builtin does natively for native values (floor and ceil:
/// math's)
pub const Builtin = enum(u32) { int, float, len, abs, str, bool, list, floor, ceil, chr, ord, id, fspath };

/// A builtin of one argument (`code`), natively for native values, as
/// Python does it; anything else (and the errors) by the builtin itself
/// (objects[callee_index]).
export fn zr_builtin(ctx: *Ctx, node: u32, code: u32, callee_index: u64, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    // (a Big: int(), floor(), ceil() itself, abs() in 128 bits, float()
    // rounded, str() its digits, bool() true)
    if (v.kind() == .big) {
        const x = value.wide(v).?;
        switch (@as(Builtin, @enumFromInt(code))) {
            .int, .floor, .ceil => {
                value.incref(v);
                out.* = v;
                return true;
            },
            .list, .ord, .chr, .id, .fspath => {},
            .abs => if (x != std.math.minInt(i128)) {
                out.* = value.intValue(if (x < 0) -x else x) orelse return oomFail(ctx, node);
                return true;
            },
            .float => {
                out.* = Value.float(@floatFromInt(x));
                return true;
            },
            .str => {
                var buf: [48]u8 = undefined;
                const s = value.newStr(std.fmt.bufPrint(&buf, "{d}", .{x}) catch unreachable) orelse return oomFail(ctx, node);
                out.* = Value.obj(.str, &s.head);
                return true;
            },
            .bool => {
                out.* = Value.boolean(true);
                return true;
            },
            .len => {},
        }
    }
    switch (@as(Builtin, @enumFromInt(code))) {
        // (int() of an int: a plain one, as int(I64) gives)
        .int => switch (v.kind()) {
            .int, .bool => {
                out.* = Value.pint(v.asInt());
                return true;
            },
            .str => {
                const s: *value.Str = @ptrCast(v.ptr());
                if (s.chars == s.len) {
                    if (parseInt(s.bytes())) |x| {
                        out.* = Value.pint(x);
                        return true;
                    }
                    // (not a number at all: Python's error; one beyond 64
                    // bits: Python's big int)
                    if (!looksLikeInt(s.bytes())) {
                        var buf: [256]u8 = undefined;
                        const r = pyRepr(s.bytes(), &buf) orelse "...";
                        return failAs(ctx, node, .ValueError, null, "invalid literal for int() with base 10: {s}", .{r});
                    }
                }
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
            // (a str as Python reads one: its syntax checked, the number
            // correctly rounded, as CPython's; one that isn't: Python's
            // error, its words)
            .str => {
                const s: *value.Str = @ptrCast(v.ptr());
                if (pyFloat(s.bytes())) |x| {
                    out.* = Value.float(x);
                    return true;
                }
                return pythonsError(ctx, node, callee_index, v, out);
            },
            else => {},
        },
        .len => switch (v.kind()) {
            .list => {
                out.* = Value.pint(@intCast(@as(*value.List, @ptrCast(@alignCast(v.ptr()))).len));
                return true;
            },
            // (a Python list, tuple or dict (data given by Python): its
            // size, without calling Python)
            .host => {
                gil.ensureAt(@src(), node, null);
                const o: *PyObject = @ptrFromInt(v.bits);
                const t_ = ph.typeOf(o);
                const n: isize = if (t_ == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyList_Type")))))
                    py.c.PyList_Size(o)
                else if (t_ == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyTuple_Type")))))
                    py.c.PyTuple_Size(o)
                else if (t_ == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyDict_Type")))))
                    py.c.PyDict_Size(o)
                else
                    -1;
                if (n >= 0) {
                    out.* = Value.pint(@intCast(n));
                    return true;
                }
            },
            .tuple => {
                out.* = Value.pint(@intCast(@as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).len));
                return true;
            },
            .dict => {
                out.* = Value.pint(@intCast(@as(*value.Dict, @ptrCast(@alignCast(v.ptr()))).len));
                return true;
            },
            .set => {
                out.* = Value.pint(@intCast(setOf(v).used));
                return true;
            },
            .str => {
                out.* = Value.pint(@intCast(@as(*value.Str, @ptrCast(v.ptr())).chars));
                return true;
            },
            .bytes => {
                out.* = Value.pint(@intCast(@as(*value.Bytes, @ptrCast(@alignCast(v.ptr()))).len));
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
            } else {
                // (of the least 64-bit int: past 64 bits, a Big for a plain
                // int; an I64's overflow)
                if (v.tag == @intFromEnum(Tag.int)) return failAs(ctx, node, .IntegerOverflow, null, "integer overflow", .{});
                out.* = value.intValue(-@as(i128, std.math.minInt(i64))) orelse return oomFail(ctx, node);
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
            // (a float, a bool, None: as Python writes them)
            .float, .bool, .none => {
                var buf: [40]u8 = undefined;
                const r = value.newStr(plainStr(v).?.of(&buf)) orelse return oomFail(ctx, node);
                out.* = Value.obj(.str, &r.head);
                return true;
            },
            // (an exception: its arguments' str, as Python's; a zrun.Error:
            // its message)
            .exc => {
                const e: *value.Exc = @ptrCast(@alignCast(v.ptr()));
                var buf: std.ArrayListUnmanaged(u8) = .empty;
                defer buf.deinit(allocator);
                const ok = if (e.kind == @intFromEnum(errors.Kind.ZrunError)) false else excStr(ctx, e, &buf);
                if (ok) {
                    const r = value.newStr(buf.items) orelse return oomFail(ctx, node);
                    out.* = Value.obj(.str, &r.head);
                    return true;
                }
            },
            else => {},
        },
        .bool => switch (v.kind()) {
            .none, .bool, .int, .float, .str, .list, .tuple, .dict, .set => {
                out.* = Value.boolean(value.truthy(v));
                return true;
            },
            else => {},
        },
        // (math.floor(), math.ceil(): an int itself, a float rounded to an
        // int (infinity, NaN and ones beyond 64 bits: Python's))
        .floor, .ceil => switch (v.kind()) {
            .int, .bool => {
                out.* = Value.pint(v.asInt());
                return true;
            },
            .float => {
                const f = v.asFloat();
                const r = if (code == @intFromEnum(Builtin.floor)) @floor(f) else @ceil(f);
                if (r == r and @abs(r) < 9.2e18) {
                    out.* = Value.pint(@intFromFloat(r));
                    return true;
                }
            },
            else => {},
        },
        // (chr() of an int: its character, UTF-8; one out of range: Python's
        // error, its words; a lone surrogate (not UTF-8): Python's str)
        .chr => if (isInt(v)) {
            const cp = v.asInt();
            if (cp < 0 or cp > 0x10FFFF) return pythonsError(ctx, node, callee_index, v, out);
            if (cp < 0xD800 or cp > 0xDFFF) {
                var buf: [4]u8 = undefined;
                const n = std.unicode.utf8Encode(@intCast(cp), &buf) catch unreachable;
                const s = value.newStr(buf[0..n]) orelse return oomFail(ctx, node);
                out.* = Value.obj(.str, &s.head);
                return true;
            }
        },
        // (ord() of a str of one character: its code point)
        .ord => if (v.kind() == .str) {
            const s: *value.Str = @ptrCast(v.ptr());
            if (s.chars == 1) {
                const cp = std.unicode.utf8Decode(s.bytes()) catch unreachable;
                out.* = Value.pint(cp);
                return true;
            }
            return pythonsError(ctx, node, callee_index, v, out);
        },
        // (id(): see idOf)
        .id => if (idOf(v)) |x| {
            out.* = Value.pint(x);
            return true;
        },
        // (os.fspath() of a str: itself)
        .fspath => if (v.kind() == .str) {
            value.incref(v);
            out.* = v;
            return true;
        },
        // (list() of a list or tuple: a new list of its items)
        .list => switch (v.kind()) {
            .list, .tuple => {
                const items = if (v.kind() == .list) @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
                const l = value.newList(items.len) orelse return oomFail(ctx, node);
                for (items) |x| {
                    value.incref(x);
                    if (!value.listPush(l, x)) return oomFail(ctx, node);
                }
                out.* = Value.obj(.list, &l.head);
                return true;
            },
            // (a dict's keys, a str's characters: a new list, as a loop
            // over them sees them)
            .dict, .str, .set => return zr_items(ctx, node, v.tag, v.bits, out),
            else => {},
        },
    }
    return zr_call_python(ctx, node, callee_index, @ptrCast(&v), 1, out);
}

/// id(v), with what Python promises: two values alive at once have the same
/// id exactly when they're the same (`is`). An object of the code's (a str,
/// list, dict, record, function...): its address; a Python object: its
/// address, as CPython's id(); None, True, False: CPython's own objects'.
/// A number (or a node) is no object here: Python's id() of one is a
/// temporary's; its id here, from its kind and bits (the same for the same
/// one), is odd (no object's address, all aligned) and below 2**63 (an
/// int of the program). Null for anything else (Python's to say).
fn idOf(v: Value) ?i64 {
    switch (v.kind()) {
        .none => return @intCast(@intFromPtr(py.Py_None())),
        .bool => return @intCast(@intFromPtr(if (v.asInt() != 0) py.Py_True() else py.Py_False())),
        .host => return @intCast(v.bits),
        .int, .float, .node => {
            // (a bijective mix of the bits, the kind apart; 62 of its bits)
            const kind_salt: u64 = @as(u64, @intFromEnum(v.kind())) *% 0xD6E8FEB86659FD93;
            const h = (v.bits ^ kind_salt) *% 0x9E3779B97F4A7C15;
            return @intCast(((h >> 2) << 1) | 1);
        },
        else => {},
    }
    if (value.counted(v.tag)) return @intCast(@intFromPtr(v.ptr()));
    return null;
}

/// An int beyond 64 bits: a Big, or Python's int beyond 128 bits (a host
/// value): not one of the program's (Gen.checkedAt: the overflow, as the
/// reference mode's I64 raises it). Its type read, no Python run.
export fn zr_huge_int(t: u64, bits: u64) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() == .big) return true;
    if (v.kind() != .host) return false;
    return ph.typeOf(@ptrFromInt(bits)) == py.types.typeObject("PyLong_Type");
}

/// The str() of a float, a bool or None (each its repr too), as Python
/// writes it, made into a buffer; null for another value
const PlainStr = union(enum) {
    float: f64,
    text: []const u8,

    fn of(self: PlainStr, buf: *[40]u8) []const u8 {
        return switch (self) {
            .float => |x| value.floatRepr(x, buf),
            .text => |t| t,
        };
    }
};

fn plainStr(v: Value) ?PlainStr {
    return switch (v.kind()) {
        .float => .{ .float = v.asFloat() },
        .bool => .{ .text = if (v.asInt() != 0) "True" else "False" },
        .none => .{ .text = "None" },
        else => null,
    };
}

/// A builtin (objects[callee_index]) given a value it refuses: called by
/// Python for its error, in its words (what strict mode allows: an error).
fn pythonsError(ctx: *Ctx, node: u32, callee_index: u64, v: Value, out: *Value) bool {
    return pythonsErrorOf(ctx, node, callee_index, @ptrCast(&v), 1, out);
}

fn pythonsErrorOf(ctx: *Ctx, node: u32, callee_index: u64, args: [*]const Value, n: u64, out: *Value) bool {
    gil.allowBegin();
    defer gil.allowEnd();
    return zr_call_python(ctx, node, callee_index, args, n, out);
}

/// The math module's functions zr_math does natively (copysign, fmod,
/// atan2, pow: two arguments; isnan, isinf, isfinite: a bool). Those from
/// `exp` on are the C library's, natively only where it's the one Python
/// runs with (Linux: the same libm, so the same results to the last bit).
pub const MathFn = enum(u32) { sqrt, fabs, degrees, radians, isnan, isinf, isfinite, copysign, modf, fmod, exp, log, log2, log10, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, expm1, log1p, atan2, pow };

pub fn mathArity(f: MathFn) u32 {
    return switch (f) {
        .copysign, .fmod, .atan2, .pow => 2,
        else => 1,
    };
}

/// Whether math's `f` is the C library's (zr_math: from the process's libm)
fn needsLibm(f: MathFn) bool {
    return switch (f) {
        .sqrt, .fabs, .degrees, .radians, .isnan, .isinf, .isfinite, .copysign, .modf => false,
        else => true,
    };
}

const F1 = *const fn (f64) callconv(.c) f64;
const F2 = *const fn (f64, f64) callconv(.c) f64;

/// The C library's math functions as Python's math module calls them: the
/// process's libm's own, found by name (Zig's compiler-rt has functions of
/// the same names, not the same to the last bit); null where they can't be
/// had (math's functions run in Python then).
const Libm = struct { fmod: F2, exp: F1, log: F1, log2: F1, log10: F1, sin: F1, cos: F1, tan: F1, asin: F1, acos: F1, atan: F1, sinh: F1, cosh: F1, tanh: F1, asinh: F1, acosh: F1, atanh: F1, expm1: F1, log1p: F1, atan2: F2, pow: F2 };
var libm: ?Libm = null;
var libm_looked = false;

/// libm's functions looked up (once: while compiling, with the GIL)
pub fn findLibm() void {
    if (libm_looked) return;
    libm_looked = true;
    if (@import("builtin").os.tag != .linux) return;
    const h = std.c.dlopen("libm.so.6", .{ .LAZY = true, .NOLOAD = true }) orelse return;
    var t: Libm = undefined;
    inline for (@typeInfo(Libm).@"struct".fields) |fd| {
        const p = std.c.dlsym(h, fd.name) orelse return;
        @field(t, fd.name) = @ptrCast(@alignCast(p));
    }
    libm = t;
}

/// A number as math's functions take it (an int, a bool, a float), or null
fn floatOf(v: Value) ?f64 {
    if (value.wide(v)) |x| return @floatFromInt(x);
    return switch (v.kind()) {
        .bool => @floatFromInt(v.asInt()),
        .float => v.asFloat(),
        else => null,
    };
}

/// math.<f>(args) (objects[callee_index]: the function) of numbers,
/// natively, as CPython computes it; a NaN from numbers that aren't, an
/// infinity from finite ones (CPython's errors, or its special cases): the
/// function itself, its result or its error in its words; anything not a
/// number, the function.
export fn zr_math(ctx: *Ctx, node: u32, code: u32, callee_index: u64, n: u64, args: [*]const Value, out: *Value) callconv(.c) bool {
    const f: MathFn = @enumFromInt(code);
    var xs: [2]f64 = undefined;
    for (args[0..n], 0..) |a, i| xs[i] = floatOf(a) orelse return zr_call_python(ctx, node, callee_index, args, n, out);
    if (needsLibm(f) and libm == null) return zr_call_python(ctx, node, callee_index, args, n, out);
    const lm = libm orelse undefined;
    const x = xs[0];
    const y = xs[1];
    switch (f) {
        .isnan, .isinf, .isfinite => {
            out.* = Value.boolean(switch (f) {
                .isnan => std.math.isNan(x),
                .isinf => std.math.isInf(x),
                else => std.math.isFinite(x),
            });
            return true;
        },
        // (its fraction and its whole part, both with its sign: C's modf)
        .modf => {
            const whole = if (std.math.isInf(x) or std.math.isNan(x)) x else @trunc(x);
            const fraction = if (std.math.isNan(x)) x else std.math.copysign(if (std.math.isInf(x)) 0.0 else x - whole, x);
            const t = value.newTuple(2) orelse return oomFail(ctx, node);
            t.slice()[0] = Value.float(fraction);
            t.slice()[1] = Value.float(whole);
            out.* = Value.obj(.tuple, &t.head);
            return true;
        },
        else => {},
    }
    const r: f64 = switch (f) {
        .sqrt => @sqrt(x),
        .fabs => @abs(x),
        // (CPython's constants: 180 / pi, pi / 180, divided as doubles)
        .degrees => x * (@as(f64, 180.0) / @as(f64, std.math.pi)),
        .radians => x * (@as(f64, std.math.pi) / @as(f64, 180.0)),
        .copysign => std.math.copysign(x, y),
        .isnan, .isinf, .isfinite, .modf => unreachable,
        // (the C library's, by the name: Libm's field)
        inline else => |g| blk: {
            const func = @field(lm, @tagName(g));
            break :blk if (@TypeOf(func) == F2) func(x, y) else func(x);
        },
    };
    var any_nan = false;
    var all_finite = true;
    for (xs[0..n]) |a| {
        any_nan = any_nan or std.math.isNan(a);
        all_finite = all_finite and std.math.isFinite(a);
    }
    if ((std.math.isNan(r) and !any_nan) or (std.math.isInf(r) and all_finite)) return pythonsErrorOf(ctx, node, callee_index, args, n, out);
    out.* = Value.float(r);
    return true;
}

/// float(s) of a str as Python reads it: whitespace around (Python's), a
/// sign, then inf, infinity or nan (any case), or digits (single
/// underscores between them) with a point and an exponent; null if it
/// isn't one (or out of what's done here).
fn pyFloat(s: []const u8) ?f64 {
    // (the whitespace around: Python's)
    var lo: usize = 0;
    var hi: usize = s.len;
    while (lo < hi) {
        const c = cpAt(s, lo);
        if (!isSpaceCp(c.cp)) break;
        lo += c.n;
    }
    while (hi > lo) {
        const c = cpBefore(s, hi);
        if (!isSpaceCp(c.cp)) break;
        hi -= c.n;
    }
    const t = s[lo..hi];
    var i: usize = 0;
    var neg = false;
    if (i < t.len and (t[i] == '+' or t[i] == '-')) {
        neg = t[i] == '-';
        i += 1;
    }
    const word = t[i..];
    const words = [_]struct { []const u8, f64 }{ .{ "inf", std.math.inf(f64) }, .{ "infinity", std.math.inf(f64) }, .{ "nan", std.math.nan(f64) } };
    for (words) |w| if (std.ascii.eqlIgnoreCase(word, w[0])) return if (neg) -w[1] else w[1];
    // digits (with single underscores between them), point, exponent
    var buf: [128]u8 = undefined;
    var n: usize = 0;
    const digits = struct {
        /// Digits from `at` (an underscore only between two): copied, their
        /// count; null for an underscore out of place
        fn run(text: []const u8, at: *usize, out: []u8, len: *usize) ?usize {
            var count: usize = 0;
            while (at.* < text.len) {
                const c = text[at.*];
                if (std.ascii.isDigit(c)) {
                    if (len.* >= out.len) return null;
                    out[len.*] = c;
                    len.* += 1;
                    count += 1;
                    at.* += 1;
                } else if (c == '_' and count > 0 and at.* + 1 < text.len and std.ascii.isDigit(text[at.* + 1])) {
                    at.* += 1;
                } else break;
            }
            return count;
        }
    }.run;
    if (neg) {
        buf[0] = '-';
        n = 1;
    }
    const whole = digits(t, &i, &buf, &n) orelse return null;
    var frac: usize = 0;
    if (i < t.len and t[i] == '.') {
        if (n >= buf.len) return null;
        buf[n] = '.';
        n += 1;
        i += 1;
        frac = digits(t, &i, &buf, &n) orelse return null;
    }
    if (whole == 0 and frac == 0) return null;
    if (i < t.len and (t[i] == 'e' or t[i] == 'E')) {
        if (n + 2 >= buf.len) return null;
        buf[n] = 'e';
        n += 1;
        i += 1;
        if (i < t.len and (t[i] == '+' or t[i] == '-')) {
            buf[n] = t[i];
            n += 1;
            i += 1;
        }
        const exp = digits(t, &i, &buf, &n) orelse return null;
        if (exp == 0) return null;
    }
    if (i != t.len) return null;
    return std.fmt.parseFloat(f64, buf[0..n]) catch null;
}

const IntParse = union(enum) { int: i64, too_big, invalid };

/// int(s, base) of an ASCII str as Python reads it (base 2 to 36, or 0:
/// from its prefix): whitespace around, a sign, a prefix (0x, 0o, 0b) the
/// base allows, digits below the base with single underscores between them
/// (one may follow the prefix); base 0 refuses leading zeros of a number
/// not zero.
fn parseIntBase(s: []const u8, base_given: u32) IntParse {
    const t = std.mem.trim(u8, s, int_space);
    var i: usize = 0;
    var neg = false;
    if (i < t.len and (t[i] == '+' or t[i] == '-')) {
        neg = t[i] == '-';
        i += 1;
    }
    var base = base_given;
    var prefixed = false;
    if (i + 1 < t.len and t[i] == '0') {
        const pb: u32 = switch (std.ascii.toLower(t[i + 1])) {
            'x' => 16,
            'o' => 8,
            'b' => 2,
            else => 0,
        };
        if (pb != 0 and (base == 0 or base == pb)) {
            base = pb;
            i += 2;
            prefixed = true;
        }
    }
    const auto = base == 0;
    if (auto) base = 10;
    var acc: i128 = 0;
    var too_big = false;
    var digits: usize = 0;
    var leading_zero = false;
    var nonzero = false;
    // (an underscore: after a digit, or the prefix)
    var after_digit = prefixed;
    while (i < t.len) : (i += 1) {
        const ch = t[i];
        if (ch == '_') {
            if (!after_digit or i + 1 >= t.len) return .invalid;
            after_digit = false;
            continue;
        }
        const d: u32 = if (std.ascii.isDigit(ch)) ch - '0' else if (std.ascii.isAlphabetic(ch)) std.ascii.toLower(ch) - 'a' + 10 else return .invalid;
        if (d >= base) return .invalid;
        if (digits == 0 and d == 0) leading_zero = true;
        if (d != 0) nonzero = true;
        digits += 1;
        after_digit = true;
        if (!too_big) {
            acc = acc * base + d;
            if (acc > std.math.maxInt(i64) + 1) too_big = true;
        }
    }
    if (digits == 0) return .invalid;
    if (auto and !prefixed and leading_zero and nonzero) return .invalid;
    if (too_big) return .too_big;
    if (neg) acc = -acc;
    if (acc > std.math.maxInt(i64)) return .too_big;
    return .{ .int = @intCast(acc) };
}

/// int(v, base) (objects[callee_index]: int), base known when compiling:
/// an ASCII str natively, with Python's error for one that isn't a number
/// in the base; a non-str, Python's TypeError; a str beyond 64 bits (a big
/// int), or not ASCII, by int itself.
export fn zr_int_base(ctx: *Ctx, node: u32, callee_index: u64, t: u64, bits: u64, base: u32, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    switch (v.kind()) {
        .str => {
            const s: *value.Str = @ptrCast(v.ptr());
            if (s.chars == s.len) switch (parseIntBase(s.bytes(), base)) {
                .int => |x| {
                    out.* = Value.pint(x);
                    return true;
                },
                .invalid => {
                    var buf: [256]u8 = undefined;
                    const r = pyRepr(s.bytes(), &buf) orelse "...";
                    return failAs(ctx, node, .ValueError, null, "invalid literal for int() with base {d}: {s}", .{ base, r });
                },
                .too_big => {},
            };
        },
        .int, .bool, .float, .none, .list, .tuple, .dict => return failAs(ctx, node, .TypeError, null, "int() can't convert non-string with explicit base", .{}),
        else => {},
    }
    const args = [2]Value{ v, Value.pint(base) };
    return zr_call_python(ctx, node, callee_index, &args, 2, out);
}

/// int(v, base), the base known only at run time: zr_int_base's, for a
/// base of 0 or 2 to 36; another, Python's error, its words
export fn zr_int_base_of(ctx: *Ctx, node: u32, callee_index: u64, t: u64, bits: u64, bt: u64, bb: u64, out: *Value) callconv(.c) bool {
    const b = Value{ .tag = bt, .bits = bb };
    if (isInt(b)) {
        const base = b.asInt();
        if (base == 0 or (base >= 2 and base <= 36)) return zr_int_base(ctx, node, callee_index, t, bits, @intCast(base), out);
    }
    const args = [2]Value{ .{ .tag = t, .bits = bits }, b };
    return pythonsErrorOf(ctx, node, callee_index, &args, 2, out);
}

/// float.hex(x): x's exact value in hexadecimal, as CPython writes it
/// (`0x1.8000000000000p+1`: 13 digits of the fraction; a subnormal's
/// `0x0.` with the exponent -1022), into `buf`
fn floatHex(x: f64, buf: *[32]u8) []const u8 {
    if (std.math.isNan(x)) return "nan";
    if (std.math.isInf(x)) return if (x < 0) "-inf" else "inf";
    const bits: u64 = @bitCast(x);
    const neg = bits >> 63 != 0;
    const exp_bits: u64 = (bits >> 52) & 0x7ff;
    const frac = bits & ((@as(u64, 1) << 52) - 1);
    if (exp_bits == 0 and frac == 0) return if (neg) "-0x0.0p+0" else "0x0.0p+0";
    const lead: u8 = if (exp_bits == 0) '0' else '1';
    const e: i64 = if (exp_bits == 0) -1022 else @as(i64, @intCast(exp_bits)) - 1023;
    return std.fmt.bufPrint(buf, "{s}0x{c}.{x:0>13}p{c}{d}", .{ if (neg) "-" else "", lead, frac, @as(u8, if (e < 0) '-' else '+'), @abs(e) }) catch "nan";
}

/// float.hex(v) (objects[callee_index]: float.hex) of a float: natively;
/// anything else, Python's
export fn zr_float_hex(ctx: *Ctx, node: u32, callee_index: u64, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() != .float) return zr_call_python(ctx, node, callee_index, @ptrCast(&v), 1, out);
    var buf: [32]u8 = undefined;
    const r = value.newStr(floatHex(v.asFloat(), &buf)) orelse return oomFail(ctx, node);
    out.* = Value.obj(.str, &r.head);
    return true;
}

/// len(v.encode("utf-8")): a str's length in bytes (a str is its UTF-8
/// bytes); anything else as Python does it.
export fn zr_utf8_len(ctx: *Ctx, node: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    if (v.kind() == .str) {
        out.* = Value.pint(@intCast(@as(*value.Str, @ptrCast(v.ptr())).len));
        return true;
    }
    gil.ensureAt(@src(), node, null);
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node,&.{v}, &objs)) return failPython(ctx, node);
    defer py.Py_DecRef(objs[0]);
    const encoded = py.c.PyObject_CallMethod(objs[0], "encode", "s", "utf-8") orelse return failPython(ctx, node);
    defer py.Py_DecRef(encoded);
    const n = py.c.PyObject_Size(encoded);
    if (n < 0) return failPython(ctx, node);
    out.* = Value.pint(@intCast(n));
    return true;
}

/// A closure's captured variable: its cell `idx` (of the function called,
/// a host value), what it holds now; Python's NameError for an empty one.
export fn zr_cell(ctx: *Ctx, node: u32, ft: u64, fb: u64, idx: u64, out: *Value) callconv(.c) bool {
    // (a variable of a Python closure's read: no Python code runs (the
    // function's and cell's own attributes), what strict mode allows)
    gil.allowBegin();
    defer gil.allowEnd();
    _ = ft;
    const f: *PyObject = @ptrFromInt(fb);
    const closure = py.c.PyObject_GetAttrString(f, "__closure__") orelse return failPython(ctx, node);
    defer py.Py_DecRef(closure);
    const cell = py.c.PyTuple_GetItem(closure, @intCast(idx)) orelse return failPython(ctx, node);
    const v = py.c.PyObject_GetAttrString(cell, "cell_contents") orelse {
        py.c.PyErr_Clear();
        return failAs(ctx, node, .NameError, null, "free variable referenced before assignment in enclosing scope", .{});
    };
    defer py.Py_DecRef(v);
    out.* = value.fromPython(v) orelse return failPython(ctx, node);
    return true;
}

/// type(v): its class, as the reference mode has it (an I64's zrun.I64, a
/// plain int's int, a record's its class...), without Python.
export fn zr_type(t: u64, bits: u64, out: *Value) callconv(.c) void {
    gil.ensure(@src());
    const v = Value{ .tag = t, .bits = bits };
    const cls: *PyObject = switch (v.kind()) {
        .none => @ptrCast(@alignCast(ph.typeOf(py.Py_None()))),
        .bool => @ptrCast(@alignCast(py.types.typeObject("PyBool_Type"))),
        .int => if (v.isPlain()) @ptrCast(@alignCast(py.types.typeObject("PyLong_Type"))) else types.I64,
        .big => @ptrCast(@alignCast(py.types.typeObject("PyLong_Type"))),
        .float => @ptrCast(@alignCast(py.types.typeObject("PyFloat_Type"))),
        .str => @ptrCast(@alignCast(py.types.typeObject("PyUnicode_Type"))),
        .list => @ptrCast(@alignCast(py.types.typeObject("PyList_Type"))),
        .tuple => @ptrCast(@alignCast(py.types.typeObject("PyTuple_Type"))),
        .dict => @ptrCast(@alignCast(py.types.typeObject("PyDict_Type"))),
        .set => @ptrCast(@alignCast(py.types.typeObject("PySet_Type"))),
        .exc => @as(errors.Kind, @enumFromInt(@as(*value.Exc, @ptrCast(@alignCast(v.ptr()))).kind)).pyClass() orelse @ptrCast(@alignCast(ph.typeOf(py.Py_None()))),
        .bytes => if (@as(*value.Bytes, @ptrCast(@alignCast(v.ptr()))).isPyBytes()) @ptrCast(@alignCast(py.types.typeObject("PyBytes_Type"))) else @import("bytes.zig").BytesType,
        // (a lambda's, a nested def's: Python's function)
        .closure => @import("compile.zig").pyFunctionType() orelse @ptrCast(@alignCast(ph.typeOf(py.Py_None()))),
        .record => @as(*value.Record, @ptrCast(@alignCast(v.ptr()))).rtype.py_class orelse @import("proxies.zig").RecordType,
        .function => @import("objects.zig").FunctionType,
        .node => @import("objects.zig").NodeType,
        .host => @ptrCast(@alignCast(ph.typeOf(@ptrFromInt(v.bits)))),
        else => @ptrCast(@alignCast(ph.typeOf(py.Py_None()))),
    };
    py.Py_IncRef(cls);
    out.* = .{ .tag = @intFromEnum(Tag.host), .bits = @intFromPtr(cls) };
}

/// type(v).__name__: natively for native values (their class's name, as
/// Python's); a Python object's, Python's
export fn zr_type_name(ctx: *Ctx, node: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    const name: []const u8 = switch (v.kind()) {
        .host, .record, .node, .function, .rt => {
            var cls = Value.none_v;
            zr_type(t, bits, &cls);
            defer value.decref(cls);
            const n = value.literal("__name__") orelse return oomFail(ctx, node);
            return zr_getattr(ctx, node, cls.tag, cls.bits, n, out);
        },
        .int => "int",
        else => value.typeName(v),
    };
    const s = value.newStr(name) orelse return oomFail(ctx, node);
    out.* = Value.obj(.str, &s.head);
    return true;
}

/// A module-level name some function assigns (`global`), read when the
/// code runs from the module's dict (objects[globals_index]), then the
/// builtins; Python's NameError if neither has it.
export fn zr_global(ctx: *Ctx, node: u32, globals_index: u64, name: *const value.Str, out: *Value) callconv(.c) bool {
    gil.ensureAt(@src(), node, null);
    const g = ctx.object(globals_index);
    const key = ph.newString(name.bytes()) orelse return failPython(ctx, node);
    defer py.Py_DecRef(key);
    const v = py.c.PyDict_GetItemWithError(g, key) orelse blk: {
        if (py.c.PyErr_Occurred() != null) return failPython(ctx, node);
        const builtins = py.c.PyEval_GetBuiltins() orelse return failPython(ctx, node);
        break :blk py.c.PyDict_GetItemWithError(builtins, key) orelse {
            if (py.c.PyErr_Occurred() != null) return failPython(ctx, node);
            return failAs(ctx, node, .NameError, null, "name '{s}' is not defined", .{name.bytes()});
        };
    };
    // (a reference of ours: the dict's is another, so a list stays itself)
    py.Py_IncRef(v);
    defer py.Py_DecRef(v);
    out.* = value.fromPython(v) orelse return failPython(ctx, node);
    return true;
}

/// A module variable a semantic assigns (`global name`): set in the
/// module's dict, as Python sets it.
export fn zr_set_global(ctx: *Ctx, node: u32, globals_index: u64, name: *const value.Str, t: u64, bits: u64) callconv(.c) bool {
    gil.ensureAt(@src(), node, null);
    const g = ctx.object(globals_index);
    const v = value.toPython(.{ .tag = t, .bits = bits }, ctx.node_maker) orelse return failPython(ctx, node);
    defer py.Py_DecRef(v);
    const key = ph.newString(name.bytes()) orelse return failPython(ctx, node);
    defer py.Py_DecRef(key);
    if (py.c.PyDict_SetItem(g, key, v) != 0) return failPython(ctx, node);
    return true;
}

/// Whether a Python object is of a builtin type (or a subclass of it).
fn ofType(o: *PyObject, comptime name: [:0]const u8) bool {
    return py.c.PyType_IsSubtype(ph.typeOf(o), @ptrCast(@alignCast(py.types.typeObject(name)))) != 0;
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
            9 => ofType(o, "PyBytes_Type"),
            10 => ofType(o, "PyByteArray_Type"),
            11 => ofType(o, "PyComplex_Type"),
            12 => ofType(o, "PySet_Type"),
            13 => ofType(o, "PyFrozenSet_Type"),
            else => false,
        };
    }
    return switch (code) {
        0 => v.kind() == .int or v.kind() == .bool or v.kind() == .big, // int (bool is an int)
        1 => v.kind() == .float,
        2 => v.kind() == .str,
        3 => v.kind() == .bool,
        4 => v.kind() == .list,
        5 => v.kind() == .tuple,
        6 => v.kind() == .dict,
        7 => v.kind() == .none,
        8 => v.kind() == .function,
        9 => v.kind() == .bytes and @as(*value.Bytes, @ptrCast(@alignCast(v.ptr()))).isPyBytes(),
        12 => v.kind() == .set,
        else => false,
    };
}

/// A value formatted for an f-string ({v!conversion:spec}), as Python does.
export fn zr_format(ctx: *Ctx, node: u32, t: u64, bits: u64, conversion: u32, spec: *const value.Str, out: *Value) callconv(.c) bool {
    const v = Value{ .tag = t, .bits = bits };
    // (`{s}` of a str, `{n}` or `{n:d}` of an int, with no conversion or
    // `!s`: natively, as Python formats them; anything else Python's)
    if (conversion == 0 or conversion == 's') {
        const sp = spec.bytes();
        if (v.kind() == .str and sp.len == 0) {
            value.incref(v);
            out.* = v;
            return true;
        }
        if (v.kind() == .int and (sp.len == 0 or std.mem.eql(u8, sp, "d"))) {
            var buf: [24]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{v.asInt()}) catch unreachable;
            const r = value.newStr(s) orelse return oomFail(ctx, node);
            out.* = Value.obj(.str, &r.head);
            return true;
        }
    }
    // (`{x}`, `{x!s}`, `{x!r}` of a float, a bool, None: their str, the same
    // as their repr)
    if ((conversion == 0 or conversion == 's' or conversion == 'r') and spec.len == 0) {
        if (plainStr(v)) |text| {
            var b: [40]u8 = undefined;
            const r = value.newStr(text.of(&b)) orelse return oomFail(ctx, node);
            out.* = Value.obj(.str, &r.head);
            return true;
        }
    }
    // (a spec of ints, floats, strs: natively, the mini-language's)
    if (conversion == 0 and spec.len > 0) {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        defer buf.deinit(allocator);
        if (formatSpec(v, spec.bytes(), &buf)) |_| {
            const r = value.newStr(buf.items) orelse return oomFail(ctx, node);
            out.* = Value.obj(.str, &r.head);
            return true;
        }
    }
    gil.ensureAt(@src(), node, null);
    var objs: [1]*PyObject = undefined;
    if (!objects(ctx, node,&.{v}, &objs)) return failPython(ctx, node);
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

/// format(v, spec) natively, as Python's format-spec mini-language does it
/// for ints, floats and strs ([[fill]align][sign][#][0][width][,_][.prec]
/// [type]); null: a spec or value done otherwise (Python's then, its
/// errors)
fn formatSpec(v: Value, spec: []const u8, out: *std.ArrayListUnmanaged(u8)) ?void {
    // The spec
    var i: usize = 0;
    var fill: []const u8 = " ";
    var alignment: u8 = 0;
    if (spec.len > 0) {
        const n = std.unicode.utf8ByteSequenceLength(spec[0]) catch return null;
        if (spec.len > n and std.mem.indexOfScalar(u8, "<>=^", spec[n]) != null) {
            fill = spec[0..n];
            alignment = spec[n];
            i = n + 1;
        } else if (std.mem.indexOfScalar(u8, "<>=^", spec[0]) != null) {
            alignment = spec[0];
            i = 1;
        }
    }
    var sign: u8 = '-';
    if (i < spec.len and std.mem.indexOfScalar(u8, "+- ", spec[i]) != null) {
        sign = spec[i];
        i += 1;
    }
    var alt = false;
    if (i < spec.len and spec[i] == '#') {
        alt = true;
        i += 1;
    }
    if (i < spec.len and spec[i] == '0') {
        if (alignment == 0) {
            fill = "0";
            alignment = '=';
        }
        i += 1;
    }
    var width: usize = 0;
    while (i < spec.len and std.ascii.isDigit(spec[i])) : (i += 1) width = width * 10 + (spec[i] - '0');
    var group: u8 = 0;
    if (i < spec.len and (spec[i] == ',' or spec[i] == '_')) {
        group = spec[i];
        i += 1;
    }
    var prec: ?usize = null;
    if (i < spec.len and spec[i] == '.') {
        i += 1;
        const from = i;
        var p: usize = 0;
        while (i < spec.len and std.ascii.isDigit(spec[i])) : (i += 1) p = p * 10 + (spec[i] - '0');
        if (i == from) return null;
        prec = p;
    }
    var ty: u8 = 0;
    if (i < spec.len) {
        ty = spec[i];
        i += 1;
    }
    if (i != spec.len) return null;
    var buf: [512]u8 = undefined;
    var body: []const u8 = undefined;
    var negative = false;
    var prefix: []const u8 = "";
    switch (v.kind()) {
        // (a str: its characters, cut to the precision)
        .str => {
            if (ty != 0 and ty != 's') return null;
            if (sign != '-' or alt or group != 0 or alignment == '=') return null;
            const s = @as(*value.Str, @ptrCast(v.ptr())).bytes();
            var end = s.len;
            if (prec) |p| {
                var k: usize = 0;
                end = 0;
                while (end < s.len and k < p) : (k += 1) end += std.unicode.utf8ByteSequenceLength(s[end]) catch 1;
            }
            body = s[0..end];
            if (alignment == 0) alignment = '<';
        },
        .int, .float => {
            const is_int = v.kind() == .int;
            if (alignment == 0) alignment = '>';
            const int_ty = ty == 0 or ty == 'd' or ty == 'x' or ty == 'X' or ty == 'o' or ty == 'b';
            if (is_int and int_ty) {
                if (prec != null) return null;
                const x = v.asInt();
                negative = x < 0;
                const mag: u64 = @abs(x);
                const base: u8 = switch (ty) {
                    'x', 'X' => 16,
                    'o' => 8,
                    'b' => 2,
                    else => 10,
                };
                if (group == ',' and base != 10) return null;
                if (alt and base != 10) prefix = switch (ty) {
                    'x' => "0x",
                    'X' => "0X",
                    'o' => "0o",
                    else => "0b",
                };
                var digits: [80]u8 = undefined;
                const raw = switch (base) {
                    16 => if (ty == 'X') std.fmt.bufPrint(&digits, "{X}", .{mag}) else std.fmt.bufPrint(&digits, "{x}", .{mag}),
                    8 => std.fmt.bufPrint(&digits, "{o}", .{mag}),
                    2 => std.fmt.bufPrint(&digits, "{b}", .{mag}),
                    else => std.fmt.bufPrint(&digits, "{d}", .{mag}),
                } catch return null;
                body = grouped(raw, group, if (base == 10) 3 else 4, &buf) orelse return null;
            } else {
                if (int_ty and ty != 0) return null;
                if (@import("builtin").os.tag != .linux) return null;
                const x: f64 = if (is_int) @floatFromInt(v.asInt()) else v.asFloat();
                negative = std.math.signbit(x) and !std.math.isNan(x);
                const mag = @abs(x);
                var tbuf: [400]u8 = undefined;
                var text: []const u8 = undefined;
                if (ty == 0 and prec == null) {
                    // (no type, no precision: as str() writes it)
                    var fb: [40]u8 = undefined;
                    if (is_int) return null;
                    text = value.floatRepr(mag, &fb);
                    @memcpy(tbuf[0..text.len], text);
                    text = tbuf[0..text.len];
                } else {
                    if (ty == 0) return null;
                    const conv: u8 = if (ty == '%') 'f' else ty;
                    if (std.mem.indexOfScalar(u8, "eEfFgG", conv) == null) return null;
                    var cf: [16]u8 = undefined;
                    const cfs = std.fmt.bufPrintZ(&cf, "%{s}.*{c}", .{ if (alt) "#" else "", conv }) catch return null;
                    const p: c_int = @intCast(prec orelse 6);
                    const got = snprintf(&tbuf, tbuf.len - 1, cfs.ptr, p, if (ty == '%') mag * 100 else mag);
                    if (got < 0 or @as(usize, @intCast(got)) >= tbuf.len - 1) return null;
                    text = tbuf[0..@intCast(got)];
                    if (ty == '%') {
                        tbuf[text.len] = '%';
                        text = tbuf[0 .. text.len + 1];
                    }
                }
                if (group != 0) {
                    // (the integer part's digits grouped)
                    const end = std.mem.indexOfAny(u8, text, ".eE%") orelse text.len;
                    for (text[0..end]) |c| if (!std.ascii.isDigit(c)) return null;
                    const g = grouped(text[0..end], group, 3, &buf) orelse return null;
                    if (g.len + text.len - end > buf.len) return null;
                    @memcpy(buf[g.len..][0 .. text.len - end], text[end..]);
                    body = buf[0 .. g.len + text.len - end];
                } else {
                    @memcpy(buf[0..text.len], text);
                    body = buf[0..text.len];
                }
            }
        },
        else => return null,
    }
    // The sign, the padding
    const sign_s: []const u8 = if (negative) "-" else if (sign == '+') "+" else if (sign == ' ') " " else "";
    var chars: usize = sign_s.len + prefix.len;
    var at: usize = 0;
    while (at < body.len) : (chars += 1) at += std.unicode.utf8ByteSequenceLength(body[at]) catch 1;
    const pad = if (width > chars) width - chars else 0;
    const left: usize, const right: usize = switch (alignment) {
        '<' => .{ 0, pad },
        '^' => .{ pad / 2, pad - pad / 2 },
        else => .{ pad, 0 },
    };
    if (alignment == '=') {
        out.appendSlice(allocator, sign_s) catch return null;
        out.appendSlice(allocator, prefix) catch return null;
        for (0..pad) |_| out.appendSlice(allocator, fill) catch return null;
        out.appendSlice(allocator, body) catch return null;
        return;
    }
    for (0..left) |_| out.appendSlice(allocator, fill) catch return null;
    out.appendSlice(allocator, sign_s) catch return null;
    out.appendSlice(allocator, prefix) catch return null;
    out.appendSlice(allocator, body) catch return null;
    for (0..right) |_| out.appendSlice(allocator, fill) catch return null;
}

/// Digits with a separator every `every` (none: themselves)
fn grouped(digits: []const u8, sep: u8, every: usize, buf: []u8) ?[]const u8 {
    if (sep == 0) {
        if (digits.len > buf.len) return null;
        @memcpy(buf[0..digits.len], digits);
        return buf[0..digits.len];
    }
    const n = digits.len + (digits.len - 1) / every;
    if (n > buf.len) return null;
    var o: usize = n;
    var k: usize = 0;
    var j = digits.len;
    while (j > 0) {
        j -= 1;
        if (k > 0 and k % every == 0) {
            o -= 1;
            buf[o] = sep;
        }
        o -= 1;
        buf[o] = digits[j];
        k += 1;
    }
    return buf[0..n];
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
    "zr_write",    "zr_print",
    "zr_binary",   "zr_compare",    "zr_unary",      "zr_truthy",        "zr_function",
    "zr_call",     "zr_object",     "zr_frame_new",  "zr_frame_release", "zr_free",
    "zr_list",     "zr_tuple",      "zr_dict",       "zr_record",        "zr_is_record",
    "zr_getattr",  "zr_setattr",    "zr_getitem",    "zr_setitem",       "zr_items",
    "zr_wrapping", "zr_read", "zr_native_fail", "zr_call_site", "zr_host_jump", "zr_host_failed", "zr_bury",
    "zr_list_len", "zr_list_at",    "zr_append",     "zr_call_method",   "zr_call_python",
    "zr_is_type",  "zr_global",     "zr_format",     "zr_concat",        "zr_unpack",
    "zr_varargs",  "zr_record_new", "zr_isinstance", "zr_call_seq",      "zr_slice",
    "zr_type",     "zr_builtin",    "zr_range",      "zr_cell",          "zr_frame_of",
    "zr_extend_items", "zr_tail_set", "zr_tail_resolve", "zr_tail_take", "zr_tail_put", "zr_tail_clear",
    "zr_int_base", "zr_utf8_len", "zr_min_max", "zr_math", "zr_huge_int", "zr_dict_view",
    "zr_new_exception", "zr_call_plain", "zr_int_base_of", "zr_float_hex",
    "zr_delitem",
    "zr_closure",
    "zr_set_global",
    "zr_set",
    "zr_to_set",
    "zr_inplace",
    "zr_lib",
    "zr_sorted",
    "zr_type_name",
    "zr_exc_wrap",
};

/// The names compiled code calls them by, and their addresses
pub fn symbols() [helper_names.len]struct { []const u8, usize } {
    var out: [helper_names.len]struct { []const u8, usize } = undefined;
    inline for (helper_names, 0..) |name, i| out[i] = .{ name, @intFromPtr(&@field(@This(), name)) };
    return out;
}
