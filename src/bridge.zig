//! Semantics run as Python inside compiled code (those the compiler can't
//! compile, and native=False ones).
//!
//! The compiled code calls zr_py_semantic for such a node: the Python
//! function gets the node and an rt over the compiled program's frames
//! (zrun.CompiledRuntime). What it asks rt to run (rt.eval/exec/loop of a
//! node) is compiled code too: a thunk (driver.zig), compiled the first
//! time; its variables (rt.load/store) are the frames' slots; functions
//! it makes and calls are the compiled ones.
//!
//! Exceptions cross both ways: a Return, Break or Continue raised by the
//! Python semantic is a status the compiled code acts on (and one from
//! compiled code is raised in Python); errors of the compiled code are
//! zrun.Errors in Python (a semantic can catch them, as in the reference
//! mode), and a rt.Throw or zrun.Error from Python goes up through
//! compiled code (Ctx.pending) to the next Python code, or out of the run.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const helpers = @import("helpers.zig");
const value = @import("value.zig");
const gil = @import("gil.zig");
const types = @import("types.zig");
const program_mod = @import("program.zig");
const grammar_mod = @import("grammar.zig");
const compile_mod = @import("compile.zig");
const driver = @import("driver.zig");

const Value = value.Value;
const Ctx = helpers.Ctx;
const NONE = program_mod.NONE;
const allocator = std.heap.c_allocator;

/// What the bridge needs of the program running (lib.zig fills it)
pub const Link = struct {
    program: *anyopaque,
    compiled: *driver.Compiled,
    data: *program_mod.Data,
    hosts: *PyObject,
    analysis: ?*PyObject,
    path: ?*PyObject,
    context: ?*PyObject,
    /// A node's semantic (its Python function, borrowed), or null
    semantic: *const fn (program: *anyopaque, idx: u32, which: compile_mod.Which) ?*PyObject,
    /// A zrun.Error at a node, with a stack (innermost first): a new
    /// reference, or null with an exception
    error_object: *const fn (program: *anyopaque, idx: u32, message: []const u8, stack: []const helpers.CallEntry) ?*PyObject,
};

/// node.name of a node only known at run time, read from the program's
/// tree as the compiler reads a known node's: true / false (an error), or
/// null for one not read here (Python's Node then: a field whose value an
/// action makes...).
pub fn nodeAttr(ctx: *Ctx, idx: u32, name: *const value.Str, out: *Value) ?bool {
    const link = linkOf(ctx);
    const d = link.data;
    const n = d.nodes[idx];
    const rid = n.ruleId();
    // (what the name is, worked out once: a literal's str is one per name)
    const attr = link.compiled.attrOf(name, d.grammar) orelse return helpers.fail(ctx, idx, "out of memory", .{});
    switch (attr) {
        .field => |field| return fieldValue(ctx, d, idx, rid, field, out),
        .kind, .rule => {
            const is_kind = attr == .kind;
            const s = link.compiled.nameStr(if (is_kind) d.grammar.kind_names[rid] else d.grammar.rule_names[rid], rid, is_kind) orelse return helpers.fail(ctx, idx, "out of memory", .{});
            out.* = Value.obj(.str, &s.head);
        },
        .text => {
            const s = link.compiled.textStr(idx, d) orelse return helpers.fail(ctx, idx, "out of memory", .{});
            value.increfObj(&s.head);
            out.* = Value.obj(.str, &s.head);
        },
        // (plain ints, as the reference mode's Node gives them)
        .start => out.* = Value.pint(n.text_start),
        .end => out.* = Value.pint(n.text_end),
        .line => out.* = Value.pint(d.lineCol(n.text_start).line),
        .column => out.* = Value.pint(d.lineCol(n.text_start).col),
        .index => out.* = Value.pint(idx),
        .parent => {
            const p = d.parents[idx];
            out.* = if (p == program_mod.NONE) Value.none_v else .{ .tag = @intFromEnum(value.Tag.node), .bits = p };
        },
        .children => {
            var count: usize = 0;
            var ch = idx + 1;
            const stop = d.end(idx);
            while (ch < stop) : (ch = d.end(ch)) {
                _ = childValue(d, ch) orelse return null;
                if (!dropped(d, ch)) count += 1;
            }
            const l = value.newList(count) orelse return helpers.fail(ctx, idx, "out of memory", .{});
            ch = idx + 1;
            while (ch < stop) : (ch = d.end(ch)) {
                if (dropped(d, ch)) continue;
                _ = value.listPush(l, childValue(d, ch).?);
            }
            out.* = Value.obj(.list, &l.head);
        },
        .other => return null,
    }
    return true;
}

/// A node's labelled field: its child (None: none), or a list of them (a
/// label that repeats); null for a child an action makes a value of
/// (Python's conversion).
fn fieldValue(ctx: *Ctx, d: *const program_mod.Data, idx: u32, rid: u32, field: u8, out: *Value) ?bool {
    const label = d.grammar.labelOf(rid, field);
    const many = label != null and label.?.many;
    var count: usize = 0;
    var only: Value = Value.none_v;
    var ch = idx + 1;
    const stop = d.end(idx);
    // (the labelled children: nodes; one an action makes a value of:
    // Python's)
    while (ch < stop) : (ch = d.end(ch)) {
        if (d.nodes[ch].fieldId() != field) continue;
        const v = childValue(d, ch) orelse return null;
        if (v.kind() == .none) continue;
        count += 1;
        only = v;
    }
    if (!many and count <= 1) {
        out.* = only;
        return true;
    }
    const l = value.newList(count) orelse return helpers.fail(ctx, idx, "out of memory", .{});
    ch = idx + 1;
    while (ch < stop) : (ch = d.end(ch)) {
        if (d.nodes[ch].fieldId() != field) continue;
        const v = childValue(d, ch).?;
        if (v.kind() != .none) _ = value.listPush(l, v);
    }
    out.* = Value.obj(.list, &l.head);
    return true;
}

fn dropped(d: *const program_mod.Data, ch: u32) bool {
    const crid = d.rule(ch);
    return crid < d.grammar.actions.len and d.grammar.actions[crid] == .drop;
}

/// A child's value as a field: the node (None: dropped), or null for one
/// whose action makes a value (Python's conversion).
fn childValue(d: *const program_mod.Data, ch: u32) ?Value {
    const crid = d.rule(ch);
    const action: grammar_mod.Action = if (crid < d.grammar.actions.len) d.grammar.actions[crid] else .none;
    return switch (action) {
        .none, .class => .{ .tag = @intFromEnum(value.Tag.node), .bits = ch },
        .drop => Value.none_v,
        else => null,
    };
}

fn linkOf(ctx: *Ctx) *const Link {
    return @ptrCast(@alignCast(ctx.link.?));
}

// ======================================================================
// Calling a Python semantic from compiled code
// ======================================================================

/// Run a node's Python semantic (which: 0 eval, 1 exec) in the frames of
/// `frame_slot` (the frame of `owner`): its status (0 error, 1 done: the
/// value in `out`, 2 Return: its value in `out`, 3 Break, 4 Continue).
pub export fn zr_py_semantic(ctx: *Ctx, which: u32, idx: u32, frame_slot: **value.Frame, owner: u32, out: *Value) callconv(.c) i32 {
    gil.ensureAt(@src(), idx, null);
    out.* = Value.none_v;
    helpers.stat("py_semantic", .{});
    const link = linkOf(ctx);
    const w: compile_mod.Which = @enumFromInt(which);
    const func = link.semantic(link.program, idx, w) orelse {
        _ = helpers.fail(ctx, idx, "no semantics for this node", .{});
        return 0;
    };
    const node = ctx.node_maker.make(idx) orelse return fromPythonError(ctx, idx, out);
    defer py.Py_DecRef(node);
    const rt = newRuntime(ctx, frame_slot, owner, idx) orelse return fromPythonError(ctx, idx, out);
    defer {
        // (no use of it after: the frames it sees are the caller's)
        asRuntime(rt).ctx = null;
        py.Py_DecRef(rt);
    }
    const r = py.c.PyObject_CallFunctionObjArgs(func, node, rt, @as(?*PyObject, null)) orelse return fromPythonError(ctx, idx, out);
    defer py.Py_DecRef(r);
    if (w == .eval) {
        // (as rt.eval gives it: an int an I64)
        out.* = (value.fromPython(r) orelse return fromPythonError(ctx, idx, out)).checked();
    }
    return 1;
}

/// rt.eval / rt.exec (which: 0, 1; 2: rt.loop) of a value only known at
/// run time, from compiled code: a node runs (its thunk, or its Python
/// semantic), a list each of its items (eval: the list of their values),
/// anything else is itself (eval). A status as zr_py_semantic's (rt.loop:
/// 1 with True or False in `out`, its Break and Continue taken).
pub export fn zr_run_value(ctx: *Ctx, which: u32, at: u32, tag: u64, bits: u64, frame_slot: **value.Frame, owner: u32, out: *Value) callconv(.c) i32 {
    // (no GIL to start with: a node's thunk run again needs none; runNode
    // takes it where it does)
    out.* = Value.none_v;
    const v = Value{ .tag = tag, .bits = bits };
    const loop = which == 2;
    const w: compile_mod.Which = if (which == 0) .eval else .exec;
    switch (v.kind()) {
        .node => {
            const status = runNode(ctx, w, @intCast(v.bits), frame_slot, owner, out);
            if (!loop) return status;
            return switch (status) {
                1, 4 => blk: {
                    value.decref(out.*);
                    out.* = Value.boolean(true);
                    break :blk 1;
                },
                3 => blk: {
                    out.* = Value.boolean(false);
                    break :blk 1;
                },
                else => status,
            };
        },
        .list, .tuple => {
            const items = if (v.kind() == .list) @as(*value.List, @ptrCast(@alignCast(v.ptr()))).slice() else @as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice();
            const results = if (w == .eval) (value.newList(items.len) orelse {
                _ = helpers.fail(ctx, at, "out of memory", .{});
                return 0;
            }) else null;
            for (items) |item| {
                var r = Value.none_v;
                const status = zr_run_value(ctx, @intFromEnum(w), at, item.tag, item.bits, frame_slot, owner, &r);
                if (status != 1) {
                    if (results) |l| value.decref(Value.obj(.list, &l.head));
                    out.* = r;
                    return status;
                }
                if (results) |l| _ = value.listPush(l, r) else value.decref(r);
            }
            if (results) |l| out.* = Value.obj(.list, &l.head);
            if (loop) out.* = Value.boolean(true);
            return 1;
        },
        else => {
            if (w == .eval) {
                value.incref(v);
                out.* = v;
            }
            if (loop) out.* = Value.boolean(true);
            return 1;
        },
    }
}

/// A node's eval or exec from compiled code that only knows it at run
/// time: its Python semantic, or its thunk (compiled the first time).
fn runNode(ctx: *Ctx, which: compile_mod.Which, idx: u32, frame_slot: **value.Frame, owner: u32, out: *Value) i32 {
    const link = linkOf(ctx);
    // (what it did last time, for the same frames: again)
    const run_ = link.compiled.runOf(idx, which, link.data.nodes.len) orelse {
        _ = helpers.fail(ctx, idx, "out of memory", .{});
        return 0;
    };
    if (run_.thunk) |t| if (run_.owner == owner) return t(ctx, frame_slot.*, out);
    if (run_.python or (!link.data.hasFrame(idx) and link.semantic(link.program, idx, which) != null and isPython(link, idx, which))) {
        run_.python = true;
        return zr_py_semantic(ctx, @intFromEnum(which), idx, frame_slot, owner, out);
    }
    // (its thunk compiled now: what strict mode allows; run outside that)
    const thunk = blk: {
        gil.allowBegin();
        defer gil.allowEnd();
        break :blk link.compiled.thunk(idx, which, owner) orelse return fromPythonError(ctx, idx, out);
    };
    run_.* = .{ .thunk = thunk, .owner = owner };
    return thunk(ctx, frame_slot.*, out);
}

/// The exception a Python semantic (at `idx`) raised, as a status (a
/// Return's value in `out`).
fn fromPythonError(ctx: *Ctx, idx: u32, out: *Value) i32 {
    switch (types.pendingControl()) {
        .ret => {
            const v = types.takeReturn() orelse return pythonFailure(ctx, idx);
            defer py.Py_DecRef(v);
            out.* = (value.fromPython(v) orelse return pythonFailure(ctx, idx)).checked();
            return 2;
        },
        .brk => {
            py.c.PyErr_Clear();
            return 3;
        },
        .cont => {
            py.c.PyErr_Clear();
            return 4;
        },
        // (the bridge's rt.tail_call makes its call: none comes this way)
        .none, .tail => {},
    }
    return pythonFailure(ctx, idx);
}

/// A Python exception (being raised) as the run's error at `idx`: a Throw
/// or a zrun.Error kept to raise again in Python (Ctx.pending); anything
/// else worded as the reference mode words it. 0.
pub fn pythonFailure(ctx: *Ctx, idx: u32) i32 {
    if (py.c.PyErr_Occurred() == null) {
        _ = helpers.fail(ctx, idx, "error", .{});
        return 0;
    }
    if (py.c.PyErr_ExceptionMatches(types.Throw) != 0 or py.c.PyErr_ExceptionMatches(types.Error) != 0) {
        var t: ?*PyObject = null;
        var v: ?*PyObject = null;
        var tb: ?*PyObject = null;
        py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        if (t) |o| py.Py_DecRef(o);
        if (tb) |o| py.Py_DecRef(o);
        const exc = v orelse {
            _ = helpers.fail(ctx, idx, "error", .{});
            return 0;
        };
        // (a Throw: where it was raised kept with it, its zrun.Error)
        if (py.c.PyObject_IsInstance(exc, types.Throw) == 1 and py.c.PyObject_HasAttrString(exc, "_zrun_error") == 0) recordThrow(ctx, idx, exc);
        if (ctx.pending) |p| py.Py_DecRef(p);
        ctx.pending = exc;
        _ = helpers.fail(ctx, idx, "a Python exception", .{});
        return 0;
    }
    // (any other exception: kept with the error, its message the error's)
    if (!ctx.failed) {
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
    const msg = helpers.pythonMessage() orelse {
        py.c.PyErr_Clear();
        _ = helpers.fail(ctx, idx, "error", .{});
        return 0;
    };
    defer py.Py_DecRef(msg);
    const text = ph.utf8(msg, "message") orelse {
        py.c.PyErr_Clear();
        _ = helpers.fail(ctx, idx, "error", .{});
        return 0;
    };
    _ = helpers.fail(ctx, idx, "{s}", .{text});
    return 0;
}

/// A Throw's zrun.Error (its message: its own, or str(value)), at the node
/// with the calls being run.
fn recordThrow(ctx: *Ctx, idx: u32, exc: *PyObject) void {
    const link = linkOf(ctx);
    const m = py.c.PyObject_GetAttrString(exc, "message") orelse return py.c.PyErr_Clear();
    defer py.Py_DecRef(m);
    const text_obj = if (m != py.Py_None()) py.c.PyObject_Str(m) else blk: {
        const val = py.c.PyObject_GetAttrString(exc, "value") orelse return py.c.PyErr_Clear();
        defer py.Py_DecRef(val);
        break :blk py.c.PyObject_Str(val);
    };
    const t = text_obj orelse return py.c.PyErr_Clear();
    defer py.Py_DecRef(t);
    const text = ph.utf8(t, "message") orelse return py.c.PyErr_Clear();
    const stack = callStack(ctx) orelse return py.c.PyErr_Clear();
    defer allocator.free(stack);
    const err = link.error_object(link.program, idx, text, stack) orelse return py.c.PyErr_Clear();
    defer py.Py_DecRef(err);
    if (py.c.PyObject_SetAttrString(exc, "_zrun_error", err) != 0) py.c.PyErr_Clear();
}

/// The calls being run, innermost first.
fn callStack(ctx: *Ctx) ?[]helpers.CallEntry {
    const n = ctx.depth;
    const out = allocator.alloc(helpers.CallEntry, n) catch return null;
    for (ctx.calls[0..n], 0..) |e, i| out[n - 1 - i] = e;
    return out;
}

/// The compiled code's error (its status was 0) raised in Python: the
/// exception kept going up (Ctx.pending), or a zrun.Error made of it; the
/// run's error forgotten (Python code has it now). Null.
fn raiseCompiledError(ctx: *Ctx) ?*PyObject {
    if (ctx.pending) |p| {
        ctx.pending = null;
        const t: *PyObject = @ptrCast(@alignCast(p.ob_type));
        py.c.PyErr_SetObject(t, p);
        py.Py_DecRef(p);
        ctx.clearError();
        return null;
    }
    // (a Python exception behind it: raised as itself, as in the reference
    // mode)
    if (ctx.exc != null or ctx.exc_class != null) {
        const e = ctx.exceptionOf() orelse return null;
        py.c.PyErr_SetObject(@ptrCast(@alignCast(ph.typeOf(e))), e);
        py.Py_DecRef(e);
        ctx.clearError();
        return null;
    }
    const link = linkOf(ctx);
    const err = link.error_object(link.program, ctx.err_node, ctx.err_msg.items, ctx.err_stack.items) orelse return null;
    defer py.Py_DecRef(err);
    py.c.PyErr_SetObject(types.Error, err);
    ctx.clearError();
    return null;
}

// ======================================================================
// zrun.CompiledRuntime: rt in a semantic run as Python
// ======================================================================

pub var RuntimeType: *PyObject = undefined;

const RuntimeObject = extern struct {
    ob_base: py.c.PyObject,
    /// Null once its semantic returned
    ctx: ?*Ctx,
    /// The frame of the code it runs in (a slot: rt.fresh replaces it)
    frame_slot: ?**value.Frame,
    owner: u32,
    /// The node whose semantic it is (errors)
    at: u32,
    /// The frame, held (an rt compiled code passes to Python: frame_slot
    /// is this field)
    own_frame: ?*value.Frame = null,
};

fn asRuntime(o: *PyObject) *RuntimeObject {
    return @ptrCast(@alignCast(o));
}

fn newRuntime(ctx: *Ctx, frame_slot: **value.Frame, owner: u32, at: u32) ?*PyObject {
    const alloc: py.c.allocfunc = @ptrCast(py.c.PyType_GetSlot(@ptrCast(RuntimeType), py.c.Py_tp_alloc));
    const obj = alloc.?(@ptrCast(RuntimeType), 0) orelse return null;
    const r = asRuntime(obj);
    r.ctx = ctx;
    r.frame_slot = frame_slot;
    r.owner = owner;
    r.at = at;
    r.own_frame = null;
    return obj;
}

/// `raise exc` in compiled code (an exception object or class: a rt.Throw,
/// a ValueError...): raised as a Python semantic's exception is (a Throw
/// goes up to whoever catches it, anything else is the run's error, worded
/// as the reference mode does). Always false.
pub export fn zr_raise(ctx: *Ctx, at: u32, t: u64, bits: u64) callconv(.c) bool {
    // (an error: Python's exceptions still, what strict mode allows)
    gil.allowBegin();
    defer gil.allowEnd();
    const v = Value{ .tag = t, .bits = bits };
    const o = value.toPython(v, ctx.node_maker) orelse return pythonFailure(ctx, at) != 0;
    defer py.Py_DecRef(o);
    const base = py.c.PyExc_BaseException;
    const is_type = py.c.PyObject_IsInstance(o, @ptrCast(@alignCast(py.types.typeObject("PyType_Type")))) == 1;
    if (is_type and py.c.PyObject_IsSubclass(o, base) == 1) {
        py.c.PyErr_SetNone(o);
    } else if (py.c.PyObject_IsInstance(o, base) == 1) {
        py.c.PyErr_SetObject(@ptrCast(@alignCast(ph.typeOf(o))), o);
    } else {
        ph.raise(py.PyExc_TypeError(), "exceptions must derive from BaseException", .{});
    }
    return pythonFailure(ctx, at) != 0;
}

/// `raise exc from cause` in compiled code: exc (a class made an instance)
/// with its __cause__ (None: no context shown, as `from None`), raised as
/// zr_raise raises it. Always false.
pub export fn zr_raise_from(ctx: *Ctx, at: u32, t: u64, bits: u64, ct: u64, cb: u64) callconv(.c) bool {
    gil.allowBegin();
    defer gil.allowEnd();
    const o = value.toPython(.{ .tag = t, .bits = bits }, ctx.node_maker) orelse return pythonFailure(ctx, at) != 0;
    defer py.Py_DecRef(o);
    const cause = value.toPython(.{ .tag = ct, .bits = cb }, ctx.node_maker) orelse return pythonFailure(ctx, at) != 0;
    defer py.Py_DecRef(cause);
    const base = py.c.PyExc_BaseException;
    const is_type = py.c.PyObject_IsInstance(o, @ptrCast(@alignCast(py.types.typeObject("PyType_Type")))) == 1;
    // (the exception: a class's instance, made now)
    const exc = if (is_type and py.c.PyObject_IsSubclass(o, base) == 1) (py.c.PyObject_CallNoArgs(o) orelse return pythonFailure(ctx, at) != 0) else blk: {
        if (py.c.PyObject_IsInstance(o, base) != 1) {
            ph.raise(py.PyExc_TypeError(), "exceptions must derive from BaseException", .{});
            return pythonFailure(ctx, at) != 0;
        }
        py.Py_IncRef(o);
        break :blk o;
    };
    defer py.Py_DecRef(exc);
    // (its cause: an exception, a class's instance, or None)
    const cause_exc: ?*PyObject = if (cause == py.Py_None()) null else if (py.c.PyObject_IsInstance(cause, base) == 1) blk: {
        py.Py_IncRef(cause);
        break :blk cause;
    } else if (py.c.PyObject_IsInstance(cause, @ptrCast(@alignCast(py.types.typeObject("PyType_Type")))) == 1 and py.c.PyObject_IsSubclass(cause, base) == 1)
        (py.c.PyObject_CallNoArgs(cause) orelse return pythonFailure(ctx, at) != 0)
    else {
        ph.raise(py.PyExc_TypeError(), "exception causes must derive from BaseException", .{});
        return pythonFailure(ctx, at) != 0;
    };
    // (PyException_SetCause takes the reference; it marks the context
    // suppressed too)
    py.c.PyException_SetCause(exc, cause_exc);
    if (cause_exc == null) if (py.c.PyObject_SetAttrString(exc, "__suppress_context__", py.Py_True()) != 0) return pythonFailure(ctx, at) != 0;
    py.c.PyErr_SetObject(@ptrCast(@alignCast(ph.typeOf(exc))), exc);
    return pythonFailure(ctx, at) != 0;
}

/// `except cls:` in compiled code: whether the error being raised is one
/// (objects[cls_index]: a class or a tuple of them). An error of the
/// compiled code itself is a zrun.Error, as Python code above would see it.
pub export fn zr_exc_matches(ctx: *Ctx, cls_index: u64) callconv(.c) bool {
    gil.allowBegin();
    defer gil.allowEnd();
    const cls = ctx.object(cls_index);
    const r = if (ctx.pending orelse ctx.exc) |e|
        py.c.PyObject_IsInstance(e, cls)
    else
        py.c.PyObject_IsSubclass(ctx.exc_class orelse types.Error, cls);
    if (r < 0) py.c.PyErr_Clear();
    return r == 1;
}

/// The error being raised, caught (`except ... as e`): the exception
/// object in `out` (a host value), the error cleared.
pub export fn zr_exc_catch(ctx: *Ctx, at: u32, out: *Value) callconv(.c) bool {
    gil.allowBegin();
    defer gil.allowEnd();
    var exc: *PyObject = undefined;
    if (ctx.pending != null or ctx.exc != null or ctx.exc_class != null) {
        exc = ctx.exceptionOf() orelse {
            py.c.PyErr_Clear();
            ctx.clearError();
            return helpers.fail(ctx, at, "out of memory", .{});
        };
    } else {
        const link = linkOf(ctx);
        exc = link.error_object(link.program, ctx.err_node, ctx.err_msg.items, ctx.err_stack.items) orelse {
            py.c.PyErr_Clear();
            ctx.clearError();
            return helpers.fail(ctx, at, "out of memory", .{});
        };
    }
    ctx.clearError();
    out.* = .{ .tag = @intFromEnum(value.Tag.host), .bits = @intFromPtr(exc) };
    return true;
}

/// An rt.Return, Break or Continue caught as an exception (zr_exc_catch's,
/// an owned reference) by an except of the jump only (Gen.tryStmt): its
/// value in `out` (rt.Return(value)'s; None), the exception given up.
pub export fn zr_control_value(ctx: *Ctx, at: u32, t: u64, bits: u64, out: *Value) callconv(.c) bool {
    // (an error's way: what strict mode allows)
    gil.allowBegin();
    defer gil.allowEnd();
    const exc: *PyObject = @ptrFromInt(bits);
    _ = t;
    defer py.Py_DecRef(exc);
    const args = py.c.PyObject_GetAttrString(exc, "args") orelse return pythonFailure(ctx, at) != 0;
    defer py.Py_DecRef(args);
    if (py.c.PyTuple_Size(args) < 1) {
        out.* = Value.none_v;
        return true;
    }
    out.* = value.fromBorrowed(py.c.PyTuple_GetItem(args, 0).?) orelse return pythonFailure(ctx, at) != 0;
    return true;
}

/// A Python function compiled code calls (`f(args)`, a library function
/// of the language...): by its compiled code (made the first time), the
/// rt values among the arguments giving it the frames to run in.
/// `checked`: through rt.call (ints handed over as I64s, as it does). True
/// / false (an error); null if it can't be compiled (Python runs it).
pub fn compiledCall(ctx: *Ctx, node: u32, callee: *PyObject, args: []const Value, checked: bool, out: *Value) ?bool {
    const link = ctx.link orelse return null;
    const lk: *const Link = @ptrCast(@alignCast(link));
    if (args.len > 63) return null;
    const pt = compile_mod.pyFunctionType() orelse return null;
    if (ph.typeOf(callee) != @as(*py.c.PyTypeObject, @ptrCast(@alignCast(pt)))) return null;
    var mask: u64 = 0;
    var frame: ?*value.Frame = null;
    var owner: u32 = 0;
    var given: [64]Value = undefined;
    var n: usize = 0;
    for (args, 0..) |a, i| {
        if (a.kind() == .rt) {
            mask |= @as(u64, 1) << @intCast(i);
            frame = @ptrFromInt(a.bits);
            owner = a.rtOwner();
            continue;
        }
        given[n] = if (checked) a.checked() else a;
        n += 1;
    }
    // (a function called before, not a closure: its code by the function,
    // no attribute looked up (the program keeps the function, and so its
    // address, for the code); a closure's is by its code object)
    const known = lk.compiled.called.get(.{ .func = callee, .nargs = args.len, .rt_mask = mask });
    var closure = false;
    const code = (if (known) |k| k else blk: {
        // (the first call: its code compiled now, what strict mode allows;
        // the code itself runs outside that)
        gil.allowBegin();
        defer gil.allowEnd();
        // (a closure: one code for all of them, the function given last)
        const closure_obj = py.c.PyObject_GetAttrString(callee, "__closure__") orelse {
            py.c.PyErr_Clear();
            return null;
        };
        py.Py_DecRef(closure_obj);
        closure = closure_obj != py.Py_None();
        break :blk lk.compiled.calledCode(callee, args.len, mask, closure);
    }) orelse {
        if (helpers.collecting) {
            gil.ensureAt(@src(), node, callee);
            if (ph.attr(callee, "__qualname__")) |nm| {
                defer py.Py_DecRef(nm);
                helpers.stat("python function {s}", .{ph.utf8(nm, "name") orelse "?"});
            }
            py.c.PyErr_Clear();
        }
        return null;
    };
    if (closure) {
        given[n] = .{ .tag = @intFromEnum(value.Tag.host), .bits = @intFromPtr(callee) };
        n += 1;
    }
    const status = code(ctx, frame, &given, node, owner, null, null, out);
    switch (status) {
        1 => {
            if (checked) out.* = out.*.checked();
            return true;
        },
        // (rt.call's: a host function's failure, worded as the reference
        // mode words it)
        0 => return if (checked) helpers.hostCodeFailed(ctx, callee) else false,
        else => {
            // (rt.Return, Break, Continue out of it: as the reference mode
            // raises them out of a call, an error here)
            return helpers.fail(ctx, node, "rt.Return, rt.Break or rt.Continue raised out of a function called with rt.call", .{});
        },
    }
}

/// The compiled code of a Python function (not a closure) compiled code
/// calls with `nargs` arguments of its values (no rt values): made the
/// first time; null if it can't have one (Python runs it). With the GIL.
pub fn hostCode(ctx: *Ctx, callee: *PyObject, nargs: usize) ?@import("driver.zig").Helper {
    const link = ctx.link orelse return null;
    const lk: *const Link = @ptrCast(@alignCast(link));
    if (nargs > 63) return null;
    const pt = compile_mod.pyFunctionType() orelse return null;
    if (ph.typeOf(callee) != @as(*py.c.PyTypeObject, @ptrCast(@alignCast(pt)))) return null;
    const closure_obj = py.c.PyObject_GetAttrString(callee, "__closure__") orelse {
        py.c.PyErr_Clear();
        return null;
    };
    py.Py_DecRef(closure_obj);
    if (closure_obj != py.Py_None()) return null;
    return lk.compiled.calledCode(callee, nargs, 0, false) orelse {
        py.c.PyErr_Clear();
        return null;
    };
}

/// obj.name(args) where obj.name is a Python function, or a bound method
/// of one (its object first): by its compiled code (compiledCall).
/// `r.name(args)` of a record whose class defines `name` as a plain
/// function: its compiled code called with the record as self, the method
/// found once per class (no Python object made for the record, the bound
/// method or the name); null if not so (Python's way then). With the GIL.
pub fn recordMethod(ctx: *Ctx, node: u32, rec: Value, name: *const value.Str, args: []const Value, out: *Value) ?bool {
    const link = ctx.link orelse return null;
    const lk: *const Link = @ptrCast(@alignCast(link));
    if (args.len >= 63) return null;
    const r: *value.Record = @ptrCast(@alignCast(rec.ptr()));
    const key: driver.Compiled.MethodKey = .{ .rtype = r.rtype, .name = name };
    const found: ?*PyObject = if (lk.compiled.methods.get(key)) |f| f else blk: {
        // (found once per class: what strict mode allows)
        gil.allowBegin();
        defer gil.allowEnd();
        const f = classFunction(r.rtype.py_class orelse return null, name.bytes());
        lk.compiled.methods.put(allocator, key, f) catch return null;
        break :blk f;
    };
    const func = found orelse return null;
    var all: [64]Value = undefined;
    all[0] = rec;
    @memcpy(all[1 .. args.len + 1], args);
    return compiledCall(ctx, node, func, all[0 .. args.len + 1], false, out);
}

/// What `cls.name` is in the class (through its bases) if it's a plain
/// function (a method), else null (a staticmethod, a classmethod, a
/// property, anything else, or nothing). Borrowed: the class keeps it.
pub fn classFunction(cls: *PyObject, name: []const u8) ?*PyObject {
    const pt = compile_mod.pyFunctionType() orelse return null;
    const mro = py.c.PyObject_GetAttrString(cls, "__mro__") orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(mro);
    const key = ph.newString(name) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(key);
    const n = py.c.PyTuple_Size(mro);
    var i: py.c.Py_ssize_t = 0;
    while (i < n) : (i += 1) {
        const c = py.c.PyTuple_GetItem(mro, i) orelse return null;
        const d = py.c.PyObject_GetAttrString(c, "__dict__") orelse {
            py.c.PyErr_Clear();
            return null;
        };
        defer py.Py_DecRef(d);
        const found = py.c.PyObject_GetItem(d, key) orelse {
            py.c.PyErr_Clear();
            continue;
        };
        // (the class's own reference keeps it: the class lives as long as
        // the compiled program's record type)
        py.Py_DecRef(found);
        return if (ph.typeOf(found) == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(pt)))) found else null;
    }
    return null;
}

pub fn compiledMethod(ctx: *Ctx, node: u32, m: *PyObject, args: []const Value, checked: bool, out: *Value) ?bool {
    if (args.len >= 63) return null;
    const pt = compile_mod.pyMethodType() orelse return null;
    if (ph.typeOf(m) != @as(*py.c.PyTypeObject, @ptrCast(@alignCast(pt)))) return compiledCall(ctx, node, m, args, checked, out);
    // (a bound method: its function and object as Python has them)
    gil.ensureAt(@src(), node, m);
    const func = py.c.PyObject_GetAttrString(m, "__func__") orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(func);
    const self_obj = py.c.PyObject_GetAttrString(m, "__self__") orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(self_obj);
    var all: [64]Value = undefined;
    all[0] = value.fromPython(self_obj) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer value.decref(all[0]);
    @memcpy(all[1 .. args.len + 1], args);
    return compiledCall(ctx, node, func, all[0 .. args.len + 1], checked, out);
}

/// The compiled run going on (the innermost): what an rt value reaching
/// Python runs in
pub threadlocal var current: ?*Ctx = null;

/// An rt value as Python sees it: an rt object over its frames (a new
/// reference; null with an exception).
pub fn runtimeObject(v: Value) ?*PyObject {
    const ctx = current orelse {
        ph.raise(py.PyExc_RuntimeError(), "an rt outside the run it belongs to", .{});
        return null;
    };
    const frame: *value.Frame = @ptrFromInt(v.bits);
    var slot: *value.Frame = frame;
    const obj = newRuntime(ctx, &slot, v.rtOwner(), 0) orelse return null;
    const r = asRuntime(obj);
    value.increfObj(&frame.head);
    r.own_frame = frame;
    r.frame_slot = &r.own_frame.?;
    return obj;
}

/// An rt object of the run going on, back: the rt value of its frames;
/// null for anything else.
pub fn runtimeValue(o: *PyObject) ?Value {
    if (o.ob_type != @as(*py.c.PyTypeObject, @ptrCast(@alignCast(RuntimeType)))) return null;
    const r = asRuntime(o);
    if (r.ctx == null or r.ctx != current) return null;
    const slot = r.frame_slot orelse return null;
    return Value.rt(slot.*, r.owner);
}

/// `rt` as a value compiled code passes (to a Python function: f(rt,
/// node...)): an rt over the frames of the code there, holding its frame.
pub export fn zr_runtime(ctx: *Ctx, at: u32, frame: *value.Frame, owner: u32, out: *Value) callconv(.c) bool {
    gil.ensureAt(@src(), at, null);
    var slot: *value.Frame = frame;
    const obj = newRuntime(ctx, &slot, owner, at) orelse {
        py.c.PyErr_Clear();
        return helpers.fail(ctx, at, "out of memory", .{});
    };
    const r = asRuntime(obj);
    value.increfObj(&frame.head);
    r.own_frame = frame;
    r.frame_slot = &r.own_frame.?;
    out.* = .{ .tag = @intFromEnum(value.Tag.host), .bits = @intFromPtr(obj) };
    return true;
}

fn live(self: ?*PyObject) ?*RuntimeObject {
    const r = asRuntime(self.?);
    if (r.ctx == null) {
        ph.raise(py.PyExc_RuntimeError(), "this rt belongs to a semantic that has returned", .{});
        return null;
    }
    return r;
}

fn ref(o: *PyObject) *PyObject {
    py.Py_IncRef(o);
    return o;
}

fn none() *PyObject {
    return ref(py.Py_None());
}

/// A node object's index (of this program), or null with TypeError.
fn nodeIndex(r: *RuntimeObject, o: *PyObject, what: []const u8) ?u32 {
    if (@import("objects.zig").asNode(o)) |n| return n.idx;
    _ = r;
    ph.raise(py.PyExc_TypeError(), "{s} must be a node", .{what});
    return null;
}

// -- running nodes --

/// rt.eval / rt.exec / rt.loop of x: a node (its compiled code, or its
/// Python semantic), a list of them (each), anything else (itself, for
/// eval).
fn run(r: *RuntimeObject, x: *PyObject, which: compile_mod.Which, loop: bool) ?*PyObject {
    if (py.PyList_Check(x) or py.PyTuple_Check(x)) {
        const seq = py.c.PySequence_Fast(x, "") orelse return null;
        defer py.Py_DecRef(seq);
        const n: usize = @intCast(py.c.PySequence_Size(seq));
        const out = if (which == .eval) (py.c.PyList_New(@intCast(n)) orelse return null) else null;
        errdefer if (out) |o| py.Py_DecRef(o);
        for (0..n) |i| {
            const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse {
                if (out) |o| py.Py_DecRef(o);
                return null;
            };
            defer py.Py_DecRef(item);
            const v = run(r, item, which, false) orelse {
                if (out) |o| py.Py_DecRef(o);
                return null;
            };
            if (out) |o| _ = py.c.PyList_SetItem(o, @intCast(i), v) else py.Py_DecRef(v);
        }
        return out orelse none();
    }
    const n = @import("objects.zig").asNode(x) orelse {
        if (which == .eval) return ref(x);
        return none();
    };
    const ctx = r.ctx.?;
    const link = linkOf(ctx);
    const idx = n.idx;
    // (a node with a Python semantic and no frame of its own: called
    // directly; anything else through its compiled code)
    _ = link;
    var out = Value.none_v;
    const status = runNode(ctx, which, idx, r.frame_slot.?, r.owner, &out);
    switch (status) {
        1 => {
            defer value.decref(out);
            if (loop) return ref(py.Py_True());
            if (which == .exec) return none();
            return value.toPython(out, ctx.node_maker);
        },
        2 => {
            defer value.decref(out);
            const v = value.toPython(out, ctx.node_maker) orelse return null;
            defer py.Py_DecRef(v);
            const exc = py.c.PyObject_CallFunctionObjArgs(types.Return, v, @as(?*PyObject, null)) orelse return null;
            defer py.Py_DecRef(exc);
            py.c.PyErr_SetObject(types.Return, exc);
            return null;
        },
        3 => {
            if (loop) return ref(py.Py_False());
            py.c.PyErr_SetNone(types.Break);
            return null;
        },
        4 => {
            if (loop) return ref(py.Py_True());
            py.c.PyErr_SetNone(types.Continue);
            return null;
        },
        // (an rt.tail_call: Python's in between, no frame to leave: the
        // call made here, its result returned)
        5 => {
            out = .{ .tag = helpers.TAIL_TAG, .bits = 0 };
            if (!helpers.zr_tail_resolve(ctx, r.at, &out)) return raiseCompiledError(ctx);
            defer value.decref(out);
            const v = value.toPython(out, ctx.node_maker) orelse return null;
            defer py.Py_DecRef(v);
            const exc = py.c.PyObject_CallFunctionObjArgs(types.Return, v, @as(?*PyObject, null)) orelse return null;
            defer py.Py_DecRef(exc);
            py.c.PyErr_SetObject(types.Return, exc);
            return null;
        },
        else => return raiseCompiledError(ctx),
    }
}

fn isPython(link: *const Link, idx: u32, which: compile_mod.Which) bool {
    return link.compiled.compiler.semanticOf(idx, which) == .python;
}

fn rtEval(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    return run(r, x.?, .eval, false);
}

fn rtExec(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    return run(r, x.?, .exec, false);
}

fn rtLoop(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    return run(r, x.?, .exec, true);
}

// -- variables --

/// The frame of `home` from the runtime's (through the frames' parents).
fn frameOf(r: *RuntimeObject, home: u32, at: u32) ?*value.Frame {
    const link = linkOf(r.ctx.?);
    const c = &link.compiled.compiler;
    var frame: ?*value.Frame = r.frame_slot.?.*;
    var owner = r.owner;
    while (owner != home) {
        if (owner == NONE) {
            failHere(r, at, "a variable isn't reachable from here");
            return null;
        }
        frame = frame.?.parent;
        owner = c.ownerOf(owner);
    }
    return frame;
}

/// A runtime error at a node, raised as a zrun.Error.
fn failHere(r: *RuntimeObject, at: u32, comptime fmt: []const u8) void {
    _ = helpers.fail(r.ctx.?, at, fmt, .{});
    _ = raiseCompiledError(r.ctx.?);
}

fn varSlot(r: *RuntimeObject, idx: u32) ?*Value {
    const link = linkOf(r.ctx.?);
    const d = link.data;
    const si = d.symbolIndex(idx) orelse {
        const msg = std.fmt.allocPrint(allocator, "'{s}' is not a variable", .{d.text(idx)}) catch return null;
        defer allocator.free(msg);
        _ = helpers.fail(r.ctx.?, idx, "{s}", .{msg});
        _ = raiseCompiledError(r.ctx.?);
        return null;
    };
    const frame = frameOf(r, d.homeOf(si), idx) orelse return null;
    const slot = link.compiled.compiler.slot_of.get(si).?;
    return &frame.slots()[slot];
}

fn rtLoad(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "a variable's name") orelse return null;
    const link = linkOf(r.ctx.?);
    const d = link.data;
    if (d.symbolIndex(idx)) |si| {
        if (d.syms[si].builtin) {
            const key = ph.newString(d.syms[si].name) orelse return null;
            defer py.Py_DecRef(key);
            return ref(py.c.PyDict_GetItem(link.hosts, key) orelse py.Py_None());
        }
    }
    const slot = varSlot(r, idx) orelse return null;
    if (slot.tag == helpers.UNSET) {
        const msg = std.fmt.allocPrint(allocator, "'{s}' has no value yet", .{d.text(idx)}) catch return null;
        defer allocator.free(msg);
        _ = helpers.fail(r.ctx.?, idx, "{s}", .{msg});
        return raiseCompiledError(r.ctx.?);
    }
    return value.toPython(slot.*, r.ctx.?.node_maker);
}

fn rtStore(self: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    var name: ?*PyObject = null;
    var v: ?*PyObject = null;
    if (py.c.PyArg_UnpackTuple(args, "store", 2, 2, &name, &v) == 0) return null;
    const idx = nodeIndex(r, name.?, "a variable's name") orelse return null;
    const slot = varSlot(r, idx) orelse return null;
    // (stored as rt.store does: an int an I64)
    const nv = (value.fromPython(v.?) orelse return null).checked();
    const old = slot.*;
    slot.* = nv;
    // (a top-level variable's function, read without a reference: kept
    // till the code's done)
    const d = linkOf(r.ctx.?).data;
    const top = if (d.symbolIndex(idx)) |si| d.homeOf(si) == program_mod.NONE else false;
    if (top) r.ctx.?.bury(old) else if (old.tag != helpers.UNSET) value.decref(old);
    return none();
}

/// rt.load(node), rt.store(node, v) on an rt value compiled code handed
/// over (a function's parameter, a variable: its frames and their owner),
/// natively (zr_call_method), as CompiledRuntime's do it; true / false (an
/// error at `at`); null for another method (Python's).
pub fn rtValueMethod(ctx: *Ctx, at: u32, rv: Value, name: []const u8, args: []const Value, out: *Value) ?bool {
    const store = std.mem.eql(u8, name, "store") and args.len == 2;
    if (!store and !(std.mem.eql(u8, name, "load") and args.len == 1)) return null;
    if (args[0].kind() != .node) return null;
    const link = linkOf(ctx);
    const c = &link.compiled.compiler;
    const d = link.data;
    const idx: u32 = @intCast(args[0].bits);
    const si = d.symbolIndex(idx) orelse return helpers.fail(ctx, idx, "'{s}' is not a variable", .{d.text(idx)});
    // (a builtin's name: its host function, Python's)
    if (d.syms[si].builtin) return null;
    var frame: ?*value.Frame = @ptrFromInt(rv.bits);
    var owner = rv.rtOwner();
    const home = d.homeOf(si);
    while (owner != home) {
        if (owner == NONE) return helpers.fail(ctx, at, "a variable isn't reachable from here", .{});
        frame = frame.?.parent;
        owner = c.ownerOf(owner);
    }
    const slot = &frame.?.slots()[c.slot_of.get(si).?];
    if (!store) {
        if (slot.tag == helpers.UNSET) return helpers.fail(ctx, idx, "'{s}' has no value yet", .{d.text(idx)});
        value.incref(slot.*);
        out.* = slot.*.checked();
        return true;
    }
    // (stored as rt.store does: an int an I64)
    const nv = args[1].checked();
    value.incref(nv);
    const old = slot.*;
    slot.* = nv;
    // (a top-level variable's function, read without a reference: kept
    // till the code's done)
    if (home == NONE) ctx.bury(old) else if (old.tag != helpers.UNSET) value.decref(old);
    out.* = Value.none_v;
    return true;
}

// -- functions --

fn rtFunction(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const fnode = nodeIndex(r, x.?, "a function's node") orelse return null;
    const ctx = r.ctx.?;
    const link = linkOf(ctx);
    const c = &link.compiled.compiler;
    const d = link.data;
    const fspec = c.specOf(fnode) orelse {
        _ = helpers.fail(ctx, fnode, "this node isn't a function kind (Language.function)", .{});
        return raiseCompiledError(ctx);
    };
    const addr = link.compiled.functionAddr(fnode) orelse return null;
    const env = frameOf(r, c.ownerOf(fnode), fnode) orelse return null;
    const name_text = if (fspec.name != 0) if (program_mod.labelled(d, fnode, fspec.name)) |nn| d.text(nn) else "<anonymous>" else "<anonymous>";
    const name = value.newStr(name_text) orelse return py.c.PyErr_NoMemory();
    defer value.decref(Value.obj(.str, &name.head));
    var nparams: u16 = 0;
    if (fspec.params != 0) {
        var ch = fnode + 1;
        const stop = d.end(fnode);
        while (ch < stop) : (ch = d.end(ch)) {
            if (d.nodes[ch].fieldId() == fspec.params) nparams += 1;
        }
    }
    const flags = helpers.FunctionFlags{ .nparams = nparams, .missing_none = fspec.missing == .none, .extra = switch (fspec.extra) {
        .@"error" => .@"error",
        .drop => .drop,
        .keep => .keep,
    } };
    var out = Value.none_v;
    if (!helpers.zr_function(ctx, @ptrFromInt(addr), env, fnode, name, flags.word(), &out)) return raiseCompiledError(ctx);
    defer value.decref(out);
    return value.toPython(out, ctx.node_maker);
}

fn rtCall(self: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const ctx = r.ctx.?;
    var f: ?*PyObject = null;
    var call_args: ?*PyObject = null;
    var receiver: ?*PyObject = null;
    var kwlist = [_:null]?[*:0]u8{ @constCast("f"), @constCast("args"), @constCast("receiver"), null };
    if (py.c.PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", @ptrCast(&kwlist), &f, &call_args, &receiver) == 0) return null;
    const fv = value.fromPython(f.?) orelse return null;
    defer value.decref(fv);
    const seq = py.c.PySequence_Fast(call_args.?, "rt.call's arguments must be a list") orelse return null;
    defer py.Py_DecRef(seq);
    const n: usize = @intCast(py.c.PySequence_Size(seq));
    const vals = allocator.alloc(Value, n) catch return py.c.PyErr_NoMemory();
    defer allocator.free(vals);
    var made: usize = 0;
    defer for (vals[0..made]) |v| value.decref(v);
    for (0..n) |i| {
        const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse return null;
        defer py.Py_DecRef(item);
        vals[i] = (value.fromPython(item) orelse return null).checked();
        made += 1;
    }
    var recv_v: Value = Value.none_v;
    const has_recv = receiver != null and receiver.? != py.Py_None();
    if (has_recv) recv_v = value.fromPython(receiver.?) orelse return null;
    defer if (has_recv) value.decref(recv_v);
    var out = Value.none_v;
    if (!helpers.zr_call(ctx, r.at, fv.tag, fv.bits, vals.ptr, n, if (has_recv) &recv_v else null, &out)) return raiseCompiledError(ctx);
    defer value.decref(out);
    return value.toPython(out, ctx.node_maker);
}

/// rt.tail_call(f, args, receiver=None) of a semantic run as Python: the
/// call made here, its result returned (raise rt.Return): what it returns
/// is the same, without the frame given up first.
fn rtTailCall(self: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const v = rtCall(self, args, kwargs) orelse return null;
    defer py.Py_DecRef(v);
    const exc = py.c.PyObject_CallFunctionObjArgs(types.Return, v, @as(?*PyObject, null)) orelse return null;
    defer py.Py_DecRef(exc);
    py.c.PyErr_SetObject(types.Return, exc);
    return null;
}

// -- errors and what's known of nodes --

fn rtError(self: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    var node: ?*PyObject = null;
    var message: ?*PyObject = null;
    var code: ?*PyObject = null;
    var kwlist = [_:null]?[*:0]u8{ @constCast("node"), @constCast("message"), @constCast("code"), null };
    if (py.c.PyArg_ParseTupleAndKeywords(args, kwargs, "OO|O", @ptrCast(&kwlist), &node, &message, &code) == 0) return null;
    const idx = if (node.? == py.Py_None()) r.at else nodeIndex(r, node.?, "node") orelse return null;
    const msg = ph.utf8(message.?, "message") orelse return null;
    _ = helpers.fail(r.ctx.?, idx, "{s}", .{msg});
    return raiseCompiledError(r.ctx.?);
}

fn rtKind(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "node") orelse return null;
    const d = linkOf(r.ctx.?).data;
    return ref(d.grammar.kinds[d.rule(idx)]);
}

fn rtText(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "node") orelse return null;
    return ph.newString(linkOf(r.ctx.?).data.text(idx));
}

fn rtSpan(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "node") orelse return null;
    const nd = linkOf(r.ctx.?).data.nodes[idx];
    return py.c.Py_BuildValue("(II)", @as(c_uint, nd.text_start), @as(c_uint, nd.text_end));
}

fn rtScope(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "node") orelse return null;
    const d = linkOf(r.ctx.?).data;
    const si = d.symbolIndex(idx) orelse return none();
    const s = d.syms[si].scope;
    if (s == NONE or s >= d.nodes.len) return none();
    return r.ctx.?.node_maker.make(s);
}

fn rtSymbol(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "node") orelse return null;
    const analysis = linkOf(r.ctx.?).analysis orelse return none();
    return py.c.PyObject_CallMethod(analysis, "resolve", "I", @as(c_uint, idx));
}

fn rtTypeOf(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "node") orelse return null;
    const analysis = linkOf(r.ctx.?).analysis orelse return none();
    return py.c.PyObject_CallMethod(analysis, "type_of", "I", @as(c_uint, idx));
}

fn rtNodeAt(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const i = py.c.PyLong_AsLongLong(x.?);
    if (i == -1 and py.c.PyErr_Occurred() != null) return null;
    const d = linkOf(r.ctx.?).data;
    if (i < 0 or i >= d.nodes.len) {
        ph.raise(py.PyExc_IndexError(), "no node {d}", .{i});
        return null;
    }
    return r.ctx.?.node_maker.make(@intCast(i));
}

/// rt.wrapping_add(a, b) and the others (wrapping.zig)
fn rtWrapping(comptime op: @import("wrapping.zig").Op) fn (?*PyObject, ?*PyObject) callconv(.c) ?*PyObject {
    return struct {
        fn f(self: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
            _ = live(self) orelse return null;
            var a: ?*PyObject = null;
            var b: ?*PyObject = null;
            if (py.c.PyArg_UnpackTuple(args, "wrapping_" ++ @tagName(op), 2, 2, &a, &b) == 0) return null;
            return @import("wrapping.zig").ofPython(op, a.?, b.?);
        }
    }.f;
}

/// rt.u8(data, i) and the others (bytes.zig)
fn rtRead(comptime r: @import("bytes.zig").Read) fn (?*PyObject, ?*PyObject) callconv(.c) ?*PyObject {
    return struct {
        fn f(self: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
            _ = live(self) orelse return null;
            var data: ?*PyObject = null;
            var at: ?*PyObject = null;
            if (py.c.PyArg_UnpackTuple(args, @tagName(r), 2, 2, &data, &at) == 0) return null;
            return @import("bytes.zig").read(r, data.?, at.?);
        }
    }.f;
}

fn rtFresh(self: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const idx = nodeIndex(r, x.?, "node") orelse return null;
    const link = linkOf(r.ctx.?);
    if (!link.data.hasFrame(idx)) return none();
    if (r.owner != idx) {
        failHere(r, idx, "rt.fresh(): this scope isn't the one being run");
        return null;
    }
    const old = r.frame_slot.?.*;
    const n = (link.compiled.compiler.layouts.get(idx) orelse return none()).syms.items.len;
    const frame = helpers.zr_frame_new(old.parent, n) orelse return py.c.PyErr_NoMemory();
    r.frame_slot.?.* = frame;
    value.decrefFrame(old);
    return none();
}

// -- properties --

/// The frame of the function around the code (and the function), or
/// null at the top level.
fn functionFrame(r: *RuntimeObject) ?struct { frame: *value.Frame, fnode: u32 } {
    const c = &linkOf(r.ctx.?).compiled.compiler;
    var frame: *value.Frame = r.frame_slot.?.*;
    var owner = r.owner;
    while (owner != NONE and !c.isFunctionNode(owner)) {
        frame = frame.parent.?;
        owner = c.ownerOf(owner);
    }
    if (owner == NONE) return null;
    return .{ .frame = frame, .fnode = owner };
}

/// A function frame's hidden slot (0: the receiver, 1: the extra arguments).
fn hidden(r: *RuntimeObject, which: usize) ?Value {
    const ff = functionFrame(r) orelse return null;
    const c = &linkOf(r.ctx.?).compiled.compiler;
    const n = (c.layouts.get(ff.fnode) orelse return null).syms.items.len;
    const v = ff.frame.slots()[n + which];
    return if (v.tag == helpers.UNSET) null else v;
}

fn getReceiver(self: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const v = hidden(r, 0) orelse return none();
    return value.toPython(v, r.ctx.?.node_maker);
}

fn getVarargs(self: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    const v = hidden(r, 1) orelse return py.c.PyTuple_New(0);
    return value.toPython(v, r.ctx.?.node_maker);
}

fn getPath(self: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    return ref(linkOf(r.ctx.?).path orelse py.Py_None());
}

fn getContext(self: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    const r = live(self) orelse return null;
    return ref(linkOf(r.ctx.?).context orelse py.Py_None());
}

fn getReturn(_: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    return ref(types.Return);
}

fn getBreak(_: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    return ref(types.Break);
}

fn getContinue(_: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    return ref(types.Continue);
}

fn getThrow(_: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    return ref(types.Throw);
}

fn method(comptime name: [:0]const u8, comptime f: anytype, comptime flags: c_int) py.c.PyMethodDef {
    return .{ .ml_name = name, .ml_meth = @ptrCast(@constCast(&f)), .ml_flags = flags, .ml_doc = null };
}

fn getter(comptime name: [:0]const u8, comptime f: anytype) py.c.PyGetSetDef {
    return .{ .name = name, .get = @ptrCast(@constCast(&f)), .set = null, .doc = null, .closure = null };
}

var methods = [_]py.c.PyMethodDef{
    method("eval", rtEval, py.c.METH_O),
    method("exec", rtExec, py.c.METH_O),
    method("loop", rtLoop, py.c.METH_O),
    method("load", rtLoad, py.c.METH_O),
    method("store", rtStore, py.c.METH_VARARGS),
    method("function", rtFunction, py.c.METH_O),
    method("call", rtCall, py.c.METH_VARARGS | py.c.METH_KEYWORDS),
    method("tail_call", rtTailCall, py.c.METH_VARARGS | py.c.METH_KEYWORDS),
    method("error", rtError, py.c.METH_VARARGS | py.c.METH_KEYWORDS),
    method("kind", rtKind, py.c.METH_O),
    method("text", rtText, py.c.METH_O),
    method("span", rtSpan, py.c.METH_O),
    method("scope", rtScope, py.c.METH_O),
    method("symbol", rtSymbol, py.c.METH_O),
    method("type_of", rtTypeOf, py.c.METH_O),
    method("node_at", rtNodeAt, py.c.METH_O),
    method("fresh", rtFresh, py.c.METH_O),
    method("wrapping_add", rtWrapping(.add), py.c.METH_VARARGS),
    method("wrapping_sub", rtWrapping(.sub), py.c.METH_VARARGS),
    method("wrapping_mul", rtWrapping(.mul), py.c.METH_VARARGS),
    method("wrapping_shl", rtWrapping(.shl), py.c.METH_VARARGS),
    method("wrapping_shr", rtWrapping(.shr), py.c.METH_VARARGS),
    method("wrapping_ushr", rtWrapping(.ushr), py.c.METH_VARARGS),
    method("u8", rtRead(.u8), py.c.METH_VARARGS),
    method("i8", rtRead(.i8), py.c.METH_VARARGS),
    method("u16le", rtRead(.u16le), py.c.METH_VARARGS),
    method("u16be", rtRead(.u16be), py.c.METH_VARARGS),
    method("i16le", rtRead(.i16le), py.c.METH_VARARGS),
    method("i16be", rtRead(.i16be), py.c.METH_VARARGS),
    method("u32le", rtRead(.u32le), py.c.METH_VARARGS),
    method("u32be", rtRead(.u32be), py.c.METH_VARARGS),
    method("i32le", rtRead(.i32le), py.c.METH_VARARGS),
    method("i32be", rtRead(.i32be), py.c.METH_VARARGS),
    method("u64le", rtRead(.u64le), py.c.METH_VARARGS),
    method("u64be", rtRead(.u64be), py.c.METH_VARARGS),
    method("i64le", rtRead(.i64le), py.c.METH_VARARGS),
    method("i64be", rtRead(.i64be), py.c.METH_VARARGS),
    .{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null },
};

var getset = [_]py.c.PyGetSetDef{
    getter("receiver", getReceiver),
    getter("varargs", getVarargs),
    getter("path", getPath),
    getter("context", getContext),
    getter("Return", getReturn),
    getter("Break", getBreak),
    getter("Continue", getContinue),
    getter("Throw", getThrow),
    .{ .name = null, .get = null, .set = null, .doc = null, .closure = null },
};

fn runtimeDealloc(obj: ?*PyObject) callconv(.c) void {
    if (asRuntime(obj.?).own_frame) |f| value.decrefFrame(f);
    const t = obj.?.ob_type;
    const free: py.c.freefunc = @ptrCast(py.c.PyType_GetSlot(t, py.c.Py_tp_free));
    free.?(obj);
    py.Py_DecRef(@ptrCast(@alignCast(t)));
}

var slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&runtimeDealloc)) },
    .{ .slot = py.c.Py_tp_methods, .pfunc = @ptrCast(&methods) },
    .{ .slot = py.c.Py_tp_getset, .pfunc = @ptrCast(&getset) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("rt in a semantic run as Python inside compiled code: the same methods as the reference mode's, over the compiled program's frames.")) },
    .{ .slot = 0, .pfunc = null },
};

var spec = py.c.PyType_Spec{
    .name = "zrun.CompiledRuntime",
    .basicsize = @sizeOf(RuntimeObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &slots,
};

pub fn init(module: *PyObject) !void {
    RuntimeType = py.c.PyType_FromSpec(&spec) orelse return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "CompiledRuntime", RuntimeType) != 0) return error.Python;
}

/// A call site of a helper compiled out of line ran often: its helper
/// compiled for what the site knows of its arguments (Compiled.specialize),
/// what the site calls from now on. Nothing (the site keeps calling the
/// generic code) if that can't be done.
pub export fn zr_specialize(ctx: *Ctx, site: u64) callconv(.c) void {
    // (compiling while the program runs: what strict mode allows)
    gil.allowBegin();
    defer gil.allowEnd();
    linkOf(ctx).compiled.specialize(@intCast(site));
}

/// rt.context in compiled code: the object program.call() was given (None:
/// none), as a value.
pub export fn zr_context(ctx: *Ctx, out: *Value) callconv(.c) bool {
    gil.ensure(@src());
    const o = linkOf(ctx).context orelse py.Py_None();
    out.* = value.fromPython(o) orelse return pythonFailure(ctx, 0) != 0;
    return true;
}

/// A hot language function's typed entry, for the kinds its arguments have
/// always been (driver's speculate)
pub export fn zr_speculate(ctx: *Ctx, fnode: u64) callconv(.c) void {
    gil.allowBegin();
    defer gil.allowEnd();
    linkOf(ctx).compiled.speculate(@intCast(fnode));
}

/// The bridge's helpers, by name (for the JIT)
pub fn symbols() [11]struct { []const u8, usize } {
    return .{
        .{ "zr_specialize", @intFromPtr(&zr_specialize) },
        .{ "zr_speculate", @intFromPtr(&zr_speculate) },
        .{ "zr_context", @intFromPtr(&zr_context) },
        .{ "zr_py_semantic", @intFromPtr(&zr_py_semantic) },
        .{ "zr_run_value", @intFromPtr(&zr_run_value) },
        .{ "zr_runtime", @intFromPtr(&zr_runtime) },
        .{ "zr_raise", @intFromPtr(&zr_raise) },
        .{ "zr_exc_matches", @intFromPtr(&zr_exc_matches) },
        .{ "zr_exc_catch", @intFromPtr(&zr_exc_catch) },
        .{ "zr_control_value", @intFromPtr(&zr_control_value) },
        .{ "zr_raise_from", @intFromPtr(&zr_raise_from) },
    };
}
