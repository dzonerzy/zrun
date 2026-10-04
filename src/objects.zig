//! The objects semantics handle, as raw C-API types (made when the module
//! loads): `Node` (a node of the program, with its fields), `Function` (a
//! function of the program: its node and the frame it was made in) and
//! `Frame` (the variables of one run of a function); and `State`, a loaded
//! program's data, which they refer to.
//!
//! Everything that can be part of a reference cycle is here, and takes part
//! in Python's cycle collector: a function stored in the frame it was made
//! in; a node stored in a variable (the program's frame, in its state,
//! refers to the node, which refers to the state). The PyOZ classes
//! (Language, Program, Runtime) only refer to these, never the other way,
//! so they need no collector support. (They can't have it in the stable ABI:
//! PyOZ 0.13.7 frees a collected class's objects with PyObject_Del there.)

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const program_mod = @import("program.zig");
const grammar_mod = @import("grammar.zig");
const types = @import("types.zig");
const value = @import("value.zig");

const NONE = program_mod.NONE;
const allocator = std.heap.c_allocator;

pub var NodeType: *PyObject = undefined;
pub var FunctionType: *PyObject = undefined;
pub var FrameType: *PyObject = undefined;
pub var StateType: *PyObject = undefined;
/// zgram.Diagnostic
pub var Diagnostic: *PyObject = undefined;

pub fn init(module: *PyObject) !void {
    StateType = py.c.PyType_FromSpec(&state_spec) orelse return error.Python;
    NodeType = py.c.PyType_FromSpec(&node_spec) orelse return error.Python;
    FunctionType = py.c.PyType_FromSpec(&function_spec) orelse return error.Python;
    FrameType = py.c.PyType_FromSpec(&frame_spec) orelse return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "Node", NodeType) != 0) return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "Function", FunctionType) != 0) return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "Frame", FrameType) != 0) return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "State", StateType) != 0) return error.Python;
    const zgram = py.c.PyImport_ImportModule("zgram") orelse return error.Python;
    defer py.Py_DecRef(zgram);
    Diagnostic = py.c.PyObject_GetAttrString(zgram, "Diagnostic") orelse return error.Python;
    const object_type: *PyObject = @ptrCast(@alignCast(py.types.typeObject("PyBaseObject_Type")));
    drop_sentinel = py.c.PyObject_CallObject(object_type, null) orelse return error.Python;
}

fn allocObject(t: *PyObject) ?*PyObject {
    const alloc: py.c.allocfunc = @ptrCast(py.c.PyType_GetSlot(@ptrCast(t), py.c.Py_tp_alloc));
    return alloc.?(@ptrCast(t), 0);
}

fn freeObject(obj: *PyObject) void {
    const t = obj.ob_type;
    const free: py.c.freefunc = @ptrCast(py.c.PyType_GetSlot(t, py.c.Py_tp_free));
    free.?(obj);
    py.Py_DecRef(@ptrCast(@alignCast(t)));
}

// ======================================================================
// State: a loaded program's data
// ======================================================================

/// What nodes read their tree and values from (owned by the State)
pub const Context = struct {
    data: *program_mod.Data,
    /// The zgram Tree (owned), for the values of scalar actions
    /// (tree.node(i).to_ast()); its capsule's memory is what data reads
    tree: *PyObject,
    /// Per node: its value, once worked out (owned references)
    values: []?*PyObject,

    fn destroy(self: *Context) void {
        for (self.values) |v| if (v) |o| py.Py_DecRef(o);
        allocator.free(self.values);
        self.data.deinit();
        allocator.destroy(self.data);
        py.Py_DecRef(self.tree);
        allocator.destroy(self);
    }
};

pub const StateObject = extern struct {
    ob_base: py.c.PyObject,
    ctx: ?*Context,
    /// The Language (keeps the grammar and the semantics alive)
    lang: ?*PyObject,
    /// The zrules Analysis (keeps the symbols' memory alive), or null
    analysis: ?*PyObject,
    diagnostics: ?*PyObject,
    /// The program's frame once it ran (for calls into it)
    globals: ?*PyObject,
};

/// A state owning `ctx` and taking references to the rest.
pub fn newState(ctx: *Context, lang: *PyObject, analysis: ?*PyObject, diagnostics: *PyObject) ?*PyObject {
    const obj = allocObject(StateType) orelse return null;
    const s: *StateObject = @ptrCast(@alignCast(obj));
    s.ctx = ctx;
    py.Py_IncRef(lang);
    s.lang = lang;
    if (analysis) |a| py.Py_IncRef(a);
    s.analysis = analysis;
    py.Py_IncRef(diagnostics);
    s.diagnostics = diagnostics;
    s.globals = null;
    return obj;
}

pub fn asState(obj: *PyObject) *StateObject {
    return @ptrCast(@alignCast(obj));
}

/// The context of a state object.
pub fn contextOf(state: *PyObject) *Context {
    return asState(state).ctx.?;
}

fn stateClear(obj: ?*PyObject) callconv(.c) c_int {
    const s: *StateObject = @ptrCast(@alignCast(obj.?));
    if (s.globals) |g| {
        s.globals = null;
        py.Py_DecRef(g);
    }
    return 0;
}

fn stateTraverse(obj: ?*PyObject, visit: py.c.visitproc, arg: ?*anyopaque) callconv(.c) c_int {
    const s: *StateObject = @ptrCast(@alignCast(obj.?));
    inline for (.{ s.globals, s.lang, s.analysis, s.diagnostics }) |o| {
        if (o) |x| {
            const r = visit.?(x, arg);
            if (r != 0) return r;
        }
    }
    if (s.ctx) |c| {
        const r = visit.?(c.tree, arg);
        if (r != 0) return r;
    }
    return visit.?(@ptrCast(@alignCast(obj.?.ob_type)), arg);
}

fn stateDealloc(obj: ?*PyObject) callconv(.c) void {
    py.c.PyObject_GC_UnTrack(obj);
    _ = stateClear(obj);
    const s: *StateObject = @ptrCast(@alignCast(obj.?));
    if (s.ctx) |c| c.destroy();
    s.ctx = null;
    inline for (.{ "lang", "analysis", "diagnostics" }) |f| {
        if (@field(s, f)) |o| py.Py_DecRef(o);
        @field(s, f) = null;
    }
    freeObject(obj.?);
}

var state_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&stateDealloc)) },
    .{ .slot = py.c.Py_tp_traverse, .pfunc = @ptrCast(@constCast(&stateTraverse)) },
    .{ .slot = py.c.Py_tp_clear, .pfunc = @ptrCast(@constCast(&stateClear)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("A loaded program's data: its tree, symbols, and frame.")) },
    .{ .slot = 0, .pfunc = null },
};

var state_spec = py.c.PyType_Spec{
    .name = "zrun.State",
    .basicsize = @sizeOf(StateObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT | py.c.Py_TPFLAGS_HAVE_GC,
    .slots = &state_slots,
};

// ======================================================================
// Node
// ======================================================================

pub const NodeObject = extern struct {
    ob_base: py.c.PyObject,
    /// The program's State (owned): keeps the context alive
    state: ?*PyObject,
    ctx: ?*Context,
    idx: u32,
};

pub fn newNode(state: *PyObject, ctx: *Context, idx: u32) ?*PyObject {
    const obj = allocObject(NodeType) orelse return null;
    const n: *NodeObject = @ptrCast(@alignCast(obj));
    py.Py_IncRef(state);
    n.state = state;
    n.ctx = ctx;
    n.idx = idx;
    return obj;
}

pub fn asNode(obj: *PyObject) ?*NodeObject {
    if (obj.ob_type != @as(*py.c.PyTypeObject, @ptrCast(@alignCast(NodeType)))) return null;
    const n: *NodeObject = @ptrCast(@alignCast(obj));
    // (a node whose state the collector cleared has nothing to read)
    if (n.state == null) return null;
    return n;
}

fn nodeClear(obj: ?*PyObject) callconv(.c) c_int {
    const n: *NodeObject = @ptrCast(@alignCast(obj.?));
    if (n.state) |s| {
        n.state = null;
        n.ctx = null;
        py.Py_DecRef(s);
    }
    return 0;
}

fn nodeTraverse(obj: ?*PyObject, visit: py.c.visitproc, arg: ?*anyopaque) callconv(.c) c_int {
    const n: *NodeObject = @ptrCast(@alignCast(obj.?));
    if (n.state) |s| {
        const r = visit.?(s, arg);
        if (r != 0) return r;
    }
    return visit.?(@ptrCast(@alignCast(obj.?.ob_type)), arg);
}

fn nodeDealloc(obj: ?*PyObject) callconv(.c) void {
    py.c.PyObject_GC_UnTrack(obj);
    _ = nodeClear(obj);
    freeObject(obj.?);
}

/// A child's value as a field: what its action makes, or a Node; null
/// with an exception; DROP for `-> drop`.
pub fn valueOf(state: *PyObject, ctx: *Context, idx: u32) ?*PyObject {
    if (ctx.values[idx]) |v| {
        py.Py_IncRef(v);
        return v;
    }
    const data = ctx.data;
    const g = data.grammar;
    const rid = data.rule(idx);
    const action: grammar_mod.Action = if (rid < g.actions.len) g.actions[rid] else .none;
    const v: *PyObject = switch (action) {
        .none, .class => return newNode(state, ctx, idx),
        .str, .int, .float, .unquote => blk: {
            // zgram's own conversion, so values are exactly parse_ast's
            const zn = py.c.PyObject_CallMethod(ctx.tree, "node", "I", @as(c_uint, idx)) orelse return null;
            defer py.Py_DecRef(zn);
            const raw = py.c.PyObject_CallMethod(zn, "to_ast", null) orelse return null;
            break :blk types.wrapOwned(raw) orelse return null;
        },
        .true_ => ref(py.Py_True()),
        .false_ => ref(py.Py_False()),
        .none_ => ref(py.Py_None()),
        .drop => ref(DROP()),
        .list, .tuple => blk: {
            const list = childValues(state, ctx, idx) orelse return null;
            if (action == .list) break :blk list;
            defer py.Py_DecRef(list);
            break :blk py.c.PyList_AsTuple(list) orelse return null;
        },
        .dict => blk: {
            const list = childValues(state, ctx, idx) orelse return null;
            defer py.Py_DecRef(list);
            const dict = py.c.PyDict_New() orelse return null;
            const n: usize = @intCast(py.c.PyList_Size(list));
            for (0..n) |i| {
                const pair = py.c.PyList_GetItem(list, @intCast(i)).?;
                if (!py.PyTuple_Check(pair) or py.c.PyTuple_Size(pair) != 2) continue;
                if (py.c.PyDict_SetItem(dict, py.c.PyTuple_GetItem(pair, 0).?, py.c.PyTuple_GetItem(pair, 1).?) != 0) {
                    py.Py_DecRef(dict);
                    return null;
                }
            }
            break :blk dict;
        },
        .first => blk: {
            var c = idx + 1;
            const stop = data.end(idx);
            while (c < stop) : (c = data.end(c)) {
                const cv = valueOf(state, ctx, c) orelse return null;
                if (cv == DROP()) {
                    py.Py_DecRef(cv);
                    continue;
                }
                break :blk cv;
            }
            break :blk ref(py.Py_None());
        },
    };
    // Only values without nodes are kept: a node holds its program, which
    // holds this cache (a cycle the collector doesn't see)
    switch (action) {
        .str, .int, .float, .unquote, .true_, .false_, .none_, .drop => {
            py.Py_IncRef(v);
            ctx.values[idx] = v;
        },
        else => {},
    }
    return v;
}

/// The values of a node's children, `-> drop` ones left out (a new list).
pub fn childValues(state: *PyObject, ctx: *Context, idx: u32) ?*PyObject {
    const data = ctx.data;
    const list = py.c.PyList_New(0) orelse return null;
    var c = idx + 1;
    const stop = data.end(idx);
    while (c < stop) : (c = data.end(c)) {
        const v = valueOf(state, ctx, c) orelse {
            py.Py_DecRef(list);
            return null;
        };
        defer py.Py_DecRef(v);
        if (v == DROP()) continue;
        if (py.c.PyList_Append(list, v) != 0) {
            py.Py_DecRef(list);
            return null;
        }
    }
    return list;
}

/// A field of a node: the labelled child's value (None if absent), or the
/// list of them for a label that repeats. Null with an exception; or null
/// without one if `field` isn't a label of the grammar.
pub fn fieldOf(state: *PyObject, ctx: *Context, idx: u32, field: u8) ?*PyObject {
    const data = ctx.data;
    const rid = data.rule(idx);
    const label = data.grammar.labelOf(rid, field);
    var list: ?*PyObject = null;
    var single: ?*PyObject = null;
    defer if (single) |s| py.Py_DecRef(s);
    var count: usize = 0;
    var c = idx + 1;
    const stop = data.end(idx);
    while (c < stop) : (c = data.end(c)) {
        if (data.nodes[c].fieldId() != field) continue;
        const v = valueOf(state, ctx, c) orelse {
            if (list) |l| py.Py_DecRef(l);
            return null;
        };
        if (v == DROP()) {
            py.Py_DecRef(v);
            continue;
        }
        count += 1;
        if (count == 1 and !(label != null and label.?.many)) {
            single = v;
            continue;
        }
        if (list == null) {
            list = py.c.PyList_New(0) orelse return null;
            if (single) |s| {
                _ = py.c.PyList_Append(list.?, s);
                py.Py_DecRef(s);
                single = null;
            }
        }
        defer py.Py_DecRef(v);
        if (py.c.PyList_Append(list.?, v) != 0) {
            py.Py_DecRef(list.?);
            return null;
        }
    }
    if (list) |l| return l;
    if (label != null and label.?.many) return py.c.PyList_New(0);
    if (single) |s| {
        single = null;
        return s;
    }
    return ref(py.Py_None());
}

fn ref(o: *PyObject) *PyObject {
    py.Py_IncRef(o);
    return o;
}

/// What a `-> drop` node's value is: left out of its parent's values
var drop_sentinel: *PyObject = undefined;
fn DROP() *PyObject {
    return drop_sentinel;
}

fn nodeGetattro(obj: ?*PyObject, name_obj: ?*PyObject) callconv(.c) ?*PyObject {
    const n: *NodeObject = @ptrCast(@alignCast(obj.?));
    const name = ph.utf8(name_obj.?, "an attribute name") orelse return null;
    if (name.len > 1 and name[0] == '_' and name[1] == '_') return py.c.PyObject_GenericGetAttr(obj, name_obj);
    const state = n.state orelse {
        ph.raise(py.PyExc_RuntimeError(), "this node's program is gone", .{});
        return null;
    };
    const ctx = n.ctx.?;
    // Labels first: a field named like an attribute (`index`) is the field
    if (ctx.data.grammar.field_ids.get(name)) |field| return fieldOf(state, ctx, n.idx, field);
    if (meta(n, name)) |result| return result;
    if (py.c.PyErr_Occurred() != null) return null;
    // No such field: say which it has
    const data = ctx.data;
    const rid = data.rule(n.idx);
    var buf: [512]u8 = undefined;
    var len: usize = 0;
    for (data.grammar.labels[rid], 0..) |l, i| {
        const fname = data.grammar.field_names[l.field - 1];
        if (len + fname.len + 2 >= buf.len) break;
        if (i > 0) {
            @memcpy(buf[len..][0..2], ", ");
            len += 2;
        }
        @memcpy(buf[len..][0..fname.len], fname);
        len += fname.len;
    }
    const have = if (len == 0) "none" else buf[0..len];
    ph.raise(py.PyExc_AttributeError(), "{s} has no field '{s}' (its fields: {s})", .{ data.grammar.kind_names[rid], name, have });
    return null;
}

/// The node attributes that aren't fields: null if `name` isn't one (or
/// with an exception).
fn meta(n: *NodeObject, name: []const u8) ?*PyObject {
    const ctx = n.ctx.?;
    const data = ctx.data;
    const node = data.nodes[n.idx];
    const rid = node.ruleId();
    const eq = std.mem.eql;
    if (eq(u8, name, "kind")) return ref(data.grammar.kinds[rid]);
    if (eq(u8, name, "rule")) {
        const r = data.grammar.rule_names[rid];
        return py.PyUnicode_FromStringAndSize(r.ptr, @intCast(r.len));
    }
    if (eq(u8, name, "text")) return ph.newString(data.text(n.idx));
    if (eq(u8, name, "span")) return py.c.Py_BuildValue("(II)", @as(c_uint, node.text_start), @as(c_uint, node.text_end));
    if (eq(u8, name, "start")) return py.c.PyLong_FromUnsignedLong(node.text_start);
    if (eq(u8, name, "end")) return py.c.PyLong_FromUnsignedLong(node.text_end);
    // (1-based, where the node starts; the column in bytes)
    if (eq(u8, name, "line")) return py.c.PyLong_FromUnsignedLong(data.lineCol(node.text_start).line);
    if (eq(u8, name, "column")) return py.c.PyLong_FromUnsignedLong(data.lineCol(node.text_start).col);
    if (eq(u8, name, "index")) return py.c.PyLong_FromUnsignedLong(n.idx);
    if (eq(u8, name, "children")) return childValues(n.state.?, ctx, n.idx);
    if (eq(u8, name, "parent")) {
        const p = data.parents[n.idx];
        if (p == NONE) return ref(py.Py_None());
        return newNode(n.state.?, ctx, p);
    }
    if (eq(u8, name, "fields")) {
        const labels = data.grammar.labels[rid];
        const list = py.c.PyList_New(@intCast(labels.len)) orelse return null;
        for (labels, 0..) |l, i| {
            const f = data.grammar.field_names[l.field - 1];
            _ = py.c.PyList_SetItem(list, @intCast(i), py.PyUnicode_FromStringAndSize(f.ptr, @intCast(f.len)));
        }
        return list;
    }
    return null;
}

fn nodeRichcompare(a: ?*PyObject, b: ?*PyObject, op: c_int) callconv(.c) ?*PyObject {
    if (op != py.c.Py_EQ and op != py.c.Py_NE) return ref(py.c.Py_NotImplemented());
    const x = asNode(a.?) orelse return ref(py.c.Py_NotImplemented());
    const y = asNode(b.?) orelse return ref(py.c.Py_NotImplemented());
    const same = x.ctx == y.ctx and x.idx == y.idx;
    return ref(if (same == (op == py.c.Py_EQ)) py.Py_True() else py.Py_False());
}

fn nodeHash(obj: ?*PyObject) callconv(.c) py.c.Py_hash_t {
    const n: *NodeObject = @ptrCast(@alignCast(obj.?));
    const h: u64 = (@as(u64, @intFromPtr(n.ctx)) >> 4) ^ (@as(u64, n.idx) *% 0x9E3779B97F4A7C15);
    return @intCast(h & 0x3FFF_FFFF_FFFF_FFFF);
}

fn nodeRepr(obj: ?*PyObject) callconv(.c) ?*PyObject {
    const n: *NodeObject = @ptrCast(@alignCast(obj.?));
    const ctx = n.ctx orelse return ph.newString("<node of a program that is gone>");
    const data = ctx.data;
    const rid = data.rule(n.idx);
    var t = data.text(n.idx);
    var dots: []const u8 = "";
    if (t.len > 30) {
        t = t[0..27];
        dots = "...";
    }
    const text = ph.newString(t) orelse return null;
    defer py.Py_DecRef(text);
    const kind = ph.newString(data.grammar.kind_names[rid]) orelse return null;
    defer py.Py_DecRef(kind);
    return py.c.PyUnicode_FromFormat("<%U %R%s>", kind, text, dots.ptr);
}

var node_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&nodeDealloc)) },
    .{ .slot = py.c.Py_tp_traverse, .pfunc = @ptrCast(@constCast(&nodeTraverse)) },
    .{ .slot = py.c.Py_tp_clear, .pfunc = @ptrCast(@constCast(&nodeClear)) },
    .{ .slot = py.c.Py_tp_getattro, .pfunc = @ptrCast(@constCast(&nodeGetattro)) },
    .{ .slot = py.c.Py_tp_richcompare, .pfunc = @ptrCast(@constCast(&nodeRichcompare)) },
    .{ .slot = py.c.Py_tp_hash, .pfunc = @ptrCast(@constCast(&nodeHash)) },
    .{ .slot = py.c.Py_tp_repr, .pfunc = @ptrCast(@constCast(&nodeRepr)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("A node of the program. Its fields are its labelled children (node.cond), converted as the grammar's -> actions say; a label that repeats is a list. Also: kind (its action's class, else its rule), rule, text, span, start, end, index, children, parent, fields (a field of the same name comes first).")) },
    .{ .slot = 0, .pfunc = null },
};

var node_spec = py.c.PyType_Spec{
    .name = "zrun.Node",
    .basicsize = @sizeOf(NodeObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT | py.c.Py_TPFLAGS_HAVE_GC,
    .slots = &node_slots,
};

// ======================================================================
// Frame
// ======================================================================

pub const Slots = std.AutoHashMapUnmanaged(u32, *PyObject);

pub const FrameObject = extern struct {
    ob_base: py.c.PyObject,
    /// The function node it runs (NONE: the program)
    scope: u32,
    /// The node that called it (NONE: none)
    call: u32,
    /// The frame it was made in (the function's definition), owned
    parent: ?*PyObject,
    /// The function's name, owned
    name: ?*PyObject,
    /// The variables, by symbol (owned values)
    slots: ?*Slots,
    /// What the function was called on (rt.call(f, args, receiver=...)),
    /// owned; null: none
    receiver: ?*PyObject,
    /// The arguments beyond its parameters (a tuple, owned; a function
    /// kind with extra="keep"), rt.varargs; null: none
    varargs: ?*PyObject,
};

pub fn newFrame(scope: u32, call: u32, parent: ?*PyObject, name: *PyObject) ?*PyObject {
    const obj = allocObject(FrameType) orelse return null;
    const f: *FrameObject = @ptrCast(@alignCast(obj));
    f.scope = scope;
    f.call = call;
    f.receiver = null;
    f.varargs = null;
    if (parent) |p| py.Py_IncRef(p);
    f.parent = parent;
    py.Py_IncRef(name);
    f.name = name;
    const slots = allocator.create(Slots) catch {
        py.Py_DecRef(obj);
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    slots.* = .empty;
    f.slots = slots;
    return obj;
}

pub fn asFrame(obj: *PyObject) *FrameObject {
    return @ptrCast(@alignCast(obj));
}

/// Set a slot (takes a new reference to `value`).
pub fn setSlot(f: *FrameObject, sym: u32, obj: *PyObject) !void {
    const slots = f.slots orelse return error.OutOfMemory;
    py.Py_IncRef(obj);
    const entry = slots.getOrPut(allocator, sym) catch {
        py.Py_DecRef(obj);
        return error.OutOfMemory;
    };
    if (entry.found_existing) py.Py_DecRef(entry.value_ptr.*);
    entry.value_ptr.* = obj;
}

fn frameClear(obj: ?*PyObject) callconv(.c) c_int {
    const f: *FrameObject = @ptrCast(@alignCast(obj.?));
    if (f.slots) |slots| {
        var it = slots.valueIterator();
        while (it.next()) |v| py.Py_DecRef(v.*);
        slots.clearRetainingCapacity();
    }
    inline for (.{ "parent", "receiver", "varargs" }) |field| {
        if (@field(f, field)) |o| {
            @field(f, field) = null;
            py.Py_DecRef(o);
        }
    }
    return 0;
}

fn frameTraverse(obj: ?*PyObject, visit: py.c.visitproc, arg: ?*anyopaque) callconv(.c) c_int {
    const f: *FrameObject = @ptrCast(@alignCast(obj.?));
    inline for (.{ f.parent, f.receiver, f.varargs }) |o| {
        if (o) |x| {
            const r = visit.?(x, arg);
            if (r != 0) return r;
        }
    }
    if (f.slots) |slots| {
        var it = slots.valueIterator();
        while (it.next()) |v| {
            const r = visit.?(v.*, arg);
            if (r != 0) return r;
        }
    }
    // (heap types visit their type)
    return visit.?(@ptrCast(@alignCast(obj.?.ob_type)), arg);
}

fn frameDealloc(obj: ?*PyObject) callconv(.c) void {
    py.c.PyObject_GC_UnTrack(obj);
    _ = frameClear(obj);
    const f: *FrameObject = @ptrCast(@alignCast(obj.?));
    if (f.slots) |slots| {
        slots.deinit(allocator);
        allocator.destroy(slots);
        f.slots = null;
    }
    if (f.name) |n| py.Py_DecRef(n);
    freeObject(obj.?);
}

var frame_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&frameDealloc)) },
    .{ .slot = py.c.Py_tp_traverse, .pfunc = @ptrCast(@constCast(&frameTraverse)) },
    .{ .slot = py.c.Py_tp_clear, .pfunc = @ptrCast(@constCast(&frameClear)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("The variables of one run of a function (or of the program).")) },
    .{ .slot = 0, .pfunc = null },
};

var frame_spec = py.c.PyType_Spec{
    .name = "zrun.Frame",
    .basicsize = @sizeOf(FrameObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT | py.c.Py_TPFLAGS_HAVE_GC,
    .slots = &frame_slots,
};

// ======================================================================
// Function
// ======================================================================

pub const FunctionObject = extern struct {
    ob_base: py.c.PyObject,
    /// The program's State (owned): a function is called only by its own
    /// program
    state: ?*PyObject,
    node: u32,
    /// The frame it was made in, owned
    env: ?*PyObject,
    /// Its name (str), owned
    name: ?*PyObject,
    /// A compiled program's function given to Python (owned; state and
    /// env are null): it comes back as itself
    native: ?*value.Function = null,
};

/// A compiled function as Python sees it (as the reference mode's),
/// keeping its program (`owner`, whose code it runs) alive.
pub fn newNativeFunction(f: *value.Function, owner: ?*PyObject) ?*PyObject {
    const name = ph.newString(f.name.bytes()) orelse return null;
    defer py.Py_DecRef(name);
    const obj = allocObject(FunctionType) orelse return null;
    const fo: *FunctionObject = @ptrCast(@alignCast(obj));
    value.increfObj(&f.head);
    py.Py_IncRef(name);
    if (owner) |o| py.Py_IncRef(o);
    fo.* = .{ .ob_base = fo.ob_base, .state = owner, .node = @intCast(f.node), .env = null, .name = name, .native = f };
    return obj;
}

pub fn newFunction(state: *PyObject, node: u32, env: ?*PyObject, name: *PyObject) ?*PyObject {
    const obj = allocObject(FunctionType) orelse return null;
    const f: *FunctionObject = @ptrCast(@alignCast(obj));
    py.Py_IncRef(state);
    f.state = state;
    f.node = node;
    if (env) |e| py.Py_IncRef(e);
    f.env = env;
    py.Py_IncRef(name);
    f.name = name;
    return obj;
}

pub fn asFunction(obj: *PyObject) ?*FunctionObject {
    if (obj.ob_type != @as(*py.c.PyTypeObject, @ptrCast(@alignCast(FunctionType)))) return null;
    return @ptrCast(@alignCast(obj));
}

fn functionClear(obj: ?*PyObject) callconv(.c) c_int {
    const f: *FunctionObject = @ptrCast(@alignCast(obj.?));
    // (a compiled one's native value first: its program, which the state
    // keeps, is still there)
    if (f.native) |n| {
        f.native = null;
        value.decref(value.Value.obj(.function, &n.head));
    }
    inline for (.{ "env", "state" }) |field| {
        if (@field(f, field)) |o| {
            @field(f, field) = null;
            py.Py_DecRef(o);
        }
    }
    return 0;
}

fn functionTraverse(obj: ?*PyObject, visit: py.c.visitproc, arg: ?*anyopaque) callconv(.c) c_int {
    const f: *FunctionObject = @ptrCast(@alignCast(obj.?));
    inline for (.{ f.env, f.state }) |o| {
        if (o) |x| {
            const r = visit.?(x, arg);
            if (r != 0) return r;
        }
    }
    return visit.?(@ptrCast(@alignCast(obj.?.ob_type)), arg);
}

fn functionDealloc(obj: ?*PyObject) callconv(.c) void {
    py.c.PyObject_GC_UnTrack(obj);
    _ = functionClear(obj);
    const f: *FunctionObject = @ptrCast(@alignCast(obj.?));
    if (f.name) |n| py.Py_DecRef(n);
    if (f.native) |n| value.decref(value.Value.obj(.function, &n.head));
    freeObject(obj.?);
}

fn functionRepr(obj: ?*PyObject) callconv(.c) ?*PyObject {
    const f: *FunctionObject = @ptrCast(@alignCast(obj.?));
    return py.c.PyUnicode_FromFormat("<function %U>", f.name.?);
}

var function_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&functionDealloc)) },
    .{ .slot = py.c.Py_tp_traverse, .pfunc = @ptrCast(@constCast(&functionTraverse)) },
    .{ .slot = py.c.Py_tp_clear, .pfunc = @ptrCast(@constCast(&functionClear)) },
    .{ .slot = py.c.Py_tp_repr, .pfunc = @ptrCast(@constCast(&functionRepr)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("A function of the program: its node, and the frame it was made in.")) },
    .{ .slot = 0, .pfunc = null },
};

var function_spec = py.c.PyType_Spec{
    .name = "zrun.Function",
    .basicsize = @sizeOf(FunctionObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT | py.c.Py_TPFLAGS_HAVE_GC,
    .slots = &function_slots,
};
