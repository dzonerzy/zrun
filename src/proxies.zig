//! Compiled values in Python, shared rather than copied: a list, a dict or
//! a record the compiled code made reaches Python (a host function, a
//! semantic run as Python) as a proxy over the same native object, so a
//! change on either side is seen by the other, as in the reference mode
//! where both are the same Python object. Their `__class__` is list, dict
//! or the record's class (isinstance() is true), and they come back to
//! compiled code as the very object they stand for.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const value = @import("value.zig");
const helpers = @import("helpers.zig");

const Value = value.Value;
const allocator = std.heap.c_allocator;

pub var ListType: *PyObject = undefined;
pub var DictType: *PyObject = undefined;
pub var RecordType: *PyObject = undefined;

/// What every proxy holds: the native object, and what its values need to
/// become Python objects (nodes: the program, kept alive).
///
/// A native object has one proxy, for its whole life (Python sees the same
/// object each time: `is`, id()): while compiled code holds the object, the
/// object holds its proxy (a reference); once it lets go and Python still
/// has the proxy, the proxy owns the object (`owns`) until it comes back.
const Proxy = extern struct {
    ob_base: py.c.PyObject,
    obj: *value.Obj,
    maker_ctx: ?*anyopaque,
    maker_fn: ?*const fn (ctx: *anyopaque, idx: u32) ?*PyObject,
    owner: ?*PyObject,
    owns: bool,
};

/// The proxy of each native object that has one (value.HAS_PROXY)
var by_obj: std.AutoHashMapUnmanaged(*value.Obj, *PyObject) = .empty;

fn asProxy(o: ?*PyObject) *Proxy {
    return @ptrCast(@alignCast(o.?));
}

fn maker(p: *Proxy) helpers.NodeMaker {
    return .{ .ctx = p.maker_ctx.?, .make_fn = p.maker_fn.?, .owner = p.owner };
}

fn list(p: *Proxy) *value.List {
    return @ptrCast(@alignCast(p.obj));
}

fn dict(p: *Proxy) *value.Dict {
    return @ptrCast(@alignCast(p.obj));
}

fn record(p: *Proxy) *value.Record {
    return @ptrCast(@alignCast(p.obj));
}

fn ref(o: *PyObject) *PyObject {
    py.Py_IncRef(o);
    return o;
}

fn none() *PyObject {
    return ref(py.Py_None());
}

fn typeIs(o: *PyObject, t: *PyObject) bool {
    return @as(*PyObject, @ptrCast(@alignCast(o.ob_type))) == t;
}

/// The proxy of the native object `v` stands for (a new reference): its
/// own, made the first time.
pub fn make(v: Value, m: helpers.NodeMaker) ?*PyObject {
    const o = v.ptr();
    if (o.flags & value.HAS_PROXY != 0) {
        const existing = by_obj.get(o).?;
        py.Py_IncRef(existing);
        return existing;
    }
    const t = switch (v.kind()) {
        .list => ListType,
        .dict => DictType,
        .record => RecordType,
        else => unreachable,
    };
    const alloc: py.c.allocfunc = @ptrCast(py.c.PyType_GetSlot(@ptrCast(t), py.c.Py_tp_alloc));
    const obj = alloc.?(@ptrCast(t), 0) orelse return null;
    const p = asProxy(obj);
    p.obj = o;
    p.owns = false;
    p.maker_ctx = m.ctx;
    p.maker_fn = m.make_fn;
    if (m.owner) |w| py.Py_IncRef(w);
    p.owner = m.owner;
    by_obj.put(allocator, o, obj) catch {
        py.Py_DecRef(obj);
        return py.c.PyErr_NoMemory();
    };
    o.flags |= value.HAS_PROXY;
    // (the object's reference)
    py.Py_IncRef(obj);
    return obj;
}

/// The native value a proxy stands for (a new reference), or null if `o`
/// isn't one.
pub fn unwrap(o: *PyObject) ?Value {
    const kind: value.Tag = if (typeIs(o, ListType)) .list else if (typeIs(o, DictType)) .dict else if (typeIs(o, RecordType)) .record else return null;
    const p = asProxy(o);
    const v = Value.obj(kind, p.obj);
    if (p.owns) {
        // (back with compiled code: the object holds its proxy again)
        p.owns = false;
        p.obj.rc = 1;
        py.Py_IncRef(o);
        return v;
    }
    value.incref(v);
    return v;
}

/// The native value a proxy stands for (borrowed, nothing changed), or
/// null if `o` isn't one.
pub fn objectOf(o: *PyObject) ?Value {
    const kind: value.Tag = if (typeIs(o, ListType)) .list else if (typeIs(o, DictType)) .dict else if (typeIs(o, RecordType)) .record else return null;
    return Value.obj(kind, asProxy(o).obj);
}

/// Python holds the proxy of an object that has one (not only the object
/// itself): the cycle collector's root (the GIL held).
pub fn heldByPython(o: *value.Obj) bool {
    const proxy = by_obj.get(o) orelse return false;
    // (a proxy owning its object: only Python has it; else the object holds
    // one reference itself)
    return asProxy(proxy).owns or ph.refcnt(proxy) > 1;
}

/// Compiled code let go of an object that has a proxy: true if it's to be
/// freed now (Python doesn't have the proxy either), else the proxy owns
/// it from now on.
pub fn released(o: *value.Obj) bool {
    const proxy = by_obj.get(o).?;
    if (ph.refcnt(proxy) == 1) {
        _ = by_obj.remove(o);
        o.flags &= ~value.HAS_PROXY;
        py.Py_DecRef(proxy);
        return true;
    }
    asProxy(proxy).owns = true;
    py.Py_DecRef(proxy);
    return false;
}

fn dealloc(o: ?*PyObject) callconv(.c) void {
    const p = asProxy(o);
    const kind: value.Tag = if (typeIs(o.?, ListType)) .list else if (typeIs(o.?, DictType)) .dict else .record;
    // (it owned the object: freed with it; else the object is being freed,
    // released() let go of it)
    if (p.owns) {
        _ = by_obj.remove(p.obj);
        p.obj.flags &= ~value.HAS_PROXY;
        value.free(kind, p.obj);
    }
    if (p.owner) |w| py.Py_DecRef(w);
    const t = o.?.ob_type;
    const free: py.c.freefunc = @ptrCast(py.c.PyType_GetSlot(t, py.c.Py_tp_free));
    free.?(o);
    py.Py_DecRef(@ptrCast(@alignCast(t)));
}

fn out(p: *Proxy, v: Value) ?*PyObject {
    return value.toPython(v, maker(p));
}

/// A Python object as a value for storing in a native container (a new
/// reference).
fn in(o: *PyObject) ?Value {
    return value.fromPython(o);
}

// ======================================================================
// zrun.List
// ======================================================================

fn listLen(o: ?*PyObject) callconv(.c) isize {
    return @intCast(list(asProxy(o)).len);
}

/// A sequence index (negatives from the end), or null with IndexError.
fn seqIndex(i: isize, n: usize, comptime what: []const u8) ?usize {
    const k = if (i < 0) i + @as(isize, @intCast(n)) else i;
    if (k < 0 or k >= n) {
        ph.raise(py.PyExc_IndexError(), what ++ " index out of range", .{});
        return null;
    }
    return @intCast(k);
}

fn listItem(o: ?*PyObject, i: isize) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const l = list(p);
    // (sq_item gets negatives already adjusted by the length)
    if (i < 0 or i >= l.len) {
        ph.raise(py.PyExc_IndexError(), "list index out of range", .{});
        return null;
    }
    return out(p, l.slice()[@intCast(i)]);
}

fn listSubscript(o: ?*PyObject, key: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const l = list(p);
    if (isSlice(key.?)) {
        var start: isize = 0;
        var stop: isize = 0;
        var step: isize = 0;
        if (py.c.PySlice_Unpack(key.?, &start, &stop, &step) < 0) return null;
        const n = py.c.PySlice_AdjustIndices(@intCast(l.len), &start, &stop, step);
        const nl = value.newList(@intCast(n)) orelse return py.c.PyErr_NoMemory();
        var i = start;
        for (0..@intCast(n)) |_| {
            const v = l.slice()[@intCast(i)];
            value.incref(v);
            _ = value.listPush(nl, v);
            i += step;
        }
        const v = Value.obj(.list, &nl.head);
        defer value.decref(v);
        return make(v, maker(p));
    }
    const i = py.c.PyNumber_AsSsize_t(key.?, py.c.PyExc_IndexError);
    if (i == -1 and py.c.PyErr_Occurred() != null) {
        if (py.c.PyErr_ExceptionMatches(py.c.PyExc_TypeError) != 0) {
            py.c.PyErr_Clear();
            ph.raise(py.PyExc_TypeError(), "list indices must be integers or slices, not {s}", .{typeNameOf(key.?)});
        }
        return null;
    }
    const k = seqIndex(i, l.len, "list") orelse return null;
    return out(p, l.slice()[k]);
}

/// The name of an object's type (copied: valid until the next call).
fn typeNameOf(o: *PyObject) []const u8 {
    const t: *PyObject = @ptrCast(@alignCast(o.ob_type));
    const n = py.c.PyObject_GetAttrString(t, "__name__") orelse {
        py.c.PyErr_Clear();
        return "object";
    };
    defer py.Py_DecRef(n);
    const s = ph.utf8(n, "name") orelse {
        py.c.PyErr_Clear();
        return "object";
    };
    const k = @min(s.len, type_name_buf.len);
    @memcpy(type_name_buf[0..k], s[0..k]);
    return type_name_buf[0..k];
}

var type_name_buf: [128]u8 = undefined;

fn listRemoveAt(l: *value.List, k: usize) Value {
    const items = l.slice();
    const v = items[k];
    std.mem.copyForwards(Value, items[k .. items.len - 1], items[k + 1 ..]);
    l.len -= 1;
    return v;
}

fn listInsertAt(l: *value.List, k: usize, v: Value) bool {
    // (room made with a None: the elements kind the list had, then v's)
    const kinds = l.head.flags & value.LIST_KINDS;
    if (!value.listPush(l, Value.none_v)) return false;
    l.head.flags |= kinds;
    const items = l.slice();
    std.mem.copyBackwards(Value, items[k + 1 ..], items[k .. items.len - 1]);
    items[k] = v;
    value.listStored(l, v);
    return true;
}

fn isSlice(o: *PyObject) bool {
    return typeIs(o, @ptrCast(@alignCast(py.types.typeObject("PySlice_Type"))));
}

fn listAssSubscript(o: ?*PyObject, key: ?*PyObject, v: ?*PyObject) callconv(.c) c_int {
    const p = asProxy(o);
    const l = list(p);
    if (isSlice(key.?)) {
        var start: isize = 0;
        var stop: isize = 0;
        var step: isize = 0;
        if (py.c.PySlice_Unpack(key.?, &start, &stop, &step) < 0) return -1;
        const n: usize = @intCast(py.c.PySlice_AdjustIndices(@intCast(l.len), &start, &stop, step));
        // The new items, as values (an iterable)
        var items: std.ArrayListUnmanaged(Value) = .empty;
        defer {
            for (items.items) |x| value.decref(x);
            items.deinit(allocator);
        }
        if (v) |seq_obj| {
            const seq = py.c.PySequence_Fast(seq_obj, "can only assign an iterable") orelse return -1;
            defer py.Py_DecRef(seq);
            const m: usize = @intCast(py.c.PySequence_Size(seq));
            for (0..m) |i| {
                const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse return -1;
                defer py.Py_DecRef(item);
                const x = in(item) orelse return -1;
                items.append(allocator, x) catch {
                    value.decref(x);
                    _ = py.c.PyErr_NoMemory();
                    return -1;
                };
            }
        }
        if (step != 1) {
            if (v != null and items.items.len != n) {
                ph.raise(py.PyExc_ValueError(), "attempt to assign sequence of size {d} to extended slice of size {d}", .{ items.items.len, n });
                return -1;
            }
            if (v == null) {
                // (deleting: from the highest index down)
                var idx: std.ArrayListUnmanaged(usize) = .empty;
                defer idx.deinit(allocator);
                var i = start;
                for (0..n) |_| {
                    idx.append(allocator, @intCast(i)) catch return -1;
                    i += step;
                }
                std.mem.sort(usize, idx.items, {}, std.sort.desc(usize));
                for (idx.items) |k| value.decref(listRemoveAt(l, k));
                return 0;
            }
            var i = start;
            for (items.items) |*x| {
                const k: usize = @intCast(i);
                const old = l.slice()[k];
                l.slice()[k] = x.*;
                value.listStored(l, x.*);
                x.* = Value.none_v;
                value.decref(old);
                i += step;
            }
            return 0;
        }
        // Step 1: the range replaced (its length may change)
        const lo: usize = @intCast(start);
        const hi: usize = lo + n;
        var k: usize = hi;
        while (k > lo) {
            k -= 1;
            value.decref(listRemoveAt(l, k));
        }
        for (items.items, 0..) |*x, j| {
            if (!listInsertAt(l, lo + j, x.*)) {
                _ = py.c.PyErr_NoMemory();
                return -1;
            }
            x.* = Value.none_v;
        }
        return 0;
    }
    const i = py.c.PyNumber_AsSsize_t(key.?, py.c.PyExc_IndexError);
    if (i == -1 and py.c.PyErr_Occurred() != null) return -1;
    const k = seqIndex(i, l.len, "list assignment") orelse return -1;
    if (v) |x| {
        const nv = in(x) orelse return -1;
        const old = l.slice()[k];
        l.slice()[k] = nv;
        value.listStored(l, nv);
        value.decref(old);
    } else value.decref(listRemoveAt(l, k));
    return 0;
}

/// Python's == between a value of a native container and an object.
fn equalTo(p: *Proxy, v: Value, x: *PyObject) c_int {
    const o = out(p, v) orelse return -1;
    defer py.Py_DecRef(o);
    return py.c.PyObject_RichCompareBool(o, x, py.c.Py_EQ);
}

fn listContains(o: ?*PyObject, x: ?*PyObject) callconv(.c) c_int {
    const p = asProxy(o);
    for (list(p).slice()) |v| {
        const r = equalTo(p, v, x.?);
        if (r != 0) return r;
    }
    return 0;
}

/// A Python list with the proxy's items (a new reference: a copy).
fn listCopy(p: *Proxy) ?*PyObject {
    const items = list(p).slice();
    const l = py.c.PyList_New(@intCast(items.len)) orelse return null;
    for (items, 0..) |v, i| {
        const o = out(p, v) orelse {
            py.Py_DecRef(l);
            return null;
        };
        _ = py.c.PyList_SetItem(l, @intCast(i), o);
    }
    return l;
}

fn listRepr(o: ?*PyObject) callconv(.c) ?*PyObject {
    const r = py.c.Py_ReprEnter(o);
    if (r != 0) return if (r > 0) ph.newString("[...]") else null;
    defer py.c.Py_ReprLeave(o);
    const copy = listCopy(asProxy(o)) orelse return null;
    defer py.Py_DecRef(copy);
    return py.c.PyObject_Repr(copy);
}

/// A plain Python container of a proxy (a copy), for comparing.
fn plain(x: *PyObject) ?*PyObject {
    if (typeIs(x, ListType)) return listCopy(asProxy(x));
    if (typeIs(x, DictType)) return dictCopy(asProxy(x));
    return ref(x);
}

fn containerCompare(o: ?*PyObject, other: ?*PyObject, op: c_int) callconv(.c) ?*PyObject {
    const a = plain(o.?) orelse return null;
    defer py.Py_DecRef(a);
    const b = plain(other.?) orelse return null;
    defer py.Py_DecRef(b);
    return py.c.PyObject_RichCompare(a, b, op);
}

fn listIter(o: ?*PyObject) callconv(.c) ?*PyObject {
    return py.c.PySeqIter_New(o);
}

fn listAppend(o: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const v = in(x.?) orelse return null;
    if (!value.listPush(list(asProxy(o)), v)) {
        value.decref(v);
        return py.c.PyErr_NoMemory();
    }
    return none();
}

fn listExtend(o: ?*PyObject, it: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    // (a copy first: extending by itself goes over what it was)
    const seq = py.c.PySequence_List(it.?) orelse return null;
    defer py.Py_DecRef(seq);
    const n: usize = @intCast(py.c.PyList_Size(seq));
    for (0..n) |i| {
        const v = in(py.c.PyList_GetItem(seq, @intCast(i)).?) orelse return null;
        if (!value.listPush(list(p), v)) {
            value.decref(v);
            return py.c.PyErr_NoMemory();
        }
    }
    return none();
}

fn listInsert(o: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
    var i_obj: ?*PyObject = null;
    var x: ?*PyObject = null;
    if (py.c.PyArg_UnpackTuple(args, "insert", 2, 2, &i_obj, &x) == 0) return null;
    const l = list(asProxy(o));
    var i = py.c.PyNumber_AsSsize_t(i_obj.?, null);
    if (i == -1 and py.c.PyErr_Occurred() != null) return null;
    const n: isize = @intCast(l.len);
    if (i < 0) i = @max(i + n, 0);
    if (i > n) i = n;
    const v = in(x.?) orelse return null;
    if (!listInsertAt(l, @intCast(i), v)) {
        value.decref(v);
        return py.c.PyErr_NoMemory();
    }
    return none();
}

fn listPop(o: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
    var i_obj: ?*PyObject = null;
    if (py.c.PyArg_UnpackTuple(args, "pop", 0, 1, &i_obj) == 0) return null;
    const p = asProxy(o);
    const l = list(p);
    if (l.len == 0) {
        ph.raise(py.PyExc_IndexError(), "pop from empty list", .{});
        return null;
    }
    var i: isize = -1;
    if (i_obj) |io| {
        i = py.c.PyNumber_AsSsize_t(io, null);
        if (i == -1 and py.c.PyErr_Occurred() != null) return null;
    }
    const k = seqIndex(i, l.len, "pop") orelse return null;
    const v = listRemoveAt(l, k);
    defer value.decref(v);
    return out(p, v);
}

fn listIndexOf(p: *Proxy, x: *PyObject) ?usize {
    for (list(p).slice(), 0..) |v, i| {
        const r = equalTo(p, v, x);
        if (r < 0) return null;
        if (r == 1) return i;
    }
    return std.math.maxInt(usize);
}

fn listRemove(o: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const k = listIndexOf(p, x.?) orelse return null;
    if (k == std.math.maxInt(usize)) {
        ph.raise(py.PyExc_ValueError(), "list.remove(x): x not in list", .{});
        return null;
    }
    value.decref(listRemoveAt(list(p), k));
    return none();
}

fn listIndex(o: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const k = listIndexOf(p, x.?) orelse return null;
    if (k == std.math.maxInt(usize)) {
        const r = py.c.PyObject_Repr(x.?) orelse return null;
        defer py.Py_DecRef(r);
        ph.raise(py.PyExc_ValueError(), "{s} is not in list", .{ph.utf8(r, "repr") orelse "?"});
        return null;
    }
    return py.c.PyLong_FromSize_t(k);
}

fn listCount(o: ?*PyObject, x: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    var n: usize = 0;
    for (list(p).slice()) |v| {
        const r = equalTo(p, v, x.?);
        if (r < 0) return null;
        if (r == 1) n += 1;
    }
    return py.c.PyLong_FromSize_t(n);
}

fn listClear(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    const l = list(asProxy(o));
    while (l.len > 0) value.decref(listRemoveAt(l, l.len - 1));
    return none();
}

fn listReverse(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    std.mem.reverse(Value, list(asProxy(o)).slice());
    return none();
}

fn listCopyMethod(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const l = list(p);
    const nl = value.newList(l.len) orelse return py.c.PyErr_NoMemory();
    for (l.slice()) |v| {
        value.incref(v);
        _ = value.listPush(nl, v);
    }
    const v = Value.obj(.list, &nl.head);
    defer value.decref(v);
    return make(v, maker(p));
}

/// sort(key=None, reverse=False): Python's sort of the items, written back.
fn listSort(o: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const copy = listCopy(p) orelse return null;
    defer py.Py_DecRef(copy);
    const sort = py.c.PyObject_GetAttrString(copy, "sort") orelse return null;
    defer py.Py_DecRef(sort);
    const r = py.c.PyObject_Call(sort, args.?, kwargs) orelse return null;
    py.Py_DecRef(r);
    const l = list(p);
    for (l.slice(), 0..) |*slot, i| {
        const v = in(py.c.PyList_GetItem(copy, @intCast(i)).?) orelse return null;
        const old = slot.*;
        slot.* = v;
        value.listStored(l, v);
        value.decref(old);
    }
    return none();
}

fn listConcat(o: ?*PyObject, other: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    if (!(typeIs(other.?, ListType) or py.PyList_Check(other.?))) {
        ph.raise(py.PyExc_TypeError(), "can only concatenate list (not \"{s}\") to list", .{typeNameOf(other.?)});
        return null;
    }
    const r = listCopyMethod(o, null) orelse return null;
    const e = listExtend(r, other) orelse {
        py.Py_DecRef(r);
        return null;
    };
    py.Py_DecRef(e);
    _ = p;
    return r;
}

/// a + b with a proxy on either side (a list on the left can't add a
/// proxy itself: Python asks the proxy): a new list of both's items.
fn listAdd(a: ?*PyObject, b: ?*PyObject) callconv(.c) ?*PyObject {
    const left_ok = typeIs(a.?, ListType) or py.PyList_Check(a.?);
    const right_ok = typeIs(b.?, ListType) or py.PyList_Check(b.?);
    if (!left_ok or !right_ok) return ref(py.c.Py_NotImplemented());
    const p = if (typeIs(a.?, ListType)) asProxy(a) else asProxy(b);
    const nl = value.newList(0) orelse return py.c.PyErr_NoMemory();
    const v = Value.obj(.list, &nl.head);
    defer value.decref(v);
    const r = make(v, maker(p)) orelse return null;
    inline for (.{ a, b }) |side| {
        const e = listExtend(r, side) orelse {
            py.Py_DecRef(r);
            return null;
        };
        py.Py_DecRef(e);
    }
    return r;
}

fn listInplaceConcat(o: ?*PyObject, other: ?*PyObject) callconv(.c) ?*PyObject {
    const e = listExtend(o, other) orelse return null;
    py.Py_DecRef(e);
    return ref(o.?);
}

fn listRepeat(o: ?*PyObject, n: isize) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const l = list(p);
    const times: usize = if (n > 0) @intCast(n) else 0;
    const nl = value.newList(l.len * times) orelse return py.c.PyErr_NoMemory();
    for (0..times) |_| {
        for (l.slice()) |v| {
            value.incref(v);
            _ = value.listPush(nl, v);
        }
    }
    const v = Value.obj(.list, &nl.head);
    defer value.decref(v);
    return make(v, maker(p));
}

fn getListClass(_: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    return ref(@ptrCast(@alignCast(py.types.typeObject("PyList_Type"))));
}

fn getDictClass(_: ?*PyObject, _: ?*anyopaque) callconv(.c) ?*PyObject {
    return ref(@ptrCast(@alignCast(py.types.typeObject("PyDict_Type"))));
}

fn method(comptime name: [:0]const u8, comptime f: anytype, comptime flags: c_int) py.c.PyMethodDef {
    return .{ .ml_name = name, .ml_meth = @ptrCast(@constCast(&f)), .ml_flags = flags, .ml_doc = null };
}

const end_method = py.c.PyMethodDef{ .ml_name = null, .ml_meth = null, .ml_flags = 0, .ml_doc = null };
const end_getset = py.c.PyGetSetDef{ .name = null, .get = null, .set = null, .doc = null, .closure = null };

var list_methods = [_]py.c.PyMethodDef{
    method("append", listAppend, py.c.METH_O),
    method("extend", listExtend, py.c.METH_O),
    method("insert", listInsert, py.c.METH_VARARGS),
    method("pop", listPop, py.c.METH_VARARGS),
    method("remove", listRemove, py.c.METH_O),
    method("index", listIndex, py.c.METH_O),
    method("count", listCount, py.c.METH_O),
    method("clear", listClear, py.c.METH_NOARGS),
    method("reverse", listReverse, py.c.METH_NOARGS),
    method("copy", listCopyMethod, py.c.METH_NOARGS),
    method("sort", listSort, py.c.METH_VARARGS | py.c.METH_KEYWORDS),
    end_method,
};

var list_getset = [_]py.c.PyGetSetDef{
    .{ .name = "__class__", .get = @ptrCast(@constCast(&getListClass)), .set = null, .doc = null, .closure = null },
    end_getset,
};

var list_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&dealloc)) },
    .{ .slot = py.c.Py_sq_length, .pfunc = @ptrCast(@constCast(&listLen)) },
    .{ .slot = py.c.Py_mp_length, .pfunc = @ptrCast(@constCast(&listLen)) },
    .{ .slot = py.c.Py_sq_item, .pfunc = @ptrCast(@constCast(&listItem)) },
    .{ .slot = py.c.Py_mp_subscript, .pfunc = @ptrCast(@constCast(&listSubscript)) },
    .{ .slot = py.c.Py_mp_ass_subscript, .pfunc = @ptrCast(@constCast(&listAssSubscript)) },
    .{ .slot = py.c.Py_sq_contains, .pfunc = @ptrCast(@constCast(&listContains)) },
    .{ .slot = py.c.Py_sq_concat, .pfunc = @ptrCast(@constCast(&listConcat)) },
    .{ .slot = py.c.Py_nb_add, .pfunc = @ptrCast(@constCast(&listAdd)) },
    .{ .slot = py.c.Py_sq_inplace_concat, .pfunc = @ptrCast(@constCast(&listInplaceConcat)) },
    .{ .slot = py.c.Py_sq_repeat, .pfunc = @ptrCast(@constCast(&listRepeat)) },
    .{ .slot = py.c.Py_tp_iter, .pfunc = @ptrCast(@constCast(&listIter)) },
    .{ .slot = py.c.Py_tp_repr, .pfunc = @ptrCast(@constCast(&listRepr)) },
    .{ .slot = py.c.Py_tp_richcompare, .pfunc = @ptrCast(@constCast(&containerCompare)) },
    .{ .slot = py.c.Py_tp_hash, .pfunc = @ptrCast(@constCast(&py.c.PyObject_HashNotImplemented)) },
    .{ .slot = py.c.Py_tp_methods, .pfunc = @ptrCast(&list_methods) },
    .{ .slot = py.c.Py_tp_getset, .pfunc = @ptrCast(&list_getset) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("A list of compiled code, shared with it (isinstance(x, list) is true).")) },
    .{ .slot = 0, .pfunc = null },
};

var list_spec = py.c.PyType_Spec{
    .name = "zrun.List",
    .basicsize = @sizeOf(Proxy),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &list_slots,
};

// ======================================================================
// zrun.Dict
// ======================================================================

fn dictLen(o: ?*PyObject) callconv(.c) isize {
    return @intCast(dict(asProxy(o)).len);
}

/// A key as a value (a new reference), or null with TypeError if it
/// can't be one (unhashable).
fn keyOf(k: *PyObject) ?Value {
    const v = in(k) orelse return null;
    if (!value.hashable(v)) {
        value.decref(v);
        // (as this Python words it: from 3.14 saying where it was used)
        const name = typeNameOf(k);
        if (ph.minor >= 14)
            ph.raise(py.PyExc_TypeError(), "cannot use '{s}' as a dict key (unhashable type: '{s}')", .{ name, name })
        else
            ph.raise(py.PyExc_TypeError(), "unhashable type: '{s}'", .{name});
        return null;
    }
    return v;
}

fn dictSubscript(o: ?*PyObject, key: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const k = keyOf(key.?) orelse return null;
    defer value.decref(k);
    const v = value.dictGet(dict(p), k) orelse {
        py.c.PyErr_SetObject(py.c.PyExc_KeyError, key.?);
        return null;
    };
    return out(p, v);
}

fn dictAssSubscript(o: ?*PyObject, key: ?*PyObject, v: ?*PyObject) callconv(.c) c_int {
    const p = asProxy(o);
    const k = keyOf(key.?) orelse return -1;
    defer value.decref(k);
    if (v) |x| {
        const nv = in(x) orelse return -1;
        defer value.decref(nv);
        if (!value.dictSet(dict(p), k, nv)) {
            _ = py.c.PyErr_NoMemory();
            return -1;
        }
        return 0;
    }
    if (!value.dictDelete(dict(p), k)) {
        py.c.PyErr_SetObject(py.c.PyExc_KeyError, key.?);
        return -1;
    }
    return 0;
}

fn dictContains(o: ?*PyObject, key: ?*PyObject) callconv(.c) c_int {
    const k = keyOf(key.?) orelse return -1;
    defer value.decref(k);
    return @intFromBool(value.dictGet(dict(asProxy(o)), k) != null);
}

/// A Python dict with the proxy's entries (a new reference: a copy).
fn dictCopy(p: *Proxy) ?*PyObject {
    const d = py.c.PyDict_New() orelse return null;
    for (value.dictEntries(dict(p))) |e| {
        if (value.isDeleted(e)) continue;
        const k = out(p, e.key) orelse {
            py.Py_DecRef(d);
            return null;
        };
        defer py.Py_DecRef(k);
        const v = out(p, e.value) orelse {
            py.Py_DecRef(d);
            return null;
        };
        defer py.Py_DecRef(v);
        if (py.c.PyDict_SetItem(d, k, v) != 0) {
            py.Py_DecRef(d);
            return null;
        }
    }
    return d;
}

/// The keys, values or (key, value) pairs, in order, as a Python list.
fn dictList(p: *Proxy, what: enum { keys, values, items }) ?*PyObject {
    const l = py.c.PyList_New(0) orelse return null;
    for (value.dictEntries(dict(p))) |e| {
        if (value.isDeleted(e)) continue;
        const item = switch (what) {
            .keys => out(p, e.key),
            .values => out(p, e.value),
            .items => blk: {
                const k = out(p, e.key) orelse break :blk null;
                defer py.Py_DecRef(k);
                const v = out(p, e.value) orelse break :blk null;
                defer py.Py_DecRef(v);
                break :blk py.c.PyTuple_Pack(2, k, v);
            },
        } orelse {
            py.Py_DecRef(l);
            return null;
        };
        defer py.Py_DecRef(item);
        if (py.c.PyList_Append(l, item) != 0) {
            py.Py_DecRef(l);
            return null;
        }
    }
    return l;
}

fn dictIter(o: ?*PyObject) callconv(.c) ?*PyObject {
    const keys = dictList(asProxy(o), .keys) orelse return null;
    defer py.Py_DecRef(keys);
    return py.c.PyObject_GetIter(keys);
}

fn dictRepr(o: ?*PyObject) callconv(.c) ?*PyObject {
    const r = py.c.Py_ReprEnter(o);
    if (r != 0) return if (r > 0) ph.newString("{...}") else null;
    defer py.c.Py_ReprLeave(o);
    const copy = dictCopy(asProxy(o)) orelse return null;
    defer py.Py_DecRef(copy);
    return py.c.PyObject_Repr(copy);
}

fn dictKeys(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    return dictList(asProxy(o), .keys);
}

fn dictValues(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    return dictList(asProxy(o), .values);
}

fn dictItems(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    return dictList(asProxy(o), .items);
}

fn dictGetMethod(o: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
    var key: ?*PyObject = null;
    var default: ?*PyObject = null;
    if (py.c.PyArg_UnpackTuple(args, "get", 1, 2, &key, &default) == 0) return null;
    const p = asProxy(o);
    const k = keyOf(key.?) orelse return null;
    defer value.decref(k);
    if (value.dictGet(dict(p), k)) |v| return out(p, v);
    return ref(default orelse py.Py_None());
}

fn dictPop(o: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
    var key: ?*PyObject = null;
    var default: ?*PyObject = null;
    if (py.c.PyArg_UnpackTuple(args, "pop", 1, 2, &key, &default) == 0) return null;
    const p = asProxy(o);
    const k = keyOf(key.?) orelse return null;
    defer value.decref(k);
    if (value.dictGet(dict(p), k)) |v| {
        const r = out(p, v) orelse return null;
        _ = value.dictDelete(dict(p), k);
        return r;
    }
    if (default) |d| return ref(d);
    py.c.PyErr_SetObject(py.c.PyExc_KeyError, key.?);
    return null;
}

fn dictSetdefault(o: ?*PyObject, args: ?*PyObject) callconv(.c) ?*PyObject {
    var key: ?*PyObject = null;
    var default: ?*PyObject = null;
    if (py.c.PyArg_UnpackTuple(args, "setdefault", 1, 2, &key, &default) == 0) return null;
    const p = asProxy(o);
    const k = keyOf(key.?) orelse return null;
    defer value.decref(k);
    if (value.dictGet(dict(p), k)) |v| return out(p, v);
    const d = default orelse py.Py_None();
    const nv = in(d) orelse return null;
    defer value.decref(nv);
    if (!value.dictSet(dict(p), k, nv)) return py.c.PyErr_NoMemory();
    return out(p, nv);
}

fn dictUpdate(o: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    // (Python's dict update rules, through a plain dict made of the arguments)
    const tmp = py.c.PyDict_New() orelse return null;
    defer py.Py_DecRef(tmp);
    const upd = py.c.PyObject_GetAttrString(tmp, "update") orelse return null;
    defer py.Py_DecRef(upd);
    const r = py.c.PyObject_Call(upd, args.?, kwargs) orelse return null;
    py.Py_DecRef(r);
    var pos: isize = 0;
    var k: ?*PyObject = null;
    var v: ?*PyObject = null;
    while (py.c.PyDict_Next(tmp, &pos, &k, &v) != 0) {
        if (dictAssSubscript(o, k, v) != 0) return null;
    }
    return none();
}

fn dictClear(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    const d = dict(asProxy(o));
    for (value.dictEntries(d)) |e| {
        if (value.isDeleted(e)) continue;
        _ = value.dictDelete(d, e.key);
    }
    return none();
}

fn dictCopyMethod(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const nd = value.newDict() orelse return py.c.PyErr_NoMemory();
    const v = Value.obj(.dict, &nd.head);
    defer value.decref(v);
    for (value.dictEntries(dict(p))) |e| {
        if (value.isDeleted(e)) continue;
        if (!value.dictSet(nd, e.key, e.value)) return py.c.PyErr_NoMemory();
    }
    return make(v, maker(p));
}

fn dictPopitem(o: ?*PyObject, _: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const d = dict(p);
    const entries = value.dictEntries(d);
    var i = entries.len;
    while (i > 0) {
        i -= 1;
        const e = entries[i];
        if (value.isDeleted(e)) continue;
        const k = out(p, e.key) orelse return null;
        defer py.Py_DecRef(k);
        const v = out(p, e.value) orelse return null;
        defer py.Py_DecRef(v);
        const pair = py.c.PyTuple_Pack(2, k, v) orelse return null;
        _ = value.dictDelete(d, e.key);
        return pair;
    }
    ph.raise(py.PyExc_KeyError(), "popitem(): dictionary is empty", .{});
    return null;
}

var dict_methods = [_]py.c.PyMethodDef{
    method("keys", dictKeys, py.c.METH_NOARGS),
    method("values", dictValues, py.c.METH_NOARGS),
    method("items", dictItems, py.c.METH_NOARGS),
    method("get", dictGetMethod, py.c.METH_VARARGS),
    method("pop", dictPop, py.c.METH_VARARGS),
    method("setdefault", dictSetdefault, py.c.METH_VARARGS),
    method("update", dictUpdate, py.c.METH_VARARGS | py.c.METH_KEYWORDS),
    method("clear", dictClear, py.c.METH_NOARGS),
    method("copy", dictCopyMethod, py.c.METH_NOARGS),
    method("popitem", dictPopitem, py.c.METH_NOARGS),
    end_method,
};

var dict_getset = [_]py.c.PyGetSetDef{
    .{ .name = "__class__", .get = @ptrCast(@constCast(&getDictClass)), .set = null, .doc = null, .closure = null },
    end_getset,
};

var dict_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&dealloc)) },
    .{ .slot = py.c.Py_mp_length, .pfunc = @ptrCast(@constCast(&dictLen)) },
    .{ .slot = py.c.Py_mp_subscript, .pfunc = @ptrCast(@constCast(&dictSubscript)) },
    .{ .slot = py.c.Py_mp_ass_subscript, .pfunc = @ptrCast(@constCast(&dictAssSubscript)) },
    .{ .slot = py.c.Py_sq_contains, .pfunc = @ptrCast(@constCast(&dictContains)) },
    .{ .slot = py.c.Py_tp_iter, .pfunc = @ptrCast(@constCast(&dictIter)) },
    .{ .slot = py.c.Py_tp_repr, .pfunc = @ptrCast(@constCast(&dictRepr)) },
    .{ .slot = py.c.Py_tp_richcompare, .pfunc = @ptrCast(@constCast(&containerCompare)) },
    .{ .slot = py.c.Py_tp_hash, .pfunc = @ptrCast(@constCast(&py.c.PyObject_HashNotImplemented)) },
    .{ .slot = py.c.Py_tp_methods, .pfunc = @ptrCast(&dict_methods) },
    .{ .slot = py.c.Py_tp_getset, .pfunc = @ptrCast(&dict_getset) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("A dict of compiled code, shared with it (isinstance(x, dict) is true).")) },
    .{ .slot = 0, .pfunc = null },
};

var dict_spec = py.c.PyType_Spec{
    .name = "zrun.Dict",
    .basicsize = @sizeOf(Proxy),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &dict_slots,
};

// ======================================================================
// zrun.Record
// ======================================================================

fn fieldIndex(r: *value.Record, name: []const u8) ?usize {
    for (r.rtype.fields, 0..) |f, i| if (std.mem.eql(u8, f, name)) return i;
    return null;
}

fn recordClass(r: *value.Record) ?*PyObject {
    return r.rtype.py_class;
}

fn recordGetattro(o: ?*PyObject, name: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const r = record(p);
    const n = ph.utf8(name.?, "attribute") orelse return null;
    if (fieldIndex(r, n)) |i| {
        if (r.fields()[i].tag == value.UNSET_TAG) {
            ph.raise(py.PyExc_AttributeError(), "'{s}' object has no attribute '{s}'", .{ r.rtype.unset_name, n });
            return null;
        }
        return out(p, r.fields()[i]);
    }
    if (std.mem.eql(u8, n, "__class__")) return ref(recordClass(r) orelse return py.c.PyObject_GenericGetAttr(o, name));
    // The class's: methods bound to the proxy, as an instance's would be
    const cls = recordClass(r) orelse return py.c.PyObject_GenericGetAttr(o, name);
    const attr = py.c.PyObject_GetAttr(cls, name.?) orelse {
        py.c.PyErr_Clear();
        ph.raise(py.PyExc_AttributeError(), "'{s}' object has no attribute '{s}'", .{ r.rtype.name, n });
        return null;
    };
    defer py.Py_DecRef(attr);
    if (py.c.PyObject_HasAttrString(attr, "__get__") == 1 and py.c.PyObject_IsInstance(attr, @ptrCast(@alignCast(py.types.typeObject("PyType_Type")))) == 0) {
        return py.c.PyObject_CallMethod(attr, "__get__", "OO", o.?, cls);
    }
    return ref(attr);
}

fn recordSetattro(o: ?*PyObject, name: ?*PyObject, v: ?*PyObject) callconv(.c) c_int {
    const p = asProxy(o);
    const r = record(p);
    const n = ph.utf8(name.?, "attribute") orelse return -1;
    const i = fieldIndex(r, n) orelse {
        // (from 3.13 Python says why it can't be added)
        if (ph.minor >= 13)
            ph.raise(py.PyExc_AttributeError(), "'{s}' object has no attribute '{s}' and no __dict__ for setting new attributes", .{ r.rtype.name, n })
        else
            ph.raise(py.PyExc_AttributeError(), "'{s}' object has no attribute '{s}'", .{ r.rtype.name, n });
        return -1;
    };
    if (r.rtype.frozen) {
        // (dataclasses.FrozenInstanceError, as the dataclass raises it)
        const dataclasses = py.c.PyImport_ImportModule("dataclasses") orelse return -1;
        defer py.Py_DecRef(dataclasses);
        const exc = py.c.PyObject_GetAttrString(dataclasses, "FrozenInstanceError") orelse return -1;
        defer py.Py_DecRef(exc);
        ph.raise(exc, "cannot assign to field '{s}'", .{n});
        return -1;
    }
    const x = v orelse {
        // (del obj.field: unset again, as a slot's)
        if (r.rtype.slots and r.fields()[i].tag != value.UNSET_TAG) {
            const old = r.fields()[i];
            r.fields()[i] = value.unset;
            value.decref(old);
            return 0;
        }
        ph.raise(py.PyExc_AttributeError(), "'{s}' object has no attribute '{s}'", .{ r.rtype.name, n });
        return -1;
    };
    const nv = in(x) orelse return -1;
    const old = r.fields()[i];
    r.fields()[i] = nv;
    value.decref(old);
    return 0;
}

/// Through the class's own method (a dataclass's __repr__, __eq__...),
/// with the proxy as self; null without one of its own.
fn classMethod(r: *value.Record, name: [*:0]const u8) ?*PyObject {
    const cls = recordClass(r) orelse return null;
    const d = py.c.PyObject_GetAttrString(cls, "__dict__") orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(d);
    // (the class's or a base's other than object's)
    const m = py.c.PyObject_GetAttrString(cls, name) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    const object_type: *PyObject = @ptrCast(@alignCast(py.types.typeObject("PyBaseObject_Type")));
    const base = py.c.PyObject_GetAttrString(object_type, name);
    defer if (base) |b| py.Py_DecRef(b) else py.c.PyErr_Clear();
    if (base != null and base.? == m) {
        py.Py_DecRef(m);
        return null;
    }
    return m;
}

fn recordRepr(o: ?*PyObject) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const r = record(p);
    if (classMethod(r, "__repr__")) |m| {
        defer py.Py_DecRef(m);
        return py.c.PyObject_CallFunctionObjArgs(m, o.?, @as(?*PyObject, null));
    }
    // (object's: the class's module and qualified name)
    if (recordClass(r)) |cls| {
        const module = py.c.PyObject_GetAttrString(cls, "__module__") orelse return null;
        defer py.Py_DecRef(module);
        const qual = py.c.PyObject_GetAttrString(cls, "__qualname__") orelse return null;
        defer py.Py_DecRef(qual);
        return py.c.PyUnicode_FromFormat("<%S.%S object at %p>", module, qual, o.?);
    }
    var buf: [256]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "<{s} object at 0x{x}>", .{ r.rtype.name, @intFromPtr(o.?) }) catch "<object>";
    return ph.newString(text);
}

fn recordStr(o: ?*PyObject) callconv(.c) ?*PyObject {
    const r = record(asProxy(o));
    if (classMethod(r, "__str__")) |m| {
        defer py.Py_DecRef(m);
        return py.c.PyObject_CallFunctionObjArgs(m, o.?, @as(?*PyObject, null));
    }
    return recordRepr(o);
}

fn recordCompare(o: ?*PyObject, other: ?*PyObject, op: c_int) callconv(.c) ?*PyObject {
    const p = asProxy(o);
    const r = record(p);
    if (op == py.c.Py_EQ or op == py.c.Py_NE) {
        var same: bool = undefined;
        if (typeIs(other.?, RecordType)) {
            const a = Value.obj(.record, &r.head);
            const b = Value.obj(.record, asProxy(other).obj);
            same = value.equal(a, b);
        } else if (!r.rtype.value_eq) {
            same = false;
        } else {
            return ref(py.c.Py_NotImplemented());
        }
        const want = if (op == py.c.Py_EQ) same else !same;
        return ref(if (want) py.Py_True() else py.Py_False());
    }
    return ref(py.c.Py_NotImplemented());
}

fn recordHash(o: ?*PyObject) callconv(.c) isize {
    const r = record(asProxy(o));
    // (a frozen dataclass compared by value: its fields' tuple's, as the
    // dataclass's __hash__; one not frozen: unhashable, as it is)
    if (r.rtype.value_eq) {
        if (!r.rtype.frozen) return py.c.PyObject_HashNotImplemented(o);
        return valueHash(r) orelse -1;
    }
    // (a plain object's: its identity, the native object's)
    return @bitCast(@intFromPtr(r) >> 4);
}

/// The hash Python gives a frozen dataclass's object compared by value:
/// its fields' tuple's (null with an exception).
pub fn valueHash(r: *value.Record) ?isize {
    const fields = r.fields();
    const t = py.c.PyTuple_New(@intCast(fields.len)) orelse return null;
    defer py.Py_DecRef(t);
    for (fields, 0..) |x, i| {
        const o = value.toPython(x, @import("adopt.zig").shared_maker) orelse return null;
        _ = py.c.PyTuple_SetItem(t, @intCast(i), o);
    }
    const h = py.c.PyObject_Hash(t);
    return if (h == -1) null else h;
}

var record_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&dealloc)) },
    .{ .slot = py.c.Py_tp_getattro, .pfunc = @ptrCast(@constCast(&recordGetattro)) },
    .{ .slot = py.c.Py_tp_setattro, .pfunc = @ptrCast(@constCast(&recordSetattro)) },
    .{ .slot = py.c.Py_tp_repr, .pfunc = @ptrCast(@constCast(&recordRepr)) },
    .{ .slot = py.c.Py_tp_str, .pfunc = @ptrCast(@constCast(&recordStr)) },
    .{ .slot = py.c.Py_tp_richcompare, .pfunc = @ptrCast(@constCast(&recordCompare)) },
    .{ .slot = py.c.Py_tp_hash, .pfunc = @ptrCast(@constCast(&recordHash)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("An object of compiled code (a record of a class), shared with it: its fields, its class's methods; isinstance() with its class is true.")) },
    .{ .slot = 0, .pfunc = null },
};

var record_spec = py.c.PyType_Spec{
    .name = "zrun.Record",
    .basicsize = @sizeOf(Proxy),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &record_slots,
};

pub fn init(module: *PyObject) !void {
    ListType = py.c.PyType_FromSpec(&list_spec) orelse return error.Python;
    DictType = py.c.PyType_FromSpec(&dict_spec) orelse return error.Python;
    RecordType = py.c.PyType_FromSpec(&record_spec) orelse return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "List", ListType) != 0) return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "Dict", DictType) != 0) return error.Python;
    if (py.c.PyModule_AddObjectRef(module, "Record", RecordType) != 0) return error.Python;
}
