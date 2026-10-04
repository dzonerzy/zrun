//! Values of compiled code, and the runtime helpers it calls.
//!
//! A value is 16 bytes: a tag and 64 bits (an int, a float's bits, a bool,
//! a pointer). Strings, lists, tuples, dicts, records, functions and frames
//! are heap objects with a reference count; literals are immortal (never
//! counted, never freed). A Python object (a host function, a value from
//! one) is a `host` value. What semantics do with values follows Python:
//! the same results, the same errors with the same messages as the
//! reference mode, integers checked at 64 bits.
//!
//! Helpers called from compiled code take values as (tag, bits) pairs,
//! borrow them, and return new references through an out pointer; they
//! report errors in the execution context (`Ctx`) and return false.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const types = @import("types.zig");
const objects = @import("objects.zig");

const allocator = std.heap.c_allocator;

pub const Tag = enum(u64) {
    none = 0,
    bool = 1,
    int = 2,
    float = 3,
    str = 4,
    list = 5,
    tuple = 6,
    dict = 7,
    record = 8,
    function = 9,
    /// A Python object (owned reference)
    host = 10,
    /// A node of the program (its index)
    node = 11,
    _,
};

pub const Value = extern struct {
    tag: u64,
    bits: u64,

    pub const none_v = Value{ .tag = @intFromEnum(Tag.none), .bits = 0 };

    pub fn int(v: i64) Value {
        return .{ .tag = @intFromEnum(Tag.int), .bits = @bitCast(v) };
    }

    pub fn float(v: f64) Value {
        return .{ .tag = @intFromEnum(Tag.float), .bits = @bitCast(v) };
    }

    pub fn boolean(v: bool) Value {
        return .{ .tag = @intFromEnum(Tag.bool), .bits = @intFromBool(v) };
    }

    pub fn obj(tag: Tag, o: *Obj) Value {
        return .{ .tag = @intFromEnum(tag), .bits = @intFromPtr(o) };
    }

    pub fn kind(self: Value) Tag {
        return @enumFromInt(self.tag);
    }

    pub fn asInt(self: Value) i64 {
        return @bitCast(self.bits);
    }

    pub fn asFloat(self: Value) f64 {
        return @bitCast(self.bits);
    }

    pub fn ptr(self: Value) *Obj {
        return @ptrFromInt(self.bits);
    }

    pub fn isHeap(self: Value) bool {
        return switch (self.kind()) {
            .str, .list, .tuple, .dict, .record, .function => true,
            else => false,
        };
    }
};

// ======================================================================
// Heap objects
// ======================================================================

/// Reference counts at or above this are immortal (literals)
pub const IMMORTAL: u64 = 1 << 62;

pub const Obj = extern struct {
    rc: u64,
    kind: u32,
    flags: u32 = 0,
};

pub const Str = extern struct {
    head: Obj,
    /// Bytes (UTF-8) and code points
    len: u64,
    chars: u64,
    /// Its hash once worked out (0: not yet; literals have theirs)
    hash: u64 = 0,
    // the bytes follow

    pub fn bytes(self: *const Str) []const u8 {
        const base: [*]const u8 = @ptrCast(self);
        return (base + @sizeOf(Str))[0..self.len];
    }
};

pub const List = extern struct {
    head: Obj,
    len: u64,
    cap: u64,
    items: ?[*]Value,

    pub fn slice(self: *const List) []Value {
        return if (self.items) |p| p[0..self.len] else &.{};
    }
};

pub const Tuple = extern struct {
    head: Obj,
    len: u64,
    // the items follow

    pub fn slice(self: *Tuple) []Value {
        const base: [*]u8 = @ptrCast(self);
        const items: [*]Value = @ptrCast(@alignCast(base + @sizeOf(Tuple)));
        return items[0..self.len];
    }
};

/// A dict: entries in insertion order, and an index of them by hash
pub const Dict = extern struct {
    head: Obj,
    /// Live entries, and entries used (deleted ones stay until a resize)
    len: u64,
    used: u64,
    cap: u64,
    entries: ?[*]Entry,
    /// Open addressing over entry indices + 1 (0: empty); cap * 2 slots
    index: ?[*]u32,

    pub const Entry = extern struct { key: Value, value: Value, hash: u64 };
};

pub const RecordType = struct {
    name: []const u8,
    fields: []const []const u8,
    /// The Python class it was made from (a dataclass), for converting
    py_class: ?*PyObject,
};

pub const Record = extern struct {
    head: Obj,
    rtype: *const RecordType,
    // the fields follow

    pub fn fields(self: *Record) []Value {
        const base: [*]u8 = @ptrCast(self);
        const items: [*]Value = @ptrCast(@alignCast(base + @sizeOf(Record)));
        return items[0..self.rtype.fields.len];
    }
};

/// A function of the program: its compiled code, the frame it was made
/// in, its node and name
pub const Function = extern struct {
    head: Obj,
    code: ?*const anyopaque,
    env: ?*Frame,
    node: u64,
    name: *Str,
};

/// The variables of a run of a function that functions made in it see (a
/// heap frame); the program's own is the globals
pub const Frame = extern struct {
    head: Obj,
    parent: ?*Frame,
    len: u64,
    // the slots follow

    pub fn slots(self: *Frame) []Value {
        const base: [*]u8 = @ptrCast(self);
        const items: [*]Value = @ptrCast(@alignCast(base + @sizeOf(Frame)));
        return items[0..self.len];
    }
};

pub const KIND_FRAME: u32 = 100;

// ----------------------------------------------------------------------
// Reference counting
// ----------------------------------------------------------------------

pub fn incref(v: Value) void {
    if (v.isHeap()) {
        const o = v.ptr();
        if (o.rc < IMMORTAL) o.rc += 1;
    } else if (v.kind() == .host) {
        py.Py_IncRef(@ptrFromInt(v.bits));
    }
}

pub fn decref(v: Value) void {
    if (v.isHeap()) {
        const o = v.ptr();
        if (o.rc >= IMMORTAL) return;
        o.rc -= 1;
        if (o.rc == 0) free(v.kind(), o);
    } else if (v.kind() == .host) {
        py.Py_DecRef(@ptrFromInt(v.bits));
    }
}

pub fn increfObj(o: *Obj) void {
    if (o.rc < IMMORTAL) o.rc += 1;
}

pub fn decrefFrame(f: *Frame) void {
    if (f.head.rc >= IMMORTAL) return;
    f.head.rc -= 1;
    if (f.head.rc == 0) freeFrame(f);
}

fn freeFrame(f: *Frame) void {
    for (f.slots()) |s| decref(s);
    if (f.parent) |p| decrefFrame(p);
    allocator.free(@as([*]u8, @ptrCast(f))[0 .. @sizeOf(Frame) + f.len * @sizeOf(Value)]);
}

pub fn free(tag: Tag, o: *Obj) void {
    switch (tag) {
        .str => {
            const s: *Str = @ptrCast(o);
            allocator.free(@as([*]u8, @ptrCast(s))[0 .. @sizeOf(Str) + s.len]);
        },
        .list => {
            const l: *List = @ptrCast(o);
            for (l.slice()) |item| decref(item);
            if (l.items) |p| allocator.free(p[0..l.cap]);
            allocator.destroy(l);
        },
        .tuple => {
            const t: *Tuple = @ptrCast(@alignCast(o));
            for (t.slice()) |item| decref(item);
            allocator.free(@as([*]align(8) u8, @ptrCast(t))[0 .. @sizeOf(Tuple) + t.len * @sizeOf(Value)]);
        },
        .dict => {
            const d: *Dict = @ptrCast(@alignCast(o));
            if (d.entries) |es| {
                for (es[0..d.used]) |e| {
                    if (e.key.tag == DELETED) continue;
                    decref(e.key);
                    decref(e.value);
                }
                allocator.free(es[0..d.cap]);
            }
            if (d.index) |ix| allocator.free(ix[0 .. d.cap * 2]);
            allocator.destroy(d);
        },
        .record => {
            const r: *Record = @ptrCast(@alignCast(o));
            for (r.fields()) |f| decref(f);
            const n = r.rtype.fields.len;
            allocator.free(@as([*]align(8) u8, @ptrCast(r))[0 .. @sizeOf(Record) + n * @sizeOf(Value)]);
        },
        .function => {
            const f: *Function = @ptrCast(@alignCast(o));
            if (f.env) |e| decrefFrame(e);
            decref(Value.obj(.str, &f.name.head));
            allocator.destroy(f);
        },
        else => {},
    }
}

/// A tag for deleted dict entries
const DELETED: u64 = 0xFFFF_FFFF;

// ----------------------------------------------------------------------
// Making objects
// ----------------------------------------------------------------------

pub fn newStr(bytes: []const u8) ?*Str {
    const mem = allocator.alignedAlloc(u8, .of(Str), @sizeOf(Str) + bytes.len) catch return null;
    const s: *Str = @ptrCast(mem.ptr);
    s.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.str) }, .len = bytes.len, .chars = std.unicode.utf8CountCodepoints(bytes) catch bytes.len };
    @memcpy((mem.ptr + @sizeOf(Str))[0..bytes.len], bytes);
    return s;
}

pub fn newList(cap: usize) ?*List {
    const l = allocator.create(List) catch return null;
    l.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.list) }, .len = 0, .cap = 0, .items = null };
    if (cap > 0) {
        const items = allocator.alloc(Value, cap) catch {
            allocator.destroy(l);
            return null;
        };
        l.items = items.ptr;
        l.cap = cap;
    }
    return l;
}

/// Append, taking the reference.
pub fn listPush(l: *List, v: Value) bool {
    if (l.len == l.cap) {
        const cap = @max(4, l.cap * 2);
        const old = if (l.items) |p| p[0..l.cap] else &[_]Value{};
        const items = allocator.realloc(@constCast(old), cap) catch return false;
        l.items = items.ptr;
        l.cap = cap;
    }
    l.items.?[l.len] = v;
    l.len += 1;
    return true;
}

pub fn newTuple(n: usize) ?*Tuple {
    const mem = allocator.alignedAlloc(u8, .of(Tuple), @sizeOf(Tuple) + n * @sizeOf(Value)) catch return null;
    const t: *Tuple = @ptrCast(mem.ptr);
    t.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.tuple) }, .len = n };
    @memset(t.slice(), Value.none_v);
    return t;
}

pub fn newRecord(rtype: *const RecordType) ?*Record {
    const n = rtype.fields.len;
    const mem = allocator.alignedAlloc(u8, .of(Record), @sizeOf(Record) + n * @sizeOf(Value)) catch return null;
    const r: *Record = @ptrCast(mem.ptr);
    r.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.record) }, .rtype = rtype };
    @memset(r.fields(), Value.none_v);
    return r;
}

pub fn newFrame(parent: ?*Frame, n: usize) ?*Frame {
    const mem = allocator.alignedAlloc(u8, .of(Frame), @sizeOf(Frame) + n * @sizeOf(Value)) catch return null;
    const f: *Frame = @ptrCast(mem.ptr);
    if (parent) |p| increfObj(&p.head);
    f.* = .{ .head = .{ .rc = 1, .kind = KIND_FRAME }, .parent = parent, .len = n };
    @memset(f.slots(), Value.none_v);
    return f;
}

pub fn newDict() ?*Dict {
    const d = allocator.create(Dict) catch return null;
    d.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.dict) }, .len = 0, .used = 0, .cap = 0, .entries = null, .index = null };
    return d;
}

// ----------------------------------------------------------------------
// Type names, truth, equality, hashing (Python's)
// ----------------------------------------------------------------------

pub fn typeName(v: Value) []const u8 {
    return switch (v.kind()) {
        .none => "NoneType",
        .bool => "bool",
        .int => "int",
        .float => "float",
        .str => "str",
        .list => "list",
        .tuple => "tuple",
        .dict => "dict",
        .record => blk: {
            const r: *Record = @ptrCast(@alignCast(v.ptr()));
            break :blk r.rtype.name;
        },
        .function => "function",
        .node => "Node",
        .host => "object",
        _ => "object",
    };
}

pub fn truthy(v: Value) bool {
    return switch (v.kind()) {
        .none => false,
        .bool, .int => v.bits != 0,
        .float => v.asFloat() != 0,
        .str => @as(*Str, @ptrCast(v.ptr())).len != 0,
        .list => @as(*List, @ptrCast(@alignCast(v.ptr()))).len != 0,
        .tuple => @as(*Tuple, @ptrCast(@alignCast(v.ptr()))).len != 0,
        .dict => @as(*Dict, @ptrCast(@alignCast(v.ptr()))).len != 0,
        .host => py.c.PyObject_IsTrue(@ptrFromInt(v.bits)) == 1,
        else => true,
    };
}

/// A number's value as a float, for comparing ints and floats (null: not
/// a number)
fn numeric(v: Value) ?f64 {
    return switch (v.kind()) {
        .bool, .int => @floatFromInt(v.asInt()),
        .float => v.asFloat(),
        else => null,
    };
}

fn isIntLike(v: Value) bool {
    return v.kind() == .int or v.kind() == .bool;
}

pub fn equal(a: Value, b: Value) bool {
    if (isIntLike(a) and isIntLike(b)) return a.asInt() == b.asInt();
    if (numeric(a)) |x| {
        if (numeric(b)) |y| return x == y;
        return false;
    }
    if (a.tag != b.tag) return false;
    return switch (a.kind()) {
        .none => true,
        .str => a.bits == b.bits or std.mem.eql(u8, @as(*Str, @ptrCast(a.ptr())).bytes(), @as(*Str, @ptrCast(b.ptr())).bytes()),
        .list => blk: {
            const x = @as(*List, @ptrCast(@alignCast(a.ptr()))).slice();
            const y = @as(*List, @ptrCast(@alignCast(b.ptr()))).slice();
            if (x.len != y.len) break :blk false;
            for (x, y) |p, q| if (!equal(p, q)) break :blk false;
            break :blk true;
        },
        .tuple => blk: {
            const x = @as(*Tuple, @ptrCast(@alignCast(a.ptr()))).slice();
            const y = @as(*Tuple, @ptrCast(@alignCast(b.ptr()))).slice();
            if (x.len != y.len) break :blk false;
            for (x, y) |p, q| if (!equal(p, q)) break :blk false;
            break :blk true;
        },
        .record => blk: {
            const x: *Record = @ptrCast(@alignCast(a.ptr()));
            const y: *Record = @ptrCast(@alignCast(b.ptr()));
            if (x.rtype != y.rtype) break :blk false;
            for (x.fields(), y.fields()) |p, q| if (!equal(p, q)) break :blk false;
            break :blk true;
        },
        .node => a.bits == b.bits,
        else => a.bits == b.bits,
    };
}

/// A hash where equal values hash alike (1, 1.0 and True too).
/// A string's hash (never 0: 0 marks one not worked out yet).
pub fn strHash(bytes: []const u8) u64 {
    const h = std.hash.Wyhash.hash(2, bytes);
    return if (h == 0) 1 else h;
}

pub inline fn hash(v: Value) u64 {
    return hashOf(v.tag, v.bits);
}

/// (a value's two words as scalars: a Value read back whole right after
/// its words were written stalls the CPU)
fn hashOf(tag: u64, bits: u64) u64 {
    const v = Value{ .tag = tag, .bits = bits };
    switch (v.kind()) {
        .bool, .int => return std.hash.Wyhash.hash(0, std.mem.asBytes(&bits)),
        .float => {
            const f = v.asFloat();
            if (f == @trunc(f) and @abs(f) < 9.2e18) {
                const i: i64 = @intFromFloat(f);
                return std.hash.Wyhash.hash(0, std.mem.asBytes(&i));
            }
            return std.hash.Wyhash.hash(1, std.mem.asBytes(&bits));
        },
        .str => {
            const s: *Str = @ptrCast(v.ptr());
            if (s.hash != 0) return s.hash;
            const h = strHash(s.bytes());
            // (literals are read-only: they come with theirs)
            s.hash = h;
            return h;
        },
        .tuple => {
            var h: u64 = 3;
            for (@as(*Tuple, @ptrCast(@alignCast(v.ptr()))).slice()) |item| h = h *% 0x100000001B3 ^ hashOf(item.tag, item.bits);
            return h;
        },
        else => return std.hash.Wyhash.hash(4, std.mem.asBytes(&bits)),
    }
}

pub fn hashable(v: Value) bool {
    return switch (v.kind()) {
        .list, .dict => false,
        .tuple => for (@as(*Tuple, @ptrCast(@alignCast(v.ptr()))).slice()) |item| {
            if (!hashable(item)) break false;
        } else true,
        else => true,
    };
}

// ----------------------------------------------------------------------
// Dicts
// ----------------------------------------------------------------------

fn dictFind(d: *Dict, key: Value, h: u64) ?usize {
    const ix = d.index orelse return null;
    const mask = d.cap * 2 - 1;
    var i = h & mask;
    while (true) : (i = (i + 1) & mask) {
        const slot = ix[i];
        if (slot == 0) return null;
        const e = &d.entries.?[slot - 1];
        if (e.key.tag != DELETED and e.hash == h and equal(e.key, key)) return slot - 1;
    }
}

fn dictGrow(d: *Dict) bool {
    const cap = @max(8, d.cap * 2);
    const entries = allocator.alloc(Dict.Entry, cap) catch return false;
    const index = allocator.alloc(u32, cap * 2) catch {
        allocator.free(entries);
        return false;
    };
    @memset(index, 0);
    var n: usize = 0;
    if (d.entries) |old| {
        for (old[0..d.used]) |e| {
            if (e.key.tag == DELETED) continue;
            entries[n] = e;
            n += 1;
        }
        allocator.free(old[0..d.cap]);
    }
    if (d.index) |old| allocator.free(old[0 .. d.cap * 2]);
    const mask = cap * 2 - 1;
    for (entries[0..n], 0..) |e, k| {
        var i = e.hash & mask;
        while (index[i] != 0) i = (i + 1) & mask;
        index[i] = @intCast(k + 1);
    }
    d.entries = entries.ptr;
    d.index = index.ptr;
    d.cap = cap;
    d.used = n;
    return true;
}

/// d[key] = value (borrowing both).
pub fn dictSet(d: *Dict, key: Value, value: Value) bool {
    const h = hash(key);
    if (dictFind(d, key, h)) |i| {
        const e = &d.entries.?[i];
        incref(value);
        decref(e.value);
        e.value = value;
        return true;
    }
    if (d.used == d.cap and !dictGrow(d)) return false;
    incref(key);
    incref(value);
    const k = d.used;
    d.entries.?[k] = .{ .key = key, .value = value, .hash = h };
    d.used += 1;
    d.len += 1;
    const mask = d.cap * 2 - 1;
    var i = h & mask;
    while (d.index.?[i] != 0) i = (i + 1) & mask;
    d.index.?[i] = @intCast(k + 1);
    return true;
}

/// d[key] (borrowed), or null.
pub fn dictGet(d: *Dict, key: Value) ?Value {
    const i = dictFind(d, key, hash(key)) orelse return null;
    return d.entries.?[i].value;
}

pub fn dictEntries(d: *Dict) []Dict.Entry {
    return if (d.entries) |es| es[0..d.used] else &.{};
}

pub fn isDeleted(e: Dict.Entry) bool {
    return e.key.tag == DELETED;
}

// ----------------------------------------------------------------------
// To and from Python
// ----------------------------------------------------------------------

/// A value as a Python object (new reference), or null with an exception.
/// Ints become zrun's checked ints, as the reference mode gives them.
pub fn toPython(v: Value, nodeObject: anytype) ?*PyObject {
    switch (v.kind()) {
        .none => {
            py.Py_IncRef(py.Py_None());
            return py.Py_None();
        },
        .bool => {
            const b = if (v.bits != 0) py.Py_True() else py.Py_False();
            py.Py_IncRef(b);
            return b;
        },
        .int => return types.fromInt(v.asInt()),
        .float => return py.c.PyFloat_FromDouble(v.asFloat()),
        .str => {
            const s: *Str = @ptrCast(v.ptr());
            return ph.newString(s.bytes());
        },
        .list => {
            const l: *List = @ptrCast(@alignCast(v.ptr()));
            const out = py.c.PyList_New(@intCast(l.len)) orelse return null;
            for (l.slice(), 0..) |item, i| {
                const o = toPython(item, nodeObject) orelse {
                    py.Py_DecRef(out);
                    return null;
                };
                _ = py.c.PyList_SetItem(out, @intCast(i), o);
            }
            return out;
        },
        .tuple => {
            const t: *Tuple = @ptrCast(@alignCast(v.ptr()));
            const items = t.slice();
            const out = py.c.PyTuple_New(@intCast(items.len)) orelse return null;
            for (items, 0..) |item, i| {
                const o = toPython(item, nodeObject) orelse {
                    py.Py_DecRef(out);
                    return null;
                };
                _ = py.c.PyTuple_SetItem(out, @intCast(i), o);
            }
            return out;
        },
        .dict => {
            const d: *Dict = @ptrCast(@alignCast(v.ptr()));
            const out = py.c.PyDict_New() orelse return null;
            for (dictEntries(d)) |e| {
                if (isDeleted(e)) continue;
                const k = toPython(e.key, nodeObject) orelse {
                    py.Py_DecRef(out);
                    return null;
                };
                defer py.Py_DecRef(k);
                const val = toPython(e.value, nodeObject) orelse {
                    py.Py_DecRef(out);
                    return null;
                };
                defer py.Py_DecRef(val);
                if (py.c.PyDict_SetItem(out, k, val) != 0) {
                    py.Py_DecRef(out);
                    return null;
                }
            }
            return out;
        },
        .record => {
            const r: *Record = @ptrCast(@alignCast(v.ptr()));
            const cls = r.rtype.py_class orelse {
                ph.raise(py.PyExc_TypeError(), "a {s} can't be given to Python", .{r.rtype.name});
                return null;
            };
            const fields = r.fields();
            const args = py.c.PyTuple_New(@intCast(fields.len)) orelse return null;
            defer py.Py_DecRef(args);
            for (fields, 0..) |f, i| {
                const o = toPython(f, nodeObject) orelse return null;
                _ = py.c.PyTuple_SetItem(args, @intCast(i), o);
            }
            return py.c.PyObject_CallObject(cls, args);
        },
        .host => {
            const o: *PyObject = @ptrFromInt(v.bits);
            py.Py_IncRef(o);
            return o;
        },
        .node => return nodeObject.make(@intCast(v.bits)),
        // (a zrun.Function, as the reference mode gives them)
        .function => return objects.newNativeFunction(@ptrCast(@alignCast(v.ptr()))),
        _ => {
            ph.raise(py.PyExc_TypeError(), "an unknown value", .{});
            return null;
        },
    }
}

/// A Python object as a value (a new reference), or null with an
/// exception (IntegerOverflow for an int outside 64 bits). Python lists,
/// tuples and dicts are copied; other objects are host values.
pub fn fromPython(o: *PyObject) ?Value {
    if (o == py.Py_None()) return Value.none_v;
    if (py.PyBool_Check(o)) return Value.boolean(o == py.Py_True());
    if (py.PyLong_Check(o)) {
        var overflow: c_int = 0;
        const n = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
        if (overflow != 0) {
            py.c.PyErr_SetString(types.IntegerOverflow, "integer overflow");
            return null;
        }
        return Value.int(n);
    }
    if (py.PyFloat_Check(o)) return Value.float(py.c.PyFloat_AsDouble(o));
    if (py.PyUnicode_Check(o)) {
        const s = ph.utf8(o, "str") orelse return null;
        const str = newStr(s) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        return Value.obj(.str, &str.head);
    }
    if (py.PyList_Check(o) or py.PyTuple_Check(o)) {
        const seq = py.c.PySequence_Fast(o, "") orelse return null;
        defer py.Py_DecRef(seq);
        const n: usize = @intCast(py.c.PySequence_Size(seq));
        if (py.PyTuple_Check(o)) {
            const t = newTuple(n) orelse {
                _ = py.c.PyErr_NoMemory();
                return null;
            };
            for (t.slice(), 0..) |*slot, i| {
                const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse {
                    decref(Value.obj(.tuple, &t.head));
                    return null;
                };
                defer py.Py_DecRef(item);
                slot.* = fromPython(item) orelse {
                    decref(Value.obj(.tuple, &t.head));
                    return null;
                };
            }
            return Value.obj(.tuple, &t.head);
        }
        const l = newList(n) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        for (0..n) |i| {
            const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse {
                decref(Value.obj(.list, &l.head));
                return null;
            };
            defer py.Py_DecRef(item);
            const v = fromPython(item) orelse {
                decref(Value.obj(.list, &l.head));
                return null;
            };
            _ = listPush(l, v);
        }
        return Value.obj(.list, &l.head);
    }
    // A compiled function given to Python, back: itself
    if (objects.asFunction(o)) |f| if (f.native) |n| {
        increfObj(&n.head);
        return Value.obj(.function, &n.head);
    };
    py.Py_IncRef(o);
    return .{ .tag = @intFromEnum(Tag.host), .bits = @intFromPtr(o) };
}
