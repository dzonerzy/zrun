//! Module state made native: a table or record a module holds at its top
//! level that semantics change (a language's globals, its library tables)
//! becomes native objects when a program compiling first refers to it, so
//! compiled code reads and changes it without Python. Python goes on seeing
//! the same objects: the module's names are rebound to their proxies
//! (proxies.zig: isinstance() is true, changes are seen both ways).
//!
//! The whole object graph is adopted at once (records of record classes,
//! lists, dicts, reached from the object), and only if nothing else holds
//! any of it: each object's reference count is exactly its references from
//! the graph and from modules' top levels. A frame, a closure, a class
//! attribute, a C extension holding one: the graph stays Python's (as
//! before: read through Python). Tuples and other objects in it are values
//! as they always are (copies of tuples; Python objects themselves).

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const value = @import("value.zig");
const proxies = @import("proxies.zig");
const helpers = @import("helpers.zig");
const compile_mod = @import("compile.zig");

const Value = value.Value;
const gpa = std.heap.c_allocator;

pub const Error = compile_mod.Error;

const Kind = union(enum) { record: *value.RecordType, list, dict };

const Entry = struct {
    kind: Kind,
    /// Its references from the graph, from modules' top levels
    internal: isize = 0,
    module: isize = 0,
    /// Its component (union-find: the index of an object in it)
    up: usize,
    /// Native objects can't stand for it (a key they can't hash, attributes
    /// beyond its fields): its component stays Python's
    unfit: bool = false,
    native: ?Value = null,
};

/// A module's top-level name referring into the graph
const Binding = struct { dict: *PyObject, key: *PyObject, obj: *PyObject };

/// What became of module state: not module state (a function, a constant...),
/// native (its object: borrowed, immortal as the module's), or Python's
/// still (why: Program.report()'s "module_state").
pub const Outcome = union(enum) { not_state, native: Value, refused: []const u8 };

/// The native object for module state `o`, made now if it can be (the
/// outcome). An object adopted before (a proxy of an immortal object): its
/// object.
///
/// What's adopted with it: the objects connected to it, through each other
/// and the other tables and records `globals` (its module's) holds (one
/// another's state: a metatable the globals' table don't reach...).
/// An object's type's name (copied into `a`), for messages (its
/// __name__: the type object's fields aren't the stable ABI's)
pub fn typeName(a: std.mem.Allocator, o: *PyObject) Error![]const u8 {
    const t: *PyObject = @ptrCast(@alignCast(ph.typeOf(o)));
    return (ph.attrString(a, t, "__name__") catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Python => {
            py.c.PyErr_Clear();
            return "object";
        },
    }) orelse "object";
}

pub fn adopt(a: std.mem.Allocator, o: *PyObject, globals: *PyObject) Error!Outcome {
    if (proxies.objectOf(o)) |v| {
        // (a proxy of a native object compiled code made: not module state
        // that stays: Python's, through the proxy)
        return if (v.ptr().rc >= value.IMMORTAL) .{ .native = v } else .not_state;
    }
    if (try kindOf(o) == null) return .not_state;
    var g = Graph{};
    defer g.deinit();
    try g.walk(o, globals);
    try g.countModules();
    const target = g.find(g.seen.getIndex(o).?);
    for (g.seen.keys(), g.seen.values(), 0..) |x, e, i| {
        if (g.find(i) != target) continue;
        if (e.unfit) return .{ .refused = try std.fmt.allocPrint(a, "a {s} in it has a key compiled code's dicts can't hash, or attributes beyond its fields", .{try typeName(a, x)}) };
        const refs = ph.refcnt(x);
        const known = e.internal + e.module;
        if (refs != known) return .{ .refused = try std.fmt.allocPrint(a, "a {s} in it is referred to from somewhere besides it and module names ({d} references, {d} of them those)", .{ try typeName(a, x), refs, known }) };
    }
    try g.build(target);
    // The module's names: the native objects' proxies (kept by them: they
    // live as long as the process, as module state does)
    for (g.bindings.items) |b| {
        if (g.find(g.seen.getIndex(b.obj).?) != target) continue;
        const v = g.seen.get(b.obj).?.native.?;
        v.ptr().rc = value.IMMORTAL;
        const proxy = proxies.make(v, shared_maker) orelse return error.Python;
        defer py.Py_DecRef(proxy);
        if (py.c.PyDict_SetItem(b.dict, b.key, proxy) != 0) return error.Python;
    }
    return .{ .native = g.seen.get(o).?.native.? };
}

/// What the proxies of module state make nodes with: the program running's
/// maker (a node is an index into a program; module state outlives them).
pub const shared_maker = helpers.NodeMaker{ .ctx = @ptrCast(@constCast(&shared_maker_tag)), .make_fn = &currentNode };
const shared_maker_tag: u8 = 0;

fn currentNode(_: *anyopaque, idx: u32) ?*PyObject {
    const ctx = @import("bridge.zig").current orelse {
        ph.raise(py.PyExc_TypeError(), "a node kept in module state is only usable while its program runs", .{});
        return null;
    };
    return ctx.node_maker.make(idx);
}

/// What `o` is in a graph (null: a value, not part of it).
fn kindOf(o: *PyObject) Error!?Kind {
    const t: *PyObject = @ptrCast(@alignCast(ph.typeOf(o)));
    if (t == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PyList_Type"))))) return .list;
    if (t == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PyDict_Type"))))) return .dict;
    if (try compile_mod.recordOf(t)) |rt| return .{ .record = rt };
    return null;
}

const Graph = struct {
    seen: std.AutoArrayHashMapUnmanaged(*PyObject, Entry) = .empty,
    bindings: std.ArrayListUnmanaged(Binding) = .empty,

    fn deinit(self: *Graph) void {
        // (the native objects: the graph's references to them, given up;
        // the module's are immortal)
        for (self.seen.values()) |e| if (e.native) |v| value.decref(v);
        self.seen.deinit(gpa);
        self.bindings.deinit(gpa);
    }

    /// Every object reached from `root` and from the tables and records of
    /// `globals`, with its references from the graph, in components.
    fn walk(self: *Graph, root: *PyObject, globals: *PyObject) Error!void {
        _ = try self.add(root);
        var gpos: py.Py_ssize_t = 0;
        var gkey: ?*PyObject = null;
        var gval: ?*PyObject = null;
        while (py.c.PyDict_Next(globals, &gpos, @ptrCast(&gkey), @ptrCast(&gval)) != 0) _ = try self.add(gval.?);
        var i: usize = 0;
        while (i < self.seen.count()) : (i += 1) {
            const o = self.seen.keys()[i];
            switch (self.seen.values()[i].kind) {
                .list => {
                    const n: usize = @intCast(py.c.PyList_Size(o));
                    for (0..n) |k| try self.reach(i, py.c.PyList_GetItem(o, @intCast(k)).?);
                },
                .dict => {
                    var pos: py.Py_ssize_t = 0;
                    var key: ?*PyObject = null;
                    var val: ?*PyObject = null;
                    while (py.c.PyDict_Next(o, &pos, @ptrCast(&key), @ptrCast(&val)) != 0) {
                        // (a key native dicts can't hash as Python does: a
                        // dataclass compared by value, a list...)
                        if (try kindOf(key.?)) |k| if (k != .record or k.record.value_eq) {
                            self.seen.values()[i].unfit = true;
                        };
                        try self.reach(i, key.?);
                        try self.reach(i, val.?);
                    }
                },
                .record => |rt| {
                    if (!try onlyFields(o, rt)) self.seen.values()[i].unfit = true;
                    for (rt.fields) |name| {
                        const v = try field(o, name) orelse continue;
                        defer py.Py_DecRef(v);
                        try self.reach(i, v);
                    }
                },
            }
        }
    }

    /// `o` in the graph (if it's a table or record): its index.
    fn add(self: *Graph, o: *PyObject) Error!?usize {
        const kind = try kindOf(o) orelse return null;
        const e = try self.seen.getOrPut(gpa, o);
        if (!e.found_existing) e.value_ptr.* = .{ .kind = kind, .up = e.index };
        return e.index;
    }

    /// Object `from` refers to `o`: counted, the two in one component.
    fn reach(self: *Graph, from: usize, o: *PyObject) Error!void {
        const i = try self.add(o) orelse return;
        self.seen.values()[i].internal += 1;
        const a = self.find(from);
        const b = self.find(i);
        if (a != b) self.seen.values()[a].up = b;
    }

    fn find(self: *Graph, i: usize) usize {
        var at = i;
        while (self.seen.values()[at].up != at) at = self.seen.values()[at].up;
        // (shortened for next time)
        var k = i;
        while (self.seen.values()[k].up != at) {
            const next = self.seen.values()[k].up;
            self.seen.values()[k].up = at;
            k = next;
        }
        return at;
    }

    /// The modules' top-level names referring into the graph (the
    /// bindings), counted.
    fn countModules(self: *Graph) Error!void {
        const modules = py.c.PyImport_GetModuleDict() orelse return error.Python;
        const module_type: *py.c.PyTypeObject = @ptrCast(@alignCast(py.types.typeObject("PyModule_Type")));
        var dicts: std.AutoHashMapUnmanaged(*PyObject, void) = .empty;
        defer dicts.deinit(gpa);
        var pos: py.Py_ssize_t = 0;
        var name: ?*PyObject = null;
        var m: ?*PyObject = null;
        while (py.c.PyDict_Next(modules, &pos, @ptrCast(&name), @ptrCast(&m)) != 0) {
            if (py.c.PyType_IsSubtype(ph.typeOf(m.?), module_type) == 0) continue;
            const d = py.c.PyModule_GetDict(m.?) orelse continue;
            // (a module under two names: its dict once)
            if ((try dicts.getOrPut(gpa, d)).found_existing) continue;
            var dpos: py.Py_ssize_t = 0;
            var key: ?*PyObject = null;
            var val: ?*PyObject = null;
            while (py.c.PyDict_Next(d, &dpos, @ptrCast(&key), @ptrCast(&val)) != 0) {
                const e = self.seen.getPtr(val.?) orelse continue;
                e.module += 1;
                try self.bindings.append(gpa, .{ .dict = d, .key = key.?, .obj = val.? });
            }
        }
    }

    /// The native objects of a component: made first (empty: there may be
    /// cycles), then filled.
    fn build(self: *Graph, component: usize) Error!void {
        for (self.seen.values(), 0..) |*e, i| {
            if (self.find(i) != component) continue;
            e.native = switch (e.kind) {
                .list => Value.obj(.list, &(value.newList(0) orelse return error.OutOfMemory).head),
                .dict => Value.obj(.dict, &(value.newDict() orelse return error.OutOfMemory).head),
                .record => |rt| Value.obj(.record, &(value.newRecord(rt) orelse return error.OutOfMemory).head),
            };
        }
        for (self.seen.keys(), self.seen.values()) |o, e| {
            const target = e.native orelse continue;
            switch (e.kind) {
                .list => {
                    const l: *value.List = @ptrCast(@alignCast(target.ptr()));
                    const n: usize = @intCast(py.c.PyList_Size(o));
                    for (0..n) |k| {
                        const v = try self.valueOf(py.c.PyList_GetItem(o, @intCast(k)).?);
                        if (!value.listPush(l, v)) {
                            value.decref(v);
                            return error.OutOfMemory;
                        }
                    }
                },
                .dict => {
                    const d: *value.Dict = @ptrCast(@alignCast(target.ptr()));
                    var pos: py.Py_ssize_t = 0;
                    var key: ?*PyObject = null;
                    var val: ?*PyObject = null;
                    while (py.c.PyDict_Next(o, &pos, @ptrCast(&key), @ptrCast(&val)) != 0) {
                        const k = try self.valueOf(key.?);
                        defer value.decref(k);
                        const v = try self.valueOf(val.?);
                        defer value.decref(v);
                        if (!value.dictSet(d, k, v)) return error.OutOfMemory;
                    }
                },
                .record => |rt| {
                    const r: *value.Record = @ptrCast(@alignCast(target.ptr()));
                    for (rt.fields, r.fields()) |name, *slot| {
                        const fo = try field(o, name) orelse continue;
                        defer py.Py_DecRef(fo);
                        slot.* = try self.valueOf(fo);
                    }
                },
            }
        }
    }

    /// A value in the graph (a new reference): an object of it, natively;
    /// anything else as values are.
    fn valueOf(self: *Graph, o: *PyObject) Error!Value {
        if (self.seen.get(o)) |e| {
            value.incref(e.native.?);
            return e.native.?;
        }
        return value.fromPython(o) orelse error.Python;
    }
};

/// An object's field (a new reference), or null if it isn't set.
fn field(o: *PyObject, name: []const u8) Error!?*PyObject {
    var buf: [256]u8 = undefined;
    if (name.len >= buf.len) return error.OutOfMemory;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    return py.c.PyObject_GetAttrString(o, @ptrCast(&buf)) orelse {
        if (py.c.PyErr_ExceptionMatches(py.PyExc_AttributeError()) == 0) return error.Python;
        py.c.PyErr_Clear();
        return null;
    };
}

/// An object with a __dict__ (a dataclass without slots): nothing in it but
/// its fields (a native record has no room for more).
fn onlyFields(o: *PyObject, rt: *value.RecordType) Error!bool {
    if (rt.slots) return true;
    const d = ph.attr(o, "__dict__") orelse {
        py.c.PyErr_Clear();
        return true;
    };
    defer py.Py_DecRef(d);
    var pos: py.Py_ssize_t = 0;
    var key: ?*PyObject = null;
    var val: ?*PyObject = null;
    while (py.c.PyDict_Next(d, &pos, @ptrCast(&key), @ptrCast(&val)) != 0) {
        const k = ph.utf8(key.?, "name") orelse return error.Python;
        for (rt.fields) |f| {
            if (std.mem.eql(u8, f, k)) break;
        } else return false;
    }
    return true;
}
