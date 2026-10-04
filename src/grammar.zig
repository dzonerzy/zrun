//! What zrun knows of a zgram grammar: per rule, its `-> action`, its kind
//! (the class the action names, else the rule's name) and its labels, with
//! which of them are lists.
//!
//! A node's fields are its labelled children, converted as zgram's AST
//! actions say: `-> str`, `-> int`, `-> float`, `-> unquote` and the
//! constants give values, `-> list` / `-> tuple` / `-> dict` / `-> first`
//! their children's values, anything else a Node. A label inside `*`/`+` (or
//! used twice) is a list, any other one the child or None.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;

const Allocator = std.mem.Allocator;

pub const Action = enum { none, str, int, float, unquote, true_, false_, none_, list, tuple, dict, first, drop, class };

pub const Label = struct { field: u8, many: bool };

pub const Grammar = struct {
    arena: std.heap.ArenaAllocator,
    rule_names: []const []const u8 = &.{},
    actions: []const Action = &.{},
    /// The kind of each rule's nodes, as str (owned references)
    kinds: []const *PyObject = &.{},
    kind_names: []const []const u8 = &.{},
    /// Each rule's labels
    labels: []const []const Label = &.{},
    /// Label names by field id - 1 (field ids as in the tree: label + 1)
    field_names: []const []const u8 = &.{},
    field_ids: std.StringHashMapUnmanaged(u8) = .empty,

    pub fn deinit(self: *Grammar) void {
        for (self.kinds) |k| py.Py_DecRef(k);
        self.arena.deinit();
    }

    /// Read the grammar of a zgram parser (rules(), actions(), labels(),
    /// fields()). Error.Python with the exception set on failure.
    pub fn init(gpa: Allocator, parser: *PyObject) !Grammar {
        var g = Grammar{ .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer g.arena.deinit();
        const a = g.arena.allocator();

        const rules = try call(parser, "rules");
        defer py.Py_DecRef(rules);
        const actions = try call(parser, "actions");
        defer py.Py_DecRef(actions);
        const fields = try call(parser, "fields");
        defer py.Py_DecRef(fields);
        const labels = py.c.PyObject_CallMethod(parser, "labels", null) orelse {
            py.c.PyErr_Clear();
            ph.raise(py.PyExc_ImportError(), "zrun needs zgram 0.3.6 or later (parser.labels())", .{});
            return error.Python;
        };
        defer py.Py_DecRef(labels);

        const n: usize = @intCast(py.c.PyList_Size(rules));
        const field_count: usize = @intCast(py.c.PyList_Size(fields));
        const field_names = try a.alloc([]const u8, field_count);
        for (field_names, 0..) |*slot, i| {
            slot.* = try a.dupe(u8, ph.utf8(py.c.PyList_GetItem(fields, @intCast(i)).?, "a label") orelse return error.Python);
            try g.field_ids.put(a, slot.*, @intCast(i + 1));
        }
        g.field_names = field_names;

        const names = try a.alloc([]const u8, n);
        const acts = try a.alloc(Action, n);
        const kind_names = try a.alloc([]const u8, n);
        const kinds = try a.alloc(*PyObject, n);
        var made: usize = 0;
        errdefer for (kinds[0..made]) |k| py.Py_DecRef(k);
        const lbls = try a.alloc([]const Label, n);
        for (0..n) |i| {
            names[i] = try a.dupe(u8, ph.utf8(py.c.PyList_GetItem(rules, @intCast(i)).?, "a rule name") orelse return error.Python);
            const act_obj = py.c.PyList_GetItem(actions, @intCast(i)).?;
            var act_name: ?[]const u8 = null;
            if (act_obj != py.Py_None()) act_name = try a.dupe(u8, ph.utf8(act_obj, "an action") orelse return error.Python);
            acts[i] = actionOf(act_name);
            kind_names[i] = if (acts[i] == .class) act_name.? else names[i];
            kinds[i] = py.PyUnicode_FromStringAndSize(kind_names[i].ptr, @intCast(kind_names[i].len)) orelse return error.Python;
            py.c.PyUnicode_InternInPlace(@ptrCast(&kinds[i]));
            made += 1;

            const pairs = py.c.PyList_GetItem(labels, @intCast(i)).?;
            const m: usize = @intCast(py.c.PyList_Size(pairs));
            const out = try a.alloc(Label, m);
            for (out, 0..) |*l, j| {
                const pair = py.c.PyList_GetItem(pairs, @intCast(j)).?;
                const label = ph.utf8(py.c.PyTuple_GetItem(pair, 0).?, "a label") orelse return error.Python;
                const many = py.c.PyObject_IsTrue(py.c.PyTuple_GetItem(pair, 1).?) == 1;
                l.* = .{ .field = g.field_ids.get(label) orelse 0, .many = many };
            }
            lbls[i] = out;
        }
        g.rule_names = names;
        g.actions = acts;
        g.kind_names = kind_names;
        g.kinds = kinds;
        g.labels = lbls;
        return g;
    }

    pub fn ruleCount(self: *const Grammar) usize {
        return self.rule_names.len;
    }

    /// The rules whose nodes have a kind or rule name: an action's class
    /// first, else the rule itself.
    pub fn hasName(self: *const Grammar, name: []const u8) bool {
        for (self.rule_names, self.kind_names) |r, k| {
            if (std.mem.eql(u8, r, name) or std.mem.eql(u8, k, name)) return true;
        }
        return false;
    }

    /// A rule's label for a field id, if it has it.
    pub fn labelOf(self: *const Grammar, rule: usize, field: u8) ?Label {
        if (rule >= self.labels.len) return null;
        for (self.labels[rule]) |l| {
            if (l.field == field) return l;
        }
        return null;
    }
};

fn call(obj: *PyObject, comptime method: [*:0]const u8) !*PyObject {
    return py.c.PyObject_CallMethod(obj, method, null) orelse error.Python;
}

fn actionOf(name: ?[]const u8) Action {
    const n = name orelse return .none;
    const table = .{
        .{ "str", Action.str },         .{ "int", Action.int },     .{ "float", Action.float },
        .{ "unquote", Action.unquote }, .{ "True", Action.true_ },  .{ "False", Action.false_ },
        .{ "None", Action.none_ },      .{ "list", Action.list },   .{ "tuple", Action.tuple },
        .{ "dict", Action.dict },       .{ "first", Action.first }, .{ "drop", Action.drop },
    };
    inline for (table) |entry| {
        if (std.mem.eql(u8, n, entry[0])) return entry[1];
    }
    return .class;
}
