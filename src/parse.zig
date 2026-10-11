//! rt.parse(grammar, text, start=None, recover=False): text parsed while
//! the program runs, with a zgram grammar (a module's parser: JSON, a
//! shell's command line, the language's own source), its tree read as the
//! program's is: node.kind, its labelled fields converted as the grammar's
//! actions say, text, span, children, parent. A syntax error raises
//! zgram.ParseError (a ValueError), as parser.parse() does; with
//! recover=True the broken text becomes error nodes instead.
//!
//! The reference mode: the parser's parse_tree(), and zrun's Node over the
//! tree (objects.zig, the program's nodes' class).

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const grammar_mod = @import("grammar.zig");
const program_mod = @import("program.zig");
const tree_mod = @import("tree.zig");
const objects = @import("objects.zig");
const ptree = @import("ptree.zig");
const value = @import("value.zig");
const gil = @import("gil.zig");

const allocator = std.heap.c_allocator;

/// zrun's view of each grammar parsed with (its rules, kinds, labels), by
/// parser: made on first parse, kept with a reference to the parser.
/// (Guarded by the GIL.)
var grammars: std.AutoHashMapUnmanaged(*PyObject, *grammar_mod.Grammar) = .empty;

/// The grammar's view (null with an exception: not a zgram parser).
pub fn grammarOf(parser: *PyObject) ?*grammar_mod.Grammar {
    if (grammars.get(parser)) |g| return g;
    if (py.c.PyObject_HasAttrString(parser, "parse_tree") == 0 or py.c.PyObject_HasAttrString(parser, "labels") == 0) {
        ph.raise(py.PyExc_TypeError(), "rt.parse: the grammar must be a zgram parser (zgram.compile()'s)", .{});
        return null;
    }
    const g = allocator.create(grammar_mod.Grammar) catch {
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    g.* = grammar_mod.Grammar.init(allocator, parser) catch |e| {
        allocator.destroy(g);
        if (e == error.OutOfMemory) _ = py.c.PyErr_NoMemory();
        return null;
    };
    // (recover=True's error nodes: rule "<error>")
    g.addErrorRule() catch |e| {
        g.deinit();
        allocator.destroy(g);
        if (e == error.OutOfMemory) _ = py.c.PyErr_NoMemory();
        return null;
    };
    grammars.put(allocator, parser, g) catch {
        g.deinit();
        allocator.destroy(g);
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    py.Py_IncRef(parser);
    return g;
}

/// What compiled code parses a grammar with (ptree.Grammar): zgram's
/// parsing (the `zgram.grammar.v1` capsule, kept) and zrun's view of its
/// rules; made once per parser, when compiling (with the GIL). Null with an
/// exception.
pub fn nativeGrammar(parser: *PyObject) ?*ptree.Grammar {
    if (ptree.grammarOf(parser)) |g| return g;
    const rules = grammarOf(parser) orelse return null;
    const capsule = py.c.PyObject_GetAttrString(parser, "capsule") orelse {
        py.c.PyErr_Clear();
        ph.raise(py.PyExc_ImportError(), "rt.parse in compiled code needs zgram 0.6 or later (GrammarParser.capsule)", .{});
        return null;
    };
    const view: *const ptree.GrammarView = @ptrCast(@alignCast(py.c.PyCapsule_GetPointer(capsule, ptree.GRAMMAR_CAPSULE) orelse {
        py.Py_DecRef(capsule);
        return null;
    }));
    if (view.abi != ptree.GRAMMAR_ABI) {
        py.Py_DecRef(capsule);
        ph.raise(py.PyExc_ImportError(), "zrun parses with zgram's grammars of ABI {d}, this zgram has {d}", .{ ptree.GRAMMAR_ABI, view.abi });
        return null;
    }
    const g = allocator.create(ptree.Grammar) catch {
        py.Py_DecRef(capsule);
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    // (the capsule kept: for as long as the process, as the grammar is)
    g.* = .{ .descriptor = view.descriptor, .parse = view.parse, .tree_free = view.tree_free, .rules = rules, .parser = parser };
    ptree.addGrammar(parser, g) catch {
        allocator.destroy(g);
        py.Py_DecRef(capsule);
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    return g;
}

/// The reference mode's rt.parse: the root of the tree, a zrun.Node (null
/// with an exception: zgram.ParseError for a syntax error); with
/// `recover`, (root, errors), each error (offset, line, column, message).
pub fn parsePython(parser: *PyObject, text: *PyObject, start: ?*PyObject, recover: bool) ?*PyObject {
    const g = grammarOf(parser) orelse return null;
    const tree = parseTree(parser, text, start, recover) orelse return null;
    defer py.Py_DecRef(tree);
    const root = rootOf(parser, g, tree) orelse return null;
    if (!recover) return root;
    defer py.Py_DecRef(root);
    const diagnostics = py.c.PyObject_GetAttrString(tree, "errors") orelse return null;
    defer py.Py_DecRef(diagnostics);
    const n = py.c.PyList_Size(diagnostics);
    const errors = py.c.PyList_New(n) orelse return null;
    for (0..@intCast(n)) |i| {
        const item = errorTuple(py.c.PyList_GetItem(diagnostics, @intCast(i)).?) orelse {
            py.Py_DecRef(errors);
            return null;
        };
        _ = py.c.PyList_SetItem(errors, @intCast(i), item);
    }
    py.Py_IncRef(root);
    return py.c.Py_BuildValue("(NN)", root, errors);
}

/// (offset, line, column, message) of a zgram Diagnostic
fn errorTuple(d: *PyObject) ?*PyObject {
    const span = py.c.PyObject_GetAttrString(d, "span") orelse return null;
    defer py.Py_DecRef(span);
    const offset = py.c.PySequence_GetItem(span, 0) orelse return null;
    const line = py.c.PyObject_GetAttrString(d, "line") orelse {
        py.Py_DecRef(offset);
        return null;
    };
    const column = py.c.PyObject_GetAttrString(d, "column") orelse {
        py.Py_DecRef(offset);
        py.Py_DecRef(line);
        return null;
    };
    const message = py.c.PyObject_GetAttrString(d, "message") orelse {
        py.Py_DecRef(offset);
        py.Py_DecRef(line);
        py.Py_DecRef(column);
        return null;
    };
    return py.c.Py_BuildValue("(NNNN)", offset, line, column, message);
}

/// A node of a tree compiled code parsed (ptree.zig) as a zrun.Node: the
/// tree parsed again in Python, once (the same parser, text, start rule and
/// recovery: the same tree), kept with the native one.
pub fn nodeObject(t: *ptree.Tree, idx: u32) ?*PyObject {
    if (t.py_root == null) {
        const parser: *PyObject = @ptrCast(@alignCast(t.grammar.parser orelse {
            ph.raise(py.PyExc_RuntimeError(), "a node of a parsed tree can't go to Python here", .{});
            return null;
        }));
        const g = grammarOf(parser) orelse return null;
        const text = ph.newString(@as(*value.Str, @ptrCast(t.text.ptr())).bytes()) orelse return null;
        defer py.Py_DecRef(text);
        const start: ?*PyObject = if (t.start) |s| ph.newString(s) orelse return null else null;
        defer if (start) |s| py.Py_DecRef(s);
        const tree = parseTree(parser, text, start, t.recover) orelse return null;
        defer py.Py_DecRef(tree);
        t.py_root = rootOf(parser, g, tree) orelse return null;
        t.py_drop = dropRoot;
    }
    const root: *objects.NodeObject = @ptrCast(@alignCast(t.py_root.?));
    if (idx == 0) {
        py.Py_IncRef(@ptrCast(root));
        return @ptrCast(root);
    }
    return objects.newNode(root.state.?, root.ctx.?, idx);
}

fn dropRoot(root: *anyopaque) void {
    // (a count, not Python running: as value.zig drops what Python owns)
    gil.allowBegin();
    defer gil.allowEnd();
    py.Py_DecRef(@ptrCast(@alignCast(root)));
}

/// parser.parse_tree(text, start=start, recover=recover)
fn parseTree(parser: *PyObject, text: *PyObject, start: ?*PyObject, recover: bool) ?*PyObject {
    const method = py.c.PyObject_GetAttrString(parser, "parse_tree") orelse return null;
    defer py.Py_DecRef(method);
    const args = py.c.PyTuple_Pack(1, text) orelse return null;
    defer py.Py_DecRef(args);
    const kwargs = py.c.PyDict_New() orelse return null;
    defer py.Py_DecRef(kwargs);
    if (start) |s| {
        if (s != py.Py_None() and py.c.PyDict_SetItemString(kwargs, "start", s) != 0) return null;
    }
    if (recover and py.c.PyDict_SetItemString(kwargs, "recover", py.Py_True()) != 0) return null;
    return py.c.PyObject_Call(method, args, kwargs);
}

/// A zgram Tree read as a program's: the root, a zrun.Node whose state
/// keeps the tree (and the parser) alive.
pub fn rootOf(parser: *PyObject, g: *grammar_mod.Grammar, tree: *PyObject) ?*PyObject {
    const capsule = py.c.PyObject_GetAttrString(tree, "capsule") orelse return null;
    defer py.Py_DecRef(capsule);
    const tv: *const tree_mod.TreeView = @ptrCast(@alignCast(py.c.PyCapsule_GetPointer(capsule, tree_mod.CAPSULE_NAME) orelse return null));
    if (tv.abi != tree_mod.TREE_ABI) {
        ph.raise(py.PyExc_ImportError(), "zrun reads zgram trees with ABI {d}, this zgram has {d}", .{ tree_mod.TREE_ABI, tv.abi });
        return null;
    }
    const data = program_mod.build(allocator, g, &.{}, tv, null) catch |e| {
        if (e == error.OutOfMemory) _ = py.c.PyErr_NoMemory();
        return null;
    };
    const context = allocator.create(objects.Context) catch {
        data.deinit();
        allocator.destroy(data);
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    const values = allocator.alloc(?*PyObject, data.nodes.len) catch {
        allocator.destroy(context);
        data.deinit();
        allocator.destroy(data);
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    @memset(values, null);
    py.Py_IncRef(tree);
    context.* = .{ .data = data, .tree = tree, .values = values };
    const diagnostics = py.c.PyObject_GetAttrString(tree, "errors") orelse return null;
    defer py.Py_DecRef(diagnostics);
    const state = objects.newState(context, parser, null, diagnostics) orelse {
        py.Py_DecRef(tree);
        allocator.free(values);
        allocator.destroy(context);
        data.deinit();
        allocator.destroy(data);
        return null;
    };
    defer py.Py_DecRef(state);
    return objects.newNode(state, context, 0);
}
