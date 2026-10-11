//! Trees parsed while the program runs, in compiled code (rt.parse; the
//! reference mode's is parse.zig). zgram's parser is called natively (the
//! `zgram.grammar.v1` capsule's zgram_parse; a standalone program links
//! zgram's runtime and the grammar's object instead), and the tree is read
//! as the program's is (bridge.nodeAttr over a Data of its own). Its nodes
//! are values of their own kind (value.PNode: the tree, an index), which
//! keep the tree alive.

const std = @import("std");
const value = @import("value.zig");
const program_mod = @import("program.zig");
const grammar_mod = @import("grammar.zig");
const tree_mod = @import("tree.zig");
const driver = @import("driver.zig");

const Value = value.Value;
const allocator = std.heap.c_allocator;

// ----------------------------------------------------------------------
// zgram's runtime (zgram's src/runtime.zig): what zrun reads of it
// ----------------------------------------------------------------------

pub const GRAMMAR_CAPSULE = "zgram.grammar.v1";
pub const GRAMMAR_ABI: u32 = 1;

/// A grammar as zgram's runtime parses with it (zgram_grammar)
pub const ZGrammar = opaque {};

pub const ZError = extern struct {
    offset: u32,
    line: u32,
    column: u32,
    message: [256]u8,
};

pub const ZTree = extern struct {
    grammar: ?*const ZGrammar = null,
    input: ?[*]const u8 = null,
    input_len: usize = 0,
    nodes: ?[*]tree_mod.FlatNode = null,
    node_count: u32 = 0,
    error_count: u32 = 0,
    errors: ?[*]ZError = null,
};

pub const ParseFn = *const fn (g: *const ZGrammar, input: [*]const u8, len: usize, start: ?[*:0]const u8, flags: i32, tree: *ZTree, err: ?*ZError) callconv(.c) i32;
pub const FreeFn = *const fn (tree: *ZTree) callconv(.c) void;

/// What the capsule points to (zgram's GrammarView)
pub const GrammarView = extern struct {
    abi: u32,
    reserved: u32,
    descriptor: *const ZGrammar,
    parse: ParseFn,
    matches: *const anyopaque,
    tree_free: FreeFn,
};

pub const Z_OK: i32 = 0;
pub const Z_SYNTAX_ERROR: i32 = 1;
pub const Z_UNKNOWN_START: i32 = -1;
pub const Z_NO_NODES: i32 = -5;
pub const Z_RECOVER: i32 = 2;

// ----------------------------------------------------------------------
// Grammars and trees
// ----------------------------------------------------------------------

/// A grammar compiled code parses with: zgram's parsing, zrun's view of
/// its rules (kinds, labels; "<error>" last), and in the JIT the parser
/// itself (a Python object: the reference mode's parse, and a node of the
/// tree in Python)
pub const Grammar = struct {
    descriptor: *const ZGrammar,
    parse: ParseFn,
    tree_free: FreeFn,
    rules: *const grammar_mod.Grammar,
    parser: ?*anyopaque = null,
};

/// Each grammar compiled code parses with, by its parser (made when
/// compiling: parse.nativeGrammar; found by zr_parse without Python)
var grammars: std.AutoHashMapUnmanaged(*const anyopaque, *Grammar) = .empty;
var grammars_lock: std.atomic.Value(bool) = .init(false);

fn lock() void {
    while (grammars_lock.cmpxchgWeak(false, true, .acquire, .monotonic) != null) std.Thread.yield() catch {};
}

fn unlock() void {
    grammars_lock.store(false, .release);
}

pub fn grammarOf(parser: *const anyopaque) ?*Grammar {
    lock();
    defer unlock();
    return grammars.get(parser);
}

pub fn addGrammar(parser: *const anyopaque, g: *Grammar) !void {
    lock();
    defer unlock();
    try grammars.put(allocator, parser, g);
}

/// A tree compiled code parsed: its nodes, its text, read as a program's
pub const Tree = struct {
    /// References: the PNodes of it
    refs: u32 = 1,
    data: *program_mod.Data,
    cache: driver.NodeCache = .{},
    z: ZTree,
    grammar: *const Grammar,
    /// The text parsed (a str value, owned: the nodes point into it), and
    /// how: from the rule (null: the first), with recovery
    text: Value,
    start: ?[:0]const u8 = null,
    recover: bool = false,
    /// The tree in Python, once one of its nodes went there (parse.zig's
    /// root zrun.Node, owned), and the function that drops it
    py_root: ?*anyopaque = null,
    py_drop: ?*const fn (*anyopaque) void = null,
};

/// The tree zgram parsed (`z`, now the tree's), read as a program's; the
/// text is referenced. Null: out of memory (`z` freed).
pub fn newTree(g: *const Grammar, z: ZTree, text: Value, start: ?[]const u8, recover: bool) ?*Tree {
    var zt = z;
    // (what program.build reads: the nodes, the input)
    const tv = tree_mod.TreeView{
        .abi = tree_mod.TREE_ABI,
        .node_count = zt.node_count,
        .nodes = zt.nodes,
        .input = zt.input,
        .input_len = zt.input_len,
        .rule_count = 0,
        .field_count = 0,
        .rule_names = null,
        .field_names = null,
    };
    const data = program_mod.build(allocator, g.rules, &.{}, &tv, null) catch {
        g.tree_free(&zt);
        return null;
    };
    const t = allocator.create(Tree) catch {
        data.deinit();
        allocator.destroy(data);
        g.tree_free(&zt);
        return null;
    };
    const start_copy: ?[:0]const u8 = if (start) |s| allocator.dupeZ(u8, s) catch {
        allocator.destroy(t);
        data.deinit();
        allocator.destroy(data);
        g.tree_free(&zt);
        return null;
    } else null;
    value.incref(text);
    t.* = .{ .data = data, .z = zt, .grammar = g, .text = text, .start = start_copy, .recover = recover };
    return t;
}

pub fn retain(t: *Tree) void {
    t.refs += 1;
}

pub fn release(t: *Tree) void {
    t.refs -= 1;
    if (t.refs != 0) return;
    if (t.py_root) |r| t.py_drop.?(r);
    t.cache.deinit();
    t.data.deinit();
    allocator.destroy(t.data);
    t.grammar.tree_free(&t.z);
    value.decref(t.text);
    if (t.start) |s| allocator.free(s);
    allocator.destroy(t);
}

/// Node `idx` of the tree as a value (a new reference); null: out of memory
pub fn node(t: *Tree, idx: u32) ?Value {
    const p = value.newPNode(t, idx) orelse return null;
    return Value.obj(.pnode, &p.head);
}
