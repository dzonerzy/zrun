//! A loaded program's native data: its tree (zgram's capsule, read in
//! place), each node's parent, the symbols of zrules' analysis (its capsule)
//! with the node each name node refers to, and what the runtime works out
//! from them once: the function a variable lives in, the functions to define
//! when a scope is entered, the line starts for errors.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const tree_mod = @import("tree.zig");
const zabi = @import("zrules_abi.zig");
const grammar_mod = @import("grammar.zig");

const Allocator = std.mem.Allocator;
pub const NONE = tree_mod.NONE;
/// A home not worked out yet
const UNSET = NONE - 1;

pub const Sym = struct {
    name: []const u8,
    /// The defining node (NONE: a builtin)
    node: u32,
    /// The scope node it is defined in (NONE: global)
    scope: u32,
    builtin: bool,
};

/// How functions are made from nodes of a kind (Language.function)
pub const FunctionSpec = struct {
    params: u8 = 0,
    /// The body: the child with this label, else the first child of the
    /// rule `body_rule`
    body: u8 = 0,
    body_rule: u16 = NO_RULE,
    name: u8 = 0,
    hoist: bool = true,
    /// Called with fewer arguments than parameters: an error, or the rest
    /// are None
    missing: enum { @"error", none } = .@"error",
    /// Called with more: an error, dropped, or kept (rt.varargs)
    extra: enum { @"error", drop, keep } = .@"error",
};

pub const NO_RULE: u16 = std.math.maxInt(u16);

/// The first child of `node` matched by a rule.
pub fn childOfRule(data: *const Data, node: u32, rule_id: u16) ?u32 {
    var c = node + 1;
    const stop = data.end(node);
    while (c < stop) : (c = data.end(c)) {
        if (data.nodes[c].ruleId() == rule_id) return c;
    }
    return null;
}

pub const Data = struct {
    arena: std.heap.ArenaAllocator,
    grammar: *const grammar_mod.Grammar,
    nodes: []const tree_mod.FlatNode,
    input: []const u8,
    parents: []u32,
    syms: []Sym = &.{},
    /// The symbol each node defines or uses (NONE: none)
    sym_of: []u32,
    /// Per symbol: the function node its variable lives in (NONE: the
    /// program's frame), worked out on first use
    home: []u32 = &.{},
    /// Per scope (a function node, or NONE for the program): the function
    /// nodes to define when it is entered (hoisted)
    hoisted: std.AutoHashMapUnmanaged(u32, std.ArrayList(u32)) = .empty,
    /// Per rule: the function spec of its kind, if it is a function kind
    functions: []const ?FunctionSpec,
    line_starts: []u32,
    /// Block scopes (a scope that isn't a function) with a frame of their
    /// own each time they run: those defining a variable a function made
    /// inside them uses (each closure keeps the one it was made in, as
    /// block scoping has it). Their variables live there.
    frame_scopes: std.AutoHashMapUnmanaged(u32, void) = .empty,

    pub fn isFunction(self: *const Data, node: u32) bool {
        const r = self.rule(node);
        return r < self.functions.len and self.functions[r] != null;
    }

    /// The function node around a node (itself excluded; NONE: the program).
    pub fn functionOf(self: *const Data, node: u32) u32 {
        var n = self.parents[node];
        while (n != NONE and n < self.nodes.len) : (n = self.parents[n]) {
            if (self.isFunction(n)) return n;
        }
        return NONE;
    }

    /// Whether a node is a block scope with frames of its own.
    pub fn hasFrame(self: *const Data, node: u32) bool {
        return self.frame_scopes.contains(node);
    }

    pub fn deinit(self: *Data) void {
        self.arena.deinit();
    }

    pub fn rule(self: *const Data, node: u32) u16 {
        return self.nodes[node].ruleId();
    }

    pub fn text(self: *const Data, node: u32) []const u8 {
        const n = self.nodes[node];
        const s = @min(n.text_start, self.input.len);
        return self.input[s..@min(@max(n.text_end, s), self.input.len)];
    }

    /// The index after the last descendant of `node`.
    pub fn end(self: *const Data, node: u32) u32 {
        return @intCast(@min(@as(u64, node) + self.nodes[node].subtree_size + 1, self.nodes.len));
    }

    pub fn symbol(self: *const Data, node: u32) ?*const Sym {
        if (node >= self.sym_of.len) return null;
        const s = self.sym_of[node];
        return if (s == NONE) null else &self.syms[s];
    }

    pub fn symbolIndex(self: *const Data, node: u32) ?u32 {
        if (node >= self.sym_of.len) return null;
        const s = self.sym_of[node];
        return if (s == NONE) null else s;
    }

    /// The node whose frame a symbol's variable lives in (NONE: the
    /// program's): its scope if that has frames of its own, else the
    /// innermost function node around its scope.
    pub fn homeOf(self: *Data, sym: u32) u32 {
        if (self.home[sym] != UNSET) return self.home[sym];
        var n = self.syms[sym].scope;
        if (n != NONE and self.hasFrame(n)) {
            self.home[sym] = n;
            return n;
        }
        while (n != NONE and n < self.nodes.len) : (n = self.parents[n]) {
            if (self.functions[self.rule(n)] != null) break;
        }
        const h = if (n < self.nodes.len) n else NONE;
        self.home[sym] = h;
        return h;
    }

    /// 1-based line and column (in bytes) of an offset.
    pub fn lineCol(self: *const Data, offset: u32) struct { line: u32, col: u32 } {
        var lo: usize = 0;
        var hi: usize = self.line_starts.len;
        while (hi - lo > 1) {
            const mid = (lo + hi) / 2;
            if (self.line_starts[mid] <= offset) lo = mid else hi = mid;
        }
        return .{ .line = @intCast(lo + 1), .col = offset - self.line_starts[lo] + 1 };
    }
};

/// Build a program's data from zgram's tree capsule and zrules' analysis
/// capsule (null: no rules). Error.Python with the exception set.
pub fn build(gpa: Allocator, grammar: *const grammar_mod.Grammar, functions: []const ?FunctionSpec, tree_view: *const tree_mod.TreeView, analysis: ?*const zabi.AnalysisView) !*Data {
    const data = try gpa.create(Data);
    errdefer gpa.destroy(data);
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    const nodes = (tree_view.nodes orelse return error.Python)[0..tree_view.node_count];
    const input = if (tree_view.input) |p| p[0..tree_view.input_len] else "";
    const parents = try tree_mod.Tree.computeParents(a, nodes);

    const sym_of = try a.alloc(u32, nodes.len);
    @memset(sym_of, NONE);
    var syms: []Sym = &.{};
    if (analysis) |view| {
        const views = (view.symbols orelse @as([*]const zabi.SymbolView, &.{}))[0..view.symbol_count];
        syms = try a.alloc(Sym, views.len);
        const use_nodes = view.use_nodes;
        for (views, syms, 0..) |v, *s, i| {
            s.* = .{
                .name = if (v.name.ptr) |p| p[0..v.name.len] else "",
                .node = v.node,
                .scope = v.scope,
                .builtin = v.flags & zabi.SYMBOL_BUILTIN != 0,
            };
            if (v.node != zabi.NONE and v.node < nodes.len) sym_of[v.node] = @intCast(i);
            if (use_nodes) |un| {
                for (un[v.uses_start..][0..v.uses_len]) |u| {
                    if (u < nodes.len) sym_of[u] = @intCast(i);
                }
            }
        }
    }
    const home = try a.alloc(u32, syms.len);
    @memset(home, UNSET);

    var line_starts: std.ArrayList(u32) = .empty;
    try line_starts.append(a, 0);
    for (input, 0..) |c, i| {
        if (c == '\n') try line_starts.append(a, @intCast(i + 1));
    }

    data.* = .{
        .arena = arena,
        .grammar = grammar,
        .nodes = nodes,
        .input = input,
        .parents = parents,
        .syms = syms,
        .sym_of = sym_of,
        .home = home,
        .functions = functions,
        .line_starts = line_starts.items,
    };

    // Block scopes with frames: a variable of one used from a function
    // made inside it (the root runs once: its variables are the program's)
    if (analysis) |view| {
        if (view.use_nodes) |un| {
            const views = (view.symbols orelse @as([*]const zabi.SymbolView, &.{}))[0..view.symbol_count];
            for (views) |v| {
                const scope = v.scope;
                if (scope == zabi.NONE or scope == 0 or scope >= nodes.len or data.isFunction(scope)) continue;
                if (data.frame_scopes.contains(scope)) continue;
                const home_fn = data.functionOf(scope);
                for (un[v.uses_start..][0..v.uses_len]) |u| {
                    if (u < nodes.len and data.functionOf(u) != home_fn) {
                        try data.frame_scopes.put(a, scope, {});
                        break;
                    }
                }
            }
        }
    }

    // Hoisted functions: each function node whose name is a symbol, by the
    // function its name lives in
    for (nodes, 0..) |n, i| {
        const spec = if (n.ruleId() < functions.len) functions[n.ruleId()] orelse continue else continue;
        if (!spec.hoist or spec.name == 0) continue;
        const name_node = labelled(data, @intCast(i), spec.name) orelse continue;
        const sym = data.symbolIndex(name_node) orelse continue;
        const scope = data.homeOf(sym);
        const entry = try data.hoisted.getOrPut(a, scope);
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        try entry.value_ptr.append(a, @intCast(i));
    }
    return data;
}

/// The first child of `node` with a field id.
pub fn labelled(data: *const Data, node: u32, field: u8) ?u32 {
    var c = node + 1;
    const stop = data.end(node);
    while (c < stop) : (c = data.end(c)) {
        if (data.nodes[c].fieldId() == field) return c;
    }
    return null;
}
