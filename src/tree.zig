//! zgram's parse tree as native code sees it: the `zgram.tree.v1` capsule
//! (zgram's src/parse_abi.zig is the definition; this is the consumer's copy).

const std = @import("std");

/// The tree interface version this code was written against
pub const TREE_ABI: u32 = 1;
pub const CAPSULE_NAME = "zgram.tree.v1";

/// "No node": the parent of the root
pub const NONE: u32 = std.math.maxInt(u32);

pub const Str = extern struct {
    ptr: [*]const u8,
    len: usize,

    pub fn slice(self: Str) []const u8 {
        return self.ptr[0..self.len];
    }
};

/// One node: 16 bytes, nodes in pre-order, a node's descendants right after it
pub const FlatNode = extern struct {
    text_start: u32,
    text_end: u32,
    /// Number of descendants
    subtree_size: u32,
    /// Child count (12 bits, saturating) | rule id (12) | field id (8)
    meta: u32,

    pub inline fn ruleId(self: FlatNode) u16 {
        return @truncate((self.meta >> 12) & 0xFFF);
    }

    /// Label the node was matched under in its parent, plus one; 0 = none
    pub inline fn fieldId(self: FlatNode) u8 {
        return @truncate(self.meta >> 24);
    }
};

pub const TreeView = extern struct {
    abi: u32,
    node_count: u32,
    nodes: ?[*]const FlatNode,
    input: ?[*]const u8,
    input_len: usize,
    rule_count: u32,
    field_count: u32,
    rule_names: ?[*]const Str,
    field_names: ?[*]const Str,
};

/// A tree being checked: the capsule's arrays plus each node's parent.
pub const Tree = struct {
    nodes: []const FlatNode,
    input: []const u8,
    parents: []const u32,

    /// Parent index of every node (NONE for the root), in one pass.
    pub fn computeParents(allocator: std.mem.Allocator, nodes: []const FlatNode) ![]u32 {
        const parents = try allocator.alloc(u32, nodes.len);
        if (nodes.len != 0) parents[0] = NONE;
        for (nodes, 0..) |n, i| {
            const last = @min(i + n.subtree_size, nodes.len - 1);
            var c = i + 1;
            while (c <= last) : (c += nodes[c].subtree_size + 1) parents[c] = @intCast(i);
        }
        return parents;
    }

    pub fn text(self: *const Tree, node: u32) []const u8 {
        const n = self.nodes[node];
        const s = @min(n.text_start, self.input.len);
        return self.input[s..@min(@max(n.text_end, s), self.input.len)];
    }

    /// Index after the last descendant of `node`.
    pub fn end(self: *const Tree, node: u32) u32 {
        return @intCast(@min(@as(u64, node) + self.nodes[node].subtree_size + 1, self.nodes.len));
    }
};
