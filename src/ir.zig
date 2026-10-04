//! Writing LLVM IR as text: a module's functions, their blocks and
//! instructions, numbered SSA names, string constants. The compiler
//! (compile.zig) writes through this; zgram's LLVM parses and compiles it.

const std = @import("std");
const value = @import("value.zig");
const Allocator = std.mem.Allocator;

pub const Module = struct {
    gpa: Allocator,
    /// Type and global declarations, constants
    head: std.ArrayList(u8) = .empty,
    /// Function definitions
    body: std.ArrayList(u8) = .empty,
    /// Declared external functions (helpers), once each
    declared: std.StringHashMapUnmanaged(void) = .empty,
    /// String constants by content: the global's name
    strings: std.StringHashMapUnmanaged(u32) = .empty,
    next_global: u32 = 0,
    /// A prefix making the module's symbols unique in the process
    prefix: []const u8,

    pub fn init(gpa: Allocator, prefix: []const u8) Module {
        return .{ .gpa = gpa, .prefix = prefix };
    }

    pub fn deinit(self: *Module) void {
        self.head.deinit(self.gpa);
        self.body.deinit(self.gpa);
        var it = self.declared.keyIterator();
        while (it.next()) |k| self.gpa.free(k.*);
        self.declared.deinit(self.gpa);
        var it2 = self.strings.keyIterator();
        while (it2.next()) |k| self.gpa.free(k.*);
        self.strings.deinit(self.gpa);
    }

    /// The whole module's text (owned by the caller).
    pub fn text(self: *Module) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        try out.appendSlice(self.gpa, self.head.items);
        try out.append(self.gpa, '\n');
        try out.appendSlice(self.gpa, self.body.items);
        return out.toOwnedSlice(self.gpa);
    }

    pub fn headPrint(self: *Module, comptime fmt: []const u8, args: anytype) !void {
        try self.head.print(self.gpa, fmt, args);
    }

    /// Declare an external function once: `declare <signature>`.
    pub fn declare(self: *Module, name: []const u8, signature: []const u8) !void {
        if (self.declared.contains(name)) return;
        try self.declared.put(self.gpa, try self.gpa.dupe(u8, name), {});
        try self.headPrint("declare {s}\n", .{signature});
    }

    /// An immortal string object for a literal (a global laid out as the
    /// runtime's Str): its global's name, `@<prefix>_s<n>`.
    pub fn string(self: *Module, bytes: []const u8) ![]const u8 {
        if (self.strings.get(bytes)) |n| return std.fmt.allocPrint(self.gpa, "@{s}_s{d}", .{ self.prefix, n });
        const n = self.next_global;
        self.next_global += 1;
        try self.strings.put(self.gpa, try self.gpa.dupe(u8, bytes), n);
        const chars = std.unicode.utf8CountCodepoints(bytes) catch bytes.len;
        const h: i64 = @bitCast(value.strHash(bytes));
        try self.headPrint("@{s}_s{d} = private unnamed_addr constant {{ i64, i32, i32, i64, i64, i64, [{d} x i8] }} {{ i64 4611686018427387904, i32 4, i32 0, i64 {d}, i64 {d}, i64 {d}, [{d} x i8] c\"", .{ self.prefix, n, bytes.len, bytes.len, chars, h, bytes.len });
        for (bytes) |c| {
            if (c >= 0x20 and c < 0x7F and c != '"' and c != '\\') {
                try self.head.append(self.gpa, c);
            } else try self.headPrint("\\{X:0>2}", .{c});
        }
        try self.headPrint("\" }}, align 8\n", .{});
        return std.fmt.allocPrint(self.gpa, "@{s}_s{d}", .{ self.prefix, n });
    }
};

/// One function being written
pub const Function = struct {
    m: *Module,
    /// Allocas, written at the top of the entry block
    entry: std.ArrayList(u8) = .empty,
    code: std.ArrayList(u8) = .empty,
    next_tmp: u32 = 0,
    next_label: u32 = 0,
    /// Whether the current block has its terminator
    terminated: bool = false,
    /// The current block's label (for phis)
    current: []const u8 = "entry",

    pub fn init(m: *Module) Function {
        return .{ .m = m };
    }

    pub fn deinit(self: *Function) void {
        self.entry.deinit(self.m.gpa);
        self.code.deinit(self.m.gpa);
    }

    /// Write the function into the module: `define <signature> { ... }`.
    pub fn finish(self: *Function, signature: []const u8) !void {
        const m = self.m;
        try m.body.print(m.gpa, "define {s} {{\nentry:\n", .{signature});
        try m.body.appendSlice(m.gpa, self.entry.items);
        try m.body.appendSlice(m.gpa, self.code.items);
        if (!self.terminated) try m.body.appendSlice(m.gpa, "  unreachable\n");
        try m.body.appendSlice(m.gpa, "}\n\n");
    }

    /// A fresh SSA name: %t<n>.
    pub fn tmp(self: *Function) ![]const u8 {
        const n = self.next_tmp;
        self.next_tmp += 1;
        return std.fmt.allocPrint(self.m.gpa, "%t{d}", .{n});
    }

    /// A fresh block label (without %): <prefix><n>.
    pub fn label(self: *Function, prefix: []const u8) ![]const u8 {
        const n = self.next_label;
        self.next_label += 1;
        return std.fmt.allocPrint(self.m.gpa, "{s}{d}", .{ prefix, n });
    }

    /// An instruction (indented, a line).
    pub fn emit(self: *Function, comptime fmt: []const u8, args: anytype) !void {
        if (self.terminated) return; // (dead code after a jump)
        try self.code.appendSlice(self.m.gpa, "  ");
        try self.code.print(self.m.gpa, fmt, args);
        try self.code.append(self.m.gpa, '\n');
    }

    /// An instruction giving a value: `%tN = <instr>`; the name.
    pub fn value(self: *Function, comptime fmt: []const u8, args: anytype) ![]const u8 {
        const t = try self.tmp();
        if (!self.terminated) {
            try self.code.print(self.m.gpa, "  {s} = ", .{t});
            try self.code.print(self.m.gpa, fmt, args);
            try self.code.append(self.m.gpa, '\n');
        }
        return t;
    }

    /// An alloca in the entry block; its name.
    pub fn alloca(self: *Function, ty: []const u8) ![]const u8 {
        const t = try self.tmp();
        try self.entry.print(self.m.gpa, "  {s} = alloca {s}, align 8\n", .{ t, ty });
        return t;
    }

    /// Start a block (a jump to it ends the current one if it has none).
    pub fn block(self: *Function, name: []const u8) !void {
        if (!self.terminated) try self.code.print(self.m.gpa, "  br label %{s}\n", .{name});
        try self.code.print(self.m.gpa, "{s}:\n", .{name});
        self.terminated = false;
        self.current = name;
    }

    /// A value from two predecessors: `phi <ty> [a, %from_a], [b, %from_b]`.
    pub fn phi(self: *Function, ty: []const u8, a: []const u8, from_a: []const u8, b: []const u8, from_b: []const u8) ![]const u8 {
        return self.value("phi {s} [ {s}, %{s} ], [ {s}, %{s} ]", .{ ty, a, from_a, b, from_b });
    }

    pub fn br(self: *Function, target: []const u8) !void {
        try self.emit("br label %{s}", .{target});
        self.terminated = true;
    }

    pub fn condBr(self: *Function, cond: []const u8, yes: []const u8, no: []const u8) !void {
        try self.emit("br i1 {s}, label %{s}, label %{s}", .{ cond, yes, no });
        self.terminated = true;
    }

    pub fn ret(self: *Function, comptime fmt: []const u8, args: anytype) !void {
        try self.emit("ret " ++ fmt, args);
        self.terminated = true;
    }
};
