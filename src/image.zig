//! A standalone build's data image: what compiled code refers to by address
//! (ir.Module.ptrConst: in the JIT, objects of the compiling process), in
//! the binary itself. Strs and Bigs are its data as they are; tables of
//! words too (a word that's a str's address its address there); value
//! graphs (constant tables), record types and closures' functions are
//! built where the code expects them as the program starts (zr_image_build,
//! from a description): their room, with the cycle collector's header
//! before it, is the image's.
//!
//! This file has both sides: the runtime's (building, from descriptions)
//! and the compiler's (describing; compile.zig emits the module).

const std = @import("std");
const value = @import("value.zig");
const set_mod = @import("set.zig");
const front = @import("front.zig");
const gc = @import("gc.zig");

const Value = value.Value;
const Tag = value.Tag;
const Allocator = std.mem.Allocator;

/// What an item built at the start is
pub const Kind = enum(u32) { value_graph, rtype, function };

/// A value's encoding (little-endian): its tag, then its contents
pub const V = enum(u8) {
    none,
    false_,
    true_,
    /// a plain int (i64)
    pint,
    /// an I64, rt's checked int (i64)
    int,
    float,
    /// u64 length, the bytes
    str,
    /// i128
    big,
    /// u64 count, the items
    list,
    tuple,
    /// u64 count, keys and values
    dict,
    set,
    /// u64 length, the bytes (a Python bytes)
    bytes,
    /// u32 the record type's item, u64 its fields' count, the fields
    record,
    /// an item of the image (u64 its tag, u32 its index): the same object
    ref,
    /// u32 the node
    node,
    /// a record's slot never assigned
    unset,
};

// ----------------------------------------------------------------------
// The runtime's side
// ----------------------------------------------------------------------

const Reader = struct {
    b: []const u8,
    at: usize = 0,

    fn u(r: *Reader, comptime T: type) T {
        const n = @sizeOf(T);
        const x = std.mem.readInt(T, r.b[r.at..][0..n], .little);
        r.at += n;
        return x;
    }

    fn bytes(r: *Reader) []const u8 {
        const n: usize = @intCast(r.u(u64));
        const s = r.b[r.at..][0..n];
        r.at += n;
        return s;
    }

    /// A str of the description, kept (the image's strs live as long as
    /// the program)
    fn str(r: *Reader) []const u8 {
        return std.heap.c_allocator.dupe(u8, r.bytes()) catch oom();
    }
};

fn oom() noreturn {
    @panic("zrun: out of memory building the program's data");
}

/// An item of the image built at `dst` from its description (`table`:
/// every item's address, by index)
export fn zr_image_build(kind: u32, dst: [*]u8, desc: [*]const u8, len: u64, table: [*]const usize) callconv(.c) void {
    var r = Reader{ .b = desc[0..@intCast(len)] };
    switch (@as(Kind, @enumFromInt(kind))) {
        .value_graph => buildAt(&r, dst, table),
        .rtype => {
            const t: *value.RecordType = @ptrCast(@alignCast(dst));
            const name = r.str();
            const unset_name = r.str();
            const n: usize = @intCast(r.u(u32));
            const fields = std.heap.c_allocator.alloc([]const u8, n) catch oom();
            for (fields) |*f| f.* = r.str();
            const flags = r.u(u8);
            const base = r.u(i32);
            t.* = .{
                .name = name,
                .unset_name = unset_name,
                .fields = fields,
                .py_class = null,
                .value_eq = flags & 1 != 0,
                .slots = flags & 2 != 0,
                .frozen = flags & 4 != 0,
                .base = if (base < 0) null else @ptrFromInt(table[@intCast(base)]),
            };
        },
        .function => {
            // (what the runtime reads of a closure's function: its names,
            // its parameters)
            const f: *front.Function = @ptrCast(@alignCast(dst));
            f.* = undefined;
            f.name = r.str();
            f.qualname = r.str();
            f.own = r.u(u32);
            f.param_count = r.u(u32);
            const n: usize = @intCast(r.u(u32));
            const locals = std.heap.c_allocator.alloc([]const u8, n) catch oom();
            for (locals) |*l| l.* = r.str();
            f.locals = locals;
            f.file = "";
            f.body = &.{};
            f.captures = &.{};
            f.children = &.{};
            f.env = &.{};
            f.heap = &.{};
            f.parent = null;
        },
    }
}

/// The value at the top of a graph, made at `dst` (an object)
fn buildAt(r: *Reader, dst: [*]u8, table: [*]const usize) void {
    switch (@as(V, @enumFromInt(r.u(u8)))) {
        .list => {
            const l = value.immortalListAt(dst);
            const n: usize = @intCast(r.u(u64));
            for (0..n) |_| if (!value.listPush(l, readValue(r, table))) oom();
        },
        .tuple => {
            const t: *value.Tuple = @ptrCast(@alignCast(dst));
            const n: usize = @intCast(r.u(u64));
            t.* = .{ .head = .{ .rc = value.IMMORTAL, .kind = @intFromEnum(Tag.tuple) }, .len = n };
            for (t.slice()) |*slot| slot.* = readValue(r, table);
        },
        .dict => {
            const d: *value.Dict = @ptrCast(@alignCast(dst));
            d.* = .{ .head = .{ .rc = value.IMMORTAL, .kind = @intFromEnum(Tag.dict) }, .len = 0, .used = 0, .cap = 0, .entries = null, .index = null };
            const n: usize = @intCast(r.u(u64));
            for (0..n) |_| {
                const k = readValue(r, table);
                const v = readValue(r, table);
                if (!value.dictSet(d, k, v)) oom();
            }
        },
        .set => {
            const s = set_mod.immortalAt(dst) orelse oom();
            const n: usize = @intCast(r.u(u64));
            for (0..n) |_| if (!set_mod.add(s, readValue(r, table))) oom();
        },
        .record => {
            const rec: *value.Record = @ptrCast(@alignCast(dst));
            const rtype: *const value.RecordType = @ptrFromInt(table[r.u(u32)]);
            _ = r.u(u64);
            rec.* = .{ .head = .{ .rc = value.IMMORTAL, .kind = @intFromEnum(Tag.record) }, .rtype = rtype };
            for (rec.fields()) |*slot| slot.* = readValue(r, table);
        },
        .bytes => {
            const b: *value.Bytes = @ptrCast(@alignCast(dst));
            const data = std.heap.c_allocator.dupe(u8, r.bytes()) catch oom();
            b.* = .{ .head = .{ .rc = value.IMMORTAL, .kind = @intFromEnum(Tag.bytes), .flags = value.PY_BYTES | value.OWNED_BYTES }, .ptr = data.ptr, .len = data.len, .py = null };
        },
        else => @panic("zrun: an image item that isn't an object"),
    }
}

/// A value of a graph (an object inside one: made, immortal)
fn readValue(r: *Reader, table: [*]const usize) Value {
    const tag: V = @enumFromInt(r.u(u8));
    return switch (tag) {
        .none => Value.none_v,
        .false_ => Value.boolean(false),
        .true_ => Value.boolean(true),
        .pint => Value.pint(r.u(i64)),
        .int => Value.int(r.u(i64)),
        .float => Value.float(@bitCast(r.u(u64))),
        .str => Value.obj(.str, &(value.literal(r.bytes()) orelse oom()).head),
        .big => Value.obj(.big, &(value.bigLiteral(r.u(i128)) orelse oom()).head),
        .ref => blk: {
            const t = r.u(u64);
            break :blk .{ .tag = t, .bits = table[r.u(u32)] };
        },
        .node => .{ .tag = @intFromEnum(Tag.node), .bits = r.u(u32) },
        .unset => value.unset,
        .list, .tuple, .dict, .set, .record, .bytes => blk: {
            // (room of its own, as the image's: the header before it)
            r.at -= 1;
            const size = sizeAhead(r);
            const mem = (std.heap.c_allocator.alignedAlloc(u8, .@"16", gc.head_size + size) catch oom()).ptr;
            @memset(mem[0..gc.head_size], 0);
            const at = mem + gc.head_size;
            buildAt(r, at, table);
            const obj: *value.Obj = @ptrCast(@alignCast(at));
            obj.rc = value.IMMORTAL;
            break :blk Value.obj(@enumFromInt(obj.kind), obj);
        },
    };
}

/// The room the object described next takes (not read)
fn sizeAhead(r: *Reader) usize {
    var p = Reader{ .b = r.b, .at = r.at };
    return switch (@as(V, @enumFromInt(p.u(u8)))) {
        .list => value.list_block,
        .tuple => @sizeOf(value.Tuple) + @as(usize, @intCast(p.u(u64))) * @sizeOf(Value),
        .dict => @sizeOf(value.Dict),
        .set => @sizeOf(set_mod.Set),
        .record => blk: {
            _ = p.u(u32);
            break :blk @sizeOf(value.Record) + @as(usize, @intCast(p.u(u64))) * @sizeOf(Value);
        },
        .bytes => @sizeOf(value.Bytes),
        else => 0,
    };
}

// ----------------------------------------------------------------------
// The compiler's side
// ----------------------------------------------------------------------

/// An item: a static object's bytes, a table of words, or a description
/// built at the start
pub const Entry = union(enum) {
    /// The object's bytes (a str, a Big: no pointers)
    static: []const u8,
    /// Words, those that are another item's address (by index) relocated
    words: struct { words: []const u64, refs: []const ?u32 },
    /// Built at the start: its kind, its room, its description
    built: struct { kind: Kind, size: usize, desc: []const u8 },
};

/// The items of a program's image, and the names the code calls them by
pub const Image = struct {
    a: Allocator,
    entries: std.ArrayListUnmanaged(Entry) = .empty,
    /// By address: the item's index
    index: std.AutoHashMapUnmanaged(usize, u32) = .empty,
    /// The code's names (ir.Module.Sym) and the item each is
    names: std.ArrayListUnmanaged(struct { name: [:0]const u8, item: u32 }) = .empty,

    pub fn indexOf(self: *const Image, addr: usize) ?u32 {
        return self.index.get(addr);
    }

    /// An item for an address (its index; reserved first, so a graph
    /// referring to itself refers to it)
    pub fn reserve(self: *Image, addr: usize) !u32 {
        const i: u32 = @intCast(self.entries.items.len);
        try self.entries.append(self.a, .{ .static = "" });
        try self.index.put(self.a, addr, i);
        return i;
    }

    pub fn set(self: *Image, i: u32, e: Entry) void {
        self.entries.items[i] = e;
    }
};

/// A description being written
pub const Writer = struct {
    a: Allocator,
    out: std.ArrayListUnmanaged(u8) = .empty,

    pub fn int(w: *Writer, comptime T: type, x: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, x, .little);
        try w.out.appendSlice(w.a, &b);
    }

    pub fn tag(w: *Writer, t: V) !void {
        try w.out.append(w.a, @intFromEnum(t));
    }

    pub fn bytes(w: *Writer, s: []const u8) !void {
        try w.int(u64, s.len);
        try w.out.appendSlice(w.a, s);
    }
};

/// The bytes of a static str (as value.Str lays one out, its bytes after)
pub fn strBytes(a: Allocator, s: *const value.Str) ![]u8 {
    const n = @sizeOf(value.Str) + s.len;
    const b = try a.alloc(u8, n);
    var copy = s.*;
    copy.index = null;
    copy.head.rc = value.IMMORTAL;
    @memcpy(b[0..@sizeOf(value.Str)], std.mem.asBytes(&copy));
    @memcpy(b[@sizeOf(value.Str)..], s.bytes());
    return b;
}

pub fn bigBytes(a: Allocator, big: *const value.Big) ![]u8 {
    var copy = big.*;
    copy.head.rc = value.IMMORTAL;
    return a.dupe(u8, std.mem.asBytes(&copy));
}
