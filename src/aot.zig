//! Compiled modules: a program compiled ahead of time, in a file
//! (`program.save(path)`, `lang.compile(source, output)`), loaded again
//! without compiling (`lang.load_compiled(path)`): its source and the
//! object of every module of compiled code made for it (the main one, the
//! ones compiled as it ran: Python functions it called, typed entries), by
//! what each is the code of (cache.zig's keys).
//!
//! Loading parses and checks the source again (fast) and makes the IR
//! again (a small part of compiling); LLVM's work, nearly all of it, is the
//! objects'. A module made for another definition of the language (its
//! grammar, semantics, host functions...), another zrun or another CPU is
//! refused, saying why: the IR it'd make wouldn't be the objects' (or, the
//! language changed in a way its definition's hash missed, its objects are
//! of no use: the code is compiled, never wrong).
//!
//! The file: "ZRUNMOD1", zrun's version (u32 length, bytes), the
//! language's definition hash (32 bytes), what the objects depend on
//! besides their IR (32: cache.salt()), the program's path (u32, bytes),
//! its source (u64, bytes), its objects (u32 count; each: key 32 bytes,
//! u64 length, bytes). Little-endian.

const std = @import("std");
const cache = @import("cache.zig");

const magic = "ZRUNMOD1";
/// zrun's version (build.zig.zon's)
pub const version = @import("build_options").version;

pub const Object = struct { key: cache.Key, bytes: []u8 };

/// A compiled module read (owned: deinit)
pub const File = struct {
    zrun: []u8,
    definition: [32]u8,
    salt: [32]u8,
    path: []u8,
    source: []u8,
    objects: []Object,

    pub fn deinit(self: *File, a: std.mem.Allocator) void {
        a.free(self.zrun);
        a.free(self.path);
        a.free(self.source);
        for (self.objects) |o| a.free(o.bytes);
        a.free(self.objects);
    }
};

pub const Error = error{ OutOfMemory, NotAModule, Truncated, Io };

/// Write a compiled module at `path` (whole or not at all: cache.writeWhole).
pub fn write(a: std.mem.Allocator, path: []const u8, definition: [32]u8, prog_path: []const u8, source: []const u8, objects: []const Object) Error!void {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(a);
    try out.appendSlice(a, magic);
    try putBytes32(a, &out, version);
    try out.appendSlice(a, &definition);
    try out.appendSlice(a, cache.salt());
    try putBytes32(a, &out, prog_path);
    try putInt(a, &out, u64, source.len);
    try out.appendSlice(a, source);
    try putInt(a, &out, u32, @intCast(objects.len));
    for (objects) |o| {
        try out.appendSlice(a, &o.key);
        try putInt(a, &out, u64, o.bytes.len);
        try out.appendSlice(a, o.bytes);
    }
    cache.writeWhole(path, out.items) catch return error.Io;
}

/// Read a compiled module.
pub fn read(a: std.mem.Allocator, path: []const u8) Error!File {
    const bytes = cache.read(a, path) orelse return error.Io;
    defer a.free(bytes);
    var r = Reader{ .bytes = bytes };
    if (!std.mem.eql(u8, try r.take(magic.len), magic)) return error.NotAModule;
    const zrun = try a.dupe(u8, try r.take(try r.int(u32)));
    errdefer a.free(zrun);
    var f = File{ .zrun = zrun, .definition = undefined, .salt = undefined, .path = &.{}, .source = &.{}, .objects = &.{} };
    @memcpy(&f.definition, try r.take(32));
    @memcpy(&f.salt, try r.take(32));
    f.path = try a.dupe(u8, try r.take(try r.int(u32)));
    errdefer a.free(f.path);
    f.source = try a.dupe(u8, try r.take(try r.int(u64)));
    errdefer a.free(f.source);
    const n = try r.int(u32);
    var objects: std.ArrayListUnmanaged(Object) = .empty;
    errdefer {
        for (objects.items) |o| a.free(o.bytes);
        objects.deinit(a);
    }
    for (0..n) |_| {
        var o: Object = undefined;
        @memcpy(&o.key, try r.take(32));
        o.bytes = try a.dupe(u8, try r.take(try r.int(u64)));
        objects.append(a, o) catch |e| {
            a.free(o.bytes);
            return e;
        };
    }
    f.objects = try objects.toOwnedSlice(a);
    return f;
}

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn take(r: *Reader, n: u64) Error![]const u8 {
        if (n > r.bytes.len - r.at) return error.Truncated;
        const s = r.bytes[r.at .. r.at + @as(usize, @intCast(n))];
        r.at += @intCast(n);
        return s;
    }

    fn int(r: *Reader, comptime T: type) Error!T {
        return std.mem.readInt(T, (try r.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
};

fn putInt(a: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), comptime T: type, x: T) !void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, x, .little);
    try out.appendSlice(a, &buf);
}

fn putBytes32(a: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), s: []const u8) !void {
    try putInt(a, out, u32, @intCast(s.len));
    try out.appendSlice(a, s);
}

/// The Python computing a language's definition hash: its grammar (what
/// its parser tells of it), its semantics' and host functions' code, the
/// rest of its definition as text (lib.zig's).
pub const definition_source =
    \\import hashlib, marshal
    \\def definition(parser, tables, hosts, rest):
    \\    h = hashlib.sha256()
    \\    def text(x):
    \\        h.update(repr(x).encode("utf-8", "replace"))
    \\        h.update(b"\0")
    \\    def code(f):
    \\        c = getattr(f, "__code__", None)
    \\        if c is not None:
    \\            h.update(marshal.dumps(c))
    \\        else:
    \\            text((type(f).__name__, getattr(f, "__name__", None), getattr(f, "signature", None)))
    \\    for m in ("rules", "actions", "fields", "labels", "literals"):
    \\        try:
    \\            text(getattr(parser, m)())
    \\        except Exception:
    \\            text(m)
    \\    for table in tables:
    \\        for k in sorted(table, key=repr):
    \\            text(k)
    \\            for f in (table[k] if isinstance(table[k], (list, tuple)) else (table[k],)):
    \\                code(f)
    \\    for k in sorted(hosts, key=repr):
    \\        text(k)
    \\        code(hosts[k])
    \\    text(rest)
    \\    return h.digest()
;
