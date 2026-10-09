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
const proxies = @import("proxies.zig");
const bytes_mod = @import("bytes.zig");
const gil = @import("gil.zig");
const gc = @import("gc.zig");
const set_mod = @import("set.zig");

/// What values are made with (and freed with: anything making one
/// elsewhere uses it too)
pub const allocator = @import("pool.zig").allocator;

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
    /// rt handed over as a value (to a function compiled code calls): the
    /// frames it runs in (bits: the frame, borrowed; the tag's upper word:
    /// the scope they're of). Valid while the call that got it runs, as an
    /// rt of the reference mode is.
    rt = 12,
    /// A plain int beyond 64 bits, within 128 (Big): Python's big ints a
    /// semantic's arithmetic makes (2**63, masks of 64 bits...), natively
    big = 13,
    /// Data read without copies (Bytes): a zrun.Bytes in Python
    bytes = 14,
    /// A function a semantic made (a lambda, a nested def as a value:
    /// Closure)
    closure = 15,
    /// A set: a Dict whose values aren't used (None)
    set = 16,
    _,
};

/// A function a semantic or helper made (front.Function.closure): its code
/// (compiled the first time it's called), the heap frame of the run it was
/// made in (its variables around), the compiled program it's of
pub const Closure = extern struct {
    head: Obj,
    env: ?*Frame,
    /// Its front.Function
    func: *const anyopaque,
    /// The compiled program it's of (driver.Compiled.id): its code runs
    /// only in that program's runs
    program: u64,
    /// Its code (driver.Helper): its own arguments, then its env (ENV_TAG)
    code: ?*const anyopaque = null,
};

/// A closure's env given to its code: the frame's address under a tag no
/// count is kept for (borrowed: the closure, or the run calling it, keeps
/// the frame)
pub const ENV_TAG: u64 = 0xFFFF_FFFD;

/// Bytes compiled code reads: part of the memory a zrun.Bytes views (bytes.zig:
/// the data a program reads, natively), or a Python `bytes` (PY_BYTES: its
/// own memory, or memory of the value's own, OWNED_BYTES: bytes made
/// natively). Read-only, as both are.
pub const Bytes = extern struct {
    head: Obj,
    ptr: [*]const u8,
    len: u64,
    /// The Python object whose memory it is (owned): a zrun.Bytes, a bytes;
    /// null for memory of its own
    py: ?*PyObject,

    pub fn slice(self: *const Bytes) []const u8 {
        return self.ptr[0..self.len];
    }

    /// A Python `bytes` (not a zrun.Bytes): what Python sees, what its
    /// slices are
    pub fn isPyBytes(self: *const Bytes) bool {
        return self.head.flags & PY_BYTES != 0;
    }
};

pub const PY_BYTES: u32 = 1 << 20;
pub const OWNED_BYTES: u32 = 1 << 21;

pub fn newBytes(ptr: [*]const u8, len: u64, owner: *PyObject) ?*Bytes {
    const b = allocator.create(Bytes) catch return null;
    py.Py_IncRef(owner);
    b.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.bytes) }, .ptr = ptr, .len = len, .py = owner };
    return b;
}

/// A Python `bytes` value of `len` bytes of its own (to fill: `ptr` is
/// writable until it's shared)
pub fn newOwnedBytes(len: usize) ?*Bytes {
    const mem = allocator.alloc(u8, @max(len, 1)) catch return null;
    const b = allocator.create(Bytes) catch {
        allocator.free(mem);
        return null;
    };
    b.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.bytes), .flags = PY_BYTES | OWNED_BYTES }, .ptr = mem.ptr, .len = len, .py = null };
    return b;
}

/// A Python `bytes` value of these bytes (copied)
pub fn bytesOf(data: []const u8) ?*Bytes {
    const b = newOwnedBytes(data.len) orelse return null;
    @memcpy(@constCast(b.ptr)[0..data.len], data);
    return b;
}

/// A part of a bytes value, of its kind (a zrun.Bytes' a view of the same
/// memory; a Python bytes' too, its owner kept; one of memory of its own:
/// copied)
pub fn bytesSlice(b: *const Bytes, from: usize, to: usize) ?*Bytes {
    if (b.head.flags & OWNED_BYTES != 0) return bytesOf(b.ptr[from..to]);
    const s = newBytes(b.ptr + from, to - from, b.py.?) orelse return null;
    s.head.flags = b.head.flags & PY_BYTES;
    return s;
}

/// The writable memory of a bytes value just made (newOwnedBytes)
pub fn bytesMemory(b: *Bytes) []u8 {
    return @constCast(b.ptr)[0..b.len];
}

/// A plain int beyond 64 bits (never one within: those are ints)
pub const Big = extern struct {
    head: Obj,
    v: i128 align(8),
};

pub fn newBig(v: i128) ?*Big {
    const b = allocator.create(Big) catch return null;
    b.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.big) }, .v = v };
    return b;
}

/// A plain int's value: an int, or a Big (a new reference), or null if
/// beyond 128 bits.
pub fn intValue(v: i128) ?Value {
    if (v >= std.math.minInt(i64) and v <= std.math.maxInt(i64)) return Value.pint(@intCast(v));
    const b = newBig(v) orelse return null;
    return Value.obj(.big, &b.head);
}

/// An int-like value's value as an i128 (an int, a bool, a Big), or null.
pub fn wide(v: Value) ?i128 {
    return switch (v.kind()) {
        .int, .bool => v.asInt(),
        .big => @as(*Big, @ptrCast(@alignCast(v.ptr()))).v,
        else => null,
    };
}

/// An int is an I64 (tag int: the program's, its arithmetic checked) or a
/// plain one (this tag: a semantic's own, as Python's: overflowing 64 bits
/// makes a big int); kind() is .int for both. (int | PLAIN: a tag test
/// `tag & ~PLAIN == int` is true for either)
pub const PLAIN: u64 = 16;
pub const PINT_TAG: u64 = @intFromEnum(Tag.int) | PLAIN;

pub const Value = extern struct {
    tag: u64,
    bits: u64,

    pub const none_v = Value{ .tag = @intFromEnum(Tag.none), .bits = 0 };

    /// An I64 (an int of the program).
    pub fn int(v: i64) Value {
        return .{ .tag = @intFromEnum(Tag.int), .bits = @bitCast(v) };
    }

    /// A plain int.
    pub fn pint(v: i64) Value {
        return .{ .tag = PINT_TAG, .bits = @bitCast(v) };
    }

    /// A plain int (not an I64).
    pub fn isPlain(self: Value) bool {
        return self.tag == PINT_TAG;
    }

    /// As rt hands values over in the reference mode (I64(): a plain int
    /// made an I64; anything else as it is).
    pub fn checked(self: Value) Value {
        return if (self.tag == PINT_TAG) .{ .tag = @intFromEnum(Tag.int), .bits = self.bits } else self;
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
        if (self.tag == PINT_TAG) return .int;
        if (self.tag & 0xFFFF_FFFF == @intFromEnum(Tag.rt)) return .rt;
        return @enumFromInt(self.tag);
    }

    /// rt as a value: the frame (borrowed) and the scope it's of.
    pub fn rt(frame: *Frame, owner: u32) Value {
        return .{ .tag = @intFromEnum(Tag.rt) | (@as(u64, owner) << 32), .bits = @intFromPtr(frame) };
    }

    /// An rt value's scope.
    pub fn rtOwner(self: Value) u32 {
        return @intCast(self.tag >> 32);
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
            .str, .list, .tuple, .dict, .record, .function, .big => true,
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

/// The tag of a slot not assigned yet (a variable, a record's field): not
/// a value (referenced counts leave it alone)
pub const UNSET_TAG: u64 = 0xFFFF_0000;
pub const unset = Value{ .tag = UNSET_TAG, .bits = 0 };

/// A list, dict or record Python has seen: it has a proxy (proxies.zig),
/// which may outlive compiled code's references
pub const HAS_PROXY: u32 = 1 << 31;

/// A function compiled code read from a variable of the program's top
/// level without a reference of its own (Gen.loadVar: calls of the
/// program's functions count nothing): that variable stored to while
/// compiled code runs, it isn't let go of but buried (Ctx.bury), kept till
/// the code's done, so a read of it never outlives it. (Its head's flags:
/// bits FunctionFlags leaves free.)
pub const FN_BORROWED: u32 = 1 << 19;

pub const Str = extern struct {
    head: Obj,
    /// Bytes (UTF-8) and code points
    len: u64,
    chars: u64,
    /// Its hash once worked out (0: not yet; literals have theirs)
    hash: u64 = 0,
    /// Not ASCII, indexed (charOffset): the byte offset of every
    /// `index_step`th code point, made the first time
    index: ?[*]u32 = null,
    // the bytes follow

    pub fn bytes(self: *const Str) []const u8 {
        const base: [*]const u8 = @ptrCast(self);
        return (base + @sizeOf(Str))[0..self.len];
    }
};

/// repr(x) of a float as Python writes it (str() the same): its shortest
/// digits that read back as it, fixed point while the point is within
/// them (from 1e-4 to below 1e16: `0.0001`, `1000000000000000.0`, always a
/// digit after the point), else exponential (`1e+16`, `1.5e-05`: two
/// exponent digits at least); `nan`, `inf`, `-inf`.
pub fn floatRepr(x: f64, buf: *[40]u8) []const u8 {
    if (std.math.isNan(x)) return "nan";
    if (std.math.isInf(x)) return if (x < 0) "-inf" else "inf";
    // (the shortest digits, Ryu's: "d.ddde±x")
    var sci: [32]u8 = undefined;
    const s = std.fmt.float.render(&sci, x, .{ .mode = .scientific }) catch return "nan";
    var i: usize = 0;
    const neg = s[0] == '-';
    if (neg) i = 1;
    var digits: [24]u8 = undefined;
    var nd: usize = 0;
    while (i < s.len and s[i] != 'e') : (i += 1) {
        if (s[i] == '.') continue;
        digits[nd] = s[i];
        nd += 1;
    }
    const exp = std.fmt.parseInt(i32, s[i + 1 ..], 10) catch 0;
    // (trailing zeros aren't digits of it)
    while (nd > 1 and digits[nd - 1] == '0') nd -= 1;
    const decpt: i32 = exp + 1;
    var w = std.Io.Writer.fixed(buf);
    if (neg) w.writeByte('-') catch {};
    if (decpt <= -4 or decpt > 16) {
        w.writeByte(digits[0]) catch {};
        if (nd > 1) {
            w.writeByte('.') catch {};
            w.writeAll(digits[1..nd]) catch {};
        }
        const e = decpt - 1;
        w.print("e{c}{d:0>2}", .{ @as(u8, if (e < 0) '-' else '+'), @abs(e) }) catch {};
    } else if (decpt <= 0) {
        w.writeAll("0.") catch {};
        for (0..@intCast(-decpt)) |_| w.writeByte('0') catch {};
        w.writeAll(digits[0..nd]) catch {};
    } else {
        const p: usize = @intCast(decpt);
        if (p >= nd) {
            w.writeAll(digits[0..nd]) catch {};
            for (0..p - nd) |_| w.writeByte('0') catch {};
            w.writeAll(".0") catch {};
        } else {
            w.writeAll(digits[0..p]) catch {};
            w.writeByte('.') catch {};
            w.writeAll(digits[p..nd]) catch {};
        }
    }
    return w.buffered();
}

const index_step = 64;

/// The byte offset of code point `i` (below its count) of a str: an ASCII
/// one's `i`; another's from its index (made the first time it's indexed:
/// a str is the same all its life), then fewer than `index_step` code
/// points on. Python's str[i] is O(1); so is this, but for those steps.
pub fn charOffset(s: *Str, i: usize) usize {
    if (s.chars == s.len) return i;
    const b = s.bytes();
    const table = @atomicLoad(?[*]u32, &s.index, .acquire) orelse makeIndex(s) orelse return stepOn(b, 0, i);
    return stepOn(b, table[i / index_step], i % index_step);
}

fn stepOn(b: []const u8, start: usize, n: usize) usize {
    var at = start;
    for (0..n) |_| at += std.unicode.utf8ByteSequenceLength(b[at]) catch 1;
    return at;
}

/// A str's index made (null: out of memory), the one another thread made
/// meanwhile if it did
fn makeIndex(s: *Str) ?[*]u32 {
    if (s.len > std.math.maxInt(u32)) return null;
    const n = s.chars / index_step + 1;
    const t = allocator.alloc(u32, n) catch return null;
    const b = s.bytes();
    var at: usize = 0;
    var chars: usize = 0;
    for (t) |*slot| {
        slot.* = @intCast(at);
        const step = @min(index_step, s.chars - chars);
        at = stepOn(b, at, step);
        chars += step;
    }
    if (@cmpxchgStrong(?[*]u32, &s.index, null, t.ptr, .acq_rel, .acquire)) |other| {
        allocator.free(t);
        return other;
    }
    return t.ptr;
}

/// A list's elements kind (its head's flags), as V8 keeps one: every item a
/// program's int (an I64: what rt gives, the ints a program makes), every
/// item a plain int (Python's, as data given brings them), every item a
/// float. An empty list has them all; storing anything else drops what it
/// isn't (listStored), never set again for the list. Compiled code reading
/// an item of a list with one knows its tag: a constant, no checks of it
/// or count after it.
pub const LIST_INTS: u32 = 1 << 0;
pub const LIST_FLOATS: u32 = 1 << 1;
pub const LIST_PINTS: u32 = 1 << 2;
pub const LIST_KINDS: u32 = LIST_INTS | LIST_FLOATS | LIST_PINTS;

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
    /// The class's name in the error reading a slot never assigned
    /// (Python's words for it: `name` before 3.13, module.qualname after)
    unset_name: []const u8,
    fields: []const []const u8,
    /// The Python class it was made from (a dataclass, or a plain class),
    /// for converting
    py_class: ?*PyObject,
    /// Records equal when their fields are (a dataclass's eq; then not
    /// hashable): else, as plain objects, by identity
    value_eq: bool = true,
    /// The class's base, a record class too (a record of this type is an
    /// instance of it)
    base: ?*const RecordType = null,
    /// A class with __slots__: its fields start unset (reading one before
    /// it's assigned is an AttributeError), its __init__ sets them
    slots: bool = false,
    /// A frozen dataclass: its fields can't be assigned
    frozen: bool = false,

    /// A record of this type is an instance of `t` (it, or a base).
    /// (one RecordType per class: compile.recordOf)
    pub fn isA(self: *const RecordType, t: *const RecordType) bool {
        var r: ?*const RecordType = self;
        while (r) |x| : (r = x.base) if (x == t) return true;
        return false;
    }
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
    /// The compiled program it's of (driver.Compiled.id, never reused): its
    /// code runs only in that program's runs
    program: u64,
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

/// A tag of a counted object: str..function (4-9), a Big (13). (By tags,
/// not kind(): reference counts are most of what code does with values)
/// The counted tags: str..function (4-9), Big, Bytes, Closure, Set; a bit
/// each
pub const counted_mask: u64 = 0x3F0 | (1 << @intFromEnum(Tag.big)) | (1 << @intFromEnum(Tag.bytes)) | (1 << @intFromEnum(Tag.closure)) | (1 << @intFromEnum(Tag.set));

pub inline fn counted(tag: u64) bool {
    return tag < 32 and (counted_mask >> @intCast(tag)) & 1 != 0;
}

pub fn incref(v: Value) void {
    if (counted(v.tag)) {
        const o = v.ptr();
        if (o.rc < IMMORTAL) o.rc += 1;
    } else if (v.tag == @intFromEnum(Tag.host)) {
        // (a count, not Python running: what strict mode allows; how the
        // object came is what it looks at)
        gil.allowBegin();
        defer gil.allowEnd();
        py.Py_IncRef(@ptrFromInt(v.bits));
    }
}

pub fn decref(v: Value) void {
    if (counted(v.tag)) {
        const o = v.ptr();
        if (o.rc >= IMMORTAL) return;
        o.rc -= 1;
        if (o.rc == 0) free(@enumFromInt(v.tag), o);
    } else if (v.tag == @intFromEnum(Tag.host)) {
        gil.allowBegin();
        defer gil.allowEnd();
        py.Py_DecRef(@ptrFromInt(v.bits));
    }
}

/// Every object reachable from some values and frames made immortal: its
/// count never changes again (shared by threads running compiled code at
/// once, no count raced; they live as long as the program that has them).
/// Into objects immortal already too (a table adopted from module state:
/// what's stored in it since isn't).
pub const Freezer = struct {
    seen: std.AutoHashMapUnmanaged(*Obj, void) = .empty,

    pub fn deinit(self: *Freezer) void {
        self.seen.deinit(allocator);
    }

    pub fn value(self: *Freezer, v: Value) error{OutOfMemory}!void {
        if (!counted(v.tag)) return;
        try self.obj(@enumFromInt(v.tag), v.ptr());
    }

    pub fn frame(self: *Freezer, f: *Frame) error{OutOfMemory}!void {
        if ((try self.seen.getOrPut(allocator, &f.head)).found_existing) return;
        f.head.rc = IMMORTAL;
        if (f.parent) |p| try self.frame(p);
        for (f.slots()) |s| if (s.tag != UNSET_TAG) try self.value(s);
    }

    fn obj(self: *Freezer, tag: Tag, o: *Obj) error{OutOfMemory}!void {
        if ((try self.seen.getOrPut(allocator, o)).found_existing) return;
        o.rc = IMMORTAL;
        switch (tag) {
            .list => for (@as(*List, @ptrCast(@alignCast(o))).slice()) |x| try self.value(x),
            .tuple => for (@as(*Tuple, @ptrCast(@alignCast(o))).slice()) |x| try self.value(x),
            .dict => {
                const d: *Dict = @ptrCast(@alignCast(o));
                if (d.entries) |es| for (es[0..d.used]) |e| {
                    try self.value(e.key);
                    try self.value(e.value);
                };
            },
            .set => for (@as(*set_mod.Set, @ptrCast(@alignCast(o))).entries()) |e| if (set_mod.isItem(e)) try self.value(e.key),
            .record => for (@as(*Record, @ptrCast(@alignCast(o))).fields()) |x| if (x.tag != UNSET_TAG) try self.value(x),
            .function => if (@as(*Function, @ptrCast(@alignCast(o))).env) |e| try self.frame(e),
            .closure => if (@as(*Closure, @ptrCast(@alignCast(o))).env) |e| try self.frame(e),
            else => {},
        }
    }
};

pub fn increfObj(o: *Obj) void {
    if (o.rc < IMMORTAL) o.rc += 1;
}

pub fn decrefFrame(f: *Frame) void {
    if (f.head.rc >= IMMORTAL) return;
    f.head.rc -= 1;
    if (f.head.rc == 0) {
        dropReferences(&f.head);
        freeBlock(&f.head);
    }
}

/// A container's references dropped (its memory kept: freeBlock())
fn dropReferences(o: *Obj) void {
    switch (o.kind) {
        @intFromEnum(Tag.list) => for (@as(*List, @ptrCast(@alignCast(o))).slice()) |item| decref(item),
        @intFromEnum(Tag.tuple) => for (@as(*Tuple, @ptrCast(@alignCast(o))).slice()) |item| decref(item),
        @intFromEnum(Tag.set) => for (@as(*set_mod.Set, @ptrCast(@alignCast(o))).entries()) |e| if (set_mod.isItem(e)) decref(e.key),
        @intFromEnum(Tag.dict) => {
            const d: *Dict = @ptrCast(@alignCast(o));
            if (d.entries) |es| for (es[0..d.used]) |e| {
                if (e.key.tag == DELETED) continue;
                decref(e.key);
                decref(e.value);
            };
        },
        @intFromEnum(Tag.record) => for (@as(*Record, @ptrCast(@alignCast(o))).fields()) |f| decref(f),
        @intFromEnum(Tag.function) => {
            const f: *Function = @ptrCast(@alignCast(o));
            if (f.env) |e| decrefFrame(e);
            decref(Value.obj(.str, &f.name.head));
        },
        @intFromEnum(Tag.closure) => if (@as(*Closure, @ptrCast(@alignCast(o))).env) |e| decrefFrame(e),
        KIND_FRAME => {
            const f: *Frame = @ptrCast(@alignCast(o));
            for (f.slots()) |s| decref(s);
            if (f.parent) |p| decrefFrame(p);
        },
        else => {},
    }
}

/// A container's memory freed (its references dropped already, or taken
/// care of by the cycle collector)
pub fn freeBlock(o: *Obj) void {
    switch (o.kind) {
        @intFromEnum(Tag.list) => {
            freeListItems(@ptrCast(@alignCast(o)));
            gc.free(o, list_block);
        },
        @intFromEnum(Tag.tuple) => gc.free(o, @sizeOf(Tuple) + @as(*Tuple, @ptrCast(@alignCast(o))).len * @sizeOf(Value)),
        @intFromEnum(Tag.set) => {
            set_mod.freeTable(@ptrCast(@alignCast(o)));
            gc.free(o, @sizeOf(set_mod.Set));
        },
        @intFromEnum(Tag.dict) => {
            const d: *Dict = @ptrCast(@alignCast(o));
            if (d.entries) |es| allocator.free(es[0..d.cap]);
            if (d.index) |ix| allocator.free(ix[0 .. d.cap * 2]);
            gc.free(o, @sizeOf(Dict));
        },
        @intFromEnum(Tag.record) => gc.free(o, @sizeOf(Record) + @as(*Record, @ptrCast(@alignCast(o))).rtype.fields.len * @sizeOf(Value)),
        @intFromEnum(Tag.function) => gc.free(o, @sizeOf(Function)),
        @intFromEnum(Tag.closure) => gc.free(o, @sizeOf(Closure)),
        KIND_FRAME => gc.free(o, @sizeOf(Frame) + @as(*Frame, @ptrCast(@alignCast(o))).len * @sizeOf(Value)),
        else => {},
    }
}

pub fn free(tag: Tag, o: *Obj) void {
    // (Python still has its proxy: the proxy keeps it)
    if (o.flags & HAS_PROXY != 0 and tag != .function and !proxies.released(o)) return;
    switch (tag) {
        .str => {
            const s: *Str = @ptrCast(o);
            if (s.index) |t| allocator.free(t[0 .. s.chars / index_step + 1]);
            allocator.free(@as([*]u8, @ptrCast(s))[0 .. @sizeOf(Str) + s.len]);
        },
        // (a list, most freed: inline)
        .list => {
            const l: *List = @ptrCast(o);
            for (l.slice()) |item| decref(item);
            freeListItems(l);
            gc.free(o, list_block);
        },
        .tuple, .dict, .record, .function, .closure, .set => {
            dropReferences(o);
            freeBlock(o);
        },
        .big => allocator.destroy(@as(*Big, @ptrCast(@alignCast(o)))),
        .bytes => {
            const b: *Bytes = @ptrCast(@alignCast(o));
            if (b.py) |owner| {
                // (a count, not Python running: what strict mode allows)
                gil.allowBegin();
                defer gil.allowEnd();
                py.Py_DecRef(owner);
            } else allocator.free(@constCast(b.ptr)[0..@max(b.len, 1)]);
            allocator.destroy(b);
        },
        else => {},
    }
}

/// A tag for deleted dict entries
pub const DELETED: u64 = 0xFFFF_FFFF;

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

/// The str of a literal (the compiled code's constants, kind names...):
/// one per content, immortal, for the process. (Not the program's memory:
/// a str stored in module state outlives the program that made it.)
pub fn literal(bytes: []const u8) ?*Str {
    // (made by compiled code on several threads at once too)
    while (!literals_lock.tryLock()) std.atomic.spinLoopHint();
    defer literals_lock.unlock();
    if (literals.get(bytes)) |s| return s;
    const s = newStr(bytes) orelse return null;
    s.head.rc = IMMORTAL;
    s.hash = strHash(bytes);
    literals.put(std.heap.c_allocator, s.bytes(), s) catch return null;
    return s;
}

var literals: std.StringHashMapUnmanaged(*Str) = .empty;
var literals_lock: std.atomic.Mutex = .unlocked;

/// The Big of a literal beyond 64 bits (as literal()'s strs: one per
/// value, immortal, for the process).
pub fn bigLiteral(v: i128) ?*Big {
    if (big_literals.get(v)) |b| return b;
    const b = newBig(v) orelse return null;
    b.head.rc = IMMORTAL;
    big_literals.put(allocator, v, b) catch return null;
    return b;
}

var big_literals: std.AutoHashMapUnmanaged(i128, *Big) = .empty;

/// Items a list has room for in its own allocation (most lists are small:
/// one allocation, none more until it grows past them)
const list_inline = 4;
const list_block = @sizeOf(List) + list_inline * @sizeOf(Value);

/// A list's own room for items (after it, in its allocation).
fn listInline(l: *List) [*]Value {
    return @ptrCast(@alignCast(@as([*]u8, @ptrCast(l)) + @sizeOf(List)));
}

/// A list's elements kind after `v` is stored in it (List's LIST_INTS...):
/// what `v` isn't dropped. Every write of a value into a list's items goes
/// through it (listPush does).
pub inline fn listStored(l: *List, v: Value) void {
    const keep: u32 = if (v.tag == PINT_TAG)
        LIST_PINTS
    else if (v.tag == @intFromEnum(Tag.int))
        LIST_INTS
    else if (v.tag == @intFromEnum(Tag.float)) LIST_FLOATS else 0;
    l.head.flags &= ~(LIST_KINDS & ~keep);
    // (a list is tracked by the cycle collector once it may hold a
    // container: until then, of numbers only, it can't be in a cycle)
    if (l.head.flags & LIST_KINDS == 0 and gc.headOf(&l.head).next == null and l.head.rc < IMMORTAL) gc.track(&l.head);
}

/// A new list, empty, room for `cap` items; untracked by the cycle
/// collector while it holds numbers only (listStored)
pub fn newList(cap: usize) ?*List {
    const l: *List = @ptrCast(@alignCast(gc.allocUntracked(list_block) orelse return null));
    l.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.list), .flags = LIST_KINDS }, .len = 0, .cap = list_inline, .items = listInline(l) };
    if (cap > list_inline) {
        const items = allocator.alloc(Value, cap) catch {
            gc.free(&l.head, list_block);
            return null;
        };
        l.items = items.ptr;
        l.cap = cap;
    }
    return l;
}

/// A list's items, given up (its own room isn't allocated apart).
fn freeListItems(l: *List) void {
    const p = l.items orelse return;
    if (p != listInline(l)) allocator.free(p[0..l.cap]);
}

/// Append, taking the reference.
/// (inline: the value stays in registers, not written to the stack and read
/// back 16 bytes at once, which stalls after two 8-byte writes)
pub inline fn listPush(l: *List, v: Value) bool {
    if (l.len == l.cap) {
        const cap = @max(list_inline * 2, l.cap * 2);
        const items = allocator.alloc(Value, cap) catch return false;
        @memcpy(items[0..l.len], l.slice());
        freeListItems(l);
        l.items = items.ptr;
        l.cap = cap;
    }
    l.items.?[l.len] = v;
    l.len += 1;
    listStored(l, v);
    return true;
}

pub fn newTuple(n: usize) ?*Tuple {
    const t: *Tuple = @ptrCast(@alignCast(gc.alloc(@sizeOf(Tuple) + n * @sizeOf(Value)) orelse return null));
    t.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.tuple) }, .len = n };
    @memset(t.slice(), Value.none_v);
    return t;
}

pub fn newRecord(rtype: *const RecordType) ?*Record {
    const n = rtype.fields.len;
    const r: *Record = @ptrCast(@alignCast(gc.alloc(@sizeOf(Record) + n * @sizeOf(Value)) orelse return null));
    r.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.record) }, .rtype = rtype };
    @memset(r.fields(), if (rtype.slots) unset else Value.none_v);
    return r;
}

/// A frame of n slots, each `fill` (a word pattern, not a byte's: a loop
/// writing it, not a call of memset, compiler_rt's going a byte at a time).
pub fn newFrame(parent: ?*Frame, n: usize, fill: Value) ?*Frame {
    const f: *Frame = @ptrCast(@alignCast(gc.alloc(@sizeOf(Frame) + n * @sizeOf(Value)) orelse return null));
    if (parent) |p| increfObj(&p.head);
    f.* = .{ .head = .{ .rc = 1, .kind = KIND_FRAME }, .parent = parent, .len = n };
    for (f.slots()) |*s| s.* = fill;
    return f;
}

/// A closure of `func` (a front.Function), its variables around in `env`
/// (a reference of its own taken).
pub fn newClosure(func: *const anyopaque, env: ?*Frame, program: u64) ?*Closure {
    const c: *Closure = @ptrCast(@alignCast(gc.alloc(@sizeOf(Closure)) orelse return null));
    if (env) |e| increfObj(&e.head);
    c.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(Tag.closure) }, .env = env, .func = func, .program = program };
    return c;
}

pub fn newDict() ?*Dict {
    const d: *Dict = @ptrCast(@alignCast(gc.alloc(@sizeOf(Dict)) orelse return null));
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
        .function, .closure => "function",
        .set => "set",
        .node => "Node",
        .host => "object",
        .rt => "CompiledRuntime",
        .big => "int",
        .bytes => if (@as(*Bytes, @ptrCast(@alignCast(v.ptr()))).isPyBytes()) "bytes" else "Bytes",
        _ => "object",
    };
}

pub fn truthy(v: Value) bool {
    return switch (v.kind()) {
        .none => false,
        .bool, .int => v.bits != 0,
        .big => true,
        .float => v.asFloat() != 0,
        .str => @as(*Str, @ptrCast(v.ptr())).len != 0,
        .list => @as(*List, @ptrCast(@alignCast(v.ptr()))).len != 0,
        .tuple => @as(*Tuple, @ptrCast(@alignCast(v.ptr()))).len != 0,
        .dict => @as(*Dict, @ptrCast(@alignCast(v.ptr()))).len != 0,
        .set => @as(*set_mod.Set, @ptrCast(@alignCast(v.ptr()))).used != 0,
        .host => blk: {
            gil.ensure(@src());
            break :blk py.c.PyObject_IsTrue(@ptrFromInt(v.bits)) == 1;
        },
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

/// An int and a float equal exactly, as Python compares them (2**53 + 1
/// isn't 2.0**53, though it is as a float).
fn intEqualsFloat(i: i64, f: f64) bool {
    if (f != @trunc(f) or !(@abs(f) < 9.3e18)) return false;
    if (f >= 9223372036854775807.0 or f < -9223372036854775808.0) return false;
    return @as(i64, @intFromFloat(f)) == i;
}

/// What a Python object is to a dict: compared and hashed by identity
/// (object's), as one of the native values (a subclass of int, float, str:
/// an IntEnum...), or by its own __eq__ and __hash__.
const HostKind = enum { identity, int, float, str, own };

fn hostKind(o: *PyObject) HostKind {
    gil.ensure(@src());
    const t = ph.typeOf(o);
    if (py.c.PyType_IsSubtype(t, exact.int()) != 0) return .int;
    if (py.c.PyType_IsSubtype(t, exact.float()) != 0) return .float;
    if (py.c.PyType_IsSubtype(t, exact.str()) != 0) return .str;
    const object = py.types.typeObject("PyBaseObject_Type");
    if (py.c.PyType_GetSlot(t, py.c.Py_tp_hash) == py.c.PyType_GetSlot(object, py.c.Py_tp_hash) and
        py.c.PyType_GetSlot(t, py.c.Py_tp_richcompare) == py.c.PyType_GetSlot(object, py.c.Py_tp_richcompare)) return .identity;
    return .own;
}

/// An int, a float or a str (native, or a Python object of a subclass of
/// one: its value, a str's characters borrowed), for comparing
const Scalar = union(enum) { int: i64, float: f64, str: []const u8, other };

fn scalarOf(v: Value) Scalar {
    switch (v.kind()) {
        .int, .bool => return .{ .int = v.asInt() },
        .float => return .{ .float = v.asFloat() },
        .str => return .{ .str = @as(*Str, @ptrCast(v.ptr())).bytes() },
        .host => {},
        else => return .other,
    }
    const o: *PyObject = @ptrFromInt(v.bits);
    switch (hostKind(o)) {
        .int => {
            var overflow: c_int = 0;
            const n = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
            if (overflow != 0 or (n == -1 and py.c.PyErr_Occurred() != null)) {
                py.c.PyErr_Clear();
                return .other;
            }
            return .{ .int = n };
        },
        .float => return .{ .float = py.c.PyFloat_AsDouble(o) },
        .str => {
            var len: py.Py_ssize_t = 0;
            const p = py.c.PyUnicode_AsUTF8AndSize(o, &len) orelse {
                py.c.PyErr_Clear();
                return .other;
            };
            return .{ .str = p[0..@intCast(len)] };
        },
        else => return .other,
    }
}

/// A record compared by value as Python sees it (its proxy; a new
/// reference), or null.
fn valueRecordObject(v: Value) ?*PyObject {
    gil.ensure(@src());
    if (v.kind() != .record) return null;
    if (!@as(*Record, @ptrCast(@alignCast(v.ptr()))).rtype.value_eq) return null;
    return proxies.make(v, @import("adopt.zig").shared_maker) orelse {
        py.c.PyErr_Clear();
        return null;
    };
}

fn hostEqual(a: Value, b: Value) bool {
    gil.ensure(@src());
    if (a.kind() == .host and b.kind() == .host and a.bits == b.bits) return true;
    const x = scalarOf(a);
    const y = scalarOf(b);
    if (x != .other and y != .other) return switch (x) {
        .int => |i| switch (y) {
            .int => |j| i == j,
            .float => |g| intEqualsFloat(i, g),
            else => false,
        },
        .float => |f| switch (y) {
            .int => |j| intEqualsFloat(j, f),
            .float => |g| f == g,
            else => false,
        },
        .str => |s| y == .str and std.mem.eql(u8, s, y.str),
        .other => unreachable,
    };
    // (Python objects: by Python's ==; with a native scalar, or a record
    // compared by value (a dataclass's: through its proxy), as Python
    // compares them)
    const pa = scalarObject(a) orelse valueRecordObject(a) orelse return false;
    defer py.Py_DecRef(pa);
    const pb = scalarObject(b) orelse valueRecordObject(b) orelse return false;
    defer py.Py_DecRef(pb);
    const r = py.c.PyObject_RichCompareBool(pa, pb, py.c.Py_EQ);
    if (r < 0) {
        py.c.PyErr_Clear();
        return false;
    }
    return r == 1;
}

/// A host value, or a scalar, as a Python object (a new reference); null
/// for the rest (they're not equal to the Python objects compared).
/// A Python int of an i128 (a new reference; null with an exception):
/// from its decimal digits.
pub fn bigObject(x: i128) ?*PyObject {
    gil.ensure(@src());
    var buf: [48]u8 = undefined;
    const s = std.fmt.bufPrintZ(&buf, "{d}", .{x}) catch unreachable;
    return py.c.PyLong_FromString(s.ptr, null, 10);
}

/// A Python int as an i128, or null if beyond 128 bits (no exception).
pub fn bigOf(o: *PyObject) ?i128 {
    // (its low 64 bits, and what's above them, which must fit an i64)
    const low = py.c.PyLong_AsUnsignedLongLongMask(o);
    if (low == std.math.maxInt(c_ulonglong) and py.c.PyErr_Occurred() != null) {
        py.c.PyErr_Clear();
        return null;
    }
    const sixty_four = py.c.PyLong_FromLong(64) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(sixty_four);
    const high_obj = py.c.PyNumber_Rshift(o, sixty_four) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(high_obj);
    var overflow: c_int = 0;
    const high = py.c.PyLong_AsLongLongAndOverflow(high_obj, &overflow);
    if (overflow != 0) return null;
    return (@as(i128, high) << 64) | @as(i128, low);
}

fn scalarObject(v: Value) ?*PyObject {
    const o: ?*PyObject = switch (v.kind()) {
        .host => blk: {
            const h: *PyObject = @ptrFromInt(v.bits);
            py.Py_IncRef(h);
            break :blk h;
        },
        .none => blk: {
            py.Py_IncRef(py.Py_None());
            break :blk py.Py_None();
        },
        .bool => py.c.PyBool_FromLong(@intFromBool(v.asInt() != 0)),
        .int => py.c.PyLong_FromLongLong(v.asInt()),
        .big => bigObject(@as(*Big, @ptrCast(@alignCast(v.ptr()))).v),
        .float => py.c.PyFloat_FromDouble(v.asFloat()),
        .str => blk: {
            const b = @as(*Str, @ptrCast(v.ptr())).bytes();
            break :blk py.c.PyUnicode_FromStringAndSize(b.ptr, @intCast(b.len));
        },
        else => null,
    };
    if (o == null) py.c.PyErr_Clear();
    return o;
}

pub fn equal(a: Value, b: Value) bool {
    if (isIntLike(a) and isIntLike(b)) return a.asInt() == b.asInt();
    // (a Big: equal to an int never (ints within 64 bits aren't Bigs), to
    // a Big of its value, to a float of exactly its value)
    if (a.kind() == .big or b.kind() == .big) {
        if (wide(a)) |x| if (wide(b)) |y| return x == y;
        const x = wide(a) orelse wide(b) orelse unreachable;
        const other = if (a.kind() == .big) b else a;
        if (other.kind() == .float) {
            const f = other.asFloat();
            return f == @trunc(f) and @abs(f) < 1.7e38 and @as(i128, @intFromFloat(f)) == x;
        }
        if (other.kind() != .host) return false;
    }
    if (a.kind() == .float and isIntLike(b)) return intEqualsFloat(b.asInt(), a.asFloat());
    if (b.kind() == .float and isIntLike(a)) return intEqualsFloat(a.asInt(), b.asFloat());
    if (a.kind() == .host or b.kind() == .host) return hostEqual(a, b);
    if (numeric(a)) |x| {
        if (numeric(b)) |y| return x == y;
        return false;
    }
    if (a.tag != b.tag) return false;
    return switch (a.kind()) {
        .none => true,
        .str => a.bits == b.bits or std.mem.eql(u8, @as(*Str, @ptrCast(a.ptr())).bytes(), @as(*Str, @ptrCast(b.ptr())).bytes()),
        .bytes => a.bits == b.bits or std.mem.eql(u8, @as(*Bytes, @ptrCast(@alignCast(a.ptr()))).slice(), @as(*Bytes, @ptrCast(@alignCast(b.ptr()))).slice()),
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
            if (a.bits == b.bits) break :blk true;
            if (x.rtype != y.rtype or !x.rtype.value_eq) break :blk false;
            for (x.fields(), y.fields()) |p, q| if (!equal(p, q)) break :blk false;
            break :blk true;
        },
        .node => a.bits == b.bits,
        // (the same items)
        .set => blk: {
            const x: *set_mod.Set = @ptrCast(@alignCast(a.ptr()));
            const y: *set_mod.Set = @ptrCast(@alignCast(b.ptr()));
            break :blk x.used == y.used and set_mod.isSubset(x, y);
        },
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
            // (one equal to a Big: hashed as it is)
            if (f == @trunc(f) and @abs(f) < 1.7e38) {
                const i: i128 = @intFromFloat(f);
                return std.hash.Wyhash.hash(2, std.mem.asBytes(&i));
            }
            return std.hash.Wyhash.hash(1, std.mem.asBytes(&bits));
        },
        .big => return std.hash.Wyhash.hash(2, std.mem.asBytes(&@as(*Big, @ptrCast(@alignCast(v.ptr()))).v)),
        // (bytes: by their bytes, as equal ones are equal)
        .bytes => return std.hash.Wyhash.hash(6, @as(*Bytes, @ptrCast(@alignCast(v.ptr()))).slice()),
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
        .host => return hostHash(@ptrFromInt(bits)),
        // (a frozen dataclass's compared by value: Python's hash of it, as
        // hostHash takes an equal Python object's)
        .record => {
            const r: *Record = @ptrCast(@alignCast(v.ptr()));
            if (!r.rtype.value_eq) return std.hash.Wyhash.hash(4, std.mem.asBytes(&bits));
            gil.ensure(@src());
            const h = proxies.valueHash(r) orelse blk: {
                py.c.PyErr_Clear();
                break :blk 0;
            };
            return std.hash.Wyhash.hash(5, std.mem.asBytes(&h));
        },
        else => return std.hash.Wyhash.hash(4, std.mem.asBytes(&bits)),
    }
}

/// A Python object's hash, alike for the values it equals (hostEqual).
fn hostHash(o: *PyObject) u64 {
    gil.ensure(@src());
    const bits = @intFromPtr(o);
    switch (scalarOf(.{ .tag = @intFromEnum(Tag.host), .bits = bits })) {
        .int => |i| return hashOf(@intFromEnum(Tag.int), @bitCast(i)),
        .float => |f| return hashOf(Value.float(f).tag, Value.float(f).bits),
        .str => |s| return strHash(s),
        .other => {},
    }
    if (hostKind(o) == .identity) return std.hash.Wyhash.hash(4, std.mem.asBytes(&bits));
    const h = py.c.PyObject_Hash(o);
    if (h == -1) py.c.PyErr_Clear();
    return std.hash.Wyhash.hash(5, std.mem.asBytes(&h));
}

pub fn hashable(v: Value) bool {
    return switch (v.kind()) {
        // (a Python object: unless its type says it isn't, as a list's does)
        .host => py.c.PyType_GetSlot(ph.typeOf(@ptrFromInt(v.bits)), py.c.Py_tp_hash) != @as(?*anyopaque, @ptrCast(@constCast(&py.c.PyObject_HashNotImplemented))),
        .list, .dict, .set => false,
        // (a dataclass compared by value isn't, unless it's frozen (by its
        // fields' values then); a plain object is, by identity)
        .record => blk: {
            const rt = @as(*Record, @ptrCast(@alignCast(v.ptr()))).rtype;
            break :blk !rt.value_eq or rt.frozen;
        },
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

/// del d[key]: whether it was there (its key and value dropped).
pub fn dictDelete(d: *Dict, key: Value) bool {
    const i = dictFind(d, key, hash(key)) orelse return false;
    const e = &d.entries.?[i];
    const k = e.key;
    const v = e.value;
    // (the entry stays, marked deleted, until a resize: the index's probe
    // chains go through it)
    e.key = .{ .tag = DELETED, .bits = 0 };
    e.value = Value.none_v;
    d.len -= 1;
    decref(k);
    decref(v);
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
    gil.ensure(@src());
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
        // (an I64, or a plain int, as the reference mode has them)
        .int => return if (v.isPlain()) py.c.PyLong_FromLongLong(v.asInt()) else types.fromInt(v.asInt()),
        .big => return bigObject(@as(*Big, @ptrCast(@alignCast(v.ptr()))).v),
        .float => return py.c.PyFloat_FromDouble(v.asFloat()),
        .str => {
            const s: *Str = @ptrCast(v.ptr());
            return ph.newString(s.bytes());
        },
        // Lists, dicts, records: proxies over the same objects (shared, as
        // in the reference mode), not copies
        .list, .dict, .record => return proxies.make(v, nodeObject),
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
        .host => {
            const o: *PyObject = @ptrFromInt(v.bits);
            py.Py_IncRef(o);
            return o;
        },
        .node => return nodeObject.make(@intCast(v.bits)),
        // (a zrun.Function, as the reference mode gives them)
        .function => return objects.newNativeFunction(@ptrCast(@alignCast(v.ptr())), nodeObject.owner),
        // (a set: a Python set of its items, a copy)
        .set => {
            const out = py.c.PySet_New(null) orelse return null;
            var it = set_mod.iterate(@ptrCast(@alignCast(v.ptr())));
            while (it.next()) |x| {
                const o = toPython(x, nodeObject) orelse {
                    py.Py_DecRef(out);
                    return null;
                };
                defer py.Py_DecRef(o);
                if (py.c.PySet_Add(out, o) != 0) {
                    py.Py_DecRef(out);
                    return null;
                }
            }
            return out;
        },
        // (a semantic's lambda or nested def: compiled code's only)
        .closure => {
            ph.raise(py.PyExc_TypeError(), "a function compiled code made (a lambda, a nested def) can't be given to Python", .{});
            return null;
        },
        // (rt reaching Python code: an rt object over its frames)
        .rt => return @import("bridge.zig").runtimeObject(v),
        // (the zrun.Bytes it's of, or a slice of it: the same memory; a
        // Python bytes: the one it is, or a bytes of its bytes)
        .bytes => {
            const b: *Bytes = @ptrCast(@alignCast(v.ptr()));
            if (b.isPyBytes()) {
                if (b.py) |o| if (py.c.PyBytes_Size(o) == @as(isize, @intCast(b.len)) and @as([*]const u8, @ptrCast(py.c.PyBytes_AsString(o))) == b.ptr) {
                    py.Py_IncRef(o);
                    return o;
                };
                return py.c.PyBytes_FromStringAndSize(@ptrCast(b.ptr), @intCast(b.len));
            }
            const whole = bytes_mod.as(b.py.?);
            if (whole.ptr == b.ptr and whole.len == b.len) {
                py.Py_IncRef(b.py.?);
                return b.py.?;
            }
            return bytes_mod.slice(b.py.?, b.ptr, b.len);
        },
        _ => {
            ph.raise(py.PyExc_TypeError(), "an unknown value", .{});
            return null;
        },
    }
}

/// A Python object as a value (a new reference), or null with an
/// exception. An I64 is an int of the program, an int a plain one (a big
/// one, beyond 64 bits: a host value, Python's). Tuples are
/// copied; a list or a dict only nothing else refers to (just made: a
/// call's result...) is copied too (no one can see the difference), one
/// that's shared is a host value, as other objects are (the same object
/// both sides see change).
pub fn fromPython(o: *PyObject) ?Value {
    return convert(o, ph.refcnt(o) == 1);
}

/// fromPython of a reference borrowed (an item of a Python list: never
/// the only one, its container has it).
pub fn fromBorrowed(o: *PyObject) ?Value {
    return convert(o, false);
}

/// The types converted (exactly these). Read when used: Python's data, on
/// Windows imported, its address known at run time only.
const exact = struct {
    inline fn int() *py.PyTypeObject {
        return py.types.typeObject("PyLong_Type");
    }
    inline fn float() *py.PyTypeObject {
        return py.types.typeObject("PyFloat_Type");
    }
    inline fn str() *py.PyTypeObject {
        return py.types.typeObject("PyUnicode_Type");
    }
    inline fn tuple() *py.PyTypeObject {
        return py.types.typeObject("PyTuple_Type");
    }
    inline fn list() *py.PyTypeObject {
        return py.types.typeObject("PyList_Type");
    }
    inline fn dict() *py.PyTypeObject {
        return py.types.typeObject("PyDict_Type");
    }
};

/// (`unique`: the reference given is the only one, the containers it's
/// in included)
fn convert(o: *PyObject, unique: bool) ?Value {
    gil.ensure(@src());
    if (o == py.Py_None()) return Value.none_v;
    // (a proxy: the compiled object itself)
    if (proxies.unwrap(o)) |v| return v;
    // (of exactly these types: a subclass (an IntEnum, a namedtuple...) is
    // a host value, itself)
    if (o == py.Py_True() or o == py.Py_False()) return Value.boolean(o == py.Py_True());
    // (a zrun.Bytes: the same memory, natively)
    if (bytes_mod.isBytes(o)) {
        const view = bytes_mod.as(o);
        const b = newBytes(view.ptr, view.len, o) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        return Value.obj(.bytes, &b.head);
    }
    const ty = ph.typeOf(o);
    // (a bytes, exactly: its memory, natively (immutable, as it is))
    if (ty == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyBytes_Type"))))) {
        const p: [*]const u8 = @ptrCast(py.c.PyBytes_AsString(o) orelse return null);
        const b = newBytes(p, @intCast(py.c.PyBytes_Size(o)), o) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        b.head.flags = PY_BYTES;
        return Value.obj(.bytes, &b.head);
    }
    // (an int: plain; zrun.I64, what rt gives Python: the program's)
    const is_i64 = @as(*PyObject, @ptrCast(@alignCast(ty))) == types.I64;
    if (ty == exact.int() or is_i64) big: {
        var overflow: c_int = 0;
        const n = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
        // (beyond 64 bits: a Big within 128; beyond, Python's own, a host
        // value; an I64 never is)
        if (overflow != 0) {
            const x = bigOf(o) orelse break :big;
            const b = newBig(x) orelse {
                _ = py.c.PyErr_NoMemory();
                return null;
            };
            return Value.obj(.big, &b.head);
        }
        return if (is_i64) Value.int(n) else Value.pint(n);
    }
    if (ty == exact.float()) return Value.float(py.c.PyFloat_AsDouble(o));
    if (ty == exact.str()) str: {
        const s = ph.utf8(o, "str") orelse {
            // (lone surrogates: not UTF-8, kept as the object)
            py.c.PyErr_Clear();
            break :str;
        };
        const str = newStr(s) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        return Value.obj(.str, &str.head);
    }
    if (ty == exact.tuple()) {
        const n: usize = @intCast(py.c.PyTuple_Size(o));
        const t = newTuple(n) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        for (t.slice(), 0..) |*slot, i| {
            // (borrowed: the tuple's)
            const item = py.c.PyTuple_GetItem(o, @intCast(i)).?;
            slot.* = convert(item, unique and ph.refcnt(item) == 1) orelse {
                // (the items not made yet are None)
                for (t.slice()[i..]) |*rest| rest.* = Value.none_v;
                decref(Value.obj(.tuple, &t.head));
                return null;
            };
        }
        return Value.obj(.tuple, &t.head);
    }
    if (unique and ty == exact.list()) {
        const n: usize = @intCast(py.c.PyList_Size(o));
        const l = newList(n) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        for (0..n) |i| {
            const item = py.c.PyList_GetItem(o, @intCast(i)).?;
            const v = convert(item, ph.refcnt(item) == 1) orelse {
                decref(Value.obj(.list, &l.head));
                return null;
            };
            _ = listPush(l, v);
        }
        return Value.obj(.list, &l.head);
    }
    if (unique and ty == exact.dict()) {
        const d = newDict() orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        var pos: py.Py_ssize_t = 0;
        var ko: ?*PyObject = null;
        var vo: ?*PyObject = null;
        while (py.c.PyDict_Next(o, &pos, @ptrCast(&ko), @ptrCast(&vo)) != 0) {
            const k = convert(ko.?, ph.refcnt(ko.?) == 1) orelse {
                decref(Value.obj(.dict, &d.head));
                return null;
            };
            defer decref(k);
            const v = convert(vo.?, ph.refcnt(vo.?) == 1) orelse {
                decref(Value.obj(.dict, &d.head));
                return null;
            };
            defer decref(v);
            if (!dictSet(d, k, v)) {
                decref(Value.obj(.dict, &d.head));
                _ = py.c.PyErr_NoMemory();
                return null;
            }
        }
        return Value.obj(.dict, &d.head);
    }
    // An rt of the compiled code, back: the rt value
    if (@import("bridge.zig").runtimeValue(o)) |v| return v;
    // A node (of the program running: nodes don't go from one program to
    // another), back: the node itself
    if (objects.asNode(o)) |n| return .{ .tag = @intFromEnum(Tag.node), .bits = n.idx };
    // A compiled function given to Python, back: itself
    if (objects.asFunction(o)) |f| if (f.native) |n| {
        increfObj(&n.head);
        return Value.obj(.function, &n.head);
    };
    py.Py_IncRef(o);
    return .{ .tag = @intFromEnum(Tag.host), .bits = @intFromPtr(o) };
}
