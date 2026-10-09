//! Sets, natively: CPython's own table (Objects/setobject.c: open
//! addressing, nine linear probes then a perturbed jump, deleted entries
//! left as dummies, the table grown four times its items), with CPython's
//! hashes of the values (Python/pyhash.c, Objects/longobject.c,
//! floatobject.c, tupleobject.c): a set built by the same operations as
//! Python's goes over its items in Python's order.
//!
//! A value CPython's hash can't be worked out for here (a Python object, a
//! record compared by identity, NaN) gets zrun's own: its set is still
//! right, its order may not be Python's.

const std = @import("std");
const value = @import("value.zig");
const ph = @import("pyhelp.zig");
const py = ph.py;
const gc = @import("gc.zig");
const gil = @import("gil.zig");

const Value = value.Value;
const Obj = value.Obj;
const allocator = value.allocator;

/// An entry with no key (never used)
const EMPTY: u64 = value.UNSET_TAG;
/// A deleted entry's (CPython's dummy; its hash -1)
const DUMMY: u64 = value.DELETED;

pub const Set = extern struct {
    head: Obj,
    /// Entries used (items and dummies), items
    fill: u64,
    used: u64,
    mask: u64,
    table: [*]Entry,
    /// Where pop() looks first
    finger: u64,

    pub const Entry = extern struct { key: Value, hash: i64 };

    pub fn entries(self: *const Set) []Entry {
        return self.table[0 .. self.mask + 1];
    }
};

pub inline fn isItem(e: Set.Entry) bool {
    return e.key.tag != EMPTY and e.key.tag != DUMMY;
}

const min_size = 8;
const linear_probes = 9;
const perturb_shift = 5;

fn newTable(n: usize) ?[*]Set.Entry {
    const t = allocator.alloc(Set.Entry, n) catch return null;
    for (t) |*e| e.* = .{ .key = .{ .tag = EMPTY, .bits = 0 }, .hash = 0 };
    return t.ptr;
}

/// An empty set
pub fn new() ?*Set {
    const s: *Set = @ptrCast(@alignCast(gc.alloc(@sizeOf(Set)) orelse return null));
    const t = newTable(min_size) orelse {
        gc.free(&s.head, @sizeOf(Set));
        return null;
    };
    s.* = .{ .head = .{ .rc = 1, .kind = @intFromEnum(value.Tag.set) }, .fill = 0, .used = 0, .mask = min_size - 1, .table = t, .finger = 0 };
    return s;
}

/// An immortal empty set made at `mem` (a standalone build's image: the
/// cycle collector's header before it zeroed)
pub fn immortalAt(mem: [*]u8) ?*Set {
    const s: *Set = @ptrCast(@alignCast(mem));
    const t = newTable(min_size) orelse return null;
    s.* = .{ .head = .{ .rc = value.IMMORTAL, .kind = @intFromEnum(value.Tag.set) }, .fill = 0, .used = 0, .mask = min_size - 1, .table = t, .finger = 0 };
    return s;
}

/// The table's memory (its items dropped already)
pub fn freeTable(s: *Set) void {
    allocator.free(s.entries());
}

// ----------------------------------------------------------------------
// Hashes: CPython's
// ----------------------------------------------------------------------

const modulus: u64 = (1 << 61) - 1;
const hash_bits = 61;

/// The process's string hash key and function (sys.hash_info), found once
/// with the GIL (init)
var sip_key: [16]u8 = undefined;
var sip_rounds: enum { none, sip24, sip13 } = .none;
var none_hash: i64 = 0;

/// What hashing needs from the process's Python: its hash secret (the
/// interpreter's own _Py_HashSecret), its string hash function, None's
/// hash. With the GIL, once (the module loading).
pub fn init() void {
    none_hash = @intCast(py.c.PyObject_Hash(py.Py_None()));
    const sys = py.c.PyImport_ImportModule("sys") orelse return py.c.PyErr_Clear();
    defer py.Py_DecRef(sys);
    const info = py.c.PyObject_GetAttrString(sys, "hash_info") orelse return py.c.PyErr_Clear();
    defer py.Py_DecRef(info);
    const algo = py.c.PyObject_GetAttrString(info, "algorithm") orelse return py.c.PyErr_Clear();
    defer py.Py_DecRef(algo);
    const cutoff = py.c.PyObject_GetAttrString(info, "cutoff") orelse return py.c.PyErr_Clear();
    defer py.Py_DecRef(cutoff);
    // (short strings hashed otherwise (a build's cutoff): zrun's hash then)
    if (py.c.PyLong_AsLong(cutoff) != 0) return;
    const name = ph.utf8(algo, "algorithm") orelse return py.c.PyErr_Clear();
    // (the interpreter's symbols, the process's: on Linux and macOS)
    const builtin = @import("builtin");
    if (builtin.os.tag != .linux and !builtin.os.tag.isDarwin()) return;
    const self_handle = std.c.dlopen(null, .{ .LAZY = true }) orelse return;
    const secret = std.c.dlsym(self_handle, "_Py_HashSecret") orelse return;
    // (siphash's k0, k1: the secret's first 16 bytes, as stored)
    @memcpy(&sip_key, @as([*]const u8, @ptrCast(secret))[0..16]);
    if (std.mem.eql(u8, name, "siphash24")) sip_rounds = .sip24 else if (std.mem.eql(u8, name, "siphash13")) sip_rounds = .sip13;
}

fn fixHash(h: i64) i64 {
    return if (h == -1) -2 else h;
}

/// An int's hash: its value modulo 2**61 - 1, its sign kept
pub fn intHash(x: i128) i64 {
    const m: u64 = @intCast(@as(u128, @abs(x)) % modulus);
    const h: i64 = @intCast(m);
    return fixHash(if (x < 0) -h else h);
}

/// A float's (_Py_HashDouble); null for NaN (its hash is its object's)
pub fn floatHash(v: f64) ?i64 {
    if (std.math.isNan(v)) return null;
    if (std.math.isInf(v)) return if (v > 0) 314159 else -314159;
    const fr = std.math.frexp(v);
    var m = fr.significand;
    var e: i64 = fr.exponent;
    var sign: i64 = 1;
    if (m < 0) {
        sign = -1;
        m = -m;
    }
    var x: u64 = 0;
    while (m != 0) {
        x = ((x << 28) & modulus) | x >> (hash_bits - 28);
        m *= 268435456.0;
        e -= 28;
        const y: u64 = @intFromFloat(m);
        m -= @floatFromInt(y);
        x += y;
        if (x >= modulus) x -= modulus;
    }
    const sh: u6 = @intCast(if (e >= 0) @mod(e, hash_bits) else hash_bits - 1 - @mod(-1 - e, hash_bits));
    x = ((x << sh) & modulus) | x >> @intCast(hash_bits - @as(u64, sh));
    const signed: i64 = @bitCast(x *% @as(u64, @bitCast(sign)));
    return fixHash(signed);
}

/// A str's: siphash of its characters as CPython stores them (one, two
/// or four bytes each, by the largest), keyed by the process's secret
pub fn strHash(s: []const u8) ?i64 {
    if (sip_rounds == .none) return null;
    if (s.len == 0) return 0;
    var max: u21 = 0;
    var n: usize = 0;
    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (it.nextCodepoint()) |cp| {
        if (cp > max) max = cp;
        n += 1;
    }
    const width: usize = if (max < 0x100) 1 else if (max < 0x10000) 2 else 4;
    // (Latin-1 and ASCII: one byte each, the UTF-8 itself if ASCII)
    if (width == 1 and n == s.len) return sip(s);
    var small: [256]u8 = undefined;
    const buf = if (n * width <= small.len) small[0 .. n * width] else allocator.alloc(u8, n * width) catch return null;
    defer if (buf.len > small.len) allocator.free(buf);
    var i: usize = 0;
    var it2 = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (it2.nextCodepoint()) |cp| : (i += width) {
        switch (width) {
            1 => buf[i] = @intCast(cp),
            2 => std.mem.writeInt(u16, buf[i..][0..2], @intCast(cp), .little),
            else => std.mem.writeInt(u32, buf[i..][0..4], cp, .little),
        }
    }
    return sip(buf);
}

fn sip(bytes: []const u8) i64 {
    const h: u64 = switch (sip_rounds) {
        .sip13 => std.crypto.auth.siphash.SipHash64(1, 3).toInt(bytes, &sip_key),
        else => std.crypto.auth.siphash.SipHash64(2, 4).toInt(bytes, &sip_key),
    };
    return fixHash(@bitCast(h));
}

const xx_prime_1: u64 = 11400714785074694791;
const xx_prime_2: u64 = 14029467366897019727;
const xx_prime_5: u64 = 2870177450012600261;

/// A tuple's (xxHash of its items' hashes); null if an item's isn't known
pub fn tupleHash(items: []const Value) ?i64 {
    var acc: u64 = xx_prime_5;
    for (items) |x| {
        const lane: u64 = @bitCast(pyHash(x) orelse return null);
        acc +%= lane *% xx_prime_2;
        acc = std.math.rotl(u64, acc, 31);
        acc *%= xx_prime_1;
    }
    acc +%= @as(u64, items.len) ^ (xx_prime_5 ^ 3527539);
    if (acc == std.math.maxInt(u64)) return 1546275796;
    return @bitCast(acc);
}

/// CPython's hash of a value, or null if it can't be worked out natively
pub fn pyHash(v: Value) ?i64 {
    return switch (v.kind()) {
        .none => none_hash,
        .bool => @intCast(v.bits & 1),
        .int => intHash(v.asInt()),
        .big => intHash(value.wide(v).?),
        .float => floatHash(v.asFloat()),
        .str => strHash(@as(*value.Str, @ptrCast(v.ptr())).bytes()),
        // (bytes, a zrun.Bytes: siphash of the bytes themselves)
        .bytes => blk: {
            const s = @as(*value.Bytes, @ptrCast(@alignCast(v.ptr()))).slice();
            if (sip_rounds == .none) break :blk null;
            break :blk if (s.len == 0) 0 else sip(s);
        },
        .tuple => tupleHash(@as(*value.Tuple, @ptrCast(@alignCast(v.ptr()))).slice()),
        // (a frozen dataclass compared by value: its fields' tuple's, as
        // dataclasses hash it)
        .record => blk: {
            const r: *value.Record = @ptrCast(@alignCast(v.ptr()));
            if (!r.rtype.value_eq or !r.rtype.frozen) break :blk null;
            break :blk tupleHash(r.fields());
        },
        // (a Python object: its hash, Python's)
        .host => blk: {
            gil.ensure(@src());
            const h = py.c.PyObject_Hash(@ptrFromInt(v.bits));
            if (h == -1) {
                py.c.PyErr_Clear();
                break :blk null;
            }
            break :blk @intCast(h);
        },
        else => null,
    };
}

/// The hash a set keeps for a value: CPython's, else zrun's (never -1)
pub fn hashOf(v: Value) i64 {
    if (pyHash(v)) |h| return h;
    return fixHash(@bitCast(value.hash(v)));
}

// ----------------------------------------------------------------------
// The table (setobject.c)
// ----------------------------------------------------------------------

fn same(e: Set.Entry, key: Value, h: i64) bool {
    return e.hash == h and (e.key.tag == key.tag and e.key.bits == key.bits or value.equal(e.key, key));
}

/// The entry of `key`, or null (set_lookkey)
pub fn find(s: *const Set, key: Value, h: i64) ?*Set.Entry {
    const mask = s.mask;
    var perturb: u64 = @bitCast(h);
    var i: u64 = @as(u64, @bitCast(h)) & mask;
    while (true) {
        var probes: usize = if (i + linear_probes <= mask) linear_probes else 0;
        var e = &s.table[i];
        while (true) {
            if (e.key.tag == EMPTY) return null;
            if (e.key.tag != DUMMY and same(e.*, key, h)) return e;
            if (probes == 0) break;
            probes -= 1;
            e = @ptrFromInt(@intFromPtr(e) + @sizeOf(Set.Entry));
        }
        perturb >>= perturb_shift;
        i = (i *% 5 +% 1 +% perturb) & mask;
    }
}

pub fn contains(s: *const Set, key: Value) bool {
    return find(s, key, hashOf(key)) != null;
}

/// Add `key` (borrowed: a reference of its own taken) with its hash
/// (set_add_entry); false: out of memory
pub fn addHashed(s: *Set, key: Value, h: i64) bool {
    const mask = s.mask;
    var perturb: u64 = @bitCast(h);
    var i: u64 = @as(u64, @bitCast(h)) & mask;
    var free_slot: ?*Set.Entry = null;
    const unused: *Set.Entry = outer: while (true) {
        var probes: usize = if (i + linear_probes <= mask) linear_probes else 0;
        var e = &s.table[i];
        while (true) {
            if (e.key.tag == EMPTY) break :outer e;
            // (the last dummy on the way: CPython's, 3.10 to 3.13 at least)
            if (e.key.tag == DUMMY) {
                free_slot = e;
            } else if (same(e.*, key, h)) return true;
            if (probes == 0) break;
            probes -= 1;
            e = @ptrFromInt(@intFromPtr(e) + @sizeOf(Set.Entry));
        }
        perturb >>= perturb_shift;
        i = (i *% 5 +% 1 +% perturb) & mask;
    };
    value.incref(key);
    // (a dummy on the way: reused)
    if (free_slot) |f| {
        s.used += 1;
        f.* = .{ .key = key, .hash = h };
        return true;
    }
    s.fill += 1;
    s.used += 1;
    unused.* = .{ .key = key, .hash = h };
    if (s.fill * 5 < mask * 3) return true;
    return resize(s, if (s.used > 50000) s.used * 2 else s.used * 4);
}

pub fn add(s: *Set, key: Value) bool {
    return addHashed(s, key, hashOf(key));
}

/// An entry for `key` in a table with no dummies, nor `key` (set_insert_clean)
fn insertClean(table: [*]Set.Entry, mask: u64, key: Value, h: i64) void {
    var perturb: u64 = @bitCast(h);
    var i: u64 = @as(u64, @bitCast(h)) & mask;
    while (true) {
        var e = &table[i];
        if (e.key.tag == EMPTY) {
            e.* = .{ .key = key, .hash = h };
            return;
        }
        if (i + linear_probes <= mask) {
            for (0..linear_probes) |_| {
                e = @ptrFromInt(@intFromPtr(e) + @sizeOf(Set.Entry));
                if (e.key.tag == EMPTY) {
                    e.* = .{ .key = key, .hash = h };
                    return;
                }
            }
        }
        perturb >>= perturb_shift;
        i = (i *% 5 +% 1 +% perturb) & mask;
    }
}

/// The table made room for more than `min_used` items (set_table_resize)
fn resize(s: *Set, min_used: u64) bool {
    var size: u64 = min_size;
    while (size <= min_used) size <<= 1;
    const t = newTable(size) orelse return false;
    const old = s.entries();
    for (old) |e| if (isItem(e)) insertClean(t, size - 1, e.key, e.hash);
    allocator.free(old);
    s.table = t;
    s.mask = size - 1;
    s.fill = s.used;
    return true;
}

/// `key` taken out (left as a dummy): whether it was there (set_discard_entry)
pub fn discard(s: *Set, key: Value) bool {
    const e = find(s, key, hashOf(key)) orelse return false;
    const k = e.key;
    e.* = .{ .key = .{ .tag = DUMMY, .bits = 0 }, .hash = -1 };
    s.used -= 1;
    value.decref(k);
    return true;
}

/// An item taken out (owned), as set.pop() takes it; null if empty
pub fn pop(s: *Set) ?Value {
    if (s.used == 0) return null;
    var i = s.finger & s.mask;
    while (!isItem(s.table[i])) i = (i + 1) & s.mask;
    const k = s.table[i].key;
    s.table[i] = .{ .key = .{ .tag = DUMMY, .bits = 0 }, .hash = -1 };
    s.used -= 1;
    s.finger = i + 1;
    return k;
}

/// Emptied, as set.clear() leaves it (a table of 8)
pub fn clear(s: *Set) bool {
    const t = newTable(min_size) orelse return false;
    const old = s.entries();
    s.table = t;
    s.mask = min_size - 1;
    s.fill = 0;
    s.used = 0;
    for (old) |e| if (isItem(e)) value.decref(e.key);
    allocator.free(old);
    return true;
}

/// other's items added (set_merge: one resize first; into an empty set,
/// the table copied or the items put in without comparing)
pub fn merge(s: *Set, other: *const Set) bool {
    if (other == s or other.used == 0) return true;
    if ((s.fill + other.used) * 5 >= s.mask * 3) {
        if (!resize(s, (s.used + other.used) * 2)) return false;
    }
    if (s.fill == 0 and s.mask == other.mask and other.fill == other.used) {
        for (other.entries(), s.entries()) |e, *d| {
            if (isItem(e)) value.incref(e.key);
            d.* = e;
        }
        s.fill = other.fill;
        s.used = other.used;
        return true;
    }
    if (s.fill == 0) {
        s.fill = other.used;
        s.used = other.used;
        for (other.entries()) |e| if (isItem(e)) {
            value.incref(e.key);
            insertClean(s.table, s.mask, e.key, e.hash);
        };
        return true;
    }
    for (other.entries()) |e| if (isItem(e)) {
        if (!addHashed(s, e.key, e.hash)) return false;
    };
    return true;
}

/// A dict's keys added (set_update_internal of a dict: one resize first)
pub fn mergeKeys(s: *Set, keys: []const Value) bool {
    if ((s.fill + keys.len) * 5 >= s.mask * 3) {
        if (!resize(s, (s.used + keys.len) * 2)) return false;
    }
    for (keys) |k| if (!add(s, k)) return false;
    return true;
}

/// A new set of s's items (set.copy(), set(s))
pub fn copy(s: *const Set) ?*Set {
    const n = new() orelse return null;
    if (!merge(n, s)) {
        value.decref(Value.obj(.set, &n.head));
        return null;
    }
    return n;
}

/// The items in the table's order (the order Python goes over them)
pub const Iterator = struct {
    s: *const Set,
    i: usize = 0,

    pub fn next(self: *Iterator) ?Value {
        while (self.i <= self.s.mask) {
            const e = self.s.table[self.i];
            self.i += 1;
            if (isItem(e)) return e.key;
        }
        return null;
    }
};

pub fn iterate(s: *const Set) Iterator {
    return .{ .s = s };
}

/// Whether every item of a is in b
pub fn isSubset(a: *const Set, b: *const Set) bool {
    if (a.used > b.used) return false;
    var it = iterate(a);
    while (it.next()) |x| if (!contains(b, x)) return false;
    return true;
}

/// a & b (set_intersection: the smaller one gone over)
pub fn intersection(a: *const Set, b: *const Set) ?*Set {
    // (itself: a copy, as CPython makes one)
    if (a == b) return copy(a);
    const out = new() orelse return null;
    var small = b;
    var big = a;
    if (b.used > a.used) {
        small = a;
        big = b;
    }
    for (small.entries()) |e| if (isItem(e) and find(big, e.key, e.hash) != null) {
        if (!addHashed(out, e.key, e.hash)) return failed(out);
    };
    return out;
}

/// a - b (set_difference: a copy less b's items if b's much smaller,
/// else a gone over)
pub fn difference(a: *const Set, b: *const Set) ?*Set {
    if (a.used >> 2 > b.used) {
        const out = copy(a) orelse return null;
        if (!differenceUpdate(out, b)) return failed(out);
        return out;
    }
    const out = new() orelse return null;
    for (a.entries()) |e| if (isItem(e) and find(b, e.key, e.hash) == null) {
        if (!addHashed(out, e.key, e.hash)) return failed(out);
    };
    return out;
}

/// s ^= other (set_symmetric_difference_update of a set)
pub fn symmetricUpdate(s: *Set, other: *const Set) bool {
    if (s == other) return clear(s);
    for (other.entries()) |e| if (isItem(e)) {
        if (find(s, e.key, e.hash)) |_| {
            _ = discard(s, e.key);
        } else if (!addHashed(s, e.key, e.hash)) return false;
    };
    return true;
}

/// a ^ b (a copy of b, a's items toggled in it: set_symmetric_difference)
pub fn symmetricDifference(a: *const Set, b: *const Set) ?*Set {
    const out = copy(b) orelse return null;
    if (!symmetricUpdate(out, a)) return failed(out);
    return out;
}

/// a | b
pub fn union_(a: *const Set, b: *const Set) ?*Set {
    const out = copy(a) orelse return null;
    if (!merge(out, b)) return failed(out);
    return out;
}

/// s &= other: s's table becomes the intersection's (set_intersection_update)
pub fn intersectionUpdate(s: *Set, other: *const Set) bool {
    const t = intersection(s, other) orelse return false;
    swapTables(s, t);
    value.decref(Value.obj(.set, &t.head));
    return true;
}

/// s -= other (set_difference_update_internal of a set: each item
/// discarded (other's intersection with s's if it's much bigger), the
/// dummies resized away if more than a quarter of the table)
pub fn differenceUpdate(s: *Set, other: *const Set) bool {
    if (s == other) return clear(s);
    const small: ?*Set = if (other.used >> 3 > s.used) intersection(s, other) orelse return false else null;
    defer if (small) |x| value.decref(Value.obj(.set, &x.head));
    var it = iterate(small orelse other);
    while (it.next()) |x| _ = discard(s, x);
    if (s.fill - s.used <= s.mask / 4) return true;
    return resize(s, if (s.used > 50000) s.used * 2 else s.used * 4);
}

/// s & items of another iterable: the items in s, in the iterable's order
pub fn intersectionItems(s: *const Set, items: []const Value) ?*Set {
    const out = new() orelse return null;
    for (items) |x| {
        const h = hashOf(x);
        if (find(s, x, h) != null and !addHashed(out, x, h)) return failed(out);
    }
    return out;
}

/// s -= items of another iterable (each discarded, the dummies resized
/// away as differenceUpdate does)
pub fn differenceUpdateItems(s: *Set, items: []const Value) bool {
    for (items) |x| _ = discard(s, x);
    if (s.fill - s.used <= s.mask / 4) return true;
    return resize(s, if (s.used > 50000) s.used * 2 else s.used * 4);
}

pub fn intersectionUpdateItems(s: *Set, items: []const Value) bool {
    const t = intersectionItems(s, items) orelse return false;
    swapTables(s, t);
    value.decref(Value.obj(.set, &t.head));
    return true;
}

fn swapTables(a: *Set, b: *Set) void {
    std.mem.swap(u64, &a.fill, &b.fill);
    std.mem.swap(u64, &a.used, &b.used);
    std.mem.swap(u64, &a.mask, &b.mask);
    std.mem.swap([*]Set.Entry, &a.table, &b.table);
}

fn failed(s: *Set) ?*Set {
    value.decref(Value.obj(.set, &s.head));
    return null;
}
