//! The cycle collector: reference counting frees what isn't referenced, this
//! frees what references only itself (a function and the frame it was made
//! in, holding it; a table holding itself).
//!
//! As CPython's: every container (list, tuple, dict, record, function,
//! closure, frame) is on a list of its thread's, in a generation (young, old), linked
//! through a header before it (Head: the object's own layout unchanged).
//! Collecting a generation counts, for each of its objects, the references
//! to it from the generation's others; one with more references than those
//! (from variables, compiled code, other generations, Python) is reachable,
//! and what it references; the rest is garbage, freed. Nothing is done as a
//! count changes: what a collection costs is paid as containers are made.
//!
//! The young generation is collected when a thread has made 2000 more
//! containers than it freed; survivors join the old one, collected with it
//! when it has grown by a quarter since; both at the end of a run and of a
//! map() worker, and on zrun.collect().
//!
//! What isn't collected: immortal objects (the state threads share: their
//! counts never change), and objects Python holds (the proxy of a list,
//! dict or record, when Python has it, not only the object: a root). What
//! Python objects (host values) and strings hold isn't gone through: a cycle
//! through Python isn't collected.
//!
//! Threads: each has its lists (with its allocator's: pool.zig), its
//! objects its own while it runs (as their counts are). A thread's left
//! over as it ends go to a list the threads share (ORPHAN, freed under its
//! lock), taken into the next collection's old generation.

const std = @import("std");
const value = @import("value.zig");
const pool = @import("pool.zig");
const proxies = @import("proxies.zig");
const gil = @import("gil.zig");

const Obj = value.Obj;
const Value = value.Value;
const Tag = value.Tag;
const IMMORTAL = value.IMMORTAL;
const ca = std.heap.c_allocator;

/// Before every tracked container: its neighbors on its generation's list
/// (null: not tracked). During a collection, `prev` holds the references
/// to it from outside the generation (CPython's trick).
pub const Head = extern struct {
    next: ?*Head = null,
    prev: ?*Head = null,
};

/// The header's room (the object 16-byte aligned after it)
pub const head_size = @sizeOf(Head);

/// On the shared list of threads gone (freed under its lock)
pub const ORPHAN: u32 = 1 << 30;
/// In the generation being collected
const COLLECTING: u32 = 1 << 29;
/// Reachable (in a collection)
const REACHABLE: u32 = 1 << 28;
/// Garbage (in a collection: freed at its end)
const GARBAGE: u32 = 1 << 27;

/// Young containers made (net of those freed) before a collection
const young_threshold = 2000;

/// A thread's generations
pub const State = struct {
    young: Head = .{},
    old: Head = .{},
    /// Containers made, less those freed, since the young generation was
    /// collected
    young_count: isize = 0,
    /// The old generation's size, and its size at the last time it was
    /// collected
    old_count: usize = 0,
    old_collected: usize = 0,
    collecting: bool = false,

    fn ready(s: *State) void {
        if (s.young.next != null) return;
        s.young = .{ .next = &s.young, .prev = &s.young };
        s.old = .{ .next = &s.old, .prev = &s.old };
    }
};

/// The lists of threads gone
var orphans: Head = .{};
var orphans_lock: std.atomic.Mutex = .unlocked;

fn lockOrphans() void {
    while (!orphans_lock.tryLock()) std.atomic.spinLoopHint();
    if (orphans.next == null) orphans = .{ .next = &orphans, .prev = &orphans };
}

pub inline fn headOf(o: *Obj) *Head {
    return @ptrFromInt(@intFromPtr(o) - head_size);
}

inline fn objOf(h: *Head) *Obj {
    return @ptrFromInt(@intFromPtr(h) + head_size);
}

inline fn link(list: *Head, h: *Head) void {
    const last = list.prev.?;
    h.prev = last;
    h.next = list;
    last.next = h;
    list.prev = h;
}

inline fn unlink(h: *Head) void {
    h.prev.?.next = h.next;
    h.next.?.prev = h.prev;
    h.next = null;
    h.prev = null;
}

/// `from`'s objects put at the end of `to`, `from` emptied
fn splice(to: *Head, from: *Head) void {
    if (from.next == from) return;
    const first = from.next.?;
    const last = from.prev.?;
    const end = to.prev.?;
    end.next = first;
    first.prev = end;
    last.next = to;
    to.prev = last;
    from.next = from;
    from.prev = from;
}

/// A container's memory, `size` bytes, tracked in this thread's young
/// generation (a collection first if it's due: no other object is being
/// made then).
pub inline fn alloc(size: usize) ?[*]u8 {
    const l = pool.current();
    const s = &l.gc;
    if (s.young_count >= young_threshold or s.young.next == null) prepare(s);
    const mem = pool.allocIn(l, head_size + size) orelse return null;
    const h: *Head = @ptrCast(@alignCast(mem));
    link(&s.young, h);
    s.young_count += 1;
    return mem + head_size;
}

/// A container's memory, `size` bytes, not tracked: one that can't be in a
/// cycle yet (a list of numbers only); track() it once it can.
pub inline fn allocUntracked(size: usize) ?[*]u8 {
    const mem = pool.allocIn(pool.current(), head_size + size) orelse return null;
    const h: *Head = @ptrCast(@alignCast(mem));
    h.* = .{};
    return mem + head_size;
}

/// An untracked container tracked from now, in this thread's young
/// generation (no collection here: it's due at the next container made).
pub noinline fn track(o: *Obj) void {
    const s = &pool.current().gc;
    s.ready();
    link(&s.young, headOf(o));
    s.young_count += 1;
}

/// (a thread's first container: its lists made; a collection due)
noinline fn prepare(s: *State) void {
    s.ready();
    if (s.young_count >= young_threshold) collectDue(s);
}

/// A container's memory freed (`size`: alloc()'s).
pub inline fn free(o: *Obj, size: usize) void {
    const h = headOf(o);
    const l = pool.current();
    if (h.next != null) {
        if (o.flags & ORPHAN != 0) {
            freeOrphan(h);
        } else {
            unlink(h);
            l.gc.young_count -= 1;
        }
    }
    pool.freeIn(l, @ptrCast(h), head_size + size);
}

noinline fn freeOrphan(h: *Head) void {
    lockOrphans();
    unlink(h);
    orphans_lock.unlock();
}


noinline fn collectDue(s: *State) void {
    if (s.collecting) return;
    // (the old generation too once it's grown by a quarter)
    _ = collect(s, s.old_count - s.old_collected > s.old_collected / 4 + young_threshold, false);
}

/// This thread's generations collected (zrun.collect(), a run's end), what
/// Python holds looked at: the objects freed.
pub fn collectHere() usize {
    return collect(pool.gcState(), true, true);
}

/// collectHere() not looking at what Python holds (the GIL not taken: a
/// map() worker's, running in parallel)
pub fn collectParallel() usize {
    return collect(pool.gcState(), true, false);
}

/// A thread's lists, as it ends: the threads' shared one's.
pub fn orphan(s: *State) void {
    if (s.young.next == null) return;
    splice(&s.old, &s.young);
    if (s.old.next != &s.old) {
        var h = s.old.next.?;
        while (h != &s.old) : (h = h.next.?) objOf(h).flags |= ORPHAN;
        lockOrphans();
        splice(&orphans, &s.old);
        orphans_lock.unlock();
    }
    s.* = .{};
}

/// `f(ctx, child)` for each container `o` references (counted: not
/// immortal).
inline fn eachChild(o: *Obj, ctx: anytype, comptime f: anytype) void {
    switch (o.kind) {
        @intFromEnum(Tag.list) => for (@as(*value.List, @ptrCast(@alignCast(o))).slice()) |v| child(v, ctx, f),
        @intFromEnum(Tag.tuple) => for (@as(*value.Tuple, @ptrCast(@alignCast(o))).slice()) |v| child(v, ctx, f),
        @intFromEnum(Tag.dict), @intFromEnum(Tag.set) => {
            const d: *value.Dict = @ptrCast(@alignCast(o));
            if (d.entries) |es| for (es[0..d.used]) |e| {
                if (e.key.tag == value.DELETED) continue;
                child(e.key, ctx, f);
                child(e.value, ctx, f);
            };
        },
        @intFromEnum(Tag.record) => for (@as(*value.Record, @ptrCast(@alignCast(o))).fields()) |v| child(v, ctx, f),
        @intFromEnum(Tag.function) => if (@as(*value.Function, @ptrCast(@alignCast(o))).env) |e| frameChild(e, ctx, f),
        @intFromEnum(Tag.closure) => if (@as(*value.Closure, @ptrCast(@alignCast(o))).env) |e| frameChild(e, ctx, f),
        value.KIND_FRAME => {
            const fr: *value.Frame = @ptrCast(@alignCast(o));
            if (fr.parent) |p| frameChild(p, ctx, f);
            for (fr.slots()) |v| child(v, ctx, f);
        },
        else => {},
    }
}

/// A container's tag: list..function (5-9), closure, set
inline fn isContainer(tag: u64) bool {
    return tag -% @intFromEnum(Tag.list) < 5 or tag -% @intFromEnum(Tag.closure) < 2;
}

inline fn child(v: Value, ctx: anytype, comptime f: anytype) void {
    if (!isContainer(v.tag)) return;
    const o = v.ptr();
    if (o.rc < IMMORTAL) f(ctx, o);
}

inline fn frameChild(fr: *value.Frame, ctx: anytype, comptime f: anytype) void {
    if (fr.head.rc < IMMORTAL) f(ctx, &fr.head);
}

/// A collection's references from outside the generation, kept in `prev`
inline fn refsOf(h: *Head) *isize {
    return @ptrCast(&h.prev);
}

const Stack = std.ArrayListUnmanaged(*Obj);

/// Collect the young generation (and the old, `full`): the objects freed.
/// `python`: whether Python holds the objects it has a proxy of looked at
/// (the GIL taken); else they're all held.
pub fn collect(s: *State, full: bool, python: bool) usize {
    if (s.collecting) return 0;
    s.ready();
    s.collecting = true;
    defer s.collecting = false;
    // (the threads gone's left over: into the old generation)
    if (full and orphans.next != null) {
        lockOrphans();
        var h = orphans.next.?;
        while (h != &orphans) : (h = h.next.?) objOf(h).flags &= ~ORPHAN;
        splice(&s.old, &orphans);
        orphans_lock.unlock();
    }
    // The generation: the young one (and the old one with it)
    var gen: Head = .{};
    gen = .{ .next = &gen, .prev = &gen };
    splice(&gen, &s.young);
    if (full) {
        splice(&gen, &s.old);
        s.old_count = 0;
    }
    s.young_count = 0;
    // Each object's count, less immortal ones (untracked: shared, their
    // counts never change), marked; `prev` its count from now
    var n: usize = 0;
    {
        var h = gen.next.?;
        while (h != &gen) {
            const next = h.next.?;
            const o = objOf(h);
            if (o.rc >= IMMORTAL) {
                unlink(h);
            } else {
                o.flags |= COLLECTING;
                n += 1;
            }
            h = next;
        }
        h = gen.next.?;
        while (h != &gen) : (h = h.next.?) refsOf(h).* = @intCast(objOf(h).rc);
    }
    // Less the references from within
    {
        var h = gen.next.?;
        while (h != &gen) : (h = h.next.?) eachChild(objOf(h), {}, struct {
            fn f(_: void, t: *Obj) void {
                if (t.flags & COLLECTING != 0) refsOf(headOf(t)).* -= 1;
            }
        }.f);
    }
    // Reachable: what a reference from outside (or Python's proxy) keeps,
    // and what that references
    var stack: Stack = .empty;
    defer stack.deinit(ca);
    var failed = false;
    {
        var h = gen.next.?;
        while (h != &gen) : (h = h.next.?) {
            const o = objOf(h);
            if (refsOf(h).* > 0 or heldByPython(o, python)) {
                o.flags |= REACHABLE;
                stack.append(ca, o) catch {
                    failed = true;
                };
            }
        }
        const Ctx = struct { stack: *Stack, failed: *bool };
        var c = Ctx{ .stack = &stack, .failed = &failed };
        while (stack.pop()) |o| eachChild(o, &c, struct {
            fn f(cx: *Ctx, t: *Obj) void {
                if (t.flags & (COLLECTING | REACHABLE) != COLLECTING) return;
                t.flags |= REACHABLE;
                cx.stack.append(ca, t) catch {
                    cx.failed.* = true;
                };
            }
        }.f);
    }
    // (the `prev` links made again: they held counts)
    {
        var prev: *Head = &gen;
        var h = gen.next.?;
        while (h != &gen) : (h = h.next.?) {
            h.prev = prev;
            prev = h;
        }
        gen.prev = prev;
    }
    // The rest garbage (unless out of memory: then nothing is, rather than
    // something referenced)
    var garbage: Head = .{};
    garbage = .{ .next = &garbage, .prev = &garbage };
    var freed: usize = 0;
    {
        var h = gen.next.?;
        while (h != &gen) {
            const next = h.next.?;
            const o = objOf(h);
            if (o.flags & REACHABLE == 0 and !failed) {
                unlink(h);
                link(&garbage, h);
                o.flags |= GARBAGE;
                freed += 1;
            } else o.flags &= ~(COLLECTING | REACHABLE);
            h = next;
        }
    }
    // The survivors: into the old generation
    s.old_count += n - freed;
    splice(&s.old, &gen);
    if (full) s.old_collected = s.old_count;
    if (freed == 0) return 0;
    // The garbage: the references it holds to what isn't garbage dropped
    // once it's freed (they may free more, and run Python's code: the
    // garbage gone by then)
    var drop: std.ArrayListUnmanaged(Value) = .empty;
    defer drop.deinit(ca);
    {
        var h = garbage.next.?;
        while (h != &garbage) : (h = h.next.?) references(objOf(h), &drop) catch {
            // (no room to keep them: the garbage kept, not freed, as
            // survivors)
            var g = garbage.next.?;
            while (g != &garbage) : (g = g.next.?) objOf(g).flags &= ~(COLLECTING | REACHABLE | GARBAGE);
            splice(&s.old, &garbage);
            return 0;
        };
    }
    {
        var h = garbage.next.?;
        while (h != &garbage) {
            const next = h.next.?;
            h.next = null;
            h.prev = null;
            const o = objOf(h);
            // (its proxy, only its own: let go of (the GIL held since
            // heldByPython: Python hasn't taken it meanwhile))
            if (o.flags & value.HAS_PROXY != 0) _ = proxies.released(o);
            value.freeBlock(o);
            h = next;
        }
    }
    for (drop.items) |v| {
        if (v.tag == frame_tag) value.decrefFrame(@ptrFromInt(v.bits)) else value.decref(v);
    }
    return freed;
}

/// The references a garbage object holds to what isn't garbage (anything
/// but a container being freed with it: GARBAGE)
fn references(o: *Obj, out: *std.ArrayListUnmanaged(Value)) !void {
    const Add = struct {
        fn value_(list: *std.ArrayListUnmanaged(Value), v: Value) !void {
            if (isContainer(v.tag) and v.ptr().flags & GARBAGE != 0) return;
            try list.append(ca, v);
        }
        fn frame(list: *std.ArrayListUnmanaged(Value), f: *value.Frame) !void {
            if (f.head.flags & GARBAGE != 0) return;
            try list.append(ca, .{ .tag = frame_tag, .bits = @intFromPtr(f) });
        }
    };
    switch (o.kind) {
        @intFromEnum(Tag.list) => for (@as(*value.List, @ptrCast(@alignCast(o))).slice()) |v| try Add.value_(out, v),
        @intFromEnum(Tag.tuple) => for (@as(*value.Tuple, @ptrCast(@alignCast(o))).slice()) |v| try Add.value_(out, v),
        @intFromEnum(Tag.dict), @intFromEnum(Tag.set) => {
            const d: *value.Dict = @ptrCast(@alignCast(o));
            if (d.entries) |es| for (es[0..d.used]) |e| {
                if (e.key.tag == value.DELETED) continue;
                try Add.value_(out, e.key);
                try Add.value_(out, e.value);
            };
        },
        @intFromEnum(Tag.record) => for (@as(*value.Record, @ptrCast(@alignCast(o))).fields()) |v| try Add.value_(out, v),
        @intFromEnum(Tag.function) => {
            const f: *value.Function = @ptrCast(@alignCast(o));
            if (f.env) |e| try Add.frame(out, e);
            try out.append(ca, Value.obj(.str, &f.name.head));
        },
        @intFromEnum(Tag.closure) => if (@as(*value.Closure, @ptrCast(@alignCast(o))).env) |e| try Add.frame(out, e),
        value.KIND_FRAME => {
            const f: *value.Frame = @ptrCast(@alignCast(o));
            if (f.parent) |p| try Add.frame(out, p);
            for (f.slots()) |v| try Add.value_(out, v);
        },
        else => {},
    }
}

/// An object Python may reach: its proxy held by Python, not only by the
/// object (a root of the collection). Known when `look` (the GIL taken);
/// else any object with a proxy (collections run as containers are made:
/// without the GIL, in parallel).
fn heldByPython(o: *Obj, look: bool) bool {
    if (o.flags & value.HAS_PROXY == 0) return false;
    if (!look) return true;
    // (asking Python, not running it: what strict mode allows)
    gil.allowBegin();
    defer gil.allowEnd();
    return proxies.heldByPython(o);
}

/// The tag `references` gives a frame (not a value's: dropped with
/// decrefFrame)
const frame_tag: u64 = 0xFFFF_FFFE;
