//! The allocator of the values compiled code makes (strs, lists, tuples,
//! dicts, records, frames...): small blocks, by size class, kept on free
//! lists when freed and handed out again, carved from slabs (malloc and
//! free for each would be most of the time a program runs). Bigger blocks
//! (or more aligned ones) are libc's.
//!
//! Per thread: compiled code runs on several threads at once (calls
//! without the GIL, map()). A thread's blocks, as it ends, go to a pool
//! the threads share, taken from before a slab's carved.
//!
//! One thread's lists (the first to carve: the one importing zrun, usually
//! running the programs) are plain globals, the others' thread-locals: a
//! thread-local of a library (zrun is one) costs a call to reach
//! (__tls_get_addr), a quarter of the time of a program allocating much.

const std = @import("std");
const builtin = @import("builtin");
const gc_mod = @import("gc.zig");

const Alignment = std.mem.Alignment;

/// Blocks up to this size come from the pool
const max_small = 512;
/// Size classes: every 16 bytes up to max_small
const granule = 16;
const classes = max_small / granule;
/// A slab: carved into blocks of one class
const slab_size = 64 * 1024;
/// A free block: the next one in its class's list
const Free = struct { next: ?*Free };

/// A free list in the shared pool (a thread's, given as it ended): its
/// first block, which links it to the next list (blocks are 16 bytes or
/// more: room for both)
const Batch = struct { first: Free, next_batch: ?*Batch };

/// A thread's blocks
pub const Lists = struct {
    /// The free ones, per class (their heads only: counting them would
    /// cost the hot path)
    free: [classes]?*Free = .{null} ** classes,
    /// The slab being carved, per class, and how much of it is left
    carve: [classes][]u8 = .{&.{}} ** classes,
    /// Blocks handed out and not freed by this thread (the tests' check
    /// that compiled code gives back what it takes: zrun._blocks())
    in_use: isize = 0,
    /// The cycle collector's buffer (as cheap to reach as the lists)
    gc: gc_mod.State = .{},
    /// Given to the shared pool as the thread ends (watch(): any thread
    /// with blocks, carved or freed by it)
    watched: bool = false,
    /// Blocks freed since the lists were trimmed (trim())
    frees: u32 = 0,

    /// All its free blocks (its slabs' too) to the shared pool: the thread
    /// is gone
    fn give(l: *Lists) void {
        for (0..classes) |c| {
            // (the slab being carved: its blocks free ones)
            const size = (c + 1) * granule;
            while (l.carve[c].len >= size) {
                const f: *Free = @ptrCast(@alignCast(l.carve[c].ptr));
                f.next = l.free[c];
                l.free[c] = f;
                l.carve[c] = l.carve[c][size..];
            }
            const b: *Batch = @ptrCast(l.free[c] orelse continue);
            l.free[c] = null;
            lockShared();
            b.next_batch = shared[c];
            shared[c] = b;
            shared_lock.unlock();
        }
    }
};

/// The home lists (a plain global's: no thread-local's call) and their
/// thread (its threadPointer(); 0: none)
var home: Lists = .{};
var home_owner: usize = 0;
/// The other threads'
threadlocal var mine: Lists = .{};

/// The shared pool: free lists of threads gone, per class (taken whole:
/// nothing walked under the lock)
var shared: [classes]?*Batch = .{null} ** classes;
var shared_lock: std.atomic.Mutex = .unlocked;
/// The hook run as a thread that took blocks ends (threadGone): a pthread
/// key's destructor; on Windows, a fiber-local slot's callback
const ExitHook = if (builtin.os.tag == .windows) struct {
    extern "kernel32" fn FlsAlloc(callback: ?*const fn (?*anyopaque) callconv(.winapi) void) callconv(.winapi) u32;
    extern "kernel32" fn FlsSetValue(index: u32, value: ?*anyopaque) callconv(.winapi) i32;
    const out_of_indexes: u32 = 0xFFFF_FFFF;

    var index: u32 = out_of_indexes;

    fn gone(_: ?*anyopaque) callconv(.winapi) void {
        threadGone();
    }

    fn make() bool {
        index = FlsAlloc(&gone);
        return index != out_of_indexes;
    }

    fn arm() void {
        _ = FlsSetValue(index, @ptrFromInt(1));
    }
} else struct {
    var key: std.c.pthread_key_t = undefined;

    fn gone(_: *anyopaque) callconv(.c) void {
        threadGone();
    }

    fn make() bool {
        return std.c.pthread_key_create(&key, &gone) == .SUCCESS;
    }

    fn arm() void {
        _ = std.c.pthread_setspecific(key, @ptrFromInt(1));
    }
};
var have_hook = false;

/// A word naming this thread, read with an instruction (the thread
/// pointer: Linux's TCB address, Windows' TEB address); 0: not known here
/// (no home lists)
pub inline fn threadPointer() usize {
    return switch (builtin.os.tag) {
        .linux => switch (builtin.cpu.arch) {
            .x86_64 => asm ("movq %%fs:0, %[ret]"
                : [ret] "=r" (-> usize),
            ),
            .aarch64 => asm ("mrs %[ret], tpidr_el0"
                : [ret] "=r" (-> usize),
            ),
            else => 0,
        },
        .windows => switch (builtin.cpu.arch) {
            .x86_64 => asm ("movq %%gs:0x30, %[ret]"
                : [ret] "=r" (-> usize),
            ),
            .aarch64 => asm ("mov %[ret], x18"
                : [ret] "=r" (-> usize),
            ),
            else => 0,
        },
        else => 0,
    };
}

/// The thread with the home lists (its threadPointer()), or 0.
pub inline fn homeOwner() usize {
    return @atomicLoad(usize, &home_owner, .monotonic);
}

/// The home lists this thread's if no thread has them (as its first
/// allocation would make them: gil.zig asks before it reads its state, code
/// that allocates nothing reading a thread-local otherwise).
pub fn claimHome() void {
    _ = mineLists();
}

/// This thread's lists
inline fn lists() *Lists {
    const tp = threadPointer();
    if (tp != 0 and tp == @atomicLoad(usize, &home_owner, .monotonic)) return &home;
    return mineLists();
}

/// (no thread has the home lists: this one's from now, as theirs went to
/// the shared pool)
noinline fn mineLists() *Lists {
    const tp = threadPointer();
    if (tp != 0 and @atomicLoad(usize, &home_owner, .acquire) == 0) {
        if (!mine.watched) watch();
        if (@cmpxchgStrong(usize, &home_owner, 0, tp, .acquire, .monotonic) == null) {
            // (the counts this thread's: carried over; its GIL state too,
            // gil.zig's home state the home lists' thread's)
            home.in_use = mine.in_use;
            mine.in_use = 0;
            home.watched = true;
            @import("gil.zig").claimHome();
            return &home;
        }
    }
    return &mine;
}

/// The home lists lent by thread `owner` (threadPointer()), waiting for
/// this one (onBigStack()), if it has them; give back with `restore`.
pub fn borrowHome(owner: usize) bool {
    const tp = threadPointer();
    if (tp == 0 or owner == 0) return false;
    return @cmpxchgStrong(usize, &home_owner, owner, tp, .monotonic, .monotonic) == null;
}

pub fn restoreHome(owner: usize) void {
    @atomicStore(usize, &home_owner, owner, .monotonic);
}

/// This thread's cycle collector buffer (given on when it ends)
pub inline fn gcState() *gc_mod.State {
    const l = lists();
    if (l == &home) return &home.gc;
    return otherGcState();
}

noinline fn otherGcState() *gc_mod.State {
    if (!mine.watched) watch();
    return &lists().gc;
}

/// Blocks this thread has and hasn't freed (zrun._blocks())
pub fn inUse() isize {
    return lists().in_use;
}

fn lockShared() void {
    while (!shared_lock.tryLock()) std.atomic.spinLoopHint();
}

/// This thread's blocks given to the shared pool when it ends (and the
/// home lists, if it has them)
noinline fn watch() void {
    mine.watched = true;
    lockShared();
    defer shared_lock.unlock();
    if (!have_hook) {
        if (!ExitHook.make()) return;
        have_hook = true;
    }
    ExitHook.arm();
}

fn threadGone() void {
    // (the buffer first: freeing its dead objects later gives blocks to
    // another thread's lists)
    gc_mod.orphan(&mine.gc);
    mine.give();
    const tp = threadPointer();
    if (tp != 0 and @atomicLoad(usize, &home_owner, .monotonic) == tp) {
        gc_mod.orphan(&home.gc);
        home.give();
        home.in_use = 0;
        home.watched = false;
        @atomicStore(usize, &home_owner, 0, .release);
    }
}

fn classOf(len: usize) usize {
    return (len + granule - 1) / granule - 1;
}

fn small(len: usize, alignment: Alignment) bool {
    return len > 0 and len <= max_small and alignment.toByteUnits() <= granule;
}

fn alloc(_: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
    const l = lists();
    l.in_use += 1;
    if (!small(len, alignment)) return std.heap.c_allocator.rawAlloc(len, alignment, ret_addr);
    const c = classOf(len);
    if (l.free[c]) |f| {
        l.free[c] = f.next;
        return @ptrCast(f);
    }
    const size = (c + 1) * granule;
    if (l.carve[c].len < size) return carved(c, ret_addr);
    const block = l.carve[c][0..size];
    l.carve[c] = l.carve[c][size..];
    return block.ptr;
}

/// A block of a class with no free ones and the slab carved: the shared
/// pool's, else a new slab's
noinline fn carved(c: usize, ret_addr: usize) ?[*]u8 {
    if (!lists().watched) watch();
    const l = lists();
    // (a gone thread's list, the others left to other threads)
    lockShared();
    const got = shared[c];
    if (got) |b| shared[c] = b.next_batch;
    shared_lock.unlock();
    if (got) |b| {
        l.free[c] = b.first.next;
        return @ptrCast(b);
    }
    const size = (c + 1) * granule;
    const slab = std.heap.c_allocator.rawAlloc(slab_size, .fromByteUnits(granule), ret_addr) orelse return null;
    l.carve[c] = slab[size..slab_size];
    return slab;
}

fn resize(_: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
    if (!small(memory.len, alignment)) {
        if (small(new_len, alignment)) return false;
        return std.heap.c_allocator.rawResize(memory, alignment, new_len, ret_addr);
    }
    // (in place within its class)
    return small(new_len, alignment) and classOf(new_len) == classOf(memory.len);
}

fn remap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
    if (resize(ctx, memory, alignment, new_len, ret_addr)) return memory.ptr;
    if (!small(memory.len, alignment) and !small(new_len, alignment)) return std.heap.c_allocator.rawRemap(memory, alignment, new_len, ret_addr);
    return null;
}

fn free(_: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
    const l = lists();
    l.in_use -= 1;
    if (!small(memory.len, alignment)) return std.heap.c_allocator.rawFree(memory, alignment, ret_addr);
    // (a thread that only frees (another's results) gives them on too:
    // push())
    push(l, classOf(memory.len), @ptrCast(@alignCast(memory.ptr)));
}

/// This thread's lists (the cycle collector's state with them: its
/// containers made and freed with one look-up)
pub const current = lists;

/// A block of `len` bytes (16-byte aligned) from `l` (this thread's),
/// inline (the cycle collector's containers)
pub inline fn allocIn(l: *Lists, len: usize) ?[*]u8 {
    l.in_use += 1;
    if (len > max_small) return std.heap.c_allocator.rawAlloc(len, .@"16", 0);
    const c = classOf(len);
    if (l.free[c]) |f| {
        l.free[c] = f.next;
        return @ptrCast(f);
    }
    const size = (c + 1) * granule;
    if (l.carve[c].len < size) return carved(c, 0);
    const block = l.carve[c][0..size];
    l.carve[c] = l.carve[c][size..];
    return block.ptr;
}

pub inline fn freeIn(l: *Lists, p: [*]u8, len: usize) void {
    l.in_use -= 1;
    if (len > max_small) return std.heap.c_allocator.rawFree(p[0..len], .@"16", 0);
    push(l, classOf(len), @ptrCast(@alignCast(p)));
}

/// A free block on a thread's list
inline fn push(l: *Lists, c: usize, f: *Free) void {
    f.next = l.free[c];
    l.free[c] = f;
    l.frees +%= 1;
    if (l.frees >= trim_every or !l.watched) freed(l);
}

/// Every so many blocks freed: the lists trimmed (a thread freeing blocks
/// it doesn't make, as many as another thread makes: Python freeing
/// map()'s results, made by its workers)
const trim_every = 16384;
/// The bytes of free blocks of a class a thread keeps
const kept_bytes = 64 * 1024;

noinline fn freed(l: *Lists) void {
    if (!l.watched) watch();
    if (l.frees < trim_every) return;
    l.frees = 0;
    for (0..classes) |c| {
        // (the first ones kept; the rest a batch of the shared pool's)
        var at = l.free[c] orelse continue;
        var n: usize = 1;
        const keep = kept_bytes / ((c + 1) * granule);
        while (n < keep) : (n += 1) at = at.next orelse break;
        const rest = at.next orelse continue;
        at.next = null;
        const b: *Batch = @ptrCast(rest);
        lockShared();
        b.next_batch = shared[c];
        shared[c] = b;
        shared_lock.unlock();
    }
}

const vtable = std.mem.Allocator.VTable{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

pub const allocator = std.mem.Allocator{ .ptr = undefined, .vtable = &vtable };
