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
const Lists = struct {
    /// The free ones, per class (their heads only: counting them would
    /// cost the hot path)
    free: [classes]?*Free = .{null} ** classes,
    /// The slab being carved, per class, and how much of it is left
    carve: [classes][]u8 = .{&.{}} ** classes,
    /// Blocks handed out and not freed by this thread (the tests' check
    /// that compiled code gives back what it takes: zrun._blocks())
    in_use: isize = 0,

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
/// (its destructor runs as a thread that took blocks ends)
var exit_key: std.c.pthread_key_t = undefined;
var have_key = false;
threadlocal var watched = false;

/// A word naming this thread, read with an instruction (Linux's thread
/// pointer: its TCB's address); 0: not known here (no home lists)
pub inline fn threadPointer() usize {
    if (builtin.os.tag != .linux) return 0;
    return switch (builtin.cpu.arch) {
        .x86_64 => asm ("mov %%fs:0, %[ret]"
            : [ret] "=r" (-> usize),
        ),
        .aarch64 => asm ("mrs %[ret], tpidr_el0"
            : [ret] "=r" (-> usize),
        ),
        else => 0,
    };
}

/// This thread's lists
inline fn lists() *Lists {
    const tp = threadPointer();
    if (tp != 0 and tp == @atomicLoad(usize, &home_owner, .monotonic)) return &home;
    return mineLists();
}

noinline fn mineLists() *Lists {
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

/// Blocks this thread has and hasn't freed (zrun._blocks())
pub fn inUse() isize {
    return lists().in_use;
}

fn lockShared() void {
    while (!shared_lock.tryLock()) std.atomic.spinLoopHint();
}

/// This thread's blocks given to the shared pool when it ends (and the
/// home lists when no thread has them: this one's from now)
fn watch() void {
    watched = true;
    lockShared();
    defer shared_lock.unlock();
    if (!have_key) {
        if (std.c.pthread_key_create(&exit_key, &threadGone) != .SUCCESS) return;
        have_key = true;
    }
    _ = std.c.pthread_setspecific(exit_key, @ptrFromInt(1));
    const tp = threadPointer();
    if (tp != 0 and home_owner == 0) {
        // (the counts this thread's: carried over)
        home.in_use = mine.in_use;
        mine.in_use = 0;
        @atomicStore(usize, &home_owner, tp, .monotonic);
    }
}

fn threadGone(_: *anyopaque) callconv(.c) void {
    mine.give();
    const tp = threadPointer();
    if (tp != 0 and @atomicLoad(usize, &home_owner, .monotonic) == tp) {
        home.give();
        home.in_use = 0;
        @atomicStore(usize, &home_owner, 0, .monotonic);
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
    if (!watched) watch();
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
    const c = classOf(memory.len);
    const f: *Free = @ptrCast(@alignCast(memory.ptr));
    f.next = l.free[c];
    l.free[c] = f;
}

const vtable = std.mem.Allocator.VTable{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

pub const allocator = std.mem.Allocator{ .ptr = undefined, .vtable = &vtable };
