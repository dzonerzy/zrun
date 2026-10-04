//! The allocator of the values compiled code makes (strs, lists, tuples,
//! dicts, records, frames...): small blocks, by size class, kept on free
//! lists when freed and handed out again, carved from slabs (malloc and
//! free for each would be most of the time a program runs). Bigger blocks
//! (or more aligned ones) are libc's.
//!
//! Not thread-safe: compiled code runs with the GIL held.

const std = @import("std");

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

var free_lists: [classes]?*Free = .{null} ** classes;
/// The slab being carved, per class, and how much of it is left
var carve: [classes][]u8 = .{&.{}} ** classes;

fn classOf(len: usize) usize {
    return (len + granule - 1) / granule - 1;
}

fn small(len: usize, alignment: Alignment) bool {
    return len > 0 and len <= max_small and alignment.toByteUnits() <= granule;
}

fn alloc(_: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
    if (!small(len, alignment)) return std.heap.c_allocator.rawAlloc(len, alignment, ret_addr);
    const c = classOf(len);
    if (free_lists[c]) |f| {
        free_lists[c] = f.next;
        return @ptrCast(f);
    }
    const size = (c + 1) * granule;
    if (carve[c].len < size) {
        const slab = std.heap.c_allocator.rawAlloc(slab_size, .fromByteUnits(granule), ret_addr) orelse return null;
        carve[c] = slab[0..slab_size];
    }
    const block = carve[c][0..size];
    carve[c] = carve[c][size..];
    return block.ptr;
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
    if (!small(memory.len, alignment)) return std.heap.c_allocator.rawFree(memory, alignment, ret_addr);
    const c = classOf(memory.len);
    const f: *Free = @ptrCast(@alignCast(memory.ptr));
    f.next = free_lists[c];
    free_lists[c] = f;
}

const vtable = std.mem.Allocator.VTable{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

pub const allocator = std.mem.Allocator{ .ptr = undefined, .vtable = &vtable };
