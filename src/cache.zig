//! Compiled code kept between runs of Python: each module's object file
//! (LLVM's work, nearly all of compiling), on disk by what it compiles.
//!
//! The key is the module's IR text (and what the object depends on besides:
//! LLVM's version, the host's CPU and features, zrun's cache format). The
//! IR refers to this process's objects only by name (ir.Module.ptrConst:
//! `<prefix>_k<n>`, defined when the code is linked) and to Python objects
//! by index: an object file made for the same text in another process is
//! the code compiling this one would make. zrun still generates the IR
//! each time (a small part of the time), the key needs it.
//!
//! Where: $XDG_CACHE_HOME/zrun, else ~/.cache/zrun; or where
//! zrun.configure(cache=path) says (cache=False: none, everything compiled
//! each time). Objects a compiled module loaded has (aot.zig) are looked at
//! first, by the same keys.

const std = @import("std");
const jit = @import("jit.zig");
const L = jit.f;

/// Bump when what the objects mean changes without the IR text showing it
const format = "zrun-cache-1";

/// Where the cache is (zrun.configure(cache=...)): the usual place
/// ($XDG_CACHE_HOME/zrun, else ~/.cache/zrun), none, or a directory given
pub const Setting = union(enum) { default, off, dir: []const u8 };

var setting: Setting = .default;
var resolved = false;
var dir_path: ?[]const u8 = null;

/// The cache from now on (the directory given: the caller's, copied).
pub fn set(s: Setting) !void {
    const a = std.heap.c_allocator;
    if (setting == .dir) a.free(setting.dir);
    setting = switch (s) {
        .dir => |d| .{ .dir = try a.dupe(u8, d) },
        else => s,
    };
    resolved = false;
    if (dir_path) |p| a.free(p);
    dir_path = null;
}

/// The cache's directory (made if needed), or null: no cache.
fn dir() ?[]const u8 {
    if (resolved) return dir_path;
    resolved = true;
    const a = std.heap.c_allocator;
    const path = switch (setting) {
        .off => return null,
        .dir => |d| a.dupe(u8, d) catch return null,
        .default => if (std.c.getenv("XDG_CACHE_HOME")) |d|
            std.fmt.allocPrint(a, "{s}/zrun", .{std.mem.span(d)}) catch return null
        else if (std.c.getenv("HOME")) |h|
            std.fmt.allocPrint(a, "{s}/.cache/zrun", .{std.mem.span(h)}) catch return null
        else
            return null,
    };
    if (!makePath(path)) {
        a.free(path);
        return null;
    }
    dir_path = path;
    return path;
}

/// What the objects depend on besides their IR: zrun's format, LLVM's
/// version, the host's CPU and features (the same for compiled modules:
/// one made on another CPU isn't this one's)
pub fn salt() *const [32]u8 {
    const S = struct {
        var value: [32]u8 = undefined;
        var made = false;
    };
    if (!S.made) {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(format);
        h.update(jit.LLVM_VERSION);
        const cpu = L("LLVMGetHostCPUName")();
        defer L("LLVMDisposeMessage")(cpu);
        const features = L("LLVMGetHostCPUFeatures")();
        defer L("LLVMDisposeMessage")(features);
        h.update(std.mem.span(cpu));
        h.update(std.mem.span(features));
        h.final(&S.value);
        S.made = true;
    }
    return &S.value;
}

/// A module's key: what its object is the code of (its IR text, salted)
pub const Key = [32]u8;

pub fn keyOf(ir_text: []const u8) Key {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(salt());
    h.update(ir_text);
    var digest: Key = undefined;
    h.final(&digest);
    return digest;
}

/// Where a module's object is kept, or null: no cache.
pub fn pathFor(a: std.mem.Allocator, key: Key) ?[:0]u8 {
    const d = dir() orelse return null;
    return std.fmt.allocPrintSentinel(a, "{s}/{s}.o", .{ d, std.fmt.bytesToHex(key, .lower) }, 0) catch null;
}

/// Objects given by compiled modules loaded (zrun's module files: aot.zig),
/// by key: looked at before the cache
var given: std.AutoHashMapUnmanaged(Key, []const u8) = .empty;
var given_lock: std.atomic.Mutex = .unlocked;

/// An object a compiled module has (the bytes: its, owned from now).
pub fn give(key: Key, bytes: []const u8) void {
    while (!given_lock.tryLock()) std.atomic.spinLoopHint();
    defer given_lock.unlock();
    const slot = given.getOrPut(std.heap.c_allocator, key) catch {
        std.heap.c_allocator.free(bytes);
        return;
    };
    if (slot.found_existing) {
        std.heap.c_allocator.free(bytes);
        return;
    }
    slot.value_ptr.* = bytes;
}

/// The object of a key given by a compiled module (borrowed: kept for
/// the process), or null.
pub fn givenObject(key: Key) ?[]const u8 {
    while (!given_lock.tryLock()) std.atomic.spinLoopHint();
    defer given_lock.unlock();
    return given.get(key);
}

/// The object of a key, from what's at hand: given, or the cache (owned
/// by the caller), or null.
pub fn objectOf(a: std.mem.Allocator, key: Key) ?[]u8 {
    if (givenObject(key)) |b| return a.dupe(u8, b) catch null;
    const path = pathFor(a, key) orelse return null;
    defer a.free(path);
    return read(a, path);
}

/// The object kept at `path` (owned by the caller), or null.
pub fn read(a: std.mem.Allocator, path: [:0]const u8) ?[]u8 {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    var buf: [65536]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, &buf, buf.len);
        if (n < 0) {
            out.deinit(a);
            return null;
        }
        if (n == 0) break;
        out.appendSlice(a, buf[0..@intCast(n)]) catch {
            out.deinit(a);
            return null;
        };
    }
    return out.toOwnedSlice(a) catch null;
}

/// Keep an object at `path` (whole or not at all: written aside, then
/// renamed). Failing is fine: compiled again next time.
pub fn write(path: [:0]const u8, bytes: []const u8) void {
    var buf: [4096]u8 = undefined;
    const tmp = std.fmt.bufPrintZ(&buf, "{s}.{d}.tmp", .{ path, std.c.getpid() }) catch return;
    const fd = std.c.open(tmp.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (n <= 0) break;
        done += @intCast(n);
    }
    _ = std.c.close(fd);
    if (done < bytes.len or std.c.rename(tmp.ptr, path.ptr) != 0) _ = std.c.unlink(tmp.ptr);
}

/// mkdir -p (true: there).
fn makePath(path: []const u8) bool {
    var buf: [4096]u8 = undefined;
    if (path.len >= buf.len) return false;
    var i: usize = 1;
    while (i <= path.len) : (i += 1) {
        if (i < path.len and path[i] != '/') continue;
        @memcpy(buf[0..i], path[0..i]);
        buf[i] = 0;
        _ = std.c.mkdir(@ptrCast(&buf), 0o755);
    }
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    // (F_OK)
    return std.c.access(@ptrCast(&buf), 0) == 0;
}
