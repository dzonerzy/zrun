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
//! each time).

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
var salt: [32]u8 = undefined;

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
    // What the objects depend on besides their IR
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(format);
    h.update(jit.LLVM_VERSION);
    const cpu = L("LLVMGetHostCPUName")();
    defer L("LLVMDisposeMessage")(cpu);
    const features = L("LLVMGetHostCPUFeatures")();
    defer L("LLVMDisposeMessage")(features);
    h.update(std.mem.span(cpu));
    h.update(std.mem.span(features));
    h.final(&salt);
    dir_path = path;
    return path;
}

/// Where a module's object is kept (its IR text's), or null: no cache.
pub fn pathFor(a: std.mem.Allocator, ir_text: []const u8) ?[:0]u8 {
    const d = dir() orelse return null;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(&salt);
    h.update(ir_text);
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.allocPrintSentinel(a, "{s}/{s}.o", .{ d, std.fmt.bytesToHex(digest, .lower) }, 0) catch null;
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
