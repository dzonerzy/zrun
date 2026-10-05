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
//! Where: the platform's place for caches (Windows: %LOCALAPPDATA%\zrun\Cache;
//! macOS: ~/Library/Caches/zrun; else $XDG_CACHE_HOME/zrun or ~/.cache/zrun);
//! or where zrun.configure(cache=path) says (cache=False: none, everything compiled
//! each time). Objects a compiled module loaded has (aot.zig) are looked at
//! first, by the same keys.

const std = @import("std");
const builtin = @import("builtin");
const jit = @import("jit.zig");
const L = jit.f;

/// Bump when what the objects mean changes without the IR text showing it
const format = "zrun-cache-1";

/// Where the cache is (zrun.configure(cache=...)): the platform's usual
/// place (default_dir), none, or a directory given
pub const Setting = union(enum) { default, off, dir: []const u8 };

var setting: Setting = .default;
var resolved = false;
var dir_path: ?[]const u8 = null;
/// The platform's place for caches, zrun's directory in it (owned by the
/// caller): %LOCALAPPDATA%\zrun\Cache on Windows, ~/Library/Caches/zrun on
/// macOS, $XDG_CACHE_HOME/zrun or ~/.cache/zrun elsewhere; null if there's
/// no home to put it in.
fn defaultDir(a: std.mem.Allocator) ?[]u8 {
    switch (builtin.os.tag) {
        .windows => {
            if (env(a, "LOCALAPPDATA")) |base| {
                defer a.free(base);
                return std.fmt.allocPrint(a, "{s}\\zrun\\Cache", .{base}) catch null;
            }
            const home = env(a, "USERPROFILE") orelse return null;
            defer a.free(home);
            return std.fmt.allocPrint(a, "{s}\\AppData\\Local\\zrun\\Cache", .{home}) catch null;
        },
        .macos => {
            const home = env(a, "HOME") orelse return null;
            defer a.free(home);
            return std.fmt.allocPrint(a, "{s}/Library/Caches/zrun", .{home}) catch null;
        },
        else => {
            // (XDG's must be absolute: a relative one is ignored)
            if (env(a, "XDG_CACHE_HOME")) |base| {
                defer a.free(base);
                if (std.fs.path.isAbsolute(base)) return std.fmt.allocPrint(a, "{s}/zrun", .{base}) catch null;
            }
            const home = env(a, "HOME") orelse return null;
            defer a.free(home);
            return std.fmt.allocPrint(a, "{s}/.cache/zrun", .{home}) catch null;
        },
    }
}

/// An environment variable of the process, not empty (owned by the
/// caller), or null.
fn env(a: std.mem.Allocator, key: []const u8) ?[]u8 {
    // (Windows: the process's block, read each time; elsewhere libc's)
    const environ: std.process.Environ = if (builtin.os.tag == .windows)
        .{ .block = .global }
    else
        .{ .block = .{ .slice = @ptrCast(std.mem.span(std.c.environ)) } };
    const v = environ.getAlloc(a, key) catch return null;
    if (v.len == 0) {
        a.free(v);
        return null;
    }
    return v;
}

/// Blocking file calls (the cache's, compiled modules'): the platform's
var threaded: std.Io.Threaded = .init_single_threaded;

pub fn io() std.Io {
    return threaded.io();
}

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
        .default => defaultDir(a) orelse return null,
    };
    std.Io.Dir.cwd().createDirPath(io(), path) catch {
        a.free(path);
        return null;
    };
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

pub fn keyOf(ir_text: []const u8, opt: u32) Key {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(salt());
    h.update(std.mem.asBytes(&opt));
    h.update(ir_text);
    var digest: Key = undefined;
    h.final(&digest);
    return digest;
}

/// Where a module's object is kept, or null: no cache.
pub fn pathFor(a: std.mem.Allocator, key: Key) ?[]u8 {
    const d = dir() orelse return null;
    return std.fmt.allocPrint(a, "{s}" ++ std.fs.path.sep_str ++ "{s}.o", .{ d, std.fmt.bytesToHex(key, .lower) }) catch null;
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

/// The bytes of the file at `path` (owned by the caller), or null.
pub fn read(a: std.mem.Allocator, path: []const u8) ?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io(), path, a, .unlimited) catch null;
}

/// Keep an object at `path` (whole or not at all). Failing is fine:
/// compiled again next time.
pub fn write(path: []const u8, bytes: []const u8) void {
    writeWhole(path, bytes) catch {};
}

/// Write a file whole or not at all: written aside (a name of this
/// process's), then renamed over `path` (replacing it, on every platform).
pub fn writeWhole(path: []const u8, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    const pid: u64 = if (builtin.os.tag == .windows) std.os.windows.GetCurrentProcessId() else @intCast(std.c.getpid());
    const tmp = try std.fmt.bufPrint(&buf, "{s}.{d}.tmp", .{ path, pid });
    const here = std.Io.Dir.cwd();
    try here.writeFile(io(), .{ .sub_path = tmp, .data = bytes });
    here.rename(tmp, here, path, io()) catch |e| {
        here.deleteFile(io(), tmp) catch {};
        return e;
    };
}
