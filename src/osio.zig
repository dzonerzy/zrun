//! The operating system's environment, working directory and files, as
//! Python's os and open() see them: POSIX's (libc's, statx on Linux) and
//! Windows' (its wide-character calls, names in UTF-8 here). An error is
//! the errno Python's OSError would carry (Windows': the one CPython maps
//! its error to).

const std = @import("std");
const builtin = @import("builtin");
const windows = builtin.os.tag == .windows;
const linux = builtin.os.tag == .linux;

pub const Error = error{ Os, OutOfMemory };

/// The errno of the last call that failed with error.Os (this thread's)
pub threadlocal var last_errno: i32 = 0;

fn failed(e: i32) error{Os} {
    last_errno = e;
    return error.Os;
}

extern "c" fn strerror(e: c_int) ?[*:0]const u8;

/// An errno's words, as Python's OSError gives them (the C library's)
pub fn errorWords(e: i32) []const u8 {
    const s = strerror(e) orelse return "Unknown error";
    return std.mem.span(s);
}

const win = struct {
    const HANDLE = ?*anyopaque;
    const INVALID: HANDLE = @ptrFromInt(std.math.maxInt(usize));
    extern "kernel32" fn GetEnvironmentVariableW(name: [*:0]const u16, buf: ?[*]u16, n: u32) callconv(.winapi) u32;
    extern "kernel32" fn GetCurrentDirectoryW(n: u32, buf: ?[*]u16) callconv(.winapi) u32;
    extern "kernel32" fn GetFileAttributesW(name: [*:0]const u16) callconv(.winapi) u32;
    extern "kernel32" fn GetFullPathNameW(name: [*:0]const u16, n: u32, buf: ?[*]u16, part: ?*?[*]u16) callconv(.winapi) u32;
    extern "kernel32" fn GetFinalPathNameByHandleW(h: HANDLE, buf: ?[*]u16, n: u32, flags: u32) callconv(.winapi) u32;
    extern "kernel32" fn CreateFileW(name: [*:0]const u16, access: u32, share: u32, sa: ?*anyopaque, disposition: u32, flags: u32, template: HANDLE) callconv(.winapi) HANDLE;
    extern "kernel32" fn ReadFile(h: HANDLE, buf: [*]u8, n: u32, read: ?*u32, overlapped: ?*anyopaque) callconv(.winapi) i32;
    extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) i32;
    extern "kernel32" fn GetStdHandle(which: u32) callconv(.winapi) HANDLE;
    extern "kernel32" fn GetLastError() callconv(.winapi) u32;

    const INVALID_ATTRIBUTES: u32 = 0xFFFFFFFF;
    const DIRECTORY: u32 = 0x10;
    const GENERIC_READ: u32 = 0x80000000;
    const SHARE_ALL: u32 = 0x7;
    const OPEN_EXISTING: u32 = 3;
    const BACKUP_SEMANTICS: u32 = 0x02000000;
    const ERROR_BROKEN_PIPE: u32 = 109;

    /// CPython's errno of a Windows error (winerror_to_errno's, what it
    /// gives the errors files' calls make)
    fn errnoOf(e: u32) i32 {
        return switch (e) {
            2, 3, 15, 53, 67, 123, 161, 206 => 2, // ENOENT
            5, 32, 33, 1920 => 13, // EACCES
            267 => 20, // ENOTDIR
            8, 14 => 12, // ENOMEM
            80, 183 => 17, // EEXIST
            else => 22, // EINVAL
        };
    }

    fn wide(a: std.mem.Allocator, s: []const u8) Error![:0]u16 {
        return std.unicode.wtf8ToWtf16LeAllocZ(a, s) catch |e| switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            // (not a name Windows has: as no file of it)
            error.InvalidWtf8 => failed(2),
        };
    }

    /// What a call filling a buffer of u16s gives (its length; more than
    /// the buffer: the size it needs), in UTF-8
    fn filled(a: std.mem.Allocator, comptime f: anytype, args: anytype) Error![]u8 {
        var n: u32 = 260;
        while (true) {
            const buf = try a.alloc(u16, n);
            defer a.free(buf);
            const got = @call(.auto, f, args ++ .{ buf.ptr, n });
            if (got == 0) return failed(errnoOf(GetLastError()));
            if (got < n) return std.unicode.wtf16LeToWtf8Alloc(a, buf[0..got]);
            n = got + 1;
        }
    }
};

/// An environment variable's value (null: it isn't set)
pub fn getenv(a: std.mem.Allocator, name: []const u8) Error!?[]u8 {
    if (windows) {
        const w = win.wide(a, name) catch |e| switch (e) {
            error.Os => return null,
            else => return e,
        };
        defer a.free(w);
        const n = win.GetEnvironmentVariableW(w.ptr, null, 0);
        if (n == 0) return null;
        const buf = try a.alloc(u16, n);
        defer a.free(buf);
        const got = win.GetEnvironmentVariableW(w.ptr, buf.ptr, n);
        if (got == 0 or got >= n) return null;
        return try std.unicode.wtf16LeToWtf8Alloc(a, buf[0..got]);
    }
    const z = try a.dupeZ(u8, name);
    defer a.free(z);
    const v = std.c.getenv(z.ptr) orelse return null;
    return try a.dupe(u8, std.mem.span(v));
}

/// The working directory
pub fn getcwd(a: std.mem.Allocator) Error![]u8 {
    if (windows) return win.filled(a, gcd, .{});
    var n: usize = 256;
    while (true) {
        const buf = try a.alloc(u8, n);
        if (std.c.getcwd(buf.ptr, n)) |_| {
            const len = std.mem.indexOfScalar(u8, buf, 0) orelse n;
            defer a.free(buf);
            return a.dupe(u8, buf[0..len]);
        }
        a.free(buf);
        const e: i32 = @intFromEnum(std.c.errno(-1));
        if (e != @intFromEnum(std.c.E.RANGE)) return failed(e);
        n *= 2;
    }
}

fn gcd(buf: [*]u16, n: u32) u32 {
    return win.GetCurrentDirectoryW(n, buf);
}

pub const Kind = enum { none, file, dir, other };

/// What's at the path, symbolic links followed (none: nothing, or it
/// can't be known: as os.path.exists() takes it)
pub fn kindOf(a: std.mem.Allocator, path: []const u8) Error!Kind {
    if (std.mem.indexOfScalar(u8, path, 0) != null) return .none;
    if (windows) {
        const w = win.wide(a, path) catch |e| switch (e) {
            error.Os => return .none,
            else => return e,
        };
        defer a.free(w);
        const attrs = win.GetFileAttributesW(w.ptr);
        if (attrs == win.INVALID_ATTRIBUTES) return .none;
        return if (attrs & win.DIRECTORY != 0) .dir else .file;
    }
    const z = try a.dupeZ(u8, path);
    defer a.free(z);
    var mode: u32 = undefined;
    if (linux) {
        const l = std.os.linux;
        var st: l.Statx = undefined;
        if (l.errno(l.statx(l.AT.FDCWD, z.ptr, 0, .{ .TYPE = true }, &st)) != .SUCCESS) return .none;
        mode = st.mode;
    } else {
        var st: std.c.Stat = undefined;
        if (std.c.stat(z.ptr, &st) != 0) return .none;
        mode = @intCast(st.mode);
    }
    const S = std.c.S;
    return if (S.ISREG(mode)) .file else if (S.ISDIR(mode)) .dir else .other;
}

/// The path absolute, its symbolic links resolved (os.path.realpath's:
/// what of it doesn't exist kept as it's written)
pub fn realpath(a: std.mem.Allocator, path: []const u8) Error![]u8 {
    if (windows) return winRealpath(a, path);
    const abs = if (path.len > 0 and path[0] == '/') try a.dupe(u8, path) else blk: {
        const cwd = try getcwd(a);
        defer a.free(cwd);
        break :blk try std.fs.path.join(a, &.{ cwd, path });
    };
    defer a.free(abs);
    // (the longest part that exists resolved, the rest after it)
    var end = abs.len;
    while (true) {
        const head = abs[0..end];
        if (try resolved(a, if (head.len == 0) "/" else head)) |r| {
            if (end == abs.len) return r;
            defer a.free(r);
            const joined = try std.fs.path.join(a, &.{ r, abs[end + 1 ..] });
            defer a.free(joined);
            return normalized(a, joined);
        }
        if (end == 0) return normalized(a, abs);
        end = std.mem.lastIndexOfScalar(u8, head, '/') orelse 0;
    }
}

fn resolved(a: std.mem.Allocator, path: []const u8) Error!?[]u8 {
    const z = try a.dupeZ(u8, path);
    defer a.free(z);
    var buf: [std.c.PATH_MAX]u8 = undefined;
    const r = std.c.realpath(z.ptr, &buf) orelse return null;
    return try a.dupe(u8, std.mem.span(r));
}

/// `.` and `..` taken out, as posixpath.normpath does
fn normalized(a: std.mem.Allocator, path: []const u8) Error![]u8 {
    var parts: std.ArrayListUnmanaged([]const u8) = .empty;
    defer parts.deinit(a);
    var it = std.mem.tokenizeScalar(u8, path, '/');
    while (it.next()) |p| {
        if (std.mem.eql(u8, p, ".")) continue;
        if (std.mem.eql(u8, p, "..")) {
            _ = parts.pop();
            continue;
        }
        try parts.append(a, p);
    }
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(a);
    for (parts.items) |p| {
        try out.append(a, '/');
        try out.appendSlice(a, p);
    }
    if (out.items.len == 0) try out.append(a, '/');
    return out.toOwnedSlice(a);
}

fn winRealpath(a: std.mem.Allocator, path: []const u8) Error![]u8 {
    const w = try win.wide(a, path);
    defer a.free(w);
    const h = win.CreateFileW(w.ptr, 0, win.SHARE_ALL, null, win.OPEN_EXISTING, win.BACKUP_SEMANTICS, null);
    if (h != win.INVALID) {
        defer _ = win.CloseHandle(h);
        if (win.filled(a, finalPath, .{h})) |r| {
            // (\\?\C:\x: C:\x; \\?\UNC\host\x: \\host\x)
            if (std.mem.startsWith(u8, r, "\\\\?\\UNC\\")) {
                defer a.free(r);
                return std.mem.concat(a, u8, &.{ "\\\\", r[8..] });
            }
            if (std.mem.startsWith(u8, r, "\\\\?\\")) {
                defer a.free(r);
                return a.dupe(u8, r[4..]);
            }
            return r;
        } else |e| if (e == error.OutOfMemory) return e;
    }
    return win.filled(a, fullPath, .{w.ptr});
}

fn finalPath(h: win.HANDLE, buf: [*]u16, n: u32) u32 {
    return win.GetFinalPathNameByHandleW(h, buf, n, 0);
}

fn fullPath(name: [*:0]const u16, buf: [*]u16, n: u32) u32 {
    return win.GetFullPathNameW(name, n, buf, null);
}

/// A file's bytes, all of it
pub fn readFile(a: std.mem.Allocator, path: []const u8) Error![]u8 {
    if (std.mem.indexOfScalar(u8, path, 0) != null) return failed(22);
    if (windows) {
        const w = try win.wide(a, path);
        defer a.free(w);
        // (a directory: what CPython's open() says of it there)
        const attrs = win.GetFileAttributesW(w.ptr);
        if (attrs != win.INVALID_ATTRIBUTES and attrs & win.DIRECTORY != 0) return failed(13);
        const h = win.CreateFileW(w.ptr, win.GENERIC_READ, win.SHARE_ALL, null, win.OPEN_EXISTING, 0, null);
        if (h == win.INVALID) return failed(win.errnoOf(win.GetLastError()));
        defer _ = win.CloseHandle(h);
        return readAllWin(a, h);
    }
    const z = try a.dupeZ(u8, path);
    defer a.free(z);
    const fd = std.c.open(z.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true });
    if (fd < 0) return failed(@intFromEnum(std.c.errno(-1)));
    defer _ = std.c.close(fd);
    // (a directory opens; reading it is the error)
    return readAllFd(a, fd);
}

/// The standard input's bytes, all that's left of it
pub fn readStdin(a: std.mem.Allocator) Error![]u8 {
    if (windows) return readAllWin(a, win.GetStdHandle(@bitCast(@as(i32, -10))));
    return readAllFd(a, 0);
}

fn readAllFd(a: std.mem.Allocator, fd: std.c.fd_t) Error![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(a);
    while (true) {
        try out.ensureUnusedCapacity(a, 65536);
        const room = out.allocatedSlice()[out.items.len..];
        const n = std.c.read(fd, room.ptr, room.len);
        if (n < 0) {
            const e = std.c.errno(-1);
            if (e == .INTR) continue;
            return failed(@intFromEnum(e));
        }
        if (n == 0) break;
        out.items.len += @intCast(n);
    }
    return out.toOwnedSlice(a);
}

fn readAllWin(a: std.mem.Allocator, h: win.HANDLE) Error![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(a);
    while (true) {
        try out.ensureUnusedCapacity(a, 65536);
        const room = out.allocatedSlice()[out.items.len..];
        var n: u32 = 0;
        if (win.ReadFile(h, room.ptr, @intCast(@min(room.len, std.math.maxInt(u32))), &n, null) == 0) {
            const e = win.GetLastError();
            // (a pipe whose writer is gone: its end)
            if (e == win.ERROR_BROKEN_PIPE) break;
            return failed(win.errnoOf(e));
        }
        if (n == 0) break;
        out.items.len += n;
    }
    return out.toOwnedSlice(a);
}
