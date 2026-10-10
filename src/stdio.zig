//! The standard streams by number (1 output, 2 error), as POSIX's write()
//! and isatty() take them: Windows' console handles there.

const std = @import("std");
const windows = @import("builtin").os.tag == .windows;

const win = struct {
    const HANDLE = ?*anyopaque;
    extern "kernel32" fn GetStdHandle(which: u32) callconv(.winapi) HANDLE;
    extern "kernel32" fn WriteFile(h: HANDLE, buf: [*]const u8, n: u32, written: ?*u32, overlapped: ?*anyopaque) callconv(.winapi) i32;
    extern "kernel32" fn GetConsoleMode(h: HANDLE, mode: *u32) callconv(.winapi) i32;

    fn handle(fd: u32) HANDLE {
        // (STD_OUTPUT_HANDLE, STD_ERROR_HANDLE: -11, -12)
        return GetStdHandle(if (fd == 2) @bitCast(@as(i32, -12)) else @bitCast(@as(i32, -11)));
    }
};

/// Some of `bytes` written to the stream: how many, or -1
pub fn write(fd: u32, bytes: []const u8) isize {
    if (windows) {
        var n: u32 = 0;
        const len: u32 = @intCast(@min(bytes.len, std.math.maxInt(u32)));
        if (win.WriteFile(win.handle(fd), bytes.ptr, len, &n, null) == 0) return -1;
        return n;
    }
    return std.c.write(@intCast(fd), bytes.ptr, bytes.len);
}

/// All of `bytes` written (what can be)
pub fn writeAll(fd: u32, bytes: []const u8) void {
    var rest = bytes;
    while (rest.len > 0) {
        const n = write(fd, rest);
        if (n <= 0) return;
        rest = rest[@intCast(n)..];
    }
}

/// Whether the stream is a terminal (its output flushed by line)
pub fn isTerminal(fd: u32) bool {
    if (windows) {
        var mode: u32 = 0;
        return win.GetConsoleMode(win.handle(fd), &mode) != 0;
    }
    return std.c.isatty(@intCast(fd)) != 0;
}
