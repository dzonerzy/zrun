//! The launcher of an executable zrun.build_executable() makes: this
//! program, then the payload (the files of a Python runtime, zrun and its
//! packages, the language's module, the program and its compiled code),
//! then a footer saying where the payload is.
//!
//! Run, it unpacks the payload once into the platform's cache directory
//! (zrun/exe/<the payload's hash>: written aside, then renamed into place,
//! whole or not at all), and runs the program there: the Python of the
//! payload on its main.py, with this program's arguments. Its exit code is
//! the program's.
//!
//! The payload: entries, one after the other, of a path (u32 length, then
//! the bytes, `/` between names), a mode (u8: 0 a file, 1 an executable
//! one), the compressed size and the size (u64 each), then the file's bytes
//! compressed (zlib); an empty path ends it. The footer: "ZRUNEXE1", the
//! payload's offset and size (u64), its SHA-256 (32 bytes). Numbers are
//! little-endian.

const std = @import("std");
const builtin = @import("builtin");

const magic = "ZRUNEXE1";
const footer_size = magic.len + 8 + 8 + 32;
const windows = builtin.os.tag == .windows;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const a = init.arena.allocator();
    const self_path = try std.process.executablePathAlloc(io, a);

    var file = std.Io.Dir.cwd().openFile(io, self_path, .{}) catch |e| fail("can't open {s}: {t}", .{ self_path, e });
    defer file.close(io);
    const size = (try file.stat(io)).size;
    if (size < footer_size) fail("{s} has no program in it", .{self_path});
    var footer: [footer_size]u8 = undefined;
    _ = try file.readPositionalAll(io, &footer, size - footer_size);
    if (!std.mem.eql(u8, footer[0..magic.len], magic)) fail("{s} has no program in it", .{self_path});
    const offset = std.mem.readInt(u64, footer[magic.len..][0..8], .little);
    const length = std.mem.readInt(u64, footer[magic.len + 8 ..][0..8], .little);
    const hash = footer[magic.len + 16 ..][0..32];
    const id = std.fmt.bytesToHex(hash[0..12], .lower);

    const base = try cacheDir(a, init.environ_map);
    const dir = try std.fs.path.join(a, &.{ base, &id });
    const python = try std.fs.path.join(a, if (windows) &.{ dir, "python", "python.exe" } else &.{ dir, "python", "bin", "python3" });
    const main_py = try std.fs.path.join(a, &.{ dir, "main.py" });

    // Unpacked already (by an earlier run): its `ok` is there
    const ok = try std.fs.path.join(a, &.{ dir, "ok" });
    if (std.Io.Dir.cwd().access(io, ok, .{})) |_| {} else |_| {
        const payload = try a.alloc(u8, @intCast(length));
        _ = try file.readPositionalAll(io, payload, offset);
        try unpack(io, a, base, dir, payload);
    }

    // The program's arguments after the Python's own
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(a, &.{ python, "-I", main_py });
    var args = try init.minimal.args.iterateAllocator(a);
    _ = args.next();
    while (args.next()) |arg| try argv.append(a, arg);

    if (!windows) {
        const err = std.process.replace(io, .{ .argv = argv.items });
        fail("can't run {s}: {t}", .{ python, err });
    }
    var child = std.process.spawn(io, .{ .argv = argv.items }) catch |e| fail("can't run {s}: {t}", .{ python, e });
    const term = try child.wait(io);
    std.process.exit(switch (term) {
        .exited => |code| code,
        else => 1,
    });
}

/// zrun/exe in the platform's cache directory (as zrun's own cache: the
/// user's local application data on Windows, XDG's or ~/.cache elsewhere).
fn cacheDir(a: std.mem.Allocator, env: *std.process.Environ.Map) ![]const u8 {
    if (windows) {
        if (env.get("LOCALAPPDATA")) |d| return std.fs.path.join(a, &.{ d, "zrun", "exe" });
        if (env.get("USERPROFILE")) |d| return std.fs.path.join(a, &.{ d, "AppData", "Local", "zrun", "exe" });
        return std.fs.path.join(a, &.{ ".", "zrun-exe" });
    }
    if (env.get("XDG_CACHE_HOME")) |d| {
        if (std.fs.path.isAbsolute(d)) return std.fs.path.join(a, &.{ d, "zrun", "exe" });
    }
    if (env.get("HOME")) |d| return std.fs.path.join(a, &.{ d, ".cache", "zrun", "exe" });
    return std.fs.path.join(a, &.{ "/tmp", "zrun-exe" });
}

/// The payload's files written into a directory of their own beside `dir`,
/// then that renamed to `dir` (another run doing the same at the same time:
/// whichever renames first wins, the other's copy is deleted).
fn unpack(io: std.Io, a: std.mem.Allocator, base: []const u8, dir: []const u8, payload: []const u8) !void {
    const here = std.Io.Dir.cwd();
    try here.createDirPath(io, base);
    // (no libc: Linux's system call, Windows' own)
    const pid: u64 = if (windows) std.os.windows.GetCurrentProcessId() else @intCast(std.os.linux.getpid());
    const tmp = try std.fmt.allocPrint(a, "{s}.{d}.tmp", .{ dir, pid });
    here.deleteTree(io, tmp) catch {};
    try here.createDirPath(io, tmp);
    var out = try here.openDir(io, tmp, .{});
    defer out.close(io);

    var at: usize = 0;
    const buffer = try a.alloc(u8, std.compress.flate.max_window_len);
    while (true) {
        const path_len = std.mem.readInt(u32, payload[at..][0..4], .little);
        at += 4;
        if (path_len == 0) break;
        const path = payload[at..][0..path_len];
        at += path_len;
        const mode = payload[at];
        at += 1;
        const packed_len: usize = @intCast(std.mem.readInt(u64, payload[at..][0..8], .little));
        const raw_len: usize = @intCast(std.mem.readInt(u64, payload[at + 8 ..][0..8], .little));
        at += 16;
        const compressed = payload[at..][0..packed_len];
        at += packed_len;

        const name = if (windows) try std.mem.replaceOwned(u8, a, path, "/", "\\") else path;
        if (std.fs.path.dirname(name)) |parent| try out.createDirPath(io, parent);
        var f = try out.createFile(io, name, .{ .permissions = if (!windows and mode == 1) .fromMode(0o755) else .default_file });
        defer f.close(io);
        var in: std.Io.Reader = .fixed(compressed);
        var inflate: std.compress.flate.Decompress = .init(&in, .zlib, buffer);
        var write_buf: [64 * 1024]u8 = undefined;
        var w = f.writer(io, &write_buf);
        const n = try inflate.reader.streamRemaining(&w.interface);
        if (n != raw_len) fail("the program's files are damaged ({s})", .{path});
        try w.interface.flush();
    }
    // (the mark it's whole)
    try out.writeFile(io, .{ .sub_path = "ok", .data = "" });
    here.rename(tmp, here, dir, io) catch {
        // (another run's copy is there: this one isn't needed)
        here.deleteTree(io, tmp) catch {};
        if (here.access(io, try std.fs.path.join(a, &.{ dir, "ok" }), .{})) |_| return else |_| {}
        fail("can't unpack the program into {s}", .{dir});
    };
}

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print(fmt ++ "\n", args);
    std.process.exit(1);
}
