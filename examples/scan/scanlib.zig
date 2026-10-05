//! scan's native host functions: a library of its own, as a YARA-like
//! engine's pattern matcher would be, called by the compiled rules directly
//! (no Python, no GIL). scan.py builds it (`zig build-lib -dynamic`) and
//! hands its functions to zrun as "zrun.native.v1" capsules.

const Bytes = extern struct { ptr: [*]const u8, len: u64 };
const Arg = extern union { i: i64, f: f64, b: Bytes };

const Native = extern struct {
    abi: u32,
    signature: [*:0]const u8,
    call: *const fn (state: ?*anyopaque, args: [*]const Arg, nargs: u64, result: *Arg) callconv(.c) c_int,
    state: ?*anyopaque,
    @"error": ?*const fn (state: ?*anyopaque, code: c_int) callconv(.c) ?[*:0]const u8,
};

const not_a_byte = 1;

fn errorText(_: ?*anyopaque, code: c_int) callconv(.c) ?[*:0]const u8 {
    return switch (code) {
        not_a_byte => "count(): a byte is 0 to 255",
        else => "failed",
    };
}

fn bytesOf(a: Arg) []const u8 {
    return a.b.ptr[0..a.b.len];
}

/// count(data, byte): how many times the byte occurs
fn count(_: ?*anyopaque, args: [*]const Arg, _: u64, out: *Arg) callconv(.c) c_int {
    const byte = args[1].i;
    if (byte < 0 or byte > 255) return not_a_byte;
    const b: u8 = @intCast(byte);
    var n: u64 = 0;
    for (bytesOf(args[0])) |x| n += @intFromBool(x == b);
    out.* = .{ .i = @intCast(n) };
    return 0;
}

/// entropy(data): Shannon's, in bits per byte (0 to 8; 0 for no data)
fn entropy(_: ?*anyopaque, args: [*]const Arg, _: u64, out: *Arg) callconv(.c) c_int {
    const data = bytesOf(args[0]);
    var counts = [_]u64{0} ** 256;
    for (data) |x| counts[x] += 1;
    var e: f64 = 0;
    if (data.len > 0) {
        const n: f64 = @floatFromInt(data.len);
        for (counts) |c| if (c != 0) {
            const p = @as(f64, @floatFromInt(c)) / n;
            e -= p * @log2(p);
        };
    }
    out.* = .{ .f = e };
    return 0;
}

export const scan_count = Native{ .abi = 1, .signature = "bi:i", .call = &count, .state = null, .@"error" = &errorText };
export const scan_entropy = Native{ .abi = 1, .signature = "b:f", .call = &entropy, .state = null, .@"error" = &errorText };
