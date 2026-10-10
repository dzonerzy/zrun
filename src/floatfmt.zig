//! C's printf of a float (`%e %E %f %F %g %G`, a precision, `#`), exactly:
//! the digits of the double's own binary value, rounded half to even, as
//! glibc's printf and CPython's dtoa give them. glibc's printf is used
//! where it's the C library (it's faster); this is what the others get
//! (Windows' C library rounds a few cases differently).

const std = @import("std");
const Big = std.math.big.int.Managed;
const allocator = std.heap.c_allocator;

/// The C library's printf, where it gives these exactly (glibc)
const c_exact = @import("builtin").os.tag == .linux;

extern "c" fn snprintf(buf: [*]u8, n: usize, fmt: [*:0]const u8, ...) c_int;

/// `%[#].{prec}{conv}` of x (not negative: the sign is the caller's), in
/// `buf`; null if it doesn't fit (or out of memory).
pub fn format(buf: []u8, conv: u8, prec: usize, alt: bool, x: f64) ?[]const u8 {
    if (c_exact) {
        var cf: [8]u8 = undefined;
        const cfs = std.fmt.bufPrintZ(&cf, "%{s}.*{c}", .{ if (alt) "#" else "", conv }) catch return null;
        if (prec > std.math.maxInt(c_int)) return null;
        const got = snprintf(buf.ptr, buf.len, cfs.ptr, @as(c_int, @intCast(prec)), x);
        if (got < 0 or @as(usize, @intCast(got)) >= buf.len) return null;
        return buf[0..@intCast(got)];
    }
    return exact(buf, conv, prec, alt, x) catch null;
}

const Error = error{ OutOfMemory, NoSpace };

/// format()'s digits worked out here, whatever the C library
pub fn exact(buf: []u8, conv: u8, prec: usize, alt: bool, x: f64) Error![]const u8 {
    const upper = std.ascii.isUpper(conv);
    if (std.math.isNan(x) or std.math.isInf(x)) {
        const s: []const u8 = if (std.math.isNan(x)) (if (upper) "NAN" else "nan") else (if (upper) "INF" else "inf");
        return put(buf, 0, s);
    }
    var w = Out{ .buf = buf };
    switch (std.ascii.toLower(conv)) {
        'f' => try fixed(&w, x, prec, alt),
        'e' => try sci(&w, x, prec, alt, upper),
        'g' => {
            const p = @max(prec, 1);
            // (the exponent %e would have, rounded to p digits)
            const k: i64 = if (x == 0) 0 else (try significant(x, p)).k;
            if (k >= -4 and k < p) {
                try fixed(&w, x, @intCast(@as(i64, @intCast(p)) - 1 - k), alt);
            } else try sci(&w, x, p - 1, alt, upper);
            // (trailing zeros of the fraction dropped, and a bare point,
            // but with `#`)
            if (!alt) {
                const s = w.slice();
                if (std.mem.indexOfScalar(u8, s, '.')) |dot| {
                    const end = std.mem.indexOfAny(u8, s, "eE") orelse s.len;
                    var cut = end;
                    while (cut > dot + 1 and s[cut - 1] == '0') cut -= 1;
                    if (cut == dot + 1) cut = dot;
                    const tail_len = s.len - end;
                    std.mem.copyForwards(u8, buf[cut..][0..tail_len], s[end..]);
                    w.len = cut + tail_len;
                }
            }
        },
        else => return error.NoSpace,
    }
    return w.slice();
}

const Out = struct {
    buf: []u8,
    len: usize = 0,

    fn add(self: *Out, s: []const u8) Error!void {
        if (self.len + s.len > self.buf.len) return error.NoSpace;
        @memcpy(self.buf[self.len..][0..s.len], s);
        self.len += s.len;
    }

    fn zeros(self: *Out, n: usize) Error!void {
        if (self.len + n > self.buf.len) return error.NoSpace;
        @memset(self.buf[self.len..][0..n], '0');
        self.len += n;
    }

    fn slice(self: *const Out) []u8 {
        return self.buf[0..self.len];
    }
};

fn put(buf: []u8, at: usize, s: []const u8) Error![]const u8 {
    if (at + s.len > buf.len) return error.NoSpace;
    @memcpy(buf[at..][0..s.len], s);
    return buf[0 .. at + s.len];
}

/// An int's decimal digits (owned)
fn decimal(q: Big) Error![]u8 {
    return q.toString(allocator, 10, .lower) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidBase => unreachable,
    };
}

/// %f: x rounded to `prec` decimals
fn fixed(w: *Out, x: f64, prec: usize, alt: bool) Error!void {
    var q = try scaled(x, @intCast(prec));
    defer q.deinit();
    const digits = try decimal(q);
    defer allocator.free(digits);
    // (at least one digit before the point)
    if (digits.len <= prec) {
        try w.add("0");
        if (prec > 0 or alt) try w.add(".");
        try w.zeros(prec - digits.len);
        try w.add(digits);
        return;
    }
    try w.add(digits[0 .. digits.len - prec]);
    if (prec > 0 or alt) try w.add(".");
    try w.add(digits[digits.len - prec ..]);
}

/// %e: x's first prec + 1 significant digits, its exponent
fn sci(w: *Out, x: f64, prec: usize, alt: bool, upper: bool) Error!void {
    var digits_buf: [1]u8 = .{'0'};
    var k: i64 = 0;
    var owned: ?[]u8 = null;
    defer if (owned) |o| allocator.free(o);
    var digits: []const u8 = &digits_buf;
    if (x != 0) {
        var s = try significant(x, prec + 1);
        defer s.q.deinit();
        owned = try decimal(s.q);
        digits = owned.?;
        k = s.k;
    }
    try w.add(digits[0..1]);
    if (prec > 0 or alt) try w.add(".");
    if (x == 0) try w.zeros(prec) else try w.add(digits[1..]);
    try w.add(if (upper) "E" else "e");
    try w.add(if (k < 0) "-" else "+");
    var eb: [8]u8 = undefined;
    const e = std.fmt.bufPrint(&eb, "{d:0>2}", .{@abs(k)}) catch unreachable;
    try w.add(e);
}

/// x's first n significant digits (an int of n digits) and its decimal
/// exponent: x ~ q * 10^(k - n + 1)
fn significant(x: f64, n: usize) Error!struct { q: Big, k: i64 } {
    const ni: i64 = @intCast(n);
    var k: i64 = @intFromFloat(@floor(std.math.log10(x)));
    var lo = try Big.init(allocator);
    defer lo.deinit();
    var hi = try Big.init(allocator);
    defer hi.deinit();
    var ten = try Big.initSet(allocator, 10);
    defer ten.deinit();
    try lo.pow(&ten, @intCast(n - 1));
    try hi.pow(&ten, @intCast(n));
    // (the estimate off by one either way; rounding up to 10^n a digit more)
    while (true) {
        var q = try scaled(x, ni - 1 - k);
        if (q.order(hi) != .lt) {
            q.deinit();
            k += 1;
        } else if (q.order(lo) == .lt) {
            q.deinit();
            k -= 1;
        } else return .{ .q = q, .k = k };
    }
}

/// x * 10^s, rounded half to even to an int
fn scaled(x: f64, s: i64) Error!Big {
    const bits: u64 = @bitCast(x);
    const biased: i64 = @intCast((bits >> 52) & 0x7ff);
    const frac = bits & ((@as(u64, 1) << 52) - 1);
    const m: u64 = if (biased == 0) frac else frac | (@as(u64, 1) << 52);
    const e: i64 = if (biased == 0) -1074 else biased - 1075;
    var num = try Big.initSet(allocator, m);
    defer num.deinit();
    var den = try Big.initSet(allocator, 1);
    defer den.deinit();
    if (e >= 0) try num.shiftLeft(&num, @intCast(e)) else try den.shiftLeft(&den, @intCast(-e));
    if (s != 0) {
        var ten = try Big.initSet(allocator, 10);
        defer ten.deinit();
        var p = try Big.init(allocator);
        defer p.deinit();
        try p.pow(&ten, @intCast(@abs(s)));
        if (s > 0) try num.mul(&num, &p) else try den.mul(&den, &p);
    }
    var q = try Big.init(allocator);
    errdefer q.deinit();
    var r = try Big.init(allocator);
    defer r.deinit();
    try q.divFloor(&r, &num, &den);
    try r.shiftLeft(&r, 1);
    const half = r.order(den);
    if (half == .gt or (half == .eq and q.isOdd())) try q.addScalar(&q, 1);
    return q;
}

test "the same digits as glibc's printf" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    const specials = [_]f64{ 0, 0.5, 1.5, 2.5, 0.125, 0.375, 2.675, 1e22, 1e23, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 9.5, 99.5, 999.5, 0.05, 0.0001, 0.00001, 123456789.0, 1e16, 9.999999e-5, 99999.5, 999999.5, 1e-7, 4.35, 0.15, 3.0, 100.0 };
    var cbuf: [1600]u8 = undefined;
    var zbuf: [1600]u8 = undefined;
    var i: usize = 0;
    while (i < 60000) : (i += 1) {
        const x: f64 = if (i < specials.len) specials[i] else switch (i % 4) {
            0 => @bitCast(rnd.int(u64) & 0x7fff_ffff_ffff_ffff),
            1 => rnd.float(f64) * std.math.pow(f64, 10, @floatFromInt(rnd.intRangeAtMost(i32, -10, 20))),
            2 => @as(f64, @floatFromInt(rnd.intRangeAtMost(i64, 0, 100000))) / 8.0,
            else => @as(f64, @floatFromInt(rnd.intRangeAtMost(i64, 0, 1000000))) / 1000.0,
        };
        if (std.math.isNan(x) or std.math.isInf(x)) continue;
        const conv = "eEfFgG"[i % 6];
        const prec: usize = if (i % 7 == 0) rnd.intRangeAtMost(usize, 0, 40) else rnd.intRangeAtMost(usize, 0, 17);
        const alt = i % 5 == 0;
        // (%f of a huge double: its 309 digits)
        var cf: [8]u8 = undefined;
        const cfs = try std.fmt.bufPrintZ(&cf, "%{s}.*{c}", .{ if (alt) "#" else "", conv });
        const got = snprintf(&cbuf, cbuf.len, cfs.ptr, @as(c_int, @intCast(prec)), x);
        const want = cbuf[0..@intCast(got)];
        const have = try exact(&zbuf, conv, prec, alt, x);
        if (!std.mem.eql(u8, want, have)) {
            std.debug.print("{s} of {e}: glibc {s}, here {s}\n", .{ cfs, x, want, have });
            return error.Differs;
        }
    }
    for ([_]f64{ std.math.inf(f64), std.math.nan(f64) }) |x| for ("efgEFG") |conv| {
        var cf: [8]u8 = undefined;
        const cfs = try std.fmt.bufPrintZ(&cf, "%.*{c}", .{conv});
        const got = snprintf(&cbuf, cbuf.len, cfs.ptr, @as(c_int, 3), x);
        try std.testing.expectEqualStrings(cbuf[0..@intCast(got)], try exact(&zbuf, conv, 3, false, x));
    };
}
