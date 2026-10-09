//! Compiled code without the GIL: a thread running it (a call made with the
//! GIL released, a map() worker) takes the GIL the first time it touches
//! Python (ensure(): Python objects' counts, conversions, Python errors,
//! host functions, semantics run as Python, compiling), and gives it back
//! when its call is done (done()). Code that never touches Python runs in
//! parallel with other threads; code that does takes turns, as Python's
//! threads do.
//!
//! A thread running Python (any thread zrun is called from) holds the GIL:
//! `held` is true there until a call gives it up (without()).
//!
//! Each thread's state is a thread-local's, but the thread with the
//! allocator's home lists (pool.zig: the thread making the values, Python's
//! usually) has it in a plain global: a library's thread-local costs a call
//! each time it's read, and ensure() is read by nearly every helper.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const pool = @import("pool.zig");

const State = struct {
    /// Whether this thread holds the GIL
    held: bool = true,
    /// ensure() took it (done() gives it back)
    taken: bool = false,
    state: py.c.PyGILState_STATE = undefined,
    /// The times this thread took it (a program's report(): calls that
    /// could run in parallel and didn't)
    takes: u64 = 0,
    /// A strict language's code running (strictBegin); it went into Python
    /// outside what strict mode allows (allowBegin): where, in `entry`
    strict: bool = false,
    entered: bool = false,
    allowed: u32 = 0,
};

/// Where this thread's strict code first went into Python (kept apart from
/// State: read only then)
threadlocal var entry: Entry = undefined;

/// Where compiled code went into Python (strict mode's message): zrun's
/// code, the program's node (if the helper knew it), what it called (its
/// name, if it was a Python object's)
pub const Entry = struct {
    at: std.builtin.SourceLocation,
    node: ?u32 = null,
    name: [96]u8 = undefined,
    name_len: usize = 0,

    pub fn calledName(self: *const Entry) ?[]const u8 {
        return if (self.name_len > 0) self.name[0..self.name_len] else null;
    }
};

/// The home thread's (pool.homeOwner()), and the others'
var home: State = .{};
threadlocal var mine: State = .{};

inline fn here() *State {
    const tp = pool.threadPointer();
    const owner = pool.homeOwner();
    if (tp != 0 and tp == owner) return &home;
    return if (owner == 0) claimed(tp) else &mine;
}

/// (no thread has the home lists: this one's from now, if it gets them)
noinline fn claimed(tp: usize) *State {
    pool.claimHome();
    return if (tp != 0 and tp == pool.homeOwner()) &home else &mine;
}

/// This thread has the allocator's home lists from now (pool.zig, on this
/// thread): its state goes with them.
pub fn claimHome() void {
    home = mine;
}

/// The times this thread took the GIL so far.
pub fn takes() u64 {
    return here().takes;
}

/// The GIL, held from here (taken if this thread hasn't it): Python's to
/// run, `at` (@src()) where, for strict mode.
pub inline fn ensure(comptime at: std.builtin.SourceLocation) void {
    const s = here();
    // (one test where the GIL is held and no strict code runs: the rest
    // out of line, the helpers it's in staying small)
    if (!s.held or s.strict) slow(s, &struct {
        const loc = at;
    }.loc, null, null);
}

/// ensure() for the program's node `node`, calling `callee` (if it's a
/// Python object's call): what strict mode's message says.
pub inline fn ensureAt(comptime at: std.builtin.SourceLocation, node: u32, callee: ?*py.c.PyObject) void {
    const s = here();
    if (!s.held or s.strict) slow(s, &struct {
        const loc = at;
    }.loc, node, callee);
}

noinline fn slow(s: *State, at: *const std.builtin.SourceLocation, node: ?u32, callee: ?*py.c.PyObject) void {
    if (!s.held) take(s);
    if (s.strict and s.allowed == 0 and !s.entered) note(s, at.*, node, callee);
}

fn note(s: *State, at: std.builtin.SourceLocation, node: ?u32, callee: ?*py.c.PyObject) void {
    entry = .{ .at = at, .node = node };
    if (callee) |o| if (ph.attr(o, "__qualname__") orelse ph.attr(o, "__name__")) |n| {
        defer py.Py_DecRef(n);
        if (ph.utf8(n, "name")) |text| {
            entry.name_len = @min(text.len, entry.name.len);
            @memcpy(entry.name[0..entry.name_len], text[0..entry.name_len]);
        }
    };
    py.c.PyErr_Clear();
    s.entered = true;
}

/// The GIL for what strict mode lets compiled code touch Python for, until
/// allowEnd(): an error being made (Python's exceptions still), code
/// compiled while the program runs, the collector asking about proxies.
pub inline fn allowBegin() void {
    const s = here();
    if (!s.held) take(s);
    s.allowed += 1;
}

pub inline fn allowEnd() void {
    here().allowed -= 1;
}

/// This thread's compiled code a strict language's from now (Language(
/// strict=True)): where it goes into Python kept, for strictEnd().
pub fn strictBegin() void {
    const s = here();
    s.strict = true;
    s.entered = false;
}

/// Where the code went into Python since strictBegin(), first (null:
/// nowhere); strict mode off.
pub fn strictEnd() ?Entry {
    const s = here();
    s.strict = false;
    defer s.entered = false;
    return if (s.entered) entry else null;
}

fn take(s: *State) void {
    s.state = py.c.PyGILState_Ensure();
    s.held = true;
    s.taken = true;
    s.takes += 1;
}

/// A call done: the GIL ensure() took given back.
pub fn done() void {
    const s = here();
    if (s.taken) {
        s.taken = false;
        s.held = false;
        py.c.PyGILState_Release(s.state);
    }
}

/// Run `f` with the GIL released (this thread holds it): compiled code
/// taking it back if it needs it.
pub fn without(comptime f: anytype, args: anytype) @typeInfo(@TypeOf(f)).@"fn".return_type.? {
    const ts = py.c.PyEval_SaveThread();
    here().held = false;
    const r = @call(.auto, f, args);
    done();
    py.c.PyEval_RestoreThread(ts);
    here().held = true;
    return r;
}

/// A thread zrun made (map()'s workers): it holds no GIL.
pub fn worker() void {
    const s = here();
    s.held = false;
    s.taken = false;
}
