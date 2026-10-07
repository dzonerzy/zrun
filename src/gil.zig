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

/// The GIL, held from here (taken if this thread hasn't it).
pub inline fn ensure() void {
    const s = here();
    if (!s.held) take(s);
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
