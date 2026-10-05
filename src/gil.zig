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

const ph = @import("pyhelp.zig");
const py = ph.py;

/// Whether this thread holds the GIL
pub threadlocal var held: bool = true;
/// ensure() took it (done() gives it back)
threadlocal var taken: bool = false;
threadlocal var state: py.c.PyGILState_STATE = undefined;

/// The GIL, held from here (taken if this thread hasn't it).
pub inline fn ensure() void {
    if (!held) take();
}

fn take() void {
    state = py.c.PyGILState_Ensure();
    held = true;
    taken = true;
}

/// A call done: the GIL ensure() took given back.
pub fn done() void {
    if (taken) {
        taken = false;
        held = false;
        py.c.PyGILState_Release(state);
    }
}

/// Run `f` with the GIL released (this thread holds it): compiled code
/// taking it back if it needs it.
pub fn without(comptime f: anytype, args: anytype) @typeInfo(@TypeOf(f)).@"fn".return_type.? {
    const ts = py.c.PyEval_SaveThread();
    held = false;
    const r = @call(.auto, f, args);
    done();
    py.c.PyEval_RestoreThread(ts);
    held = true;
    return r;
}

/// A thread zrun made (map()'s workers): it holds no GIL.
pub fn worker() void {
    held = false;
    taken = false;
}
