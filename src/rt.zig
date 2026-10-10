//! zrun's runtime for programs compiled to run without Python (standalone
//! builds): the helpers compiled code calls, the values, the cycle
//! collector, as the extension has them. Linked with a stub of the Python
//! C API (pystub.zig): a strict language's compiled code never takes the
//! paths that would need Python.

const std = @import("std");

/// A panic (zrun's bug, out of memory) said and the process stopped: no
/// stack trace (what printing one takes, its debug info readers, would be
/// most of a small program)
pub const panic = std.debug.FullPanic(stop);

fn stop(msg: []const u8, _: ?usize) noreturn {
    @branchHint(.cold);
    _ = std.c.write(2, "zrun: ", 6);
    _ = std.c.write(2, msg.ptr, msg.len);
    _ = std.c.write(2, "\n", 1);
    std.c.abort();
}

comptime {
    _ = @import("pystub.zig");
    _ = @import("helpers.zig");
    _ = @import("bridge.zig");
    _ = @import("value.zig");
    _ = @import("gc.zig");
    _ = @import("set.zig");
    _ = @import("image.zig");
    _ = @import("standalone.zig");
}
