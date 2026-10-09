//! zrun's runtime for programs compiled to run without Python (standalone
//! builds): the helpers compiled code calls, the values, the cycle
//! collector, as the extension has them. Linked with a stub of the Python
//! C API (pystub.zig): a strict language's compiled code never takes the
//! paths that would need Python.

comptime {
    _ = @import("pystub.zig");
    _ = @import("helpers.zig");
    _ = @import("bridge.zig");
    _ = @import("value.zig");
    _ = @import("gc.zig");
    _ = @import("set.zig");
}
