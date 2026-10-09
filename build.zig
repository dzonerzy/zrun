//! Build script for zrun.s native module
//!
//! You can use this directly with `zig build`, or use `pyoz build` for
//! automatic Python configuration detection.

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Strip option (can be set via -Dstrip=true or from pyoz CLI)
    const strip = b.option(bool, "strip", "Strip debug symbols from the binary") orelse false;

    // Python Stable ABI (abi3): one wheel for CPython 3.10+. `pyoz build`
    // enables it from `abi3 = true` in pyproject.toml; with plain
    // `zig build`, pass -Dabi3=true.
    const abi3 = b.option(bool, "abi3", "Build for the Python Stable ABI (abi3)") orelse false;

    const pyoz_dep = b.dependency("PyOZ", .{
        .target = target,
        .optimize = optimize,
        .abi3 = abi3,
    });

    const user_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        // libc is required for the Python C API
        .link_libc = true,
        .imports = &.{
            .{ .name = "PyOZ", .module = pyoz_dep.module("PyOZ") },
        },
    });

    // LLVM's C API headers (only the prototypes: the functions come from
    // zgram's LLVM, through its zgram.llvm.v1 capsule)
    user_lib_mod.addIncludePath(b.path("vendor/llvm/include"));

    // Single source for the version string returned by zrun.version()
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", @import("build.zig.zon").version);
    user_lib_mod.addOptions("build_options", build_options);

    const lib = b.addLibrary(.{
        .name = "zrun",
        .linkage = .dynamic,
        .root_module = user_lib_mod,
    });

    // Extensions get the Python C API from the interpreter at load time, not
    // from a library: macOS needs this (-undefined dynamic_lookup).
    if (target.result.os.tag == .macos) lib.linker_allow_shlib_undefined = true;

    // On Windows, link against the Python stable ABI library (python3.lib).
    // These options are passed automatically by `pyoz build`.
    if (b.option([]const u8, "python-lib-dir", "Python library directory")) |lib_dir| {
        user_lib_mod.addLibraryPath(.{ .cwd_relative = lib_dir });
    }
    if (b.option([]const u8, "python-lib-name", "Python library name")) |lib_name| {
        user_lib_mod.linkSystemLibrary(lib_name, .{ .use_pkg_config = .no });
    }

    // The runtime of programs compiled to run without Python (`zig build
    // rt`): the helpers, as a static library
    const rt_mod = b.createModule(.{
        .root_source_file = b.path("src/rt.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
        .imports = &.{
            .{ .name = "PyOZ", .module = pyoz_dep.module("PyOZ") },
        },
    });
    rt_mod.addIncludePath(b.path("vendor/llvm/include"));
    rt_mod.addOptions("build_options", build_options);
    const rt_lib = b.addLibrary(.{ .name = "zrun_rt", .linkage = .static, .root_module = rt_mod });
    const rt_step = b.step("rt", "The runtime of programs compiled to run without Python");
    rt_step.dependOn(&b.addInstallArtifact(rt_lib, .{}).step);

    // .pyd for Windows, .so otherwise (by the target, for cross builds)
    const ext = if (target.result.os.tag == .windows) ".pyd" else ".so";
    const install = b.addInstallArtifact(lib, .{
        .dest_sub_path = b.fmt("zrun{s}", .{ext}),
    });
    b.getInstallStep().dependOn(&install.step);
}
