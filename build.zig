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
    build_options.addOption(bool, "standalone", false);
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

    // The runtime of programs compiled to run without Python: the helpers,
    // as a static library (the same code, in a process without Python: no
    // path falls back to it)
    const rt_options = b.addOptions();
    rt_options.addOption([]const u8, "version", @import("build.zig.zon").version);
    rt_options.addOption(bool, "standalone", true);
    rt_options.addOption(bool, "has_rt", false);
    // `zig build rt`: for this target
    const rt_lib = runtime(b, target, optimize, strip, pyoz_dep, rt_options);
    const rt_step = b.step("rt", "The runtime of programs compiled to run without Python");
    rt_step.dependOn(&b.addInstallArtifact(rt_lib, .{}).step);
    // One in the extension for each target zrun.build_native makes programs
    // for (x86-64 Linux and Windows: LLVM's x86 backend, zgram's), for any
    // machine of it (a wheel's: the baseline CPU), the Python headers its
    // shared code declares with vendored (the build's own may be another
    // target's)
    build_options.addOption(bool, "has_rt", true);
    for ([_][2][]const u8{ .{ "linux", "x86_64-linux-gnu" }, .{ "windows", "x86_64-windows-gnu" } }) |rt| {
        const query = std.Target.Query.parse(.{ .arch_os_abi = rt[1], .cpu_features = "baseline" }) catch unreachable;
        const rt_target = b.resolveTargetQuery(query);
        const include: []const []const u8 = &.{b.pathFromRoot(b.fmt("vendor/python/{s}/include", .{rt[0]}))};
        const dep = b.dependency("PyOZ", .{ .target = rt_target, .optimize = optimize, .abi3 = true, .@"python-include-dirs" = include });
        const lib_rt = runtime(b, rt_target, optimize, true, dep, rt_options);
        user_lib_mod.addAnonymousImport(b.fmt("zrun_rt_{s}", .{rt[0]}), .{ .root_source_file = lib_rt.getEmittedBin() });
    }

    // Zig's own tests: those of the parts with no Python (floatfmt's
    // digits against glibc's printf)
    const unit = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/floatfmt.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
    }) });
    const test_step = b.step("test", "Zig's tests of zrun's parts with no Python");
    test_step.dependOn(&b.addRunArtifact(unit).step);

    // .pyd for Windows, .so otherwise (by the target, for cross builds)
    const ext = if (target.result.os.tag == .windows) ".pyd" else ".so";
    const install = b.addInstallArtifact(lib, .{
        .dest_sub_path = b.fmt("zrun{s}", .{ext}),
    });
    b.getInstallStep().dependOn(&install.step);
}

/// The runtime of standalone programs for a target, as a static library
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: bool, pyoz_dep: *std.Build.Dependency, options: *std.Build.Step.Options) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .root_source_file = b.path("src/rt.zig"),
        .target = target,
        .optimize = optimize,
        .strip = strip,
        .link_libc = true,
        .imports = &.{
            .{ .name = "PyOZ", .module = pyoz_dep.module("PyOZ") },
        },
    });
    mod.addIncludePath(b.path("vendor/llvm/include"));
    mod.addOptions("build_options", options);
    const lib = b.addLibrary(.{ .name = "zrun_rt", .linkage = .static, .root_module = mod });
    // (a section per function and datum: a program linked with it keeps
    // only what it calls, --gc-sections)
    lib.link_function_sections = true;
    lib.link_data_sections = true;
    return lib;
}
