//! A standalone program's runtime (rt.zig): the program as its image gives
//! it (Desc: its compiled code, its tables, its text), run from `main`,
//! its errors written as the reference mode words them. The bridge finds
//! the code it compiles as a program runs in the JIT here, compiled ahead
//! of time (standalone_build.zig): a node's thunk, a Python function's
//! code.

const std = @import("std");
const helpers = @import("helpers.zig");
const value = @import("value.zig");
const gc = @import("gc.zig");
const tree_mod = @import("tree.zig");
const program_mod = @import("program.zig");
const grammar_mod = @import("grammar.zig");

const Ctx = helpers.Ctx;
const Value = value.Value;
const allocator = std.heap.c_allocator;
/// A Python function's compiled code (driver.Helper's signature)
const Helper = *const fn (*Ctx, ?*value.Frame, [*]const Value, u32, u32, ?*const Value, ?*const Value, *Value) callconv(.c) i32;

/// The program, as the image lays it out (standalone_build.zig writes it:
/// every field a word)
pub const Desc = extern struct {
    /// bool (ctx, globals): the top level
    main: *const fn (*Ctx, *value.Frame) callconv(.c) bool,
    globals: u64,
    max_depth: u64,
    /// The objects the code refers to by index (Ctx.objects): stand-ins
    objects: [*]const usize,
    objects_len: u64,
    thunks: [*]const ThunkEntry,
    thunks_len: u64,
    called: [*]const CalledEntry,
    called_len: u64,
    path: [*]const u8,
    path_len: u64,
    source: [*]const u8,
    source_len: u64,
    /// The program's tree (tree.FlatNode each), each node's parent
    nodes: [*]const tree_mod.FlatNode,
    nodes_len: u64,
    parents: [*]const u32,
    /// The grammar's rules, kinds, actions and labels, described
    /// (standalone_build.describeGrammar)
    grammar: [*]const u8,
    grammar_len: u64,
    /// By node: the symbol it defines or uses (NONE: none), its owner (the
    /// scope whose frame its code runs in: Compiler.ownerOf)
    sym_of: [*]const u32,
    owners: [*]const u32,
    /// The symbols, as rt.load / rt.store / rt.scope of an rt value find
    /// their variables
    syms: [*]const SymInfo,
    syms_len: u64,
    /// The image's items built (once, before anything runs)
    init: *const fn () callconv(.c) void,
    /// Built with prune=True (1): what wasn't compiled may have been left
    /// out by name
    pruned: u64,
    /// The language's setup (build_native(setup=...)): its object's index
    /// + 1 (0: none), called with the program's path and arguments before
    /// the top level runs
    setup: u64,
};

/// A symbol: its scope node (NONE: global), the node whose frame holds its
/// variable (NONE: the program's), its slot there (NONE: none compiled),
/// a builtin's (1)
pub const SymInfo = extern struct { scope: u32, home: u32, slot: u32, builtin: u32 };

/// A node's symbol, or null
pub fn symbolOf(node: u32) ?*const SymInfo {
    const p = program.?;
    if (node >= p.nodes_len) return null;
    const s = p.sym_of[node];
    return if (s == program_mod.NONE or s >= p.syms_len) null else &p.syms[s];
}

pub fn ownerOf(node: u32) u32 {
    const p = program.?;
    return if (node < p.nodes_len) p.owners[node] else program_mod.NONE;
}

/// The program's tree and grammar, as the bridge reads nodes (made at the
/// start from the image's)
var data: program_mod.Data = undefined;
var grammar: grammar_mod.Grammar = undefined;
pub var nodes: @import("driver.zig").NodeCache = .{};

pub fn programData() *const program_mod.Data {
    return &data;
}

/// The grammar and the tree made from the image's descriptions
fn makeData(desc: *const Desc) void {
    grammar = .{ .arena = std.heap.ArenaAllocator.init(allocator) };
    const a = grammar.arena.allocator();
    var at: usize = 0;
    const g = desc.grammar[0..desc.grammar_len];
    const u = struct {
        fn int(comptime T: type, b: []const u8, pos: *usize) T {
            const x = std.mem.readInt(T, b[pos.*..][0..@sizeOf(T)], .little);
            pos.* += @sizeOf(T);
            return x;
        }
        fn str(b: []const u8, pos: *usize) []const u8 {
            const n: usize = @intCast(int(u64, b, pos));
            const s = b[pos.*..][0..n];
            pos.* += n;
            return s;
        }
    };
    const n_rules: usize = u.int(u32, g, &at);
    const rule_names = a.alloc([]const u8, n_rules) catch oom();
    const kind_names = a.alloc([]const u8, n_rules) catch oom();
    const actions = a.alloc(grammar_mod.Action, n_rules) catch oom();
    const labels = a.alloc([]const grammar_mod.Label, n_rules) catch oom();
    for (0..n_rules) |i| {
        rule_names[i] = u.str(g, &at);
        kind_names[i] = u.str(g, &at);
        actions[i] = @enumFromInt(u.int(u8, g, &at));
        const n: usize = u.int(u32, g, &at);
        const ls = a.alloc(grammar_mod.Label, n) catch oom();
        for (ls) |*l| {
            l.field = u.int(u8, g, &at);
            l.many = u.int(u8, g, &at) != 0;
        }
        labels[i] = ls;
    }
    const n_fields: usize = u.int(u32, g, &at);
    const field_names = a.alloc([]const u8, n_fields) catch oom();
    for (field_names, 0..) |*f, i| {
        f.* = u.str(g, &at);
        grammar.field_ids.put(a, f.*, @intCast(i + 1)) catch oom();
    }
    grammar.rule_names = rule_names;
    grammar.kind_names = kind_names;
    grammar.actions = actions;
    grammar.labels = labels;
    grammar.field_names = field_names;
    const src = desc.source[0..desc.source_len];
    var line_starts: std.ArrayListUnmanaged(u32) = .empty;
    line_starts.append(a, 0) catch oom();
    for (src, 0..) |ch, i| if (ch == '\n') line_starts.append(a, @intCast(i + 1)) catch oom();
    data = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .grammar = &grammar,
        .nodes = desc.nodes[0..desc.nodes_len],
        .input = src,
        .parents = @constCast(desc.parents[0..desc.nodes_len]),
        .sym_of = &.{},
        .functions = &.{},
        .line_starts = line_starts.items,
    };
}

fn oom() noreturn {
    @panic("zrun: out of memory");
}

/// A node's thunk (driver.Thunk): for its eval or exec in a scope's frames
pub const ThunkEntry = extern struct { node: u64, which: u64, owner: u64, code: *const anyopaque };

/// A Python function's code (driver.Helper), for calls of `nargs`
/// arguments, those of `rt_mask` rt values
pub const CalledEntry = extern struct { func: usize, nargs: u64, rt_mask: u64, code: *const anyopaque };

/// A Python object the code holds (a host function as a value, a class
/// it names): what's left of it, its name
pub const Standin = extern struct {
    refcnt: isize,
    type: ?*anyopaque,
    name: [*]const u8,
    name_len: u64,
};

/// The program running (null: not a standalone one)
pub var program: ?*const Desc = null;

/// The thunk compiled for a node's eval or exec, or null
pub fn thunkOf(node: u32, which: u32, owner: u32) ?*const anyopaque {
    const p = program orelse return null;
    for (p.thunks[0..p.thunks_len]) |t| {
        if (t.node == node and t.which == which and t.owner == owner) return t.code;
    }
    return null;
}

/// A Python function's code for a call, or null
pub fn calledOf(func: usize, nargs: usize, rt_mask: u64) ?*const anyopaque {
    const p = program orelse return null;
    for (p.called[0..p.called_len]) |e| {
        if (e.func == func and e.nargs == nargs and e.rt_mask == rt_mask) return e.code;
    }
    return null;
}

/// A stand-in's name (a Python object's __name__)
pub fn nameOf(o: *const anyopaque) []const u8 {
    const s: *const Standin = @ptrCast(@alignCast(o));
    return s.name[0..s.name_len];
}

fn noNode(_: *anyopaque, _: u32) callconv(.c) ?*anyopaque {
    return null;
}

// The program's main, zr_rt_main (the image's `main` calls it): 0, or 1
// after a runtime error written to the standard error. The runtime
// library's alone (the extension has no use for it)
comptime {
    if (helpers.standalone) @export(&rtMain, .{ .name = "zr_rt_main" });
}

fn rtMain(desc: *const Desc, argc: c_int, argv: [*]const [*:0]const u8) callconv(.c) c_int {
    program = desc;
    makeData(desc);
    desc.init();
    var objects: std.ArrayListUnmanaged(*@import("pyhelp.zig").PyObject) =.{ .items = @ptrCast(@constCast(desc.objects[0..desc.objects_len])), .capacity = desc.objects_len };
    var dummy: u8 = 0;
    var ctx = Ctx{
        .node_maker = .{ .ctx = &dummy, .make_fn = @ptrCast(&noNode) },
        .objects = &objects,
        .max_depth = @intCast(desc.max_depth),
    };
    const globals = helpers.zr_frame_new(null, desc.globals) orelse {
        writeErr("zrun: out of memory\n");
        return 1;
    };
    const ok = runSetup(desc, &ctx, argv[1..@intCast(@max(argc, 1))]) and desc.main(&ctx, globals);
    if (!ok) {
        helpers.flushOutput();
        report(desc, &ctx);
        return 1;
    }
    helpers.flushOutput();
    return 0;
}

/// The language's setup, if the build has one: its code called (compiled
/// ahead for two values) with the program's path, a str, and its
/// arguments, a list of strs. False with the error in `ctx`.
fn runSetup(desc: *const Desc, ctx: *Ctx, argv: []const [*:0]const u8) bool {
    if (desc.setup == 0) return true;
    const args = programArgs(argv) orelse return helpers.fail(ctx, 0, "out of memory", .{});
    const callee = desc.objects[desc.setup - 1];
    const code_p = calledOf(callee, 2, 0) orelse return helpers.fail(ctx, 0, "the setup wasn't compiled ahead of time", .{});
    const code: Helper = @ptrCast(@alignCast(code_p));
    const path = value.newStr(desc.path[0..desc.path_len]) orelse return helpers.fail(ctx, 0, "out of memory", .{});
    const list = value.newList(args.len) orelse return helpers.fail(ctx, 0, "out of memory", .{});
    var given = [2]Value{ Value.obj(.str, &path.head), Value.obj(.list, &list.head) };
    defer for (given) |v| value.decref(v);
    for (args) |a| {
        const s = value.newStr(a) orelse return helpers.fail(ctx, 0, "out of memory", .{});
        if (!value.listPush(list, Value.obj(.str, &s.head))) return helpers.fail(ctx, 0, "out of memory", .{});
    }
    var out: Value = Value.none_v;
    const status = code(ctx, null, &given, 0, 0, null, null, &out);
    value.decref(out);
    return switch (status) {
        1 => true,
        0 => false,
        else => helpers.fail(ctx, 0, "rt.Return, rt.Break or rt.Continue raised out of the setup", .{}),
    };
}

/// The program's arguments (the program's name not among them), as UTF-8:
/// main's on POSIX; on Windows its command line's, UTF-16 (main's are the
/// code page's), or null
fn programArgs(argv: []const [*:0]const u8) ?[]const []const u8 {
    if (@import("builtin").os.tag != .windows) {
        const out = allocator.alloc([]const u8, argv.len) catch return null;
        for (argv, out) |a, *o| o.* = std.mem.span(a);
        return out;
    }
    var n: c_int = 0;
    const wide = win.CommandLineToArgvW(win.GetCommandLineW(), &n) orelse return null;
    const count: usize = @intCast(@max(n, 1));
    const out = allocator.alloc([]const u8, count - 1) catch return null;
    for (out, wide[1..count]) |*o, w| o.* = std.unicode.utf16LeToUtf8Alloc(allocator, std.mem.span(w)) catch return null;
    return out;
}

const win = struct {
    extern "kernel32" fn GetCommandLineW() callconv(.winapi) [*:0]const u16;
    extern "shell32" fn CommandLineToArgvW(line: [*:0]const u16, n: *c_int) callconv(.winapi) ?[*]const [*:0]const u16;
};

fn writeErr(s: []const u8) void {
    @import("stdio.zig").writeAll(2, s);
}

/// The line and column (1-based, in characters) of a source offset
fn lineCol(src: []const u8, offset: usize) struct { line: usize, col: usize, start: usize } {
    const at = @min(offset, src.len);
    var line: usize = 1;
    var start: usize = 0;
    for (src[0..at], 0..) |ch, i| if (ch == '\n') {
        line += 1;
        start = i + 1;
    };
    return .{ .line = line, .col = chars(src[start..at]) + 1, .start = start };
}

fn chars(s: []const u8) usize {
    var n: usize = 0;
    for (s) |ch| {
        if (ch & 0xC0 != 0x80) n += 1;
    }
    return n;
}

/// A runtime error, as the reference mode's zrun.Error renders it: where,
/// the message, the line with the node underlined, the calls that led
/// there (innermost first)
fn report(desc: *const Desc, ctx: *Ctx) void {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(allocator);
    const src = desc.source[0..desc.source_len];
    const path = desc.path[0..desc.path_len];
    const msg = if (ctx.err_msg.items.len > 0) ctx.err_msg.items else "error";
    if (ctx.err_node < desc.nodes_len) {
        const nd = desc.nodes[ctx.err_node];
        const span = [2]u32{ nd.text_start, nd.text_end };
        const at = lineCol(src, span[0]);
        out.print(allocator, "{s}:{d}:{d}: error: {s} [runtime]\n", .{ path, at.line, at.col, msg }) catch return;
        const line_end = std.mem.indexOfScalarPos(u8, src, at.start, '\n') orelse src.len;
        const line = src[at.start..line_end];
        out.print(allocator, "{d:>5} | {s}\n", .{ at.line, line }) catch return;
        const end = @min(@max(span[1], span[0] + 1), line_end);
        const width = @max(chars(src[@min(span[0], line_end)..end]), 1);
        out.appendSlice(allocator, "      | ") catch return;
        out.appendNTimes(allocator, ' ', at.col - 1) catch return;
        out.appendNTimes(allocator, '^', width) catch return;
        out.append(allocator, '\n') catch return;
    } else out.print(allocator, "{s}: error: {s} [runtime]\n", .{ path, msg }) catch return;
    for (ctx.err_stack.items) |e| {
        const name = e.name.bytes();
        if (e.node < desc.nodes_len) {
            const at = lineCol(src, desc.nodes[e.node].text_start);
            out.print(allocator, "  in {s}(), called at {s}:{d}:{d}\n", .{ name, path, at.line, at.col }) catch return;
        } else out.print(allocator, "  in {s}()\n", .{name}) catch return;
    }
    writeErr(out.items);
}
