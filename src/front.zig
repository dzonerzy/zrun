//! The semantics compiler's front: a semantic's Python source (read with
//! Python's `ast` module, walked from here) checked against the compilable
//! subset and turned into zrun's form of it (the types below), which the
//! partial evaluator specializes for each program.
//!
//! The subset: values (int, float, str, bool, None, lists, tuples, dicts,
//! sets, records), assignment (also to fields and items, augmented), if,
//! while, for, break, continue, return, raise (rt's control flow and
//! errors), assert, pass, try, with, del, import, global, nonlocal, nested
//! defs and lambdas (closures); expressions with operators, comparisons,
//! conditional expressions, comprehensions, f-strings and `:=`; calls (rt,
//! other semantics, helper functions of the module, records, builtins,
//! methods of values), `*args` among their arguments. Outside it: a nested
//! class, yield, await, match, defaults and *args of nested functions. A
//! construct outside the subset is an error at its line, when the semantic
//! is registered.
//!
//! Module-level names (helper functions, record classes, constants) are
//! kept by name: they are resolved when a program is compiled, since a
//! helper may be defined after the semantic using it.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;

const Allocator = std.mem.Allocator;

pub const Pos = struct { line: u32 = 0, col: u32 = 0 };

pub const BinOp = enum { add, sub, mul, div, floordiv, mod, pow, lshift, rshift, bitor, bitxor, bitand };
pub const UnaryOp = enum { neg, pos, not_, invert };
pub const CmpOp = enum { eq, ne, lt, le, gt, ge, is, is_not, in, not_in };

pub const Expr = struct {
    pos: Pos,
    kind: Kind,

    pub const Kind = union(enum) {
        int: i64,
        /// An int literal beyond 64 bits (a reference kept for as long as
        /// the language: functions read are)
        big: *PyObject,
        /// Another constant (bytes, complex, `...`): Python's object, kept
        /// the same
        object: *PyObject,
        float: f64,
        str: []const u8,
        bool: bool,
        none,
        /// A local variable (parameters first)
        local: u32,
        /// A name of the module, a builtin, or a captured variable,
        /// resolved when the program is compiled
        global: []const u8,
        attr: struct { obj: *Expr, name: []const u8 },
        index: struct { obj: *Expr, index: *Expr },
        slice: struct { obj: *Expr, lo: ?*Expr, hi: ?*Expr, step: ?*Expr },
        call: struct { func: *Expr, args: []const *Expr, keywords: []const Keyword },
        /// A call of a function defined inside this one (or one around
        /// it): `func`, read as one of its own (Reader.nestedDef), given
        /// its arguments and then the variables around it that it reads
        /// (as they are when it's called: what a closure reads then)
        call_nested: struct { func: *const Function, args: []const *Expr },
        /// A function defined inside this one as a value (a lambda, a def's
        /// name read): a closure of `func` (Function.closure), the
        /// variables around it in the heap frames of the runs it's in
        make_closure: *const Function,
        /// `(name := value)`: value, stored to the local `slot` too
        named: struct { slot: u32, value: *Expr },
        /// `*value` among a call's arguments: its items there
        starred: *Expr,
        /// What an `import` inside the function binds, imported when the
        /// code compiles (as a module's own import, done once): the module
        /// `module` (its top package unless `leaf`: `import a.b` binds
        /// `a`), or its attribute `attr` (`from module import attr`)
        import_: struct { module: []const u8, attr: ?[]const u8 = null, leaf: bool = true },
        binary: struct { op: BinOp, left: *Expr, right: *Expr },
        unary: struct { op: UnaryOp, operand: *Expr },
        and_: []const *Expr,
        or_: []const *Expr,
        /// `a < b <= c`: first, then each (op, operand)
        compare: struct { first: *Expr, ops: []const CmpOp, rest: []const *Expr },
        cond: struct { test_: *Expr, then: *Expr, else_: *Expr },
        list: []const *Expr,
        tuple: []const *Expr,
        /// `{a, b}`; `folded`: made as CPython makes one of more than two
        /// constants (a frozenset of them first, copied)
        set_: struct { items: []const *Expr, folded: bool },
        set_comp: Comp,
        dict: struct { keys: []const *Expr, values: []const *Expr },
        list_comp: Comp,
        /// (x for ...): the list it gives, compiled (all() and any() of
        /// one stop at the deciding item, as Python's do)
        gen_exp: Comp,
        dict_comp: struct { key: *Expr, value: *Expr, generators: []const Generator },
        fstring: []const FPart,
        /// The function being run, called out of line with the arguments
        /// it was given (made by the compiler: a helper whose first `if`
        /// runs inline, the rest out of line)
        outline,
    };
};

pub const Keyword = struct { name: []const u8, value: *Expr };

pub const Comp = struct { elt: *Expr, generators: []const Generator };

pub const Generator = struct { target: Target, iter: *Expr, ifs: []const *Expr };

/// A piece of an f-string: text, or a value with its conversion (`!r`) and
/// format spec
pub const FPart = union(enum) {
    text: []const u8,
    value: struct { expr: *Expr, conversion: u8, spec: []const FPart },
};

pub const Target = union(enum) {
    local: u32,
    /// A module variable (`global name` in the function)
    global: []const u8,
    tuple: []const Target,
    attr: struct { obj: *Expr, name: []const u8 },
    index: struct { obj: *Expr, index: *Expr },
};

pub const Stmt = struct {
    pos: Pos,
    kind: Kind,

    pub const Kind = union(enum) {
        assign: struct { targets: []const Target, value: *Expr },
        aug: struct { target: Target, op: BinOp, value: *Expr },
        expr: *Expr,
        if_: struct { test_: *Expr, body: []const Stmt, else_: []const Stmt },
        while_: struct { test_: *Expr, body: []const Stmt, else_: []const Stmt },
        for_: struct { target: Target, iter: *Expr, body: []const Stmt, else_: []const Stmt },
        return_: ?*Expr,
        raise_: ?*Expr,
        assert_: struct { test_: *Expr, msg: ?*Expr },
        /// try: body, except handlers, else, finally
        try_: struct { body: []const Stmt, handlers: []const Handler, else_: []const Stmt, finally: []const Stmt },
        break_,
        continue_,
        pass,
        /// `del name` of a local: unset from here
        del_local: u32,
        /// `del obj[index]`
        del_item: struct { obj: *Expr, index: *Expr },
        /// `raise exc from cause`
        raise_from: struct { exc: *Expr, cause: *Expr },
        /// Statements run in turn (one Python statement read as several:
        /// `import a, b`, `del x, y`)
        seq: []const Stmt,
    };
};

/// `except T as name:` (no T: a bare except)
pub const Handler = struct {
    pos: Pos,
    type_: ?*Expr,
    name: ?u32,
    body: []const Stmt,
};

/// A semantic or helper function, read
pub const Function = struct {
    arena: std.heap.ArenaAllocator,
    name: []const u8,
    file: []const u8,
    /// The line of its `def` (or first decorator) in the file
    first_line: u32,
    /// Its parameters are locals 0 .. param_count; those from `required`
    /// on have defaults (the function's __defaults__). `*args`: the last,
    /// a tuple of the arguments after the others
    param_count: u32,
    required: u32 = 0,
    vararg: bool = false,
    /// Every local's name, by slot
    locals: []const []const u8,
    body: []const Stmt,
    /// The function object (owned): its module globals and closure, for
    /// resolving the global names
    py_function: *PyObject,
    /// How big it is: its expressions (whether to run it inline)
    size: u32 = 0,

    // A function defined in another (Reader.nestedFunction): its locals
    // its own parameters (`own`), its env (a closure's: the heap frame of
    // the run it was made in), the variables around it it reads
    // (`captures`), then what it assigns. A closure is given its own
    // parameters and its env; another (only called where it's defined) its
    // own, None and the captures' values as they are then.

    /// The function it's defined in (null: a semantic or helper)
    parent: ?*Function = null,
    /// Python's name of it (`f.<locals>.<lambda>`: what errors say)
    qualname: []const u8 = "",
    own: u32 = 0,
    captures: []const []const u8 = &.{},
    /// Made a value (a lambda, a def's name read)
    escapes: bool = false,
    /// Its `nonlocal` names (among the captures)
    nonlocals: bool = false,
    /// Run as a closure: its captures in its env's frames (`env`: where)
    closure: bool = false,
    env: []const EnvRef = &.{},
    /// Its locals a closure defined in it reads: in a heap frame each run
    /// makes (their index there, by slot; null: a stack local)
    heap: []?u32 = &.{},
    heap_len: u32 = 0,
    /// Whether a run makes that frame (heap locals, or closures made in it)
    makes_frame: bool = false,
    children: []const *Function = &.{},

    /// Where a closure's capture is: `depth` frames up from its env (0: the
    /// env itself), slot `index` there
    pub const EnvRef = struct { depth: u32, index: u32 };

    /// Whether local `slot` is the function's own (not its env or a capture)
    /// Its parameters given one argument each (all but `*args`)
    pub fn positional(self: *const Function) u32 {
        return self.param_count - @intFromBool(self.vararg);
    }

    /// Whether a call with `n` arguments gives what it takes
    pub fn takes(self: *const Function, n: usize) bool {
        return n >= self.required and (self.vararg or n <= self.param_count);
    }

    pub fn ownsLocal(self: *const Function, slot: usize) bool {
        if (self.parent == null) return true;
        return slot < self.own or slot >= self.own + 1 + self.captures.len;
    }

    fn slotOf(self: *const Function, name: []const u8) ?usize {
        for (self.locals, 0..) |l, i| if (std.mem.eql(u8, l, name)) return i;
        return null;
    }

    pub fn destroy(self: *Function, gpa: Allocator) void {
        py.Py_DecRef(self.py_function);
        self.arena.deinit();
        gpa.destroy(self);
    }
};

/// Why reading failed: an exception was set (Python), or the source is
/// outside the subset (`message` at `pos`)
pub const Failure = struct {
    message: [256]u8 = undefined,
    len: usize = 0,
    pos: Pos = .{},

    pub fn text(self: *const Failure) []const u8 {
        return self.message[0..self.len];
    }
};

const ReadError = error{ Python, OutOfMemory, Unsupported };

/// A function's source (inspect.getsource's), or that of a frozen module's
/// function from the module's file (Python 3.12's posixpath...: its code
/// names `<frozen posixpath>`, its module's __file__ the file); null with
/// the exception
const source_helper =
    \\import inspect, linecache, sys
    \\def source_of(f):
    \\    try:
    \\        return inspect.getsource(f)
    \\    except OSError:
    \\        code = getattr(f, "__code__", None)
    \\        name = code.co_filename if code is not None else ""
    \\        if name.startswith("<frozen ") and name.endswith(">"):
    \\            path = getattr(sys.modules.get(name[8:-1]), "__file__", None)
    \\            lines = linecache.getlines(path) if path else []
    \\            if len(lines) >= code.co_firstlineno:
    \\                return "".join(inspect.getblock(lines[code.co_firstlineno - 1:]))
    \\        raise
    \\def folded_order(func, elts):
    \\    # (a set literal of constants: the frozenset CPython made of them,
    \\    # in the function's code, laid out as the order they were put in
    \\    # makes it (the source's; a .pyc's, made by another process: that
    \\    # one's): an order of the items making one laid out the same, made
    \\    # here as compiled code makes it)
    \\    import ast, itertools
    \\    vals = [ast.literal_eval(e) for e in elts]
    \\    want = frozenset(vals)
    \\    def frozensets(code):
    \\        for c in code.co_consts:
    \\            if isinstance(c, frozenset):
    \\                yield c
    \\            elif hasattr(c, "co_consts"):
    \\                yield from frozensets(c)
    \\    def same(a, b):
    \\        return len(a) == len(b) and all(type(x) is type(y) and x == y for x, y in zip(a, b))
    \\    for c in frozensets(func.__code__):
    \\        if c != want:
    \\            continue
    \\        live = list(c)
    \\        used, first = set(), []
    \\        for x in live:
    \\            i = next((i for i, v in enumerate(vals) if i not in used and type(v) is type(x) and v == x), None)
    \\            if i is None:
    \\                break
    \\            used.add(i)
    \\            first.append(i)
    \\        else:
    \\            rest = [i for i in range(len(vals)) if i not in used]
    \\            # (more than four: its copy is a smaller table, the items
    \\            # put in in the frozenset's order: that order. Up to four:
    \\            # its copy the same size, slot for slot: an order laying it
    \\            # out the same (every order tried))
    \\            if len(first) > 4:
    \\                return first + rest
    \\            for t in [first] + [list(p) for p in itertools.permutations(first)]:
    \\                if same(list(frozenset(vals[i] for i in t)), live):
    \\                    return t + rest
    \\    return None
;
var helper_ns: ?*PyObject = null;

fn helper(name: [*:0]const u8) ?*PyObject {
    const ns = helper_ns orelse blk: {
        const ns = @import("compile.zig").runPython(source_helper) orelse return null;
        helper_ns = ns;
        break :blk ns;
    };
    return py.c.PyDict_GetItemString(ns, name);
}

fn sourceOf(func: *PyObject) ?*PyObject {
    const f = helper("source_of") orelse return null;
    return py.c.PyObject_CallFunctionObjArgs(f, func, @as(?*PyObject, null));
}

/// Read a Python function: its source, parsed, checked against the
/// subset. On error.Unsupported, `failure` says what and where (with the
/// function's file and first line, the caller makes it a CompileError).
pub fn read(gpa: Allocator, func: *PyObject, failure: *Failure) ReadError!*Function {
    var r = Reader{
        .gpa = gpa,
        .arena = std.heap.ArenaAllocator.init(gpa),
        .failure = failure,
        .py_function = func,
    };
    errdefer r.arena.deinit();
    const a = r.arena.allocator();

    // The source, dedented, parsed: a module with the def in it
    const inspect = py.c.PyImport_ImportModule("inspect") orelse return error.Python;
    defer py.Py_DecRef(inspect);
    const textwrap = py.c.PyImport_ImportModule("textwrap") orelse return error.Python;
    defer py.Py_DecRef(textwrap);
    const ast_mod = py.c.PyImport_ImportModule("ast") orelse return error.Python;
    defer py.Py_DecRef(ast_mod);
    const raw = sourceOf(func) orelse {
        // (one whose source isn't there (a builtin, a module without its
        // file): not compiled, Python runs it)
        if (py.c.PyErr_ExceptionMatches(py.PyExc_OSError()) != 0 or py.c.PyErr_ExceptionMatches(py.PyExc_TypeError()) != 0) {
            py.c.PyErr_Clear();
            const fname = try r.strAttr(func, "__qualname__");
            return r.unsupported(.{}, "the source of {s}() isn't available", .{fname});
        }
        return error.Python;
    };
    defer py.Py_DecRef(raw);
    const source = py.c.PyObject_CallMethod(textwrap, "dedent", "(O)", raw) orelse return error.Python;
    defer py.Py_DecRef(source);
    const module = py.c.PyObject_CallMethod(ast_mod, "parse", "(O)", source) orelse return error.Python;
    defer py.Py_DecRef(module);

    const code = ph.attr(func, "__code__") orelse return error.Python;
    defer py.Py_DecRef(code);
    const file = try r.strAttr(code, "co_filename");
    r.file = file;
    const first = try intAttr(code, "co_firstlineno");
    const name = try r.strAttr(func, "__name__");
    r.qualname = try r.strAttr(func, "__qualname__");

    const body = try listAttr(module, "body");
    defer py.Py_DecRef(body);
    const def = py.c.PyList_GetItem(body, 0) orelse return error.Python;
    if (!try isKind(def, "FunctionDef")) {
        return r.unsupported(.{}, "only functions defined with def can be compiled", .{});
    }
    r.line_offset = @intCast(@max(first - 1, 0));
    // (columns too: dedent took the def's indentation off every line)
    const raw_text = ph.utf8(raw, "source") orelse return error.Python;
    var indent: u32 = 0;
    while (indent < raw_text.len and (raw_text[indent] == ' ' or raw_text[indent] == '\t')) indent += 1;
    r.col_offset = indent;

    // Parameters: plain ones, the last ones with defaults maybe (their
    // values the function's __defaults__, made when it was defined)
    const args = ph.attr(def, "args") orelse return error.Python;
    defer py.Py_DecRef(args);
    inline for (.{ "posonlyargs", "kwonlyargs", "kw_defaults" }) |f| {
        const l = try listAttr(args, f);
        defer py.Py_DecRef(l);
        if (py.c.PyList_Size(l) != 0) return r.unsupported(try r.posOf(def), "only plain parameters can be compiled (no keyword-only or positional-only ones)", .{});
    }
    const defaults = try listAttr(args, "defaults");
    const n_defaults: usize = @intCast(py.c.PyList_Size(defaults));
    py.Py_DecRef(defaults);
    {
        const o = ph.attr(args, "kwarg") orelse return error.Python;
        defer py.Py_DecRef(o);
        if (o != py.Py_None()) return r.unsupported(try r.posOf(def), "**kwargs can't be compiled", .{});
    }
    const params = try listAttr(args, "args");
    defer py.Py_DecRef(params);
    const n_params: usize = @intCast(py.c.PyList_Size(params));
    for (0..n_params) |i| {
        const p = py.c.PyList_GetItem(params, @intCast(i)).?;
        const pname = try r.strAttr(p, "arg");
        _ = try r.declare(pname);
    }
    // (*args: a parameter after the others)
    const vararg = blk: {
        const o = ph.attr(args, "vararg") orelse return error.Python;
        defer py.Py_DecRef(o);
        if (o == py.Py_None()) break :blk false;
        _ = try r.declare(try r.strAttr(o, "arg"));
        break :blk true;
    };

    // Locals: every name the body assigns (Python's rule: then local
    // everywhere in the function)
    const stmts = try listAttr(def, "body");
    defer py.Py_DecRef(stmts);
    try r.scopeDecls(stmts);
    if (r.nonlocal_names.items.len > 0) return r.unsupported(try r.posOf(def), "`nonlocal` outside a nested function", .{});
    try r.collectAssigned(stmts);

    const out = try r.stmtList(stmts);
    const f = try gpa.create(Function);
    errdefer gpa.destroy(f);
    f.* = .{
        .arena = r.arena,
        .name = name,
        .file = file,
        .first_line = @intCast(first),
        .param_count = @intCast(n_params + @intFromBool(vararg)),
        .required = @intCast(n_params - n_defaults),
        .vararg = vararg,
        .locals = r.locals.items,
        .body = out,
        .py_function = func,
        .size = r.exprs,
        .children = r.children.items,
        .qualname = r.qualname,
    };
    for (r.children.items) |ch| ch.parent = f;
    // (the function's arena from here: r's is its)
    resolveClosures(f, f.arena.allocator()) catch |e| {
        r.arena = f.arena;
        return e;
    };
    py.Py_IncRef(func);
    _ = a;
    return f;
}

/// Which functions defined in `top` (all through) run as closures, and
/// which locals are in heap frames: one made a value, with `nonlocal`
/// names, or reading a variable in a heap frame is a closure; what a
/// closure reads is in a heap frame of the function it's a local of (as
/// many times as that changes something). Then where each closure's
/// captures are, up its env's frames.
fn resolveClosures(top: *Function, a: Allocator) ReadError!void {
    var all: std.ArrayList(*Function) = .empty;
    try collectFunctions(top, &all, a);
    for (all.items) |f| {
        f.heap = try a.alloc(?u32, f.locals.len);
        @memset(f.heap, null);
    }
    var changed = true;
    while (changed) {
        changed = false;
        for (all.items) |f| {
            const parent = f.parent orelse continue;
            if (!f.closure) {
                var reads_heap = false;
                for (f.captures) |c| if (ownerOf(parent, c)) |o| {
                    if (o.f.heap[o.slot] != null) reads_heap = true;
                };
                if (f.escapes or f.nonlocals or reads_heap) {
                    f.closure = true;
                    changed = true;
                }
            }
            if (f.closure) for (f.captures) |c| if (ownerOf(parent, c)) |o| {
                if (o.f.heap[o.slot] == null) {
                    o.f.heap[o.slot] = o.f.heap_len;
                    o.f.heap_len += 1;
                    changed = true;
                }
            };
        }
    }
    for (all.items) |f| {
        f.makes_frame = f.heap_len > 0;
        for (f.children) |ch| {
            if (ch.closure) f.makes_frame = true;
        }
        if (!f.closure) continue;
        // (given its own parameters and its env)
        f.param_count = f.own + 1;
        f.required = f.own + 1;
        const env = try a.alloc(Function.EnvRef, f.captures.len);
        for (f.captures, env) |c, *e| {
            // (up the functions it's in: the frame of the one it's a local
            // of; those between are closures reading it too)
            var p = f.parent.?;
            var depth: u32 = 0;
            while (true) {
                const slot = p.slotOf(c).?;
                if (p.ownsLocal(slot)) {
                    e.* = .{ .depth = depth, .index = p.heap[slot].? };
                    break;
                }
                p = p.parent.?;
                depth += 1;
            }
        }
        f.env = env;
    }
}

fn collectFunctions(f: *Function, out: *std.ArrayList(*Function), a: Allocator) ReadError!void {
    try out.append(a, f);
    for (f.children) |ch| try collectFunctions(ch, out, a);
}

/// The function `name` is a local of, from `f` up, and its slot there
fn ownerOf(f: *Function, name: []const u8) ?struct { f: *Function, slot: usize } {
    var p: ?*Function = f;
    while (p) |x| : (p = x.parent) {
        const slot = x.slotOf(name) orelse return null;
        if (x.ownsLocal(slot)) return .{ .f = x, .slot = slot };
    }
    return null;
}

const Reader = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator,
    failure: *Failure,
    line_offset: u32 = 0,
    col_offset: u32 = 0,
    locals: std.ArrayList([]const u8) = .empty,
    /// Names bound by the comprehensions being read, innermost last
    comp_scope: std.ArrayList(struct { name: []const u8, slot: u32 }) = .empty,
    /// Expressions read
    exprs: u32 = 0,
    /// The function this one is defined in (a nested def's reader), whose
    /// memory it uses
    parent: ?*Reader = null,
    /// The functions defined in this one and around it, by name, those
    /// read so far (and the one being read: its own calls)
    nested: std.ArrayList(Nested) = .empty,
    /// The function object read: its globals and closure the nested
    /// functions' too
    py_function: ?*PyObject = null,
    file: []const u8 = "",
    /// The names of the functions defined in this one (collectAssigned),
    /// read or not yet
    defs: std.ArrayList([]const u8) = .empty,
    /// The functions defined in this one (defs and lambdas), read
    children: std.ArrayList(*Function) = .empty,
    /// Python's name of the function read (Function.qualname)
    qualname: []const u8 = "",
    /// Its `global` and `nonlocal` names: not its locals
    global_names: std.ArrayList([]const u8) = .empty,
    nonlocal_names: std.ArrayList([]const u8) = .empty,

    /// A function defined in one (nestedDef): its own parameters' count,
    /// the variables around it it reads (its parameters after those)
    const Nested = struct { name: []const u8, func: *Function, own: usize, captures: []const []const u8 };

    fn alloc(self: *Reader) Allocator {
        if (self.parent) |p| return p.alloc();
        return self.arena.allocator();
    }

    /// Whether `name` is a function defined in this one or one around it
    /// not read yet (called before it's defined: not compiled)
    fn definesLater(self: *Reader, name: []const u8) bool {
        var r: ?*Reader = self;
        while (r) |x| : (r = x.parent) {
            if (contains(x.defs.items, name)) return true;
        }
        return false;
    }

    fn nestedNamed(self: *Reader, name: []const u8) ?Nested {
        var i = self.nested.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.nested.items[i].name, name)) return self.nested.items[i];
        }
        return null;
    }

    fn unsupported(self: *Reader, pos: Pos, comptime fmt: []const u8, args: anytype) ReadError {
        const s = std.fmt.bufPrint(&self.failure.message, fmt, args) catch self.failure.message[0..];
        self.failure.len = s.len;
        self.failure.pos = pos;
        return error.Unsupported;
    }

    fn posOf(self: *Reader, node: *PyObject) ReadError!Pos {
        const line = intAttrOr(node, "lineno", 0) catch return error.Python;
        const col = intAttrOr(node, "col_offset", 0) catch return error.Python;
        return .{ .line = @intCast(@max(line, 0) + self.line_offset), .col = @intCast(@max(col, 0) + 1 + self.col_offset) };
    }

    fn strAttr(self: *Reader, obj: *PyObject, name: [*:0]const u8) ReadError![]const u8 {
        const v = ph.attr(obj, name) orelse return error.Python;
        defer py.Py_DecRef(v);
        const s = ph.utf8(v, std.mem.span(name)) orelse return error.Python;
        return self.alloc().dupe(u8, s);
    }

    /// A slot for a local name (the existing one if declared).
    fn declare(self: *Reader, name: []const u8) ReadError!u32 {
        for (self.locals.items, 0..) |l, i| {
            if (std.mem.eql(u8, l, name)) return @intCast(i);
        }
        try self.locals.append(self.alloc(), try self.alloc().dupe(u8, name));
        return @intCast(self.locals.items.len - 1);
    }

    /// Whether assigning `name` makes it a local (not `global` or
    /// `nonlocal` in the function)
    fn bindsLocal(self: *Reader, name: []const u8) bool {
        return !contains(self.global_names.items, name) and !contains(self.nonlocal_names.items, name);
    }

    /// The `global` and `nonlocal` statements of a function's body (its
    /// own: not those of a def or class in it)
    fn scopeDecls(self: *Reader, stmts: *PyObject) ReadError!void {
        const n: usize = @intCast(py.c.PyList_Size(stmts));
        for (0..n) |i| {
            const s = py.c.PyList_GetItem(stmts, @intCast(i)).?;
            const k = try kindOf(s);
            if (eq(k, "FunctionDef") or eq(k, "AsyncFunctionDef") or eq(k, "ClassDef")) continue;
            if (eq(k, "Global") or eq(k, "Nonlocal")) {
                const names = try listAttr(s, "names");
                defer py.Py_DecRef(names);
                const out = if (eq(k, "Global")) &self.global_names else &self.nonlocal_names;
                for (0..@intCast(py.c.PyList_Size(names))) |j| {
                    const o = py.c.PyList_GetItem(names, @intCast(j)).?;
                    try out.append(self.alloc(), try self.alloc().dupe(u8, ph.utf8(o, "a name") orelse return error.Python));
                }
                continue;
            }
            inline for (.{ "body", "orelse", "finalbody" }) |f| {
                if (py.c.PyObject_HasAttrString(s, f) == 1) {
                    const l = try listAttr(s, f);
                    defer py.Py_DecRef(l);
                    try self.scopeDecls(l);
                }
            }
            if (eq(k, "Try")) {
                const hs = try listAttr(s, "handlers");
                defer py.Py_DecRef(hs);
                for (0..@intCast(py.c.PyList_Size(hs))) |j| {
                    const body = try listAttr(py.c.PyList_GetItem(hs, @intCast(j)).?, "body");
                    defer py.Py_DecRef(body);
                    try self.scopeDecls(body);
                }
            }
        }
    }

    fn lookupLocal(self: *Reader, name: []const u8) ?u32 {
        var i = self.comp_scope.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.comp_scope.items[i].name, name)) return self.comp_scope.items[i].slot;
        }
        for (self.locals.items, 0..) |l, j| {
            if (std.mem.eql(u8, l, name)) return @intCast(j);
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Locals
    // ------------------------------------------------------------------

    fn collectAssigned(self: *Reader, stmts: *PyObject) ReadError!void {
        const n: usize = @intCast(py.c.PyList_Size(stmts));
        for (0..n) |i| try self.collectStmt(py.c.PyList_GetItem(stmts, @intCast(i)).?);
    }

    fn collectStmt(self: *Reader, s: *PyObject) ReadError!void {
        const k = try kindOf(s);
        // (a def inside: its name a function of its own, its body its own)
        if (eq(k, "FunctionDef")) {
            try self.defs.append(self.alloc(), try self.strAttr(s, "name"));
            return;
        }
        // (what an import binds, what a del deletes: locals; a walrus's
        // targets in its expressions too)
        try self.collectNamed(s);
        if (eq(k, "Import") or eq(k, "ImportFrom")) {
            const names = try listAttr(s, "names");
            defer py.Py_DecRef(names);
            const n: usize = @intCast(py.c.PyList_Size(names));
            for (0..n) |i| _ = try self.declare(try self.importBound(py.c.PyList_GetItem(names, @intCast(i)).?, eq(k, "Import")));
            return;
        }
        if (eq(k, "Delete")) {
            const targets = try listAttr(s, "targets");
            defer py.Py_DecRef(targets);
            const n: usize = @intCast(py.c.PyList_Size(targets));
            for (0..n) |i| try self.collectTarget(py.c.PyList_GetItem(targets, @intCast(i)).?);
            return;
        }
        if (eq(k, "Assign")) {
            const targets = try listAttr(s, "targets");
            defer py.Py_DecRef(targets);
            const n: usize = @intCast(py.c.PyList_Size(targets));
            for (0..n) |i| try self.collectTarget(py.c.PyList_GetItem(targets, @intCast(i)).?);
        } else if (eq(k, "AugAssign") or eq(k, "AnnAssign")) {
            const t = ph.attr(s, "target") orelse return error.Python;
            defer py.Py_DecRef(t);
            try self.collectTarget(t);
        } else if (eq(k, "For")) {
            const t = ph.attr(s, "target") orelse return error.Python;
            defer py.Py_DecRef(t);
            try self.collectTarget(t);
        } else if (eq(k, "With")) {
            const items = try listAttr(s, "items");
            defer py.Py_DecRef(items);
            for (0..@intCast(py.c.PyList_Size(items))) |i| {
                const v = ph.attr(py.c.PyList_GetItem(items, @intCast(i)).?, "optional_vars") orelse return error.Python;
                defer py.Py_DecRef(v);
                if (v != py.Py_None()) try self.collectTarget(v);
            }
        }
        inline for (.{ "body", "orelse", "finalbody" }) |f| {
            if (py.c.PyObject_HasAttrString(s, f) == 1) {
                const l = try listAttr(s, f);
                defer py.Py_DecRef(l);
                try self.collectAssigned(l);
            }
        }
        // (a try's handlers: `as` names and bodies)
        if (eq(k, "Try")) {
            const hs = try listAttr(s, "handlers");
            defer py.Py_DecRef(hs);
            const n: usize = @intCast(py.c.PyList_Size(hs));
            for (0..n) |i| {
                const h = py.c.PyList_GetItem(hs, @intCast(i)).?;
                const name = ph.attr(h, "name") orelse return error.Python;
                defer py.Py_DecRef(name);
                if (name != py.Py_None()) {
                    const text = try self.alloc().dupe(u8, ph.utf8(name, "a name") orelse return error.Python);
                    if (self.lookupLocal(text) == null) _ = try self.declare(text);
                }
                const body = try listAttr(h, "body");
                defer py.Py_DecRef(body);
                try self.collectAssigned(body);
            }
        }
    }

    /// The targets of the walrus expressions of a statement (its own, not
    /// those of a def, lambda or class in it): locals of the function
    fn collectNamed(self: *Reader, node: *PyObject) ReadError!void {
        const ast_mod = py.c.PyImport_ImportModule("ast") orelse return error.Python;
        defer py.Py_DecRef(ast_mod);
        const kids_it = py.c.PyObject_CallMethod(ast_mod, "iter_child_nodes", "(O)", node) orelse return error.Python;
        defer py.Py_DecRef(kids_it);
        const kids = py.c.PySequence_List(kids_it) orelse return error.Python;
        defer py.Py_DecRef(kids);
        const n: usize = @intCast(py.c.PyList_Size(kids));
        for (0..n) |i| {
            const kid = py.c.PyList_GetItem(kids, @intCast(i)).?;
            const k = try kindOf(kid);
            if (eq(k, "FunctionDef") or eq(k, "AsyncFunctionDef") or eq(k, "Lambda") or eq(k, "ClassDef")) continue;
            // (a statement in it: its own visit, from collectAssigned)
            if (isStatement(k)) continue;
            if (eq(k, "NamedExpr")) {
                const t = ph.attr(kid, "target") orelse return error.Python;
                defer py.Py_DecRef(t);
                const name = try self.strAttr(t, "id");
                if (self.bindsLocal(name)) _ = try self.declare(name);
            }
            try self.collectNamed(kid);
        }
    }

    /// The name an import's alias binds: `as` its own; `import a.b` binds
    /// `a`; `from m import x` binds `x`
    fn importBound(self: *Reader, alias: *PyObject, plain_import: bool) ReadError![]const u8 {
        const as_o = ph.attr(alias, "asname") orelse return error.Python;
        defer py.Py_DecRef(as_o);
        if (as_o != py.Py_None()) return self.strAttr(alias, "asname");
        const name = try self.strAttr(alias, "name");
        if (plain_import) if (std.mem.indexOfScalar(u8, name, '.')) |dot| return name[0..dot];
        return name;
    }

    fn collectTarget(self: *Reader, t: *PyObject) ReadError!void {
        const k = try kindOf(t);
        if (eq(k, "Name")) {
            const name = try self.strAttr(t, "id");
            if (self.bindsLocal(name)) _ = try self.declare(name);
        } else if (eq(k, "Tuple") or eq(k, "List")) {
            const elts = try listAttr(t, "elts");
            defer py.Py_DecRef(elts);
            const n: usize = @intCast(py.c.PyList_Size(elts));
            for (0..n) |i| try self.collectTarget(py.c.PyList_GetItem(elts, @intCast(i)).?);
        }
    }

    // ------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------

    fn stmtList(self: *Reader, stmts: *PyObject) ReadError![]const Stmt {
        const n: usize = @intCast(py.c.PyList_Size(stmts));
        const out = try self.alloc().alloc(Stmt, n);
        for (out, 0..) |*o, i| o.* = try self.stmt(py.c.PyList_GetItem(stmts, @intCast(i)).?);
        return out;
    }

    fn stmtListAttr(self: *Reader, s: *PyObject, name: [*:0]const u8) ReadError![]const Stmt {
        const l = try listAttr(s, name);
        defer py.Py_DecRef(l);
        return self.stmtList(l);
    }

    fn stmt(self: *Reader, s: *PyObject) ReadError!Stmt {
        const pos = try self.posOf(s);
        const k = try kindOf(s);
        const kind: Stmt.Kind = blk: {
            if (eq(k, "Expr")) break :blk .{ .expr = try self.exprAttr(s, "value") };
            if (eq(k, "Assign")) {
                const targets = try listAttr(s, "targets");
                defer py.Py_DecRef(targets);
                const n: usize = @intCast(py.c.PyList_Size(targets));
                const ts = try self.alloc().alloc(Target, n);
                for (ts, 0..) |*t, i| t.* = try self.target(py.c.PyList_GetItem(targets, @intCast(i)).?);
                break :blk .{ .assign = .{ .targets = ts, .value = try self.exprAttr(s, "value") } };
            }
            if (eq(k, "AnnAssign")) {
                const value = ph.attr(s, "value") orelse return error.Python;
                defer py.Py_DecRef(value);
                const t = ph.attr(s, "target") orelse return error.Python;
                defer py.Py_DecRef(t);
                if (value == py.Py_None()) break :blk .pass;
                const ts = try self.alloc().alloc(Target, 1);
                ts[0] = try self.target(t);
                break :blk .{ .assign = .{ .targets = ts, .value = try self.expr(value) } };
            }
            if (eq(k, "AugAssign")) {
                const t = ph.attr(s, "target") orelse return error.Python;
                defer py.Py_DecRef(t);
                const op = ph.attr(s, "op") orelse return error.Python;
                defer py.Py_DecRef(op);
                break :blk .{ .aug = .{ .target = try self.target(t), .op = try self.binOp(op, pos), .value = try self.exprAttr(s, "value") } };
            }
            if (eq(k, "If")) break :blk .{ .if_ = .{ .test_ = try self.exprAttr(s, "test"), .body = try self.stmtListAttr(s, "body"), .else_ = try self.stmtListAttr(s, "orelse") } };
            if (eq(k, "While")) break :blk .{ .while_ = .{ .test_ = try self.exprAttr(s, "test"), .body = try self.stmtListAttr(s, "body"), .else_ = try self.stmtListAttr(s, "orelse") } };
            if (eq(k, "For")) {
                const t = ph.attr(s, "target") orelse return error.Python;
                defer py.Py_DecRef(t);
                break :blk .{ .for_ = .{ .target = try self.target(t), .iter = try self.exprAttr(s, "iter"), .body = try self.stmtListAttr(s, "body"), .else_ = try self.stmtListAttr(s, "orelse") } };
            }
            if (eq(k, "Return")) break :blk .{ .return_ = try self.optExprAttr(s, "value") };
            if (eq(k, "Raise")) {
                const cause = ph.attr(s, "cause") orelse return error.Python;
                defer py.Py_DecRef(cause);
                if (cause != py.Py_None()) break :blk .{ .raise_from = .{ .exc = try self.exprAttr(s, "exc"), .cause = try self.expr(cause) } };
                break :blk .{ .raise_ = try self.optExprAttr(s, "exc") };
            }
            if (eq(k, "Delete")) {
                const targets = try listAttr(s, "targets");
                defer py.Py_DecRef(targets);
                const n: usize = @intCast(py.c.PyList_Size(targets));
                const out = try self.alloc().alloc(Stmt, n);
                for (out, 0..) |*o, i| {
                    const t = py.c.PyList_GetItem(targets, @intCast(i)).?;
                    const tk = try kindOf(t);
                    o.pos = try self.posOf(t);
                    if (eq(tk, "Name")) {
                        const name = try self.strAttr(t, "id");
                        o.kind = .{ .del_local = self.lookupLocal(name) orelse try self.declare(name) };
                    } else if (eq(tk, "Subscript")) {
                        const sl = ph.attr(t, "slice") orelse return error.Python;
                        defer py.Py_DecRef(sl);
                        if (eq(try kindOf(sl), "Slice")) return self.unsupported(o.pos, "`del` of a slice can't be compiled", .{});
                        o.kind = .{ .del_item = .{ .obj = try self.exprAttr(t, "value"), .index = try self.indexExpr(sl) } };
                    } else return self.unsupported(o.pos, "`del` of {s} can't be compiled", .{tk});
                }
                break :blk if (n == 1) out[0].kind else .{ .seq = out };
            }
            if (eq(k, "Import") or eq(k, "ImportFrom")) {
                const plain = eq(k, "Import");
                var module: []const u8 = "";
                if (!plain) {
                    // (a relative one: its dots before the name, resolved
                    // with the function's package when compiling)
                    const level: usize = @intCast(try intAttrOr(s, "level", 0));
                    const m = ph.attr(s, "module") orelse return error.Python;
                    defer py.Py_DecRef(m);
                    const base = if (m == py.Py_None()) "" else try self.strAttr(s, "module");
                    const dots = try self.alloc().alloc(u8, level + base.len);
                    @memset(dots[0..level], '.');
                    @memcpy(dots[level..], base);
                    module = dots;
                }
                const names = try listAttr(s, "names");
                defer py.Py_DecRef(names);
                const n: usize = @intCast(py.c.PyList_Size(names));
                const out = try self.alloc().alloc(Stmt, n);
                for (out, 0..) |*o, i| {
                    const alias = py.c.PyList_GetItem(names, @intCast(i)).?;
                    const name = try self.strAttr(alias, "name");
                    if (eq(name, "*")) return self.unsupported(pos, "`from ... import *` can't be compiled", .{});
                    const as_o = ph.attr(alias, "asname") orelse return error.Python;
                    defer py.Py_DecRef(as_o);
                    const bound = try self.importBound(alias, plain);
                    const value = try self.new(pos, .{ .import_ = if (plain)
                        .{ .module = name, .leaf = as_o != py.Py_None() }
                    else
                        .{ .module = module, .attr = name } });
                    const targets = try self.alloc().alloc(Target, 1);
                    targets[0] = .{ .local = self.lookupLocal(bound) orelse try self.declare(bound) };
                    o.* = .{ .pos = pos, .kind = .{ .assign = .{ .targets = targets, .value = value } } };
                }
                break :blk if (n == 1) out[0].kind else .{ .seq = out };
            }
            if (eq(k, "Assert")) break :blk .{ .assert_ = .{ .test_ = try self.exprAttr(s, "test"), .msg = try self.optExprAttr(s, "msg") } };
            if (eq(k, "Try")) {
                const hs = try listAttr(s, "handlers");
                defer py.Py_DecRef(hs);
                const n: usize = @intCast(py.c.PyList_Size(hs));
                const handlers = try self.alloc().alloc(Handler, n);
                for (handlers, 0..) |*h, i| {
                    const ho = py.c.PyList_GetItem(hs, @intCast(i)).?;
                    const name = ph.attr(ho, "name") orelse return error.Python;
                    defer py.Py_DecRef(name);
                    var slot: ?u32 = null;
                    if (name != py.Py_None()) {
                        const text = try self.alloc().dupe(u8, ph.utf8(name, "a name") orelse return error.Python);
                        slot = self.lookupLocal(text) orelse try self.declare(text);
                    }
                    h.* = .{ .pos = try self.posOf(ho), .type_ = try self.optExprAttr(ho, "type"), .name = slot, .body = try self.stmtListAttr(ho, "body") };
                }
                break :blk .{ .try_ = .{ .body = try self.stmtListAttr(s, "body"), .handlers = handlers, .else_ = try self.stmtListAttr(s, "orelse"), .finally = try self.stmtListAttr(s, "finalbody") } };
            }
            if (eq(k, "Break")) break :blk .break_;
            if (eq(k, "Continue")) break :blk .continue_;
            if (eq(k, "Pass")) break :blk .pass;
            // (read before the body: scopeDecls)
            if (eq(k, "Global") or eq(k, "Nonlocal")) break :blk .pass;
            if (eq(k, "With")) {
                const items = try listAttr(s, "items");
                defer py.Py_DecRef(items);
                const body = try listAttr(s, "body");
                defer py.Py_DecRef(body);
                break :blk .{ .seq = try self.withItems(items, 0, body, pos) };
            }
            // (a def inside: read as a function of its own; nothing to run
            // where it's defined)
            if (eq(k, "FunctionDef")) {
                try self.nestedDef(s, pos);
                break :blk .pass;
            }
            return self.unsupported(pos, "{s} can't be compiled", .{stmtName(k)});
        };
        return .{ .pos = pos, .kind = kind };
    }

    /// `with item[i], ...: body`, as PEP 343 runs it (the rest of the items
    /// inside, in turn):
    ///
    ///     mgr = EXPR
    ///     value = mgr.__enter__()
    ///     normal = True
    ///     try:
    ///         try:
    ///             VAR = value
    ///             BODY
    ///         except BaseException as e:
    ///             normal = False
    ///             if not mgr.__exit__(type(e), e, e.__traceback__):
    ///                 raise
    ///     finally:
    ///         if normal:
    ///             mgr.__exit__(None, None, None)
    ///
    /// (the methods called as methods: a record's compiled, as any; its
    /// variables locals of their own, named so no Python name is them)
    fn withItems(self: *Reader, items: *PyObject, i: usize, body: *PyObject, pos: Pos) ReadError![]const Stmt {
        const n: usize = @intCast(py.c.PyList_Size(items));
        if (i == n) return self.stmtList(body);
        const item = py.c.PyList_GetItem(items, @intCast(i)).?;
        const id = self.locals.items.len;
        const mgr = try self.declare(try std.fmt.allocPrint(self.alloc(), "<with {d} mgr>", .{id}));
        const val = try self.declare(try std.fmt.allocPrint(self.alloc(), "<with {d} value>", .{id}));
        const normal = try self.declare(try std.fmt.allocPrint(self.alloc(), "<with {d} normal>", .{id}));
        const exc = try self.declare(try std.fmt.allocPrint(self.alloc(), "<with {d} exc>", .{id}));
        const B = struct {
            r: *Reader,
            pos: Pos,
            fn e(b: @This(), kind: Expr.Kind) ReadError!*Expr {
                return b.r.new(b.pos, kind);
            }
            fn local(b: @This(), slot: u32) ReadError!*Expr {
                return b.e(.{ .local = slot });
            }
            fn call(b: @This(), func: *Expr, args: []const *Expr) ReadError!*Expr {
                return b.e(.{ .call = .{ .func = func, .args = try b.r.alloc().dupe(*Expr, args), .keywords = &.{} } });
            }
            fn typeOf(b: @This(), x: *Expr) ReadError!*Expr {
                return b.call(try b.e(.{ .global = "type" }), &.{x});
            }
            fn assign(b: @This(), slot: u32, v: *Expr) ReadError!Stmt {
                const ts = try b.r.alloc().alloc(Target, 1);
                ts[0] = .{ .local = slot };
                return .{ .pos = b.pos, .kind = .{ .assign = .{ .targets = ts, .value = v } } };
            }
            fn list(b: @This(), stmts: []const Stmt) ReadError![]const Stmt {
                return b.r.alloc().dupe(Stmt, stmts);
            }
        };
        const b = B{ .r = self, .pos = pos };
        // (the item's expression and target, read in the function's scope)
        const ctx_e = try self.exprAttr(item, "context_expr");
        const vars = ph.attr(item, "optional_vars") orelse return error.Python;
        defer py.Py_DecRef(vars);
        const target_ = if (vars == py.Py_None()) null else try self.target(vars);
        // The inner try's body: VAR = value, the rest
        const rest = try self.withItems(items, i + 1, body, pos);
        var inner: std.ArrayList(Stmt) = .empty;
        if (target_) |t| {
            const ts = try self.alloc().alloc(Target, 1);
            ts[0] = t;
            try inner.append(self.alloc(), .{ .pos = pos, .kind = .{ .assign = .{ .targets = ts, .value = try b.local(val) } } });
        }
        try inner.appendSlice(self.alloc(), rest);
        // except BaseException as e: normal = False; if not exit(...): raise
        const exit_m = try b.e(.{ .attr = .{ .obj = try b.local(mgr), .name = "__exit__" } });
        const exit_call = try b.call(exit_m, &.{ try b.typeOf(try b.local(exc)), try b.local(exc), try b.e(.{ .attr = .{ .obj = try b.local(exc), .name = "__traceback__" } }) });
        const reraise = try b.list(&.{.{ .pos = pos, .kind = .{ .raise_ = null } }});
        const handler_body = try b.list(&.{
            try b.assign(normal, try b.e(.{ .bool = false })),
            .{ .pos = pos, .kind = .{ .if_ = .{ .test_ = try b.e(.{ .unary = .{ .op = .not_, .operand = exit_call } }), .body = reraise, .else_ = &.{} } } },
        });
        const handlers = try self.alloc().alloc(Handler, 1);
        handlers[0] = .{ .pos = pos, .type_ = try b.e(.{ .global = "BaseException" }), .name = exc, .body = handler_body };
        const inner_try = Stmt{ .pos = pos, .kind = .{ .try_ = .{ .body = inner.items, .handlers = handlers, .else_ = &.{}, .finally = &.{} } } };
        // finally: if normal: exit(mgr, None, None, None)
        const none = try b.e(.none);
        const normal_exit = try b.list(&.{.{ .pos = pos, .kind = .{ .expr = try b.call(try b.e(.{ .attr = .{ .obj = try b.local(mgr), .name = "__exit__" } }), &.{ none, none, none }) } }});
        const finally = try b.list(&.{.{ .pos = pos, .kind = .{ .if_ = .{ .test_ = try b.local(normal), .body = normal_exit, .else_ = &.{} } } }});
        const outer_try = Stmt{ .pos = pos, .kind = .{ .try_ = .{ .body = try b.list(&.{inner_try}), .handlers = &.{}, .else_ = &.{}, .finally = finally } } };
        return b.list(&.{
            try b.assign(mgr, ctx_e),
            try b.assign(val, try b.call(try b.e(.{ .attr = .{ .obj = try b.local(mgr), .name = "__enter__" } }), &.{})),
            try b.assign(normal, try b.e(.{ .bool = true })),
            outer_try,
        });
    }

    /// A def inside the function: read as a function of its own
    /// (nestedFunction), its name the functions around it call (from here
    /// on) or read as a value.
    fn nestedDef(self: *Reader, def: *PyObject, pos: Pos) ReadError!void {
        const decos = try listAttr(def, "decorator_list");
        defer py.Py_DecRef(decos);
        if (py.c.PyList_Size(decos) != 0) return self.unsupported(try self.posOf(py.c.PyList_GetItem(decos, 0).?), "a nested def with decorators can't be compiled", .{});
        _ = try self.nestedFunction(def, try self.strAttr(def, "name"), pos, false);
    }

    /// A def or lambda inside the function, read as a function of its own
    /// (lambda lifting): its parameters, then its env (a closure's: the
    /// heap frame of the run of this one it's made in), then the variables
    /// of the functions around it that it reads (or that the ones defined
    /// there it calls read), then what it assigns. Called where it's defined
    /// (call_nested), it's given those variables as they are then, what a
    /// closure would read then; a closure (resolveClosures) reads them in
    /// their heap frames.
    fn nestedFunction(self: *Reader, node: *PyObject, name: []const u8, pos: Pos, lambda: bool) ReadError!*Function {
        const what = if (lambda) "a lambda" else "a nested def";
        const args = ph.attr(node, "args") orelse return error.Python;
        defer py.Py_DecRef(args);
        inline for (.{ "posonlyargs", "kwonlyargs", "kw_defaults", "defaults" }) |field| {
            const l = try listAttr(args, field);
            defer py.Py_DecRef(l);
            if (py.c.PyList_Size(l) != 0) return self.unsupported(pos, "{s}'s parameters must be plain ones (no defaults, keyword-only or positional-only ones)", .{what});
        }
        inline for (.{ "vararg", "kwarg" }) |field| {
            const o = ph.attr(args, field) orelse return error.Python;
            defer py.Py_DecRef(o);
            if (o != py.Py_None()) return self.unsupported(pos, "{s}'s *args and **kwargs can't be compiled", .{what});
        }
        var sub = Reader{
            .gpa = self.gpa,
            .arena = std.heap.ArenaAllocator.init(self.gpa),
            .failure = self.failure,
            .line_offset = self.line_offset,
            .col_offset = self.col_offset,
            .parent = self,
            .py_function = self.py_function,
            .file = self.file,
            .qualname = try std.fmt.allocPrint(self.alloc(), "{s}.<locals>.{s}", .{ self.qualname, name }),
        };
        // Its own: its parameters, what it assigns (not its `global` and
        // `nonlocal` names)
        const params = try listAttr(args, "args");
        defer py.Py_DecRef(params);
        const n_own: usize = @intCast(py.c.PyList_Size(params));
        for (0..n_own) |i| _ = try sub.declare(try sub.strAttr(py.c.PyList_GetItem(params, @intCast(i)).?, "arg"));
        // (a lambda's body: an expression, its walruses its own)
        const body: ?*PyObject = if (lambda) null else try listAttr(node, "body");
        defer if (body) |b| py.Py_DecRef(b);
        if (body) |b| {
            try sub.scopeDecls(b);
            try sub.collectAssigned(b);
        } else try sub.collectNamed(node);
        // What it reads of the functions around it: those variables, and
        // what the functions defined there that it calls read
        var captures: std.ArrayList([]const u8) = .empty;
        for (try self.loadedNames(node)) |n| {
            if ((!lambda and eq(n, name)) or sub.lookupLocal(n) != null or contains(sub.global_names.items, n)) continue;
            if (self.lookupLocal(n) != null) {
                if (!contains(captures.items, n)) try captures.append(self.alloc(), n);
            } else if (self.nestedNamed(n)) |other| {
                for (other.captures) |c| if (!contains(captures.items, c)) try captures.append(self.alloc(), c);
            }
        }
        for (sub.nonlocal_names.items) |n| {
            if (!contains(captures.items, n)) return self.unsupported(pos, "no variable {s} around {s} for its `nonlocal`", .{ n, name });
        }
        // (its locals again, in a function's order: its parameters, its
        // env, those, then what it assigns)
        sub.locals.shrinkRetainingCapacity(n_own);
        sub.defs.clearRetainingCapacity();
        _ = try sub.declare("");
        for (captures.items) |c| _ = try sub.declare(c);
        if (body) |b| try sub.collectAssigned(b) else try sub.collectNamed(node);
        const f = try self.alloc().create(Function);
        // (visible to the functions around it from here, to itself, and to
        // those defined in it)
        if (!lambda) try self.nested.append(self.alloc(), .{ .name = name, .func = f, .own = n_own, .captures = captures.items });
        try sub.nested.appendSlice(self.alloc(), self.nested.items);
        const out = if (body) |b| try sub.stmtList(b) else blk: {
            const ret = try self.alloc().alloc(Stmt, 1);
            ret[0] = .{ .pos = pos, .kind = .{ .return_ = try sub.exprAttr(node, "body") } };
            break :blk ret;
        };
        const n_params: u32 = @intCast(n_own + 1 + captures.items.len);
        f.* = .{
            // (its memory the outermost function's: its own arena empty)
            .arena = sub.arena,
            .name = name,
            .file = self.file,
            .first_line = pos.line,
            .param_count = n_params,
            .required = n_params,
            .locals = sub.locals.items,
            .body = out,
            .py_function = self.py_function.?,
            .size = sub.exprs,
            .own = @intCast(n_own),
            .qualname = sub.qualname,
            .captures = captures.items,
            .nonlocals = sub.nonlocal_names.items.len > 0,
            .children = sub.children.items,
        };
        for (sub.children.items) |ch| ch.parent = f;
        try self.children.append(self.alloc(), f);
        return f;
    }

    /// The names an AST node (a nested def) reads, all through, in order
    /// (a `nonlocal` name's too: it reads and writes the variable around)
    fn loadedNames(self: *Reader, node: *PyObject) ReadError![]const []const u8 {
        const ast_mod = py.c.PyImport_ImportModule("ast") orelse return error.Python;
        defer py.Py_DecRef(ast_mod);
        const walk = py.c.PyObject_CallMethod(ast_mod, "walk", "(O)", node) orelse return error.Python;
        defer py.Py_DecRef(walk);
        const all = py.c.PySequence_List(walk) orelse return error.Python;
        defer py.Py_DecRef(all);
        var out: std.ArrayList([]const u8) = .empty;
        const n: usize = @intCast(py.c.PyList_Size(all));
        for (0..n) |i| {
            const x = py.c.PyList_GetItem(all, @intCast(i)).?;
            if (eq(try kindOf(x), "Nonlocal")) {
                const names = try listAttr(x, "names");
                defer py.Py_DecRef(names);
                for (0..@intCast(py.c.PyList_Size(names))) |j| {
                    const id = try self.alloc().dupe(u8, ph.utf8(py.c.PyList_GetItem(names, @intCast(j)).?, "a name") orelse return error.Python);
                    if (!contains(out.items, id)) try out.append(self.alloc(), id);
                }
                continue;
            }
            if (!eq(try kindOf(x), "Name")) continue;
            const ctx = ph.attr(x, "ctx") orelse return error.Python;
            defer py.Py_DecRef(ctx);
            if (!eq(try kindOf(ctx), "Load")) continue;
            const id = try self.strAttr(x, "id");
            if (!contains(out.items, id)) try out.append(self.alloc(), id);
        }
        return out.items;
    }

    fn target(self: *Reader, t: *PyObject) ReadError!Target {
        const k = try kindOf(t);
        if (eq(k, "Name")) {
            const name = try self.strAttr(t, "id");
            if (contains(self.global_names.items, name)) return .{ .global = name };
            return .{ .local = self.lookupLocal(name) orelse try self.declare(name) };
        }
        if (eq(k, "Tuple") or eq(k, "List")) {
            const elts = try listAttr(t, "elts");
            defer py.Py_DecRef(elts);
            const n: usize = @intCast(py.c.PyList_Size(elts));
            const ts = try self.alloc().alloc(Target, n);
            for (ts, 0..) |*x, i| x.* = try self.target(py.c.PyList_GetItem(elts, @intCast(i)).?);
            return .{ .tuple = ts };
        }
        if (eq(k, "Attribute")) return .{ .attr = .{ .obj = try self.exprAttr(t, "value"), .name = try self.strAttr(t, "attr") } };
        if (eq(k, "Subscript")) {
            const sl = ph.attr(t, "slice") orelse return error.Python;
            defer py.Py_DecRef(sl);
            if (eq(try kindOf(sl), "Slice")) return self.unsupported(try self.posOf(t), "assigning to a slice can't be compiled", .{});
            return .{ .index = .{ .obj = try self.exprAttr(t, "value"), .index = try self.indexExpr(sl) } };
        }
        return self.unsupported(try self.posOf(t), "assigning to {s} can't be compiled", .{k});
    }

    // ------------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------------

    fn exprAttr(self: *Reader, obj: *PyObject, name: [*:0]const u8) ReadError!*Expr {
        const v = ph.attr(obj, name) orelse return error.Python;
        defer py.Py_DecRef(v);
        return self.expr(v);
    }

    fn optExprAttr(self: *Reader, obj: *PyObject, name: [*:0]const u8) ReadError!?*Expr {
        const v = ph.attr(obj, name) orelse return error.Python;
        defer py.Py_DecRef(v);
        if (v == py.Py_None()) return null;
        return try self.expr(v);
    }

    /// A folded set literal's items' order in the frozenset CPython made of
    /// them (folded_order), or null if it isn't found
    fn foldedOrder(self: *Reader, elts: *PyObject, n: usize) ReadError!?[]const usize {
        const f = helper("folded_order") orelse return error.Python;
        const r = py.c.PyObject_CallFunctionObjArgs(f, self.py_function, elts, @as(?*PyObject, null)) orelse {
            py.c.PyErr_Clear();
            return null;
        };
        defer py.Py_DecRef(r);
        if (r == py.Py_None() or py.c.PyList_Size(r) != @as(isize, @intCast(n))) return null;
        const out = try self.alloc().alloc(usize, n);
        for (out, 0..) |*slot, i| {
            const x = py.c.PyLong_AsLongLong(py.c.PyList_GetItem(r, @intCast(i)).?);
            if (x < 0 or x >= n) {
                py.c.PyErr_Clear();
                return null;
            }
            slot.* = @intCast(x);
        }
        return out;
    }

    fn exprList(self: *Reader, obj: *PyObject, name: [*:0]const u8) ReadError![]const *Expr {
        const l = try listAttr(obj, name);
        defer py.Py_DecRef(l);
        const n: usize = @intCast(py.c.PyList_Size(l));
        const out = try self.alloc().alloc(*Expr, n);
        for (out, 0..) |*o, i| {
            const item = py.c.PyList_GetItem(l, @intCast(i)).?;
            if (eq(try kindOf(item), "Starred")) return self.unsupported(try self.posOf(item), "*unpacking can't be compiled", .{});
            o.* = try self.expr(item);
        }
        return out;
    }

    /// A call's positional arguments, `*x` among them (starred)
    fn callArgs(self: *Reader, call: *PyObject) ReadError![]const *Expr {
        const l = try listAttr(call, "args");
        defer py.Py_DecRef(l);
        const n: usize = @intCast(py.c.PyList_Size(l));
        const out = try self.alloc().alloc(*Expr, n);
        for (out, 0..) |*o, i| {
            const item = py.c.PyList_GetItem(l, @intCast(i)).?;
            o.* = if (eq(try kindOf(item), "Starred"))
                try self.new(try self.posOf(item), .{ .starred = try self.exprAttr(item, "value") })
            else
                try self.expr(item);
        }
        return out;
    }

    fn new(self: *Reader, pos: Pos, kind: Expr.Kind) ReadError!*Expr {
        const e = try self.alloc().create(Expr);
        e.* = .{ .pos = pos, .kind = kind };
        self.exprs += 1;
        return e;
    }

    /// A subscript's index (Python 3.8's Index wrapper unwrapped).
    fn indexExpr(self: *Reader, sl: *PyObject) ReadError!*Expr {
        if (eq(try kindOf(sl), "Index")) return self.exprAttr(sl, "value");
        return self.expr(sl);
    }

    fn expr(self: *Reader, e: *PyObject) ReadError!*Expr {
        const pos = try self.posOf(e);
        const k = try kindOf(e);
        if (eq(k, "Constant")) return self.constant(e, pos);
        if (eq(k, "Name")) {
            const name = try self.strAttr(e, "id");
            if (self.lookupLocal(name)) |slot| return self.new(pos, .{ .local = slot });
            // (a def's name as a value: a closure of it)
            if (self.nestedNamed(name)) |n| {
                n.func.escapes = true;
                return self.new(pos, .{ .make_closure = n.func });
            }
            if (self.definesLater(name))
                return self.unsupported(pos, "the nested function {s} used before its def can't be compiled", .{name});
            return self.new(pos, .{ .global = name });
        }
        if (eq(k, "Lambda")) {
            const f = try self.nestedFunction(e, "<lambda>", pos, true);
            f.escapes = true;
            return self.new(pos, .{ .make_closure = f });
        }
        if (eq(k, "Attribute")) return self.new(pos, .{ .attr = .{ .obj = try self.exprAttr(e, "value"), .name = try self.strAttr(e, "attr") } });
        if (eq(k, "Subscript")) {
            const sl = ph.attr(e, "slice") orelse return error.Python;
            defer py.Py_DecRef(sl);
            const obj = try self.exprAttr(e, "value");
            if (eq(try kindOf(sl), "Slice")) {
                return self.new(pos, .{ .slice = .{ .obj = obj, .lo = try self.optExprAttr(sl, "lower"), .hi = try self.optExprAttr(sl, "upper"), .step = try self.optExprAttr(sl, "step") } });
            }
            return self.new(pos, .{ .index = .{ .obj = obj, .index = try self.indexExpr(sl) } });
        }
        if (eq(k, "Call")) {
            // (a function defined in this one, or one around it: its code,
            // given what it reads around it after its arguments)
            const func_o = ph.attr(e, "func") orelse return error.Python;
            defer py.Py_DecRef(func_o);
            if (eq(try kindOf(func_o), "Name")) {
                const fname = try self.strAttr(func_o, "id");
                if (self.lookupLocal(fname) == null) if (self.nestedNamed(fname)) |n| {
                    const kw = try listAttr(e, "keywords");
                    defer py.Py_DecRef(kw);
                    if (py.c.PyList_Size(kw) != 0) return self.unsupported(pos, "keyword arguments to the nested function {s} can't be compiled", .{fname});
                    const own = try self.exprList(e, "args");
                    if (own.len != n.own) return self.unsupported(pos, "{s}() takes {d} arguments", .{ fname, n.own });
                    // (its own arguments, None for its env (a closure's: the
                    // compiler gives it), the captures' values)
                    const all = try self.alloc().alloc(*Expr, own.len + 1 + n.captures.len);
                    @memcpy(all[0..own.len], own);
                    all[own.len] = try self.new(pos, .none);
                    for (n.captures, all[own.len + 1 ..]) |c, *slot| {
                        const s = self.lookupLocal(c) orelse return self.unsupported(pos, "{s} isn't reachable where the nested function {s} is called", .{ c, fname });
                        slot.* = try self.new(pos, .{ .local = s });
                    }
                    return self.new(pos, .{ .call_nested = .{ .func = n.func, .args = all } });
                };
            }
            const func = try self.exprAttr(e, "func");
            const args = try self.callArgs(e);
            const kws = try listAttr(e, "keywords");
            defer py.Py_DecRef(kws);
            const n: usize = @intCast(py.c.PyList_Size(kws));
            const keywords = try self.alloc().alloc(Keyword, n);
            for (keywords, 0..) |*kw, i| {
                const item = py.c.PyList_GetItem(kws, @intCast(i)).?;
                const arg = ph.attr(item, "arg") orelse return error.Python;
                defer py.Py_DecRef(arg);
                if (arg == py.Py_None()) return self.unsupported(pos, "**unpacking can't be compiled", .{});
                kw.* = .{ .name = try self.strAttr(item, "arg"), .value = try self.exprAttr(item, "value") };
            }
            return self.new(pos, .{ .call = .{ .func = func, .args = args, .keywords = keywords } });
        }
        // (x := value): the function's local x (a comprehension's too)
        if (eq(k, "NamedExpr")) {
            const t = ph.attr(e, "target") orelse return error.Python;
            defer py.Py_DecRef(t);
            const name = try self.strAttr(t, "id");
            if (contains(self.global_names.items, name)) return self.unsupported(pos, "`:=` to a `global` name can't be compiled", .{});
            const slot = for (self.locals.items, 0..) |l, i| {
                if (eq(l, name)) break @as(u32, @intCast(i));
            } else try self.declare(name);
            return self.new(pos, .{ .named = .{ .slot = slot, .value = try self.exprAttr(e, "value") } });
        }
        if (eq(k, "BinOp")) {
            const op = ph.attr(e, "op") orelse return error.Python;
            defer py.Py_DecRef(op);
            return self.new(pos, .{ .binary = .{ .op = try self.binOp(op, pos), .left = try self.exprAttr(e, "left"), .right = try self.exprAttr(e, "right") } });
        }
        if (eq(k, "UnaryOp")) {
            const op = ph.attr(e, "op") orelse return error.Python;
            defer py.Py_DecRef(op);
            const ok = try kindOf(op);
            const u: UnaryOp = if (eq(ok, "USub")) .neg else if (eq(ok, "UAdd")) .pos else if (eq(ok, "Not")) .not_ else .invert;
            return self.new(pos, .{ .unary = .{ .op = u, .operand = try self.exprAttr(e, "operand") } });
        }
        if (eq(k, "BoolOp")) {
            const op = ph.attr(e, "op") orelse return error.Python;
            defer py.Py_DecRef(op);
            const values = try self.exprList(e, "values");
            return self.new(pos, if (eq(try kindOf(op), "And")) .{ .and_ = values } else .{ .or_ = values });
        }
        if (eq(k, "Compare")) {
            const ops_l = try listAttr(e, "ops");
            defer py.Py_DecRef(ops_l);
            const n: usize = @intCast(py.c.PyList_Size(ops_l));
            const ops = try self.alloc().alloc(CmpOp, n);
            for (ops, 0..) |*o, i| o.* = try self.cmpOp(py.c.PyList_GetItem(ops_l, @intCast(i)).?);
            return self.new(pos, .{ .compare = .{ .first = try self.exprAttr(e, "left"), .ops = ops, .rest = try self.exprList(e, "comparators") } });
        }
        if (eq(k, "IfExp")) return self.new(pos, .{ .cond = .{ .test_ = try self.exprAttr(e, "test"), .then = try self.exprAttr(e, "body"), .else_ = try self.exprAttr(e, "orelse") } });
        if (eq(k, "List")) return self.new(pos, .{ .list = try self.exprList(e, "elts") });
        if (eq(k, "Tuple")) return self.new(pos, .{ .tuple = try self.exprList(e, "elts") });
        if (eq(k, "Set")) {
            // (more than two constants: CPython makes a frozenset of them,
            // the set its copy: its table as that makes it)
            const elts = try listAttr(e, "elts");
            defer py.Py_DecRef(elts);
            const n: usize = @intCast(py.c.PyList_Size(elts));
            var folded = n > 2;
            for (0..n) |i| {
                if (!try isConstant(py.c.PyList_GetItem(elts, @intCast(i)).?)) folded = false;
            }
            const items = try self.exprList(e, "elts");
            // (folded: its items in the order of CPython's frozenset of them
            // (constants: when they're made doesn't matter), the frozenset
            // compiled code makes then laid out as that one)
            if (folded) if (try self.foldedOrder(elts, n)) |order| {
                const sorted = try self.alloc().alloc(*Expr, n);
                for (sorted, order) |*slot, i| slot.* = items[i];
                return self.new(pos, .{ .set_ = .{ .items = sorted, .folded = true } });
            };
            return self.new(pos, .{ .set_ = .{ .items = items, .folded = folded } });
        }
        if (eq(k, "SetComp")) {
            const mark = self.comp_scope.items.len;
            defer self.comp_scope.shrinkRetainingCapacity(mark);
            const gens = try self.generators(e);
            return self.new(pos, .{ .set_comp = .{ .elt = try self.exprAttr(e, "elt"), .generators = gens } });
        }
        if (eq(k, "Dict")) {
            const keys_l = try listAttr(e, "keys");
            defer py.Py_DecRef(keys_l);
            const n: usize = @intCast(py.c.PyList_Size(keys_l));
            for (0..n) |i| {
                if (py.c.PyList_GetItem(keys_l, @intCast(i)).? == py.Py_None()) return self.unsupported(pos, "**unpacking can't be compiled", .{});
            }
            return self.new(pos, .{ .dict = .{ .keys = try self.exprList(e, "keys"), .values = try self.exprList(e, "values") } });
        }
        if (eq(k, "ListComp") or eq(k, "GeneratorExp")) {
            const mark = self.comp_scope.items.len;
            defer self.comp_scope.shrinkRetainingCapacity(mark);
            const gens = try self.generators(e);
            const comp = Comp{ .elt = try self.exprAttr(e, "elt"), .generators = gens };
            return self.new(pos, if (eq(k, "ListComp")) .{ .list_comp = comp } else .{ .gen_exp = comp });
        }
        if (eq(k, "DictComp")) {
            const mark = self.comp_scope.items.len;
            defer self.comp_scope.shrinkRetainingCapacity(mark);
            const gens = try self.generators(e);
            return self.new(pos, .{ .dict_comp = .{ .key = try self.exprAttr(e, "key"), .value = try self.exprAttr(e, "value"), .generators = gens } });
        }
        if (eq(k, "JoinedStr")) return self.new(pos, .{ .fstring = try self.fparts(e) });
        if (eq(k, "FormattedValue")) {
            const parts = try self.alloc().alloc(FPart, 1);
            parts[0] = try self.fvalue(e);
            return self.new(pos, .{ .fstring = parts });
        }
        return self.unsupported(pos, "{s} can't be compiled", .{exprName(k)});
    }

    fn constant(self: *Reader, e: *PyObject, pos: Pos) ReadError!*Expr {
        const v = ph.attr(e, "value") orelse return error.Python;
        defer py.Py_DecRef(v);
        if (v == py.Py_None()) return self.new(pos, .none);
        if (py.PyBool_Check(v)) return self.new(pos, .{ .bool = v == py.Py_True() });
        if (py.PyLong_Check(v)) {
            var overflow: c_int = 0;
            const n = py.c.PyLong_AsLongLongAndOverflow(v, &overflow);
            if (overflow != 0) {
                // (the AST's: kept, as the function read is)
                py.Py_IncRef(v);
                return self.new(pos, .{ .big = v });
            }
            return self.new(pos, .{ .int = n });
        }
        if (py.PyFloat_Check(v)) return self.new(pos, .{ .float = py.c.PyFloat_AsDouble(v) });
        if (py.PyUnicode_Check(v)) {
            const s = ph.utf8(v, "a string") orelse return error.Python;
            return self.new(pos, .{ .str = try self.alloc().dupe(u8, s) });
        }
        py.Py_IncRef(v);
        return self.new(pos, .{ .object = v });
    }

    fn generators(self: *Reader, e: *PyObject) ReadError![]const Generator {
        const gens = try listAttr(e, "generators");
        defer py.Py_DecRef(gens);
        const n: usize = @intCast(py.c.PyList_Size(gens));
        const out = try self.alloc().alloc(Generator, n);
        for (out, 0..) |*g, i| {
            const item = py.c.PyList_GetItem(gens, @intCast(i)).?;
            const is_async = intAttrOr(item, "is_async", 0) catch return error.Python;
            if (is_async != 0) return self.unsupported(try self.posOf(e), "async comprehensions can't be compiled", .{});
            // The iterable is read outside the comprehension's names; the
            // target binds new ones (a slot of their own)
            const iter = try self.exprAttr(item, "iter");
            const t = ph.attr(item, "target") orelse return error.Python;
            defer py.Py_DecRef(t);
            const tgt = try self.compTarget(t);
            g.* = .{ .target = tgt, .iter = iter, .ifs = try self.exprList(item, "ifs") };
        }
        return out;
    }

    fn compTarget(self: *Reader, t: *PyObject) ReadError!Target {
        const k = try kindOf(t);
        if (eq(k, "Name")) {
            const name = try self.strAttr(t, "id");
            // A fresh slot: the comprehension's variable doesn't leak
            try self.locals.append(self.alloc(), name);
            const slot: u32 = @intCast(self.locals.items.len - 1);
            try self.comp_scope.append(self.alloc(), .{ .name = name, .slot = slot });
            return .{ .local = slot };
        }
        if (eq(k, "Tuple") or eq(k, "List")) {
            const elts = try listAttr(t, "elts");
            defer py.Py_DecRef(elts);
            const n: usize = @intCast(py.c.PyList_Size(elts));
            const ts = try self.alloc().alloc(Target, n);
            for (ts, 0..) |*x, i| x.* = try self.compTarget(py.c.PyList_GetItem(elts, @intCast(i)).?);
            return .{ .tuple = ts };
        }
        return self.unsupported(try self.posOf(t), "a comprehension's variable must be a name or a tuple of names", .{});
    }

    fn fparts(self: *Reader, joined: *PyObject) ReadError![]const FPart {
        const values = try listAttr(joined, "values");
        defer py.Py_DecRef(values);
        const n: usize = @intCast(py.c.PyList_Size(values));
        const out = try self.alloc().alloc(FPart, n);
        for (out, 0..) |*p, i| {
            const v = py.c.PyList_GetItem(values, @intCast(i)).?;
            if (eq(try kindOf(v), "Constant")) {
                const c = ph.attr(v, "value") orelse return error.Python;
                defer py.Py_DecRef(c);
                const s = ph.utf8(c, "f-string text") orelse return error.Python;
                p.* = .{ .text = try self.alloc().dupe(u8, s) };
            } else p.* = try self.fvalue(v);
        }
        return out;
    }

    fn fvalue(self: *Reader, v: *PyObject) ReadError!FPart {
        const conversion = intAttrOr(v, "conversion", -1) catch return error.Python;
        const spec_obj = ph.attr(v, "format_spec") orelse return error.Python;
        defer py.Py_DecRef(spec_obj);
        const spec: []const FPart = if (spec_obj == py.Py_None()) &.{} else try self.fparts(spec_obj);
        return .{ .value = .{
            .expr = try self.exprAttr(v, "value"),
            .conversion = if (conversion < 0) 0 else @intCast(conversion),
            .spec = spec,
        } };
    }

    fn binOp(self: *Reader, op: *PyObject, pos: Pos) ReadError!BinOp {
        const k = try kindOf(op);
        const table = .{
            .{ "Add", BinOp.add },     .{ "Sub", BinOp.sub },           .{ "Mult", BinOp.mul },
            .{ "Div", BinOp.div },     .{ "FloorDiv", BinOp.floordiv }, .{ "Mod", BinOp.mod },
            .{ "Pow", BinOp.pow },     .{ "LShift", BinOp.lshift },     .{ "RShift", BinOp.rshift },
            .{ "BitOr", BinOp.bitor }, .{ "BitXor", BinOp.bitxor },     .{ "BitAnd", BinOp.bitand },
        };
        inline for (table) |entry| {
            if (eq(k, entry[0])) return entry[1];
        }
        return self.unsupported(pos, "the operator {s} can't be compiled", .{k});
    }

    fn cmpOp(self: *Reader, op: *PyObject) ReadError!CmpOp {
        _ = self;
        const k = try kindOf(op);
        const table = .{
            .{ "Eq", CmpOp.eq }, .{ "NotEq", CmpOp.ne },     .{ "Lt", CmpOp.lt }, .{ "LtE", CmpOp.le },
            .{ "Gt", CmpOp.gt }, .{ "GtE", CmpOp.ge },       .{ "Is", CmpOp.is }, .{ "IsNot", CmpOp.is_not },
            .{ "In", CmpOp.in }, .{ "NotIn", CmpOp.not_in },
        };
        inline for (table) |entry| {
            if (eq(k, entry[0])) return entry[1];
        }
        return .eq;
    }
};

// ----------------------------------------------------------------------
// Dump: the read form as text (Language.ir(), for tests and debugging)
// ----------------------------------------------------------------------

pub fn dump(f: *const Function, out: *std.ArrayList(u8), gpa: Allocator) !void {
    var d = Dumper{ .f = f, .out = out, .gpa = gpa };
    try d.print("def {s}(", .{f.name});
    for (0..f.param_count) |i| try d.print("{s}{s}", .{ if (i > 0) ", " else "", f.locals[i] });
    try d.print("):\n", .{});
    try d.stmts(f.body, 1);
}

const Dumper = struct {
    f: *const Function,
    out: *std.ArrayList(u8),
    gpa: Allocator,

    fn print(self: *Dumper, comptime fmt: []const u8, args: anytype) !void {
        try self.out.print(self.gpa, fmt, args);
    }

    fn indent(self: *Dumper, n: usize) !void {
        for (0..n) |_| try self.print("    ", .{});
    }

    fn stmts(self: *Dumper, body: []const Stmt, depth: usize) Allocator.Error!void {
        if (body.len == 0) {
            try self.indent(depth);
            try self.print("pass\n", .{});
        }
        for (body) |s| try self.stmt(s, depth);
    }

    fn stmt(self: *Dumper, s: Stmt, depth: usize) Allocator.Error!void {
        try self.indent(depth);
        switch (s.kind) {
            .assign => |a| {
                for (a.targets) |t| {
                    try self.target(t);
                    try self.print(" = ", .{});
                }
                try self.expr(a.value);
                try self.print("\n", .{});
            },
            .aug => |a| {
                try self.target(a.target);
                try self.print(" {s}= ", .{@tagName(a.op)});
                try self.expr(a.value);
                try self.print("\n", .{});
            },
            .expr => |e| {
                try self.expr(e);
                try self.print("\n", .{});
            },
            .if_ => |i| {
                try self.print("if ", .{});
                try self.expr(i.test_);
                try self.print(":\n", .{});
                try self.stmts(i.body, depth + 1);
                if (i.else_.len != 0) {
                    try self.indent(depth);
                    try self.print("else:\n", .{});
                    try self.stmts(i.else_, depth + 1);
                }
            },
            .while_ => |w| {
                try self.print("while ", .{});
                try self.expr(w.test_);
                try self.print(":\n", .{});
                try self.stmts(w.body, depth + 1);
                if (w.else_.len != 0) {
                    try self.indent(depth);
                    try self.print("else:\n", .{});
                    try self.stmts(w.else_, depth + 1);
                }
            },
            .for_ => |w| {
                try self.print("for ", .{});
                try self.target(w.target);
                try self.print(" in ", .{});
                try self.expr(w.iter);
                try self.print(":\n", .{});
                try self.stmts(w.body, depth + 1);
                if (w.else_.len != 0) {
                    try self.indent(depth);
                    try self.print("else:\n", .{});
                    try self.stmts(w.else_, depth + 1);
                }
            },
            .return_ => |r| {
                try self.print("return", .{});
                if (r) |e| {
                    try self.print(" ", .{});
                    try self.expr(e);
                }
                try self.print("\n", .{});
            },
            .raise_ => |r| {
                try self.print("raise", .{});
                if (r) |e| {
                    try self.print(" ", .{});
                    try self.expr(e);
                }
                try self.print("\n", .{});
            },
            .assert_ => |a| {
                try self.print("assert ", .{});
                try self.expr(a.test_);
                if (a.msg) |m| {
                    try self.print(", ", .{});
                    try self.expr(m);
                }
                try self.print("\n", .{});
            },
            .try_ => |t| {
                try self.print("try:\n", .{});
                try self.stmts(t.body, depth + 1);
                for (t.handlers) |h| {
                    try self.indent(depth);
                    try self.print("except", .{});
                    if (h.type_) |e| {
                        try self.print(" ", .{});
                        try self.expr(e);
                    }
                    if (h.name) |slot| try self.print(" as {s}#{d}", .{ self.f.locals[slot], slot });
                    try self.print(":\n", .{});
                    try self.stmts(h.body, depth + 1);
                }
                if (t.else_.len > 0) {
                    try self.indent(depth);
                    try self.print("else:\n", .{});
                    try self.stmts(t.else_, depth + 1);
                }
                if (t.finally.len > 0) {
                    try self.indent(depth);
                    try self.print("finally:\n", .{});
                    try self.stmts(t.finally, depth + 1);
                }
            },
            .break_ => try self.print("break\n", .{}),
            .continue_ => try self.print("continue\n", .{}),
            .pass => try self.print("pass\n", .{}),
            .del_local => |slot| try self.print("del {s}#{d}\n", .{ self.f.locals[slot], slot }),
            .del_item => |d| {
                try self.print("del ", .{});
                try self.expr(d.obj);
                try self.print("[", .{});
                try self.expr(d.index);
                try self.print("]\n", .{});
            },
            .raise_from => |r| {
                try self.print("raise ", .{});
                try self.expr(r.exc);
                try self.print(" from ", .{});
                try self.expr(r.cause);
                try self.print("\n", .{});
            },
            // (the first where this one's indented already)
            .seq => |ss| for (ss, 0..) |x, i| try self.stmt(x, if (i == 0) 0 else depth),
        }
    }

    fn target(self: *Dumper, t: Target) Allocator.Error!void {
        switch (t) {
            .local => |slot| try self.print("{s}#{d}", .{ self.f.locals[slot], slot }),
            .global => |name| try self.print("global {s}", .{name}),
            .tuple => |ts| {
                try self.print("(", .{});
                for (ts, 0..) |x, i| {
                    if (i > 0) try self.print(", ", .{});
                    try self.target(x);
                }
                try self.print(")", .{});
            },
            .attr => |a| {
                try self.expr(a.obj);
                try self.print(".{s}", .{a.name});
            },
            .index => |x| {
                try self.expr(x.obj);
                try self.print("[", .{});
                try self.expr(x.index);
                try self.print("]", .{});
            },
        }
    }

    fn list(self: *Dumper, items: []const *Expr) Allocator.Error!void {
        for (items, 0..) |e, i| {
            if (i > 0) try self.print(", ", .{});
            try self.expr(e);
        }
    }

    fn expr(self: *Dumper, e: *const Expr) Allocator.Error!void {
        switch (e.kind) {
            .int => |v| try self.print("{d}", .{v}),
            .big => |v| {
                const s = py.c.PyObject_Str(v) orelse {
                    py.c.PyErr_Clear();
                    return self.print("<int>", .{});
                };
                defer py.Py_DecRef(s);
                const text = ph.utf8(s, "int") orelse {
                    py.c.PyErr_Clear();
                    return self.print("<int>", .{});
                };
                try self.print("{s}", .{text});
            },
            .object => |v| {
                const s = py.c.PyObject_Repr(v) orelse {
                    py.c.PyErr_Clear();
                    return self.print("<constant>", .{});
                };
                defer py.Py_DecRef(s);
                const text = ph.utf8(s, "constant") orelse {
                    py.c.PyErr_Clear();
                    return self.print("<constant>", .{});
                };
                try self.print("{s}", .{text});
            },
            .float => |v| try self.print("{d}", .{v}),
            .str => |v| try self.print("\"{s}\"", .{v}),
            .bool => |v| try self.print("{s}", .{if (v) "True" else "False"}),
            .none => try self.print("None", .{}),
            .local => |slot| try self.print("{s}#{d}", .{ self.f.locals[slot], slot }),
            .global => |name| try self.print("global {s}", .{name}),
            .attr => |a| {
                try self.expr(a.obj);
                try self.print(".{s}", .{a.name});
            },
            .index => |x| {
                try self.expr(x.obj);
                try self.print("[", .{});
                try self.expr(x.index);
                try self.print("]", .{});
            },
            .slice => |x| {
                try self.expr(x.obj);
                try self.print("[", .{});
                if (x.lo) |v| try self.expr(v);
                try self.print(":", .{});
                if (x.hi) |v| try self.expr(v);
                if (x.step) |v| {
                    try self.print(":", .{});
                    try self.expr(v);
                }
                try self.print("]", .{});
            },
            .call_nested => |c| {
                try self.print("{s}<nested>(", .{c.func.name});
                try self.list(c.args);
                try self.print(")", .{});
            },
            .make_closure => |f| try self.print("{s}<closure>", .{f.name}),
            .named => |n| {
                try self.print("({s}#{d} := ", .{ self.f.locals[n.slot], n.slot });
                try self.expr(n.value);
                try self.print(")", .{});
            },
            .starred => |x| {
                try self.print("*", .{});
                try self.expr(x);
            },
            .import_ => |m| {
                if (m.attr) |name| try self.print("import {s}.{s}", .{ m.module, name }) else try self.print("import {s}{s}", .{ m.module, if (m.leaf) "" else " (top)" });
            },
            .call => |c| {
                try self.expr(c.func);
                try self.print("(", .{});
                try self.list(c.args);
                for (c.keywords, 0..) |kw, i| {
                    if (i > 0 or c.args.len > 0) try self.print(", ", .{});
                    try self.print("{s}=", .{kw.name});
                    try self.expr(kw.value);
                }
                try self.print(")", .{});
            },
            .binary => |b| {
                try self.print("({s} ", .{@tagName(b.op)});
                try self.expr(b.left);
                try self.print(" ", .{});
                try self.expr(b.right);
                try self.print(")", .{});
            },
            .unary => |u| {
                try self.print("({s} ", .{@tagName(u.op)});
                try self.expr(u.operand);
                try self.print(")", .{});
            },
            .and_, .or_ => |items| {
                try self.print("({s} ", .{@tagName(e.kind)});
                try self.list(items);
                try self.print(")", .{});
            },
            .compare => |c| {
                try self.print("(compare ", .{});
                try self.expr(c.first);
                for (c.ops, c.rest) |op, r| {
                    try self.print(" {s} ", .{@tagName(op)});
                    try self.expr(r);
                }
                try self.print(")", .{});
            },
            .cond => |c| {
                try self.print("(", .{});
                try self.expr(c.then);
                try self.print(" if ", .{});
                try self.expr(c.test_);
                try self.print(" else ", .{});
                try self.expr(c.else_);
                try self.print(")", .{});
            },
            .list => |items| {
                try self.print("[", .{});
                try self.list(items);
                try self.print("]", .{});
            },
            .tuple => |items| {
                try self.print("(tuple ", .{});
                try self.list(items);
                try self.print(")", .{});
            },
            .set_ => |s| {
                try self.print("{{", .{});
                try self.list(s.items);
                try self.print("}}", .{});
            },
            .set_comp => |c| {
                try self.print("{{", .{});
                try self.expr(c.elt);
                try self.generators(c.generators);
                try self.print("}}", .{});
            },
            .dict => |d| {
                try self.print("{{", .{});
                for (d.keys, d.values, 0..) |k, v, i| {
                    if (i > 0) try self.print(", ", .{});
                    try self.expr(k);
                    try self.print(": ", .{});
                    try self.expr(v);
                }
                try self.print("}}", .{});
            },
            .list_comp => |c| {
                try self.print("[", .{});
                try self.expr(c.elt);
                try self.generators(c.generators);
                try self.print("]", .{});
            },
            .gen_exp => |c| {
                try self.print("(", .{});
                try self.expr(c.elt);
                try self.generators(c.generators);
                try self.print(")", .{});
            },
            .dict_comp => |c| {
                try self.print("{{", .{});
                try self.expr(c.key);
                try self.print(": ", .{});
                try self.expr(c.value);
                try self.generators(c.generators);
                try self.print("}}", .{});
            },
            .fstring => |parts| {
                try self.print("f\"", .{});
                try self.fparts(parts);
                try self.print("\"", .{});
            },
            .outline => try self.print("<out of line>", .{}),
        }
    }

    fn generators(self: *Dumper, gens: []const Generator) Allocator.Error!void {
        for (gens) |g| {
            try self.print(" for ", .{});
            try self.target(g.target);
            try self.print(" in ", .{});
            try self.expr(g.iter);
            for (g.ifs) |c| {
                try self.print(" if ", .{});
                try self.expr(c);
            }
        }
    }

    fn fparts(self: *Dumper, parts: []const FPart) Allocator.Error!void {
        for (parts) |p| switch (p) {
            .text => |t| try self.print("{s}", .{t}),
            .value => |v| {
                try self.print("{{", .{});
                try self.expr(v.expr);
                if (v.conversion != 0) try self.print("!{c}", .{v.conversion});
                if (v.spec.len != 0) {
                    try self.print(":", .{});
                    try self.fparts(v.spec);
                }
                try self.print("}}", .{});
            },
        };
    }
};

// ----------------------------------------------------------------------
// Python ast access
// ----------------------------------------------------------------------

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// An ast node's class name (borrowed from the interned type name).
fn kindOf(obj: *PyObject) ReadError![]const u8 {
    const t: *PyObject = @ptrCast(@alignCast(obj.ob_type));
    const name = ph.attr(t, "__name__") orelse return error.Python;
    defer py.Py_DecRef(name);
    const s = ph.utf8(name, "a type name") orelse return error.Python;
    // (type names are interned and live as long as their class)
    return s;
}

fn isKind(obj: *PyObject, name: []const u8) ReadError!bool {
    return eq(try kindOf(obj), name);
}

/// Whether CPython's compiler takes an expression for a constant (its AST
/// optimizer folds it): a literal, a sign before a number, a tuple of them
fn isConstant(e: *PyObject) ReadError!bool {
    const k = try kindOf(e);
    if (eq(k, "Constant")) return true;
    if (eq(k, "UnaryOp")) {
        const op = ph.attr(e, "op") orelse return error.Python;
        defer py.Py_DecRef(op);
        const ok = try kindOf(op);
        if (!eq(ok, "USub") and !eq(ok, "UAdd")) return false;
        const x = ph.attr(e, "operand") orelse return error.Python;
        defer py.Py_DecRef(x);
        if (!try isKind(x, "Constant")) return false;
        const v = ph.attr(x, "value") orelse return error.Python;
        defer py.Py_DecRef(v);
        return py.PyLong_Check(v) or py.PyFloat_Check(v);
    }
    if (eq(k, "Tuple")) {
        const elts = try listAttr(e, "elts");
        defer py.Py_DecRef(elts);
        for (0..@intCast(py.c.PyList_Size(elts))) |i| {
            if (!try isConstant(py.c.PyList_GetItem(elts, @intCast(i)).?)) return false;
        }
        return true;
    }
    return false;
}

fn listAttr(obj: *PyObject, name: [*:0]const u8) ReadError!*PyObject {
    return ph.attr(obj, name) orelse error.Python;
}

/// Whether an AST class name is a statement's
fn isStatement(k: []const u8) bool {
    const stmts = [_][]const u8{ "Expr", "Assign", "AugAssign", "AnnAssign", "If", "While", "For", "Return", "Raise", "Assert", "Try", "Break", "Continue", "Pass", "Delete", "Import", "ImportFrom", "With", "Global", "Nonlocal", "Match", "AsyncFor", "AsyncWith", "TryStar" };
    for (stmts) |s| if (eq(s, k)) return true;
    return false;
}

fn contains(names: []const []const u8, name: []const u8) bool {
    for (names) |n| if (eq(n, name)) return true;
    return false;
}

fn intAttr(obj: *PyObject, name: [*:0]const u8) ReadError!i64 {
    return (ph.attrInt(obj, name) catch return error.Python) orelse 0;
}

fn intAttrOr(obj: *PyObject, name: [*:0]const u8, default: i64) error{Python}!i64 {
    if (py.c.PyObject_HasAttrString(obj, name) != 1) return default;
    return (try ph.attrInt(obj, name)) orelse default;
}

fn stmtName(k: []const u8) []const u8 {
    const table = .{
        .{ "TryStar", "`try`" },                .{ "With", "`with`" },
        .{ "AsyncWith", "`async with`" },       .{ "FunctionDef", "a nested `def`" },
        .{ "AsyncFunctionDef", "`async def`" }, .{ "ClassDef", "a nested `class`" },
        .{ "Global", "`global`" },              .{ "Nonlocal", "`nonlocal`" },
        .{ "Delete", "`del`" },                 .{ "Import", "`import`" },
        .{ "ImportFrom", "`import`" },          .{ "AsyncFor", "`async for`" },
        .{ "Match", "`match`" },
    };
    inline for (table) |entry| {
        if (eq(k, entry[0])) return entry[1];
    }
    return k;
}

fn exprName(k: []const u8) []const u8 {
    const table = .{
        .{ "Lambda", "`lambda`" }, .{ "Yield", "`yield`" },      .{ "YieldFrom", "`yield from`" },
        .{ "Await", "`await`" },   .{ "NamedExpr", "`:=`" },     .{ "SetComp", "a set comprehension" },
        .{ "Set", "a set" },       .{ "Starred", "*unpacking" },
    };
    inline for (table) |entry| {
        if (eq(k, entry[0])) return entry[1];
    }
    return k;
}
