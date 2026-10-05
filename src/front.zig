//! The semantics compiler's front: a semantic's Python source (read with
//! Python's `ast` module, walked from here) checked against the compilable
//! subset and turned into zrun's form of it (the types below), which the
//! partial evaluator specializes for each program.
//!
//! The subset: values (int, float, str, bool, None, lists, tuples, dicts,
//! records), assignment (also to fields and items, augmented), if, while,
//! for, break, continue, return, raise (rt's control flow and errors),
//! assert, pass; expressions with operators, comparisons, conditional
//! expressions, comprehensions and f-strings; calls (rt, other semantics,
//! helper functions of the module, records, builtins, methods of values).
//! Outside it: try, with, lambda, nested def and class, yield, await,
//! global and nonlocal, del, star arguments, walrus. A construct outside the
//! subset is an error at its line, when the semantic is registered.
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
        binary: struct { op: BinOp, left: *Expr, right: *Expr },
        unary: struct { op: UnaryOp, operand: *Expr },
        and_: []const *Expr,
        or_: []const *Expr,
        /// `a < b <= c`: first, then each (op, operand)
        compare: struct { first: *Expr, ops: []const CmpOp, rest: []const *Expr },
        cond: struct { test_: *Expr, then: *Expr, else_: *Expr },
        list: []const *Expr,
        tuple: []const *Expr,
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
    /// on have defaults (the function's __defaults__)
    param_count: u32,
    required: u32 = 0,
    /// Every local's name, by slot
    locals: []const []const u8,
    body: []const Stmt,
    /// The function object (owned): its module globals and closure, for
    /// resolving the global names
    py_function: *PyObject,
    /// How big it is: its expressions (whether to run it inline)
    size: u32 = 0,

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

/// Read a Python function: its source, parsed, checked against the
/// subset. On error.Unsupported, `failure` says what and where (with the
/// function's file and first line, the caller makes it a CompileError).
pub fn read(gpa: Allocator, func: *PyObject, failure: *Failure) ReadError!*Function {
    var r = Reader{
        .gpa = gpa,
        .arena = std.heap.ArenaAllocator.init(gpa),
        .failure = failure,
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
    const raw = py.c.PyObject_CallMethod(inspect, "getsource", "(O)", func) orelse return error.Python;
    defer py.Py_DecRef(raw);
    const source = py.c.PyObject_CallMethod(textwrap, "dedent", "(O)", raw) orelse return error.Python;
    defer py.Py_DecRef(source);
    const module = py.c.PyObject_CallMethod(ast_mod, "parse", "(O)", source) orelse return error.Python;
    defer py.Py_DecRef(module);

    const code = ph.attr(func, "__code__") orelse return error.Python;
    defer py.Py_DecRef(code);
    const file = try r.strAttr(code, "co_filename");
    const first = try intAttr(code, "co_firstlineno");
    const name = try r.strAttr(func, "__name__");

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
    inline for (.{ "vararg", "kwarg" }) |f| {
        const o = ph.attr(args, f) orelse return error.Python;
        defer py.Py_DecRef(o);
        if (o != py.Py_None()) return r.unsupported(try r.posOf(def), "*args and **kwargs can't be compiled", .{});
    }
    const params = try listAttr(args, "args");
    defer py.Py_DecRef(params);
    const n_params: usize = @intCast(py.c.PyList_Size(params));
    for (0..n_params) |i| {
        const p = py.c.PyList_GetItem(params, @intCast(i)).?;
        const pname = try r.strAttr(p, "arg");
        _ = try r.declare(pname);
    }

    // Locals: every name the body assigns (Python's rule: then local
    // everywhere in the function)
    const stmts = try listAttr(def, "body");
    defer py.Py_DecRef(stmts);
    try r.collectAssigned(stmts);

    const out = try r.stmtList(stmts);
    const f = try gpa.create(Function);
    py.Py_IncRef(func);
    f.* = .{
        .arena = r.arena,
        .name = name,
        .file = file,
        .first_line = @intCast(first),
        .param_count = @intCast(n_params),
        .required = @intCast(n_params - n_defaults),
        .locals = r.locals.items,
        .body = out,
        .py_function = func,
        .size = r.exprs,
    };
    _ = a;
    return f;
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

    fn alloc(self: *Reader) Allocator {
        return self.arena.allocator();
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

    fn collectTarget(self: *Reader, t: *PyObject) ReadError!void {
        const k = try kindOf(t);
        if (eq(k, "Name")) {
            _ = try self.declare(try self.strAttr(t, "id"));
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
                if (cause != py.Py_None()) return self.unsupported(pos, "`raise ... from ...` can't be compiled", .{});
                break :blk .{ .raise_ = try self.optExprAttr(s, "exc") };
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
            return self.unsupported(pos, "{s} can't be compiled", .{stmtName(k)});
        };
        return .{ .pos = pos, .kind = kind };
    }

    fn target(self: *Reader, t: *PyObject) ReadError!Target {
        const k = try kindOf(t);
        if (eq(k, "Name")) {
            const name = try self.strAttr(t, "id");
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
            return self.new(pos, .{ .global = name });
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
            const func = try self.exprAttr(e, "func");
            const args = try self.exprList(e, "args");
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
        return self.unsupported(pos, "this constant (bytes, complex, ...) can't be compiled", .{});
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
        }
    }

    fn target(self: *Dumper, t: Target) Allocator.Error!void {
        switch (t) {
            .local => |slot| try self.print("{s}#{d}", .{ self.f.locals[slot], slot }),
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

fn listAttr(obj: *PyObject, name: [*:0]const u8) ReadError!*PyObject {
    return ph.attr(obj, name) orelse error.Python;
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
