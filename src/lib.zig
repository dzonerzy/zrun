//! zrun: execution for languages defined with zgram and checked with zrules.
//!
//! A language's semantics are Python functions, one per kind of node
//! (`@lang.eval("BinOp")`, `@lang.exec("While")`), given the node and `rt`,
//! the runtime. `lang.load(source)` parses and checks a program;
//! `program.run()` runs it; `program.call(name, *args)` calls one of its
//! functions. Everything but the semantics themselves is native: the nodes
//! and their fields, variables (by zrules symbol), frames, calls, errors.
//!
//! This is the reference mode, where the semantics run as Python; the
//! compiled modes (docs/toolkit.md) must agree with it on every program.

const std = @import("std");
const pyoz = @import("PyOZ");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const types = @import("types.zig");
const objects = @import("objects.zig");
const grammar_mod = @import("grammar.zig");
const program_mod = @import("program.zig");
const tree_mod = @import("tree.zig");
const zabi = @import("zrules_abi.zig");
const front = @import("front.zig");
const driver = @import("driver.zig");
const compile_mod = @import("compile.zig");
const bridge = @import("bridge.zig");
const helpers = @import("helpers.zig");
const value_mod = @import("value.zig");

const allocator = std.heap.c_allocator;
const NONE = program_mod.NONE;
const FunctionSpec = program_mod.FunctionSpec;

fn ref(o: *PyObject) *PyObject {
    py.Py_IncRef(o);
    return o;
}

fn none() *PyObject {
    return ref(py.Py_None());
}

fn optional(o: ?*PyObject) ?*PyObject {
    const x = o orelse return null;
    return if (x == py.Py_None()) null else x;
}

// ============================================================================
// Language
// ============================================================================

const Language = struct {
    _parser: ?*PyObject = null,
    _rules: ?*PyObject = null,
    _grammar: ?*grammar_mod.Grammar = null,
    /// Semantics by kind or rule name (dicts of str -> callable)
    _evals: ?*PyObject = null,
    _execs: ?*PyObject = null,
    /// Host functions by builtin name
    _hosts: ?*PyObject = null,
    /// Per rule: how its nodes make functions (Language.function)
    _functions: []?FunctionSpec = &.{},
    /// Per rule: its semantics, worked out from the dicts (borrowed)
    _eval_of: []?*PyObject = &.{},
    _exec_of: []?*PyObject = &.{},
    _resolved: bool = false,
    _max_depth: u32 = 1000,
    /// The semantics read by the compiler's front, by function object
    /// (each holds a reference to its function); those marked
    /// native=False aren't here
    _read: std.AutoHashMapUnmanaged(*PyObject, *front.Function) = .empty,
    /// Semantics the compiler couldn't compile: compiled programs run them
    /// as Python (bridge.zig); learned as programs are compiled
    _python: driver.PythonSet = .empty,
    /// List and dict literals of the semantics compiled programs build at
    /// run time (they escape where known ones can't follow)
    _escaping: std.AutoHashMapUnmanaged(*const front.Expr, void) = .empty,

    pub fn __new__(args: pyoz.Args(struct { parser: *PyObject, rules: ?*PyObject = null, max_depth: i64 = 1000 })) ?Language {
        const v = args.value;
        var lang = Language{};
        const g = allocator.create(grammar_mod.Grammar) catch return oom(Language);
        g.* = grammar_mod.Grammar.init(allocator, v.parser) catch |e| {
            allocator.destroy(g);
            if (e == error.OutOfMemory) _ = py.c.PyErr_NoMemory();
            return null;
        };
        lang._grammar = g;
        const n = g.ruleCount();
        lang._functions = allocator.alloc(?FunctionSpec, n) catch return lang.fail();
        @memset(lang._functions, null);
        lang._eval_of = allocator.alloc(?*PyObject, n) catch return lang.fail();
        lang._exec_of = allocator.alloc(?*PyObject, n) catch return lang.fail();
        lang._evals = py.c.PyDict_New() orelse return lang.fail();
        lang._execs = py.c.PyDict_New() orelse return lang.fail();
        lang._hosts = py.c.PyDict_New() orelse return lang.fail();
        lang._parser = ref(v.parser);
        if (optional(v.rules)) |r| lang._rules = ref(r);
        if (v.max_depth < 1) {
            ph.raise(py.PyExc_ValueError(), "max_depth must be at least 1", .{});
            return lang.fail();
        }
        lang._max_depth = @intCast(@min(v.max_depth, 1_000_000));
        return lang;
    }

    fn fail(self: *Language) ?Language {
        if (py.c.PyErr_Occurred() == null) _ = py.c.PyErr_NoMemory();
        self.release();
        return null;
    }

    fn release(self: *Language) void {
        inline for (.{ "_parser", "_rules", "_evals", "_execs", "_hosts" }) |f| {
            if (@field(self, f)) |o| py.Py_DecRef(o);
            @field(self, f) = null;
        }
        if (self._grammar) |g| {
            g.deinit();
            allocator.destroy(g);
            self._grammar = null;
        }
        if (self._functions.len != 0) allocator.free(self._functions);
        if (self._eval_of.len != 0) allocator.free(self._eval_of);
        if (self._exec_of.len != 0) allocator.free(self._exec_of);
        self._functions = &.{};
        self._eval_of = &.{};
        self._exec_of = &.{};
        var it = self._read.valueIterator();
        while (it.next()) |f| f.*.destroy(allocator);
        self._read.deinit(allocator);
        self._read = .empty;
        var pit = self._python.valueIterator();
        while (pit.next()) |r| allocator.free(r.*);
        self._python.deinit(allocator);
        self._python = .empty;
        self._escaping.deinit(allocator);
        self._escaping = .empty;
    }

    /// `lang.python_semantics()`: the semantics compiled programs run as
    /// Python, {name: why}: native=False ones, and those the compiler
    /// couldn't compile (learned as programs are compiled; why: where in
    /// the semantic, and what): what to rewrite for speed.
    pub fn python_semantics(self: *Language) ?*PyObject {
        self.resolve();
        const out = py.c.PyDict_New() orelse return null;
        for ([_][]?*PyObject{ self._eval_of, self._exec_of }) |table| {
            for (table) |fo| {
                const f = fo orelse continue;
                const reason = self._python.get(f) orelse continue;
                const name = py.c.PyObject_GetAttrString(f, "__name__") orelse return null;
                defer py.Py_DecRef(name);
                const r = ph.newString(reason) orelse return null;
                defer py.Py_DecRef(r);
                if (py.c.PyDict_SetItem(out, name, r) != 0) return null;
            }
        }
        return out;
    }

    /// `lang.ir(fn)`: a semantic (or any function) as the compiler's front
    /// reads it, as text: for tests and for seeing what gets compiled.
    pub fn ir(self: *Language, func: *PyObject) ?*PyObject {
        if (!self.readSemantic(func)) return null;
        const f = self._read.get(func).?;
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(allocator);
        front.dump(f, &out, allocator) catch return oom(*PyObject);
        return ph.newString(out.items);
    }

    /// A semantic compiled programs run as Python, and why.
    fn markPython(self: *Language, func: *PyObject, reason: []const u8) bool {
        if (self._python.contains(func)) return true;
        const r = allocator.dupe(u8, reason) catch {
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        self._python.put(allocator, func, r) catch {
            allocator.free(r);
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        return true;
    }

    /// Read a semantic with the compiler's front (once per function);
    /// false with zrun.CompileError (or another exception) set.
    fn readSemantic(self: *Language, func: *PyObject) bool {
        if (self._read.contains(func)) return true;
        var failure = front.Failure{};
        const f = front.read(allocator, func, &failure) catch |e| switch (e) {
            error.Unsupported => {
                raiseCompileError(func, &failure);
                return false;
            },
            error.OutOfMemory => {
                _ = py.c.PyErr_NoMemory();
                return false;
            },
            error.Python => return false,
        };
        self._read.put(allocator, func, f) catch {
            f.destroy(allocator);
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        return true;
    }

    // (No __traverse__: PyOZ 0.13.7 frees a collected class's objects with
    // PyObject_Del in the stable ABI, which crashes. A language is in a
    // cycle only through its semantics' module, which lives as long anyway.)
    pub fn __del__(self: *Language) void {
        self.release();
    }

    /// `@lang.eval(kind, native=True)`: the semantics of an expression kind
    /// (a kind or rule name, or a list of them): `fn(node, rt) -> value`.
    /// It is compiled with the program; native=False runs it as Python.
    pub fn eval(self: *Language, args: pyoz.Args(struct { kind: *PyObject, native: bool = true })) ?*PyObject {
        return self.registrar(args.value.kind, 0, args.value.native);
    }

    /// `@lang.exec(kind, native=True)`: the semantics of a statement kind:
    /// `fn(node, rt) -> None`.
    pub fn exec(self: *Language, args: pyoz.Args(struct { kind: *PyObject, native: bool = true })) ?*PyObject {
        return self.registrar(args.value.kind, 1, args.value.native);
    }

    fn registrar(self: *Language, kind: *PyObject, which: u8, native: bool) ?*PyObject {
        // Check the names now: a typo is an error where it's written
        if (!self.checkKinds(kind)) return null;
        var r = Registrar{ ._which = which, ._native = native };
        r._lang = ref(Module.selfObject(Language, self));
        r._kind = ref(kind);
        return Module.toPy(Registrar, r);
    }

    fn checkKinds(self: *Language, kind: *PyObject) bool {
        const g = self._grammar.?;
        if (py.PyUnicode_Check(kind)) {
            const name = ph.utf8(kind, "a kind") orelse return false;
            if (g.hasName(name)) return true;
            ph.raise(py.PyExc_ValueError(), "the grammar has no rule or -> class named '{s}'", .{name});
            return false;
        }
        const seq = py.c.PySequence_Fast(kind, "a kind is a str or a list of them") orelse return false;
        defer py.Py_DecRef(seq);
        const n: usize = @intCast(py.c.PySequence_Size(seq));
        for (0..n) |i| {
            const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse return false;
            defer py.Py_DecRef(item);
            if (!py.PyUnicode_Check(item)) {
                ph.raise(py.PyExc_TypeError(), "a kind is a str or a list of them", .{});
                return false;
            }
            if (!self.checkKinds(item)) return false;
        }
        return true;
    }

    /// Register `fn` for each name in `kind` (str or sequence). A native one
    /// is read by the compiler's front first; one it can't read, and a
    /// native=False one, compiled programs run as Python (the reason kept:
    /// lang.python_semantics()).
    fn register(self: *Language, which: u8, kind: *PyObject, func: *PyObject, native: bool) bool {
        if (!py.PyCallable_Check(func)) {
            ph.raise(py.PyExc_TypeError(), "semantics must be callable: fn(node, rt)", .{});
            return false;
        }
        if (!native) {
            if (!self.markPython(func, "native=False")) return false;
        } else if (!self.readSemantic(func)) {
            if (py.c.PyErr_ExceptionMatches(types.CompileError) == 0) return false;
            // (outside the compilable subset: run as Python, saying why)
            var t: ?*PyObject = null;
            var v: ?*PyObject = null;
            var tb: ?*PyObject = null;
            py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
            py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
            defer inline for (.{ t, v, tb }) |o| {
                if (o) |x| py.Py_DecRef(x);
            };
            const msg = py.c.PyObject_Str(v.?) orelse return false;
            defer py.Py_DecRef(msg);
            const text = ph.utf8(msg, "message") orelse return false;
            if (!self.markPython(func, text)) return false;
        }
        const dict = if (which == 0) self._evals.? else self._execs.?;
        if (py.PyUnicode_Check(kind)) {
            if (py.c.PyDict_SetItem(dict, kind, func) != 0) return false;
        } else {
            const seq = py.c.PySequence_Fast(kind, "a kind is a str or a list of them") orelse return false;
            defer py.Py_DecRef(seq);
            const n: usize = @intCast(py.c.PySequence_Size(seq));
            for (0..n) |i| {
                const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse return false;
                defer py.Py_DecRef(item);
                if (py.c.PyDict_SetItem(dict, item, func) != 0) return false;
            }
        }
        self._resolved = false;
        return true;
    }

    /// `lang.function(kind, params="params", body="body", name="name",
    /// hoist=True, missing="error", extra="error")`: nodes of `kind` define
    /// functions: their parameters (a list of name nodes, or nodes with a
    /// `name` field), body and name are the fields of those labels. With
    /// hoist, a scope's functions are defined when it is entered (callable
    /// before their definition); without, when the definition runs. A call
    /// with fewer arguments than parameters is an error, or (missing="none")
    /// gives the rest None; one with more is an error, or (extra="drop")
    /// drops them, or (extra="keep") keeps them for the body (rt.varargs).
    pub fn function(self: *Language, args: pyoz.Args(struct { kind: *PyObject, params: ?*PyObject = null, body: ?*PyObject = null, name: ?*PyObject = null, hoist: bool = true, missing: ?*PyObject = null, extra: ?*PyObject = null })) ?*PyObject {
        const v = args.value;
        const g = self._grammar.?;
        const kind = ph.utf8(v.kind, "kind") orelse return null;
        if (!g.hasName(kind)) {
            ph.raise(py.PyExc_ValueError(), "the grammar has no rule or -> class named '{s}'", .{kind});
            return null;
        }
        var spec = FunctionSpec{ .hoist = v.hoist };
        if (optional(v.missing)) |m| {
            const s = ph.utf8(m, "missing") orelse return null;
            spec.missing = std.meta.stringToEnum(@TypeOf(spec.missing), s) orelse {
                ph.raise(py.PyExc_ValueError(), "missing= is \"error\" or \"none\", not '{s}'", .{s});
                return null;
            };
        }
        if (optional(v.extra)) |e| {
            const s = ph.utf8(e, "extra") orelse return null;
            spec.extra = std.meta.stringToEnum(@TypeOf(spec.extra), s) orelse {
                ph.raise(py.PyExc_ValueError(), "extra= is \"error\", \"drop\" or \"keep\", not '{s}'", .{s});
                return null;
            };
        }
        spec.params = self.labelId(v.params, "params", true) orelse return null;
        // The body: a label, or a rule (an unlabelled block)
        if (optional(v.body)) |b| {
            const given = ph.utf8(b, "body") orelse return null;
            if (g.field_ids.get(given)) |id| {
                spec.body = id;
            } else for (g.rule_names, 0..) |r, i| {
                if (std.mem.eql(u8, r, given)) {
                    spec.body_rule = @intCast(i);
                    break;
                }
            } else {
                ph.raise(py.PyExc_ValueError(), "the grammar has no label or rule '{s}' (the function's body)", .{given});
                return null;
            }
        } else {
            spec.body = self.labelId(null, "body", false) orelse return null;
        }
        spec.name = self.labelId(v.name, "name", true) orelse return null;
        for (g.rule_names, g.kind_names, 0..) |r, k, i| {
            if (std.mem.eql(u8, r, kind) or std.mem.eql(u8, k, kind)) self._functions[i] = spec;
        }
        return none();
    }

    /// A label's field id; 0 for "none" (None given, or the default label
    /// missing from the grammar when `optional_default`).
    fn labelId(self: *Language, given: ?*PyObject, default: []const u8, optional_default: bool) ?u8 {
        const g = self._grammar.?;
        if (given) |o| {
            if (o == py.Py_None()) return 0;
            const label = ph.utf8(o, "a label") orelse return null;
            return g.field_ids.get(label) orelse {
                ph.raise(py.PyExc_ValueError(), "the grammar has no label '{s}'", .{label});
                return null;
            };
        }
        if (g.field_ids.get(default)) |id| return id;
        if (optional_default) return 0;
        ph.raise(py.PyExc_ValueError(), "the grammar has no label '{s}': say which label is the function's {s}", .{ default, default });
        return null;
    }

    /// `@lang.host` / `lang.host("name")(fn)` / `lang.host("name", fn)`: a
    /// Python function the program calls by a builtin's name.
    pub fn host(self: *Language, args: pyoz.Args(struct { func: *PyObject, implementation: ?*PyObject = null })) ?*PyObject {
        const v = args.value;
        if (optional(v.implementation)) |impl| {
            if (!self.addHost(v.func, impl)) return null;
            return ref(impl);
        }
        if (py.PyUnicode_Check(v.func)) {
            var r = Registrar{ ._which = 2 };
            r._lang = ref(Module.selfObject(Language, self));
            r._kind = ref(v.func);
            return Module.toPy(Registrar, r);
        }
        const name = py.c.PyObject_GetAttrString(v.func, "__name__") orelse return null;
        defer py.Py_DecRef(name);
        if (!self.addHost(name, v.func)) return null;
        return ref(v.func);
    }

    fn addHost(self: *Language, name: *PyObject, func: *PyObject) bool {
        if (!py.PyUnicode_Check(name)) {
            ph.raise(py.PyExc_TypeError(), "a host function's name must be a str", .{});
            return false;
        }
        if (!py.PyCallable_Check(func)) {
            ph.raise(py.PyExc_TypeError(), "a host function must be callable", .{});
            return false;
        }
        return py.c.PyDict_SetItem(self._hosts.?, name, func) == 0;
    }

    fn resolve(self: *Language) void {
        if (self._resolved) return;
        const g = self._grammar.?;
        for (0..g.ruleCount()) |i| {
            self._eval_of[i] = lookup(self._evals.?, g.kinds[i], g.rule_names[i]);
            self._exec_of[i] = lookup(self._execs.?, g.kinds[i], g.rule_names[i]);
        }
        self._resolved = true;
    }

    /// A rule's semantics: by its kind, else by its rule name (borrowed).
    fn lookup(dict: *PyObject, kind: *PyObject, rule_name: []const u8) ?*PyObject {
        if (py.c.PyDict_GetItem(dict, kind)) |f| return f;
        const r = ph.newString(rule_name) orelse {
            py.c.PyErr_Clear();
            return null;
        };
        defer py.Py_DecRef(r);
        return py.c.PyDict_GetItem(dict, r);
    }

    /// `lang.load(source, path=None)`: parse and check a program. Raises
    /// zrun.LoadError with its errors.
    pub fn load(self: *Language, args: pyoz.Args(struct { source: *PyObject, path: ?*PyObject = null })) ?Program {
        const v = args.value;
        if (!py.PyUnicode_Check(v.source)) {
            ph.raise(py.PyExc_TypeError(), "source must be a str", .{});
            return null;
        }
        // (recover: every syntax error is reported, not only the first)
        const tree = parseTree(self._parser.?, v.source) orelse return null;
        var prog = Program{};
        prog._lang = ref(Module.selfObject(Language, self));
        prog._source = ref(v.source);
        if (optional(v.path)) |p| prog._path = ref(p);
        if (!prog.setup(self, tree)) {
            prog.release();
            return null;
        }
        return prog;
    }

    pub const __doc__: [*:0]const u8 = "Language(parser, rules=None, *, max_depth=1000): a language to run programs of: its zgram parser, its zrules rules (for names and their variables), and its semantics (eval, exec, function, host). load(source) parses and checks a program.";
    pub const eval__doc__: [*:0]const u8 = "@lang.eval(kind): the semantics of an expression kind (a -> class or rule name, or a list of them): fn(node, rt) -> value.";
    pub const eval__params__ = "kind";
    pub const exec__doc__: [*:0]const u8 = "@lang.exec(kind): the semantics of a statement kind: fn(node, rt).";
    pub const exec__params__ = "kind";
    pub const function__doc__: [*:0]const u8 = "function(kind, params='params', body='body', name='name', hoist=True): nodes of kind define functions, with those labels for their parameters, body and name.";
    pub const host__doc__: [*:0]const u8 = "@lang.host, @lang.host('name') or lang.host('name', fn): a Python function the program calls through the builtin of that name.";
    pub const load__doc__: [*:0]const u8 = "load(source, path=None): parse and check a program; a Program. Raises zrun.LoadError listing its errors.";
};

/// Raise zrun.CompileError for a semantic outside the subset: "file:line:
/// col: in name(): reason", with file, line and column attributes.
fn raiseCompileError(func: *PyObject, failure: *const front.Failure) void {
    var file: []const u8 = "<unknown>";
    var line: u32 = failure.pos.line;
    var name: []const u8 = "?";
    var file_buf: [512]u8 = undefined;
    var name_buf: [128]u8 = undefined;
    if (ph.attr(func, "__code__")) |code| {
        defer py.Py_DecRef(code);
        if (ph.attr(code, "co_filename")) |f| {
            defer py.Py_DecRef(f);
            if (ph.utf8(f, "file")) |s| {
                const n = @min(s.len, file_buf.len);
                @memcpy(file_buf[0..n], s[0..n]);
                file = file_buf[0..n];
            } else py.c.PyErr_Clear();
        } else py.c.PyErr_Clear();
        if (line == 0) line = @intCast((ph.attrInt(code, "co_firstlineno") catch null) orelse 0);
    } else py.c.PyErr_Clear();
    if (ph.attr(func, "__name__")) |n| {
        defer py.Py_DecRef(n);
        if (ph.utf8(n, "name")) |s| {
            const k = @min(s.len, name_buf.len);
            @memcpy(name_buf[0..k], s[0..k]);
            name = name_buf[0..k];
        } else py.c.PyErr_Clear();
    } else py.c.PyErr_Clear();
    py.c.PyErr_Clear();
    var buf: [1024]u8 = undefined;
    const msg = std.fmt.bufPrint(&buf, "{s}:{d}:{d}: in {s}(): {s}", .{ file, line, failure.pos.col, name, failure.text() }) catch buf[0..];
    const text = ph.newString(msg) orelse return;
    defer py.Py_DecRef(text);
    const exc = py.c.PyObject_CallFunctionObjArgs(types.CompileError, text, @as(?*PyObject, null)) orelse return;
    defer py.Py_DecRef(exc);
    const attrs = .{
        .{ "file", ph.newString(file) },
        .{ "line", py.c.PyLong_FromUnsignedLong(line) },
        .{ "column", py.c.PyLong_FromUnsignedLong(failure.pos.col) },
        .{ "reason", ph.newString(failure.text()) },
    };
    inline for (attrs) |a| {
        if (a[1]) |v| {
            _ = py.c.PyObject_SetAttrString(exc, a[0], v);
            py.Py_DecRef(v);
        }
    }
    py.c.PyErr_SetObject(types.CompileError, exc);
}

/// parser.parse_tree(source, recover=True)
fn parseTree(parser: *PyObject, source: *PyObject) ?*PyObject {
    const method = py.c.PyObject_GetAttrString(parser, "parse_tree") orelse return null;
    defer py.Py_DecRef(method);
    const args = py.c.PyTuple_Pack(1, source) orelse return null;
    defer py.Py_DecRef(args);
    const kwargs = py.c.PyDict_New() orelse return null;
    defer py.Py_DecRef(kwargs);
    if (py.c.PyDict_SetItemString(kwargs, "recover", py.Py_True()) != 0) return null;
    return py.c.PyObject_Call(method, args, kwargs);
}

/// The data of an object of one of the module's classes (null if it isn't one).
fn unwrap(comptime T: type, o: *PyObject) ?*T {
    return Module.fromPy(*T, o) catch null;
}

fn oom(comptime T: type) ?T {
    _ = py.c.PyErr_NoMemory();
    return null;
}

// ============================================================================
// Registrar: what the decorators return
// ============================================================================

const Registrar = struct {
    _lang: ?*PyObject = null,
    _kind: ?*PyObject = null,
    /// 0 eval, 1 exec, 2 host
    _which: u8 = 0,
    _native: bool = true,

    pub fn __call__(self: *Registrar, func: *PyObject) ?*PyObject {
        const lang = unwrap(Language, self._lang.?) orelse return null;
        const ok = if (self._which == 2) lang.addHost(self._kind.?, func) else lang.register(self._which, self._kind.?, func, self._native);
        if (!ok) return null;
        return ref(func);
    }

    pub fn __del__(self: *Registrar) void {
        if (self._lang) |o| py.Py_DecRef(o);
        if (self._kind) |o| py.Py_DecRef(o);
    }

    pub const __doc__: [*:0]const u8 = "A decorator registering the function it decorates.";
};

// ============================================================================
// Program
// ============================================================================

const Program = struct {
    /// The Language (never refers to its programs)
    _lang: ?*PyObject = null,
    _source: ?*PyObject = null,
    _path: ?*PyObject = null,
    /// Everything else, in a State (objects.zig): the tree, the analysis,
    /// the native data, the program's frame. Nodes and functions refer to
    /// it; this class refers to nothing that refers back to it
    _state: ?*PyObject = null,
    /// The program compiled (once it ran compiled)
    _compiled: ?*driver.Compiled = null,

    fn release(self: *Program) void {
        if (self._compiled) |c| c.destroy();
        self._compiled = null;
        inline for (.{ "_state", "_lang", "_source", "_path" }) |f| {
            if (@field(self, f)) |o| py.Py_DecRef(o);
            @field(self, f) = null;
        }
    }

    pub fn __del__(self: *Program) void {
        self.release();
    }

    fn state(self: *const Program) *objects.StateObject {
        return objects.asState(self._state.?);
    }

    fn ctx(self: *const Program) *objects.Context {
        return self.state().ctx.?;
    }

    /// Check the tree (syntax errors, the rules) and build the native data.
    fn setup(self: *Program, lang: *Language, tree_obj: *PyObject) bool {
        defer py.Py_DecRef(tree_obj);
        // The diagnostics: the rules' (syntax errors included), or the
        // tree's syntax errors
        var analysis: ?*PyObject = null;
        defer if (analysis) |a| py.Py_DecRef(a);
        var diagnostics: *PyObject = undefined;
        if (lang._rules) |rules| {
            analysis = py.c.PyObject_CallMethod(rules, "analyze", "(O)", tree_obj) orelse return false;
            diagnostics = py.c.PyObject_GetAttrString(analysis.?, "diagnostics") orelse return false;
        } else {
            diagnostics = py.c.PyObject_GetAttrString(tree_obj, "errors") orelse return false;
        }
        defer py.Py_DecRef(diagnostics);
        if (!self.raiseIfErrors(diagnostics)) return false;

        const capsule = py.c.PyObject_GetAttrString(tree_obj, "capsule") orelse return false;
        defer py.Py_DecRef(capsule);
        const tv: *const tree_mod.TreeView = @ptrCast(@alignCast(py.c.PyCapsule_GetPointer(capsule, tree_mod.CAPSULE_NAME) orelse return false));
        if (tv.abi != tree_mod.TREE_ABI) {
            ph.raise(py.PyExc_ImportError(), "zrun reads zgram trees with ABI {d}, this zgram has {d}", .{ tree_mod.TREE_ABI, tv.abi });
            return false;
        }
        var av: ?*const zabi.AnalysisView = null;
        if (analysis) |checked| {
            const ac = py.c.PyObject_GetAttrString(checked, "capsule") orelse {
                py.c.PyErr_Clear();
                ph.raise(py.PyExc_ImportError(), "zrun needs zrules 0.1.5 or later (Analysis.capsule)", .{});
                return false;
            };
            defer py.Py_DecRef(ac);
            const view: *const zabi.AnalysisView = @ptrCast(@alignCast(py.c.PyCapsule_GetPointer(ac, zabi.ANALYSIS_CAPSULE) orelse return false));
            if (view.abi != zabi.ANALYSIS_ABI) {
                ph.raise(py.PyExc_ImportError(), "zrun reads zrules analyses with ABI {d}, this zrules has {d}", .{ zabi.ANALYSIS_ABI, view.abi });
                return false;
            }
            av = view;
        }
        const data = program_mod.build(allocator, lang._grammar.?, lang._functions, tv, av) catch |e| {
            if (e == error.OutOfMemory) _ = py.c.PyErr_NoMemory();
            return false;
        };
        const context = allocator.create(objects.Context) catch {
            data.deinit();
            allocator.destroy(data);
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        const values = allocator.alloc(?*PyObject, data.nodes.len) catch {
            allocator.destroy(context);
            data.deinit();
            allocator.destroy(data);
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        @memset(values, null);
        context.* = .{ .data = data, .tree = ref(tree_obj), .values = values };
        self._state = objects.newState(context, self._lang.?, analysis, diagnostics) orelse {
            py.Py_DecRef(tree_obj);
            allocator.free(values);
            allocator.destroy(context);
            data.deinit();
            allocator.destroy(data);
            return false;
        };
        return true;
    }

    /// Raise LoadError if the diagnostics have an error.
    fn raiseIfErrors(self: *Program, diags: *PyObject) bool {
        const n: usize = @intCast(py.c.PyList_Size(diags));
        var errors: usize = 0;
        for (0..n) |i| {
            const d = py.c.PyList_GetItem(diags, @intCast(i)).?;
            const sev = py.c.PyObject_GetAttrString(d, "severity") orelse return false;
            defer py.Py_DecRef(sev);
            if (py.c.PyUnicode_CompareWithASCIIString(sev, "error") == 0) errors += 1;
        }
        if (errors == 0) return true;
        // The message: every diagnostic rendered with the source
        const lines = py.c.PyList_New(0) orelse return false;
        defer py.Py_DecRef(lines);
        const path = self._path orelse py.Py_None();
        for (0..n) |i| {
            const d = py.c.PyList_GetItem(diags, @intCast(i)).?;
            const r = if (path == py.Py_None())
                py.c.PyObject_CallMethod(d, "render", "(O)", self._source.?)
            else
                py.c.PyObject_CallMethod(d, "render", "(OO)", self._source.?, path);
            const rendered = r orelse return false;
            defer py.Py_DecRef(rendered);
            if (py.c.PyList_Append(lines, rendered) != 0) return false;
        }
        const nl = ph.newString("\n") orelse return false;
        defer py.Py_DecRef(nl);
        const message = py.c.PyUnicode_Join(nl, lines) orelse return false;
        defer py.Py_DecRef(message);
        const exc = py.c.PyObject_CallFunctionObjArgs(types.LoadError, message, @as(?*PyObject, null)) orelse return false;
        defer py.Py_DecRef(exc);
        if (py.c.PyObject_SetAttrString(exc, "diagnostics", diags) != 0) return false;
        py.c.PyErr_SetObject(types.LoadError, exc);
        return false;
    }

    fn language(self: *const Program) *Language {
        return unwrap(Language, self._lang.?).?;
    }

    /// `program.run(mode="python")`: run the program from its start, its
    /// semantics as Python ("python") or compiled to native code
    /// ("compiled"). Raises zrun.Error on a runtime error, the same in
    /// every mode.
    pub fn run(self: *Program, args: pyoz.Args(struct { mode: ?*PyObject = null })) ?*PyObject {
        const mode: []const u8 = if (optional(args.value.mode)) |m| ph.utf8(m, "mode") orelse return null else "python";
        if (std.mem.eql(u8, mode, "python")) return onBigStack(runHere, .{self}, self.language()._max_depth);
        if (std.mem.eql(u8, mode, "compiled")) {
            if (!self.ensureCompiled()) return null;
            return onBigStack(runCompiled, .{self}, self.language()._max_depth);
        }
        ph.raise(py.PyExc_ValueError(), "mode must be 'python' or 'compiled', not '{s}'", .{mode});
        return null;
    }

    /// What the compiler needs of the language (and of this program's tree).
    fn langView(self: *Program) compile_mod.LangView {
        const lang = self.language();
        lang.resolve();
        const st = self.state();
        const path: ?[]const u8 = if (self._path) |p| (if (p == py.Py_None()) null else ph.utf8(p, "path")) else null;
        return .{
            .grammar = lang._grammar.?,
            .eval_of = lang._eval_of,
            .exec_of = lang._exec_of,
            .functions = lang._functions,
            .read = &lang._read,
            .hosts = lang._hosts.?,
            .tree = st.ctx.?.tree,
            .analysis = st.analysis,
            .path = path,
            .python = &lang._python,
            .escaping = &lang._escaping,
        };
    }

    fn ensureCompiled(self: *Program) bool {
        if (self._compiled != null) return true;
        self._compiled = driver.compileProgram(self.ctx().data, self.langView(), &self.language()._python, types.CompileError) orelse return false;
        return true;
    }

    /// `program.compiled_ir()`: the LLVM IR the program compiles to (before
    /// LLVM optimizes it), as text.
    pub fn compiled_ir(self: *Program) ?*PyObject {
        return driver.irText(self.ctx().data, self.langView(), &self.language()._python, types.CompileError);
    }

    /// bridge.Link: a node's semantic (borrowed).
    fn linkSemantic(raw: *anyopaque, idx: u32, which: compile_mod.Which) ?*PyObject {
        const self: *Program = @ptrCast(@alignCast(raw));
        const lang = self.language();
        const rid = self.ctx().data.rule(idx);
        const table = if (which == .eval) lang._eval_of else lang._exec_of;
        return if (rid < table.len) table[rid] else null;
    }

    /// bridge.Link: a zrun.Error at a node, with a stack.
    fn linkErrorObject(raw: *anyopaque, idx: u32, message: []const u8, stack: []const helpers.CallEntry) ?*PyObject {
        const self: *Program = @ptrCast(@alignCast(raw));
        var entries: std.ArrayListUnmanaged(Runtime.StackEntry) = .empty;
        defer entries.deinit(allocator);
        for (stack) |e| entries.append(allocator, .{ .name = e.name.bytes(), .call = e.node }) catch return null;
        return Runtime.errorObject(self, idx, message, "runtime", entries.items);
    }

    fn makeNodeObject(raw: *anyopaque, idx: u32) ?*PyObject {
        const self: *Program = @ptrCast(@alignCast(raw));
        return objects.newNode(self._state.?, self.ctx(), idx);
    }

    fn runCompiled(self: *Program) ?*PyObject {
        const c = self._compiled.?;
        const st = self.state();
        const link = bridge.Link{
            .program = self,
            .compiled = c,
            .data = self.ctx().data,
            .hosts = self.language()._hosts.?,
            .analysis = st.analysis,
            .path = self._path,
            .context = null,
            .semantic = &linkSemantic,
            .error_object = &linkErrorObject,
        };
        var ectx = helpers.Ctx{
            .node_maker = .{ .ctx = self, .make_fn = &makeNodeObject, .owner = Module.selfObject(Program, self) },
            .objects = c.objects(),
            .max_depth = self.language()._max_depth,
            .link = @constCast(&link),
        };
        defer ectx.deinit();
        // (semantics run as Python recurse through Python: room for
        // max_depth calls, as in the reference mode)
        const saved_limit = py.c.Py_GetRecursionLimit();
        const want: c_int = @intCast(@min(@as(u64, ectx.max_depth) * 40 + 1000, std.math.maxInt(c_int)));
        if (want > saved_limit) py.c.Py_SetRecursionLimit(want);
        defer py.c.Py_SetRecursionLimit(saved_limit);
        const globals = helpers.zr_frame_new(null, c.globals) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        defer value_mod.decrefFrame(globals);
        if (c.main(&ectx, globals)) return none();
        // An exception from Python going out of the run: itself (a Throw
        // nothing caught: the zrun.Error it is)
        if (ectx.pending) |p| {
            ectx.pending = null;
            const t: *PyObject = @ptrCast(@alignCast(p.ob_type));
            py.c.PyErr_SetObject(t, p);
            py.Py_DecRef(p);
            return uncaught();
        }
        if (py.c.PyErr_Occurred() != null) py.c.PyErr_Clear();
        // The error, as the reference mode makes it
        var entries: std.ArrayListUnmanaged(Runtime.StackEntry) = .empty;
        defer entries.deinit(allocator);
        for (ectx.err_stack.items) |e| entries.append(allocator, .{ .name = e.name.bytes(), .call = e.node }) catch return null;
        const exc = Runtime.errorObject(self, ectx.err_node, ectx.err_msg.items, "runtime", entries.items) orelse return null;
        defer py.Py_DecRef(exc);
        py.c.PyErr_SetObject(types.Error, exc);
        return null;
    }

    fn runHere(self: *Program) ?*PyObject {
        var rt = Runtime.begin(self) orelse return null;
        defer rt.end();
        const frame = objects.newFrame(NONE, NONE, null, name_program orelse return null) orelse return null;
        rt.self()._frame = frame;
        const st = self.state();
        if (st.globals) |g| py.Py_DecRef(g);
        st.globals = ref(frame);
        const r = rt.self();
        if (!r.hoist(NONE)) return uncaught();
        if (!r.execNode(0)) return uncaught();
        return none();
    }

    /// A run ended by an exception: a rt.Throw nothing caught becomes the
    /// zrun.Error it is (made where it was raised); null.
    fn uncaught() ?*PyObject {
        if (py.c.PyErr_ExceptionMatches(types.Throw) == 0) return null;
        var t: ?*PyObject = null;
        var v: ?*PyObject = null;
        var tb: ?*PyObject = null;
        py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        const err = if (v) |exc| py.c.PyObject_GetAttrString(exc, "_zrun_error") else null;
        if (err) |e| {
            inline for (.{ t, v, tb }) |o| if (o) |x| py.Py_DecRef(x);
            py.c.PyErr_SetObject(types.Error, e);
            py.Py_DecRef(e);
        } else {
            py.c.PyErr_Clear();
            py.c.PyErr_Restore(t, v, tb);
        }
        return null;
    }

    /// `program.call(name, *args)`: call a function the program defines at
    /// its top level (running the program first if it hasn't run).
    fn callEntry(self: *Program, name: *PyObject, args: *PyObject) ?*PyObject {
        return onBigStack(callHere, .{ self, name, args }, self.language()._max_depth);
    }

    fn callHere(self: *Program, name: *PyObject, args: *PyObject) ?*PyObject {
        const st = self.state();
        if (st.globals == null) {
            const r = self.runHere() orelse return null;
            py.Py_DecRef(r);
        }
        const wanted = ph.utf8(name, "name") orelse return null;
        const data = self.ctx().data;
        for (data.syms, 0..) |s, i| {
            if (s.builtin or !std.mem.eql(u8, s.name, wanted)) continue;
            if (data.homeOf(@intCast(i)) != NONE) continue;
            const g = objects.asFrame(st.globals.?);
            const f = g.slots.?.get(@intCast(i)) orelse continue;
            var rt = Runtime.begin(self) orelse return null;
            defer rt.end();
            rt.self()._frame = ref(st.globals.?);
            return rt.self().callValue(f, args, null) orelse uncaught();
        }
        ph.raise(py.PyExc_KeyError(), "the program defines no function '{s}'", .{wanted});
        return null;
    }

    /// `program.call(name, *args)`: call a function the program defines
    /// at its top level (running the program first if it hasn't run)
    pub fn get_call(self: *const Program) ?*PyObject {
        const alloc: py.c.allocfunc = @ptrCast(py.c.PyType_GetSlot(@ptrCast(CallerType), py.c.Py_tp_alloc));
        const o = alloc.?(@ptrCast(CallerType), 0) orelse return null;
        const c: *CallerObject = @ptrCast(@alignCast(o));
        c.program = ref(Module.selfObject(Program, self));
        return o;
    }

    /// The program's source
    pub fn get_source(self: *const Program) ?*PyObject {
        return ref(self._source.?);
    }

    /// The zgram Tree
    pub fn get_tree(self: *const Program) ?*PyObject {
        return ref(self.ctx().tree);
    }

    /// The zrules Analysis (None without rules)
    pub fn get_analysis(self: *const Program) ?*PyObject {
        return ref(self.state().analysis orelse py.Py_None());
    }

    /// The warnings found loading it
    pub fn get_diagnostics(self: *const Program) ?*PyObject {
        return ref(self.state().diagnostics.?);
    }

    /// The root node
    pub fn get_root(self: *const Program) ?*PyObject {
        return objects.newNode(self._state.?, self.ctx(), 0);
    }

    pub const __doc__: [*:0]const u8 = "A program loaded by Language.load(): run() runs it, call(name, *args) calls one of its functions. Also: source, tree, analysis, diagnostics (its warnings), root.";
    pub const run__doc__: [*:0]const u8 = "Run the program from its start. Raises zrun.Error on a runtime error.";
};

var name_program: ?*PyObject = null;

/// Run `f(args)` on a thread of its own with a stack for `max_depth`
/// calls of the language: semantics recurse through Python and native
/// frames, more than a default stack holds (8 MB on Linux, 1 MB on
/// Windows). The calling thread waits without the GIL; the result and any
/// exception come back to it.
fn onBigStack(comptime f: anytype, args: anytype, max_depth: u32) ?*PyObject {
    const Ctx = struct {
        args: @TypeOf(args),
        result: ?*PyObject = null,
        t: ?*PyObject = null,
        v: ?*PyObject = null,
        tb: ?*PyObject = null,

        fn work(c: *@This()) void {
            const g = py.c.PyGILState_Ensure();
            c.result = @call(.auto, f, c.args);
            if (c.result == null) py.c.PyErr_Fetch(@ptrCast(&c.t), @ptrCast(&c.v), @ptrCast(&c.tb));
            py.c.PyGILState_Release(g);
        }
    };
    var ctx = Ctx{ .args = args };
    // About 64 KB per call leaves room for deep semantics and debug builds
    const stack: usize = @min(@as(usize, max_depth) * 64 * 1024 + 16 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
    const ts = py.c.PyEval_SaveThread();
    const thread = std.Thread.spawn(.{ .stack_size = stack }, Ctx.work, .{&ctx}) catch {
        py.c.PyEval_RestoreThread(ts);
        _ = py.c.PyErr_NoMemory();
        return null;
    };
    thread.join();
    py.c.PyEval_RestoreThread(ts);
    if (ctx.result == null) py.c.PyErr_Restore(ctx.t, ctx.v, ctx.tb);
    return ctx.result;
}

/// `program.call`: a callable taking `(name, *args)` (PyOZ methods take
/// fixed arguments, and its types can't get methods added: the property
/// returns this)
const CallerObject = extern struct {
    ob_base: py.c.PyObject,
    program: ?*PyObject,
};

var CallerType: *PyObject = undefined;

fn callerCall(self_obj: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const c: *CallerObject = @ptrCast(@alignCast(self_obj.?));
    if (kwargs != null and py.c.PyDict_Size(kwargs) != 0) {
        ph.raise(py.PyExc_TypeError(), "call(name, *args) takes no keyword arguments", .{});
        return null;
    }
    const n = py.c.PyTuple_Size(args);
    if (n < 1) {
        ph.raise(py.PyExc_TypeError(), "call(name, *args) needs the function's name", .{});
        return null;
    }
    const self = unwrap(Program, c.program.?) orelse return null;
    const rest = py.c.PyTuple_GetSlice(args, 1, n) orelse return null;
    defer py.Py_DecRef(rest);
    return self.callEntry(py.c.PyTuple_GetItem(args, 0).?, rest);
}

fn callerDealloc(obj: ?*PyObject) callconv(.c) void {
    const c: *CallerObject = @ptrCast(@alignCast(obj.?));
    if (c.program) |p| py.Py_DecRef(p);
    const t = obj.?.ob_type;
    const free: py.c.freefunc = @ptrCast(py.c.PyType_GetSlot(t, py.c.Py_tp_free));
    free.?(obj);
    py.Py_DecRef(@ptrCast(@alignCast(t)));
}

var caller_slots = [_]py.c.PyType_Slot{
    .{ .slot = py.c.Py_tp_call, .pfunc = @ptrCast(@constCast(&callerCall)) },
    .{ .slot = py.c.Py_tp_dealloc, .pfunc = @ptrCast(@constCast(&callerDealloc)) },
    .{ .slot = py.c.Py_tp_doc, .pfunc = @ptrCast(@constCast("call(name, *args): call a function the program defines at its top level (running the program first if it hasn't run).")) },
    .{ .slot = 0, .pfunc = null },
};

var caller_spec = py.c.PyType_Spec{
    .name = "zrun.ProgramCall",
    .basicsize = @sizeOf(CallerObject),
    .itemsize = 0,
    .flags = py.c.Py_TPFLAGS_DEFAULT,
    .slots = &caller_slots,
};

// ============================================================================
// Runtime: rt
// ============================================================================

const Runtime = struct {
    _program: ?*PyObject = null,
    _p: ?*Program = null,
    _lang: ?*Language = null,
    _frame: ?*PyObject = null,
    _at: u32 = NONE,
    _depth: u32 = 0,
    /// The frames of the calls being run, outermost first (borrowed: each
    /// is referenced by the call running it)
    _calls: std.ArrayListUnmanaged(*PyObject) = .empty,
    _context: ?*PyObject = null,
    _saved_limit: c_int = 0,

    /// A runtime for one run of a program (or a call into it): the object
    /// and its data.
    const Handle = struct {
        obj: *PyObject,

        fn self(h: Handle) *Runtime {
            return unwrap(Runtime, h.obj).?;
        }

        fn end(h: Handle) void {
            const r = h.self();
            _ = py.c.Py_SetRecursionLimit(r._saved_limit);
            py.Py_DecRef(h.obj);
        }
    };

    fn begin(p: *Program) ?Handle {
        var r = Runtime{};
        r._program = ref(Module.selfObject(Program, p));
        r._p = p;
        r._lang = p.language();
        const rt_obj = Module.toPy(Runtime, r) orelse return null;
        const h = Handle{ .obj = rt_obj };
        const s = h.self();
        // Semantics recurse through Python: room for max_depth calls
        s._saved_limit = py.c.Py_GetRecursionLimit();
        const want: c_int = @intCast(@min(@as(u64, s._lang.?._max_depth) * 40 + 1000, std.math.maxInt(c_int)));
        if (want > s._saved_limit) py.c.Py_SetRecursionLimit(want);
        return h;
    }

    pub fn __del__(self: *Runtime) void {
        inline for (.{ "_program", "_frame", "_context" }) |f| {
            if (@field(self, f)) |o| py.Py_DecRef(o);
            @field(self, f) = null;
        }
        self._calls.deinit(allocator);
    }

    fn obj(self: *Runtime) *PyObject {
        return Module.selfObject(Runtime, self);
    }

    fn data(self: *const Runtime) *program_mod.Data {
        return self._p.?.ctx().data;
    }

    fn ctx(self: *Runtime) *objects.Context {
        return self._p.?.ctx();
    }

    /// The program's State object (borrowed: the program holds it)
    fn stateObj(self: *Runtime) *PyObject {
        return self._p.?._state.?;
    }

    fn node(self: *Runtime, idx: u32) ?*PyObject {
        return objects.newNode(self.stateObj(), self.ctx(), idx);
    }

    /// The node index of a Node object of this program, or null with
    /// TypeError.
    fn nodeIndex(self: *Runtime, o: *PyObject, what: []const u8) ?u32 {
        if (objects.asNode(o)) |n| {
            if (n.ctx == self.ctx()) return n.idx;
        }
        ph.raise(py.PyExc_TypeError(), "{s} must be a node of the program", .{what});
        return null;
    }

    // ------------------------------------------------------------------
    // Running nodes
    // ------------------------------------------------------------------

    /// `rt.eval(x)`: the value of an expression node (a list: of each);
    /// a value that isn't a node is its own value.
    pub fn eval(self: *Runtime, x: *PyObject) ?*PyObject {
        return self.evalObj(x);
    }

    fn evalObj(self: *Runtime, x: *PyObject) ?*PyObject {
        if (objects.asNode(x)) |n| {
            if (n.ctx == self.ctx()) return self.evalNode(n.idx);
        }
        if (py.PyList_Check(x)) {
            const n: usize = @intCast(py.c.PyList_Size(x));
            const out = py.c.PyList_New(@intCast(n)) orelse return null;
            for (0..n) |i| {
                const v = self.evalObj(py.c.PyList_GetItem(x, @intCast(i)).?) orelse {
                    py.Py_DecRef(out);
                    return null;
                };
                _ = py.c.PyList_SetItem(out, @intCast(i), v);
            }
            return out;
        }
        return types.wrap(x);
    }

    fn evalNode(self: *Runtime, idx: u32) ?*PyObject {
        if (!self.data().hasFrame(idx)) return self.evalHere(idx);
        const saved = self.enterScope(idx) orelse return null;
        defer self.leaveScope(saved);
        return self.evalHere(idx);
    }

    /// A block scope with frames of its own, entered: its frame (a new
    /// one) is the current frame, its hoisted functions are defined. The
    /// frame it replaces (for leaveScope), or null with an exception.
    fn enterScope(self: *Runtime, idx: u32) ?*PyObject {
        const saved = self._frame.?;
        const frame = objects.newFrame(idx, NONE, saved, objects.asFrame(saved).name.?) orelse return null;
        self._frame = frame;
        if (!self.hoist(idx)) {
            self.leaveScope(saved);
            return null;
        }
        return saved;
    }

    fn leaveScope(self: *Runtime, saved: *PyObject) void {
        const frame = self._frame.?;
        self._frame = saved;
        py.Py_DecRef(frame);
    }

    /// `rt.fresh(node)`: from here, the variables of the block scope `node`
    /// (being run) are new ones, closures made so far keeping theirs: a
    /// loop's variable is a new one each time round (Lua's `for`,
    /// JavaScript's `for (let ...)`). Nothing for a scope whose variables no
    /// closure sees.
    pub fn fresh(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "node") orelse return null;
        if (!self.data().hasFrame(idx)) return none();
        const cur = self._frame.?;
        const f = objects.asFrame(cur);
        if (f.scope != idx) return self.fail(idx, "rt.fresh(): this scope isn't the one being run", .{});
        const frame = objects.newFrame(idx, NONE, f.parent, f.name.?) orelse return null;
        self._frame = frame;
        py.Py_DecRef(cur);
        return none();
    }

    fn evalHere(self: *Runtime, idx: u32) ?*PyObject {
        const prev = self._at;
        self._at = idx;
        defer self._at = prev;
        const lang = self._lang.?;
        lang.resolve();
        const rid = self.data().rule(idx);
        if (rid < lang._eval_of.len) {
            if (lang._eval_of[rid]) |f| {
                const n = self.node(idx) orelse return null;
                defer py.Py_DecRef(n);
                const r = py.c.PyObject_CallFunctionObjArgs(f, n, self.obj(), @as(?*PyObject, null)) orelse return self.raised(idx);
                return types.wrapOwned(r) orelse self.raised(idx);
            }
        }
        return self.defaultEval(idx);
    }

    fn defaultEval(self: *Runtime, idx: u32) ?*PyObject {
        if (self.data().symbolIndex(idx) != null) return self.loadNode(idx);
        const values = objects.childValues(self.stateObj(), self.ctx(), idx) orelse return self.raised(idx);
        defer py.Py_DecRef(values);
        if (py.c.PyList_Size(values) == 1) return self.evalObj(py.c.PyList_GetItem(values, 0).?);
        const g = self.data().grammar;
        return self.fail(idx, "no semantics to evaluate {s}", .{g.kind_names[self.data().rule(idx)]});
    }

    /// `rt.exec(x)`: run a statement node (a list: each in order).
    pub fn exec(self: *Runtime, x: *PyObject) ?*PyObject {
        if (!self.execObj(x)) return null;
        return none();
    }

    fn execObj(self: *Runtime, x: *PyObject) bool {
        if (objects.asNode(x)) |n| {
            if (n.ctx == self.ctx()) return self.execNode(n.idx);
        }
        if (py.PyList_Check(x) or py.PyTuple_Check(x)) {
            const seq = py.c.PySequence_Fast(x, "") orelse return false;
            defer py.Py_DecRef(seq);
            const n: usize = @intCast(py.c.PySequence_Size(seq));
            for (0..n) |i| {
                const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse return false;
                defer py.Py_DecRef(item);
                if (!self.execObj(item)) return false;
            }
        }
        return true;
    }

    fn execNode(self: *Runtime, idx: u32) bool {
        if (!self.data().hasFrame(idx)) return self.execHere(idx);
        const saved = self.enterScope(idx) orelse return false;
        defer self.leaveScope(saved);
        return self.execHere(idx);
    }

    fn execHere(self: *Runtime, idx: u32) bool {
        const prev = self._at;
        self._at = idx;
        defer self._at = prev;
        const lang = self._lang.?;
        lang.resolve();
        const rid = self.data().rule(idx);
        if (rid < lang._exec_of.len) {
            if (lang._exec_of[rid]) |f| {
                const n = self.node(idx) orelse return false;
                defer py.Py_DecRef(n);
                const r = py.c.PyObject_CallFunctionObjArgs(f, n, self.obj(), @as(?*PyObject, null)) orelse {
                    _ = self.raised(idx);
                    return false;
                };
                py.Py_DecRef(r);
                return true;
            }
        }
        // No exec semantics: a function definition (unless hoisted), an
        // expression run for its effect, or the children in order
        if (rid < lang._functions.len) {
            if (lang._functions[rid]) |spec| {
                if (spec.hoist) return true;
                return self.defineFunction(idx, spec);
            }
        }
        if (rid < lang._eval_of.len and lang._eval_of[rid] != null) {
            const v = self.evalNode(idx) orelse return false;
            py.Py_DecRef(v);
            return true;
        }
        const values = objects.childValues(self.stateObj(), self.ctx(), idx) orelse {
            _ = self.raised(idx);
            return false;
        };
        defer py.Py_DecRef(values);
        return self.execObj(values);
    }

    /// `rt.loop(body)`: run a loop's body once; False if it broke out.
    pub fn loop(self: *Runtime, body: *PyObject) ?*PyObject {
        if (self.execObj(body)) return ref(py.Py_True());
        switch (types.pendingControl()) {
            .brk => {
                py.c.PyErr_Clear();
                return ref(py.Py_False());
            },
            .cont => {
                py.c.PyErr_Clear();
                return ref(py.Py_True());
            },
            else => return null,
        }
    }

    // ------------------------------------------------------------------
    // Variables
    // ------------------------------------------------------------------

    /// `rt.load(name)`: the value of the variable a name node refers to;
    /// a builtin's is the host function of its name.
    pub fn load(self: *Runtime, name: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(name, "a variable's name") orelse return null;
        return self.loadNode(idx);
    }

    fn loadNode(self: *Runtime, idx: u32) ?*PyObject {
        const d = self.data();
        const si = d.symbolIndex(idx) orelse return self.notAVariable(idx);
        const sym = d.syms[si];
        if (sym.builtin) {
            const key = ph.newString(sym.name) orelse return null;
            defer py.Py_DecRef(key);
            const h = py.c.PyDict_GetItem(self._lang.?._hosts.?, key) orelse
                return self.fail(idx, "no host function for the builtin '{s}'", .{sym.name});
            return ref(h);
        }
        const f = self.frameFor(si, idx) orelse return null;
        const v = f.slots.?.get(si) orelse return self.fail(idx, "'{s}' has no value yet", .{sym.name});
        return ref(v);
    }

    /// `rt.store(name, value)`: set the variable a name node refers to.
    pub fn store(self: *Runtime, name: *PyObject, value: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(name, "a variable's name") orelse return null;
        if (!self.storeNode(idx, value)) return null;
        return none();
    }

    fn storeNode(self: *Runtime, idx: u32, value: *PyObject) bool {
        const d = self.data();
        const si = d.symbolIndex(idx) orelse {
            _ = self.notAVariable(idx);
            return false;
        };
        if (d.syms[si].builtin) {
            _ = self.fail(idx, "can't assign to the builtin '{s}'", .{d.syms[si].name});
            return false;
        }
        const f = self.frameFor(si, idx) orelse return false;
        const v = types.wrap(value) orelse {
            _ = self.raised(idx);
            return false;
        };
        defer py.Py_DecRef(v);
        objects.setSlot(f, si, v) catch {
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        return true;
    }

    fn notAVariable(self: *Runtime, idx: u32) ?*PyObject {
        if (self._p.?.state().analysis == null) return self.fail(idx, "variables need rules with a scopes() rule", .{});
        return self.fail(idx, "'{s}' is not a variable", .{self.data().text(idx)});
    }

    /// The frame a symbol's variable is in: the innermost frame running
    /// its function, through the frames functions were made in.
    fn frameFor(self: *Runtime, sym: u32, at: u32) ?*objects.FrameObject {
        const home = self.data().homeOf(sym);
        var fo = self._frame;
        while (fo) |o| {
            const f = objects.asFrame(o);
            if (f.scope == home) return f;
            fo = f.parent;
        }
        _ = self.fail(at, "'{s}' isn't reachable from here", .{self.data().syms[sym].name});
        return null;
    }

    // ------------------------------------------------------------------
    // Functions
    // ------------------------------------------------------------------

    /// `rt.function(node)`: a function made from a node of a function kind,
    /// seeing the variables of where it's made.
    pub fn function(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "a function's node") orelse return null;
        return self.makeFunction(idx);
    }

    fn makeFunction(self: *Runtime, idx: u32) ?*PyObject {
        const d = self.data();
        const spec = self.specOf(idx) orelse {
            return self.fail(idx, "{s} isn't a function kind (Language.function)", .{d.grammar.kind_names[d.rule(idx)]});
        };
        const name_text = if (spec.name != 0) if (program_mod.labelled(d, idx, spec.name)) |nn| d.text(nn) else "<anonymous>" else "<anonymous>";
        const name = ph.newString(name_text) orelse return null;
        defer py.Py_DecRef(name);
        return objects.newFunction(self.stateObj(), idx, self._frame, name);
    }

    fn specOf(self: *Runtime, idx: u32) ?FunctionSpec {
        const rid = self.data().rule(idx);
        const fns = self._lang.?._functions;
        return if (rid < fns.len) fns[rid] else null;
    }

    fn defineFunction(self: *Runtime, idx: u32, spec: FunctionSpec) bool {
        const name_node = program_mod.labelled(self.data(), idx, spec.name) orelse return true;
        const f = self.makeFunction(idx) orelse return false;
        defer py.Py_DecRef(f);
        return self.storeNode(name_node, f);
    }

    /// Define the hoisted functions of the scope just entered.
    fn hoist(self: *Runtime, scope_node: u32) bool {
        const list = self.data().hoisted.get(scope_node) orelse return true;
        for (list.items) |fnode| {
            if (!self.defineFunction(fnode, self.specOf(fnode).?)) return false;
        }
        return true;
    }

    /// `rt.call(f, args, receiver=None)`: call a function of the program, or
    /// a host function, with a list of arguments; `receiver` is what a
    /// method is called on (`rt.receiver` in its body; a host function gets
    /// it as its first argument).
    pub fn call(self: *Runtime, a: pyoz.Args(struct { f: *PyObject, args: *PyObject, receiver: ?*PyObject = null })) ?*PyObject {
        const v = a.value;
        const tuple = py.c.PySequence_Tuple(v.args) orelse return null;
        defer py.Py_DecRef(tuple);
        return self.callValue(v.f, tuple, optional(v.receiver));
    }

    fn callValue(self: *Runtime, f: *PyObject, args: *PyObject, receiver: ?*PyObject) ?*PyObject {
        if (objects.asFunction(f)) |fo| return self.callFunction(fo, args, receiver);
        if (py.PyCallable_Check(f)) {
            const all = if (receiver) |r| prepend(r, args) orelse return null else ref(args);
            defer py.Py_DecRef(all);
            const wrapped = wrapAll(all) orelse return null;
            defer py.Py_DecRef(wrapped);
            const r = py.c.PyObject_CallObject(f, wrapped) orelse return self.hostFailed(f);
            return types.wrapOwned(r) orelse self.raised(self._at);
        }
        const tname = typeName(f);
        return self.fail(self._at, "'{s}' value is not callable", .{tname});
    }

    fn hostFailed(self: *Runtime, f: *PyObject) ?*PyObject {
        if (types.pendingControl() != .none or py.c.PyErr_ExceptionMatches(types.Error) != 0) return null;
        // (a host function can throw an error of the language too)
        if (py.c.PyErr_ExceptionMatches(types.Throw) != 0) {
            self.recordThrow(self._at);
            return null;
        }
        var buf: [512]u8 = undefined;
        const message = ph.takeError(&buf);
        var name_buf: [128]u8 = undefined;
        const name = blk: {
            const n = py.c.PyObject_GetAttrString(f, "__name__") orelse {
                py.c.PyErr_Clear();
                break :blk "host function";
            };
            defer py.Py_DecRef(n);
            const s = ph.utf8(n, "name") orelse {
                py.c.PyErr_Clear();
                break :blk "host function";
            };
            const len = @min(s.len, name_buf.len);
            @memcpy(name_buf[0..len], s[0..len]);
            break :blk name_buf[0..len];
        };
        return self.fail(self._at, "{s}: {s}", .{ name, message });
    }

    fn callFunction(self: *Runtime, fo: *objects.FunctionObject, args: *PyObject, receiver: ?*PyObject) ?*PyObject {
        if (fo.state != self.stateObj()) return self.fail(self._at, "a function of another program can't be called here", .{});
        const fnode = fo.node;
        const spec = self.specOf(fnode).?;
        const program = self.stateObj();
        // The parameters: a list, one, or none
        const params = if (spec.params != 0) objects.fieldOf(program, self.ctx(), fnode, spec.params) orelse return null else py.c.PyList_New(0) orelse return null;
        defer py.Py_DecRef(params);
        const plist = if (py.PyList_Check(params)) ref(params) else if (params == py.Py_None()) py.c.PyList_New(0) orelse return null else py.c.PyList_New(1) orelse return null;
        defer py.Py_DecRef(plist);
        if (!py.PyList_Check(params) and params != py.Py_None()) _ = py.c.PyList_SetItem(plist, 0, ref(params));
        const n_params: usize = @intCast(py.c.PyList_Size(plist));
        const n_args: usize = @intCast(py.c.PyTuple_Size(args));
        const fname = ph.utf8(fo.name.?, "name") orelse return null;
        if ((n_args < n_params and spec.missing == .@"error") or (n_args > n_params and spec.extra == .@"error")) {
            return self.fail(self._at, "{s}() takes {d} argument{s}, {d} given", .{ fname, n_params, if (n_params == 1) "" else "s", n_args });
        }
        const max = self._lang.?._max_depth;
        if (self._depth >= max) return self.fail(self._at, "call stack too deep (more than {d} calls)", .{max});

        const frame = objects.newFrame(fnode, self._at, fo.env, fo.name.?) orelse return null;
        if (receiver) |r| objects.asFrame(frame).receiver = ref(r);
        if (spec.extra == .keep) {
            objects.asFrame(frame).varargs = py.c.PyTuple_GetSlice(args, @intCast(@min(n_params, n_args)), @intCast(n_args)) orelse {
                py.Py_DecRef(frame);
                return null;
            };
        }
        const saved = self._frame;
        self._frame = frame;
        self._depth += 1;
        self._calls.append(allocator, frame) catch {
            self._frame = saved;
            self._depth -= 1;
            py.Py_DecRef(frame);
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        defer {
            _ = self._calls.pop();
            self._depth -= 1;
            self._frame = saved;
            py.Py_DecRef(frame);
        }

        if (!self.hoist(fnode)) return null;
        for (0..n_params) |i| {
            const p = py.c.PyList_GetItem(plist, @intCast(i)).?;
            const target = self.paramName(p) orelse return null;
            const arg = if (i < n_args) py.c.PyTuple_GetItem(args, @intCast(i)).? else py.Py_None();
            if (!self.storeNode(target, arg)) return null;
        }
        const body = if (spec.body != 0)
            objects.fieldOf(program, self.ctx(), fnode, spec.body) orelse return null
        else if (program_mod.childOfRule(self.data(), fnode, spec.body_rule)) |b|
            objects.valueOf(program, self.ctx(), b) orelse return null
        else
            none();
        defer py.Py_DecRef(body);
        if (self.execObj(body)) return none();
        if (types.pendingControl() == .ret) return types.takeReturn();
        if (types.pendingControl() != .none) {
            // Break or Continue outside a loop: an error of the language
            py.c.PyErr_Clear();
            return self.fail(self._at, "break or continue outside a loop", .{});
        }
        return null;
    }

    /// The node a parameter's value is stored under: the parameter itself
    /// if it defines a variable, else its `name` field.
    fn paramName(self: *Runtime, p: *PyObject) ?u32 {
        const d = self.data();
        if (objects.asNode(p)) |n| {
            if (d.symbolIndex(n.idx)) |si| {
                if (d.syms[si].node == n.idx) return n.idx;
            }
            if (d.grammar.field_ids.get("name")) |field| {
                if (program_mod.labelled(d, n.idx, field)) |c| return c;
            }
            return n.idx;
        }
        ph.raise(py.PyExc_TypeError(), "a function's parameters must be nodes", .{});
        return null;
    }

    // ------------------------------------------------------------------
    // Helpers for semantics
    // ------------------------------------------------------------------

    /// `rt.error(node, message, code="runtime")`: stop the program with a
    /// runtime error at the node.
    pub fn @"error"(self: *Runtime, args: pyoz.Args(struct { node: *PyObject, message: *PyObject, code: ?*PyObject = null })) ?*PyObject {
        const v = args.value;
        const idx = if (v.node == py.Py_None()) self._at else self.nodeIndex(v.node, "node") orelse return null;
        const msg = ph.utf8(v.message, "message") orelse return null;
        const code = if (optional(v.code)) |c| ph.utf8(c, "code") orelse return null else "runtime";
        return self.raiseError(idx, msg, code);
    }

    pub fn kind(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "node") orelse return null;
        const d = self.data();
        return ref(d.grammar.kinds[d.rule(idx)]);
    }

    pub fn text(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "node") orelse return null;
        return ph.newString(self.data().text(idx));
    }

    pub fn span(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "node") orelse return null;
        const nd = self.data().nodes[idx];
        return py.c.Py_BuildValue("(II)", @as(c_uint, nd.text_start), @as(c_uint, nd.text_end));
    }

    /// What the method being run was called on (rt.call(..., receiver=)):
    /// the innermost one through the frames functions were made in; None
    /// outside a method
    pub fn get_receiver(self: *const Runtime) ?*PyObject {
        var fo = self._frame;
        while (fo) |o| {
            const f = objects.asFrame(o);
            if (f.receiver) |r| return ref(r);
            fo = f.parent;
        }
        return none();
    }

    /// `rt.scope(node)`: the scope node the name `node` refers to is
    /// defined in (a function, a struct, the program if the rules make it a
    /// scope), or None for builtins and the global scope.
    pub fn scope(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "node") orelse return null;
        const d = self.data();
        const si = d.symbolIndex(idx) orelse return none();
        const s = d.syms[si].scope;
        if (s == NONE or s >= d.nodes.len) return none();
        return self.node(s);
    }

    /// `rt.symbol(node)`: zrules' Symbol for the name `node` defines or
    /// uses (its name, type, defining node, scope...), or None.
    pub fn symbol(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "node") orelse return null;
        const analysis = self._p.?.state().analysis orelse return none();
        return py.c.PyObject_CallMethod(analysis, "resolve", "I", @as(c_uint, idx));
    }

    /// `rt.type_of(node)`: the type zrules' types() rule gave a node, as
    /// text (`int`, `list[float]`, `Point?`), or None.
    pub fn type_of(self: *Runtime, n: *PyObject) ?*PyObject {
        const idx = self.nodeIndex(n, "node") orelse return null;
        const analysis = self._p.?.state().analysis orelse return none();
        return py.c.PyObject_CallMethod(analysis, "type_of", "I", @as(c_uint, idx));
    }

    /// `rt.node(index)`: the node at an index of the tree (symbols refer to
    /// nodes by index).
    pub fn node_at(self: *Runtime, index: i64) ?*PyObject {
        if (index < 0 or index >= self.data().nodes.len) {
            ph.raise(py.PyExc_IndexError(), "no node {d}", .{index});
            return null;
        }
        return self.node(@intCast(index));
    }

    /// `rt.path`: the name the program was loaded under (lang.load(source,
    /// path)), or None.
    pub fn get_path(self: *const Runtime) ?*PyObject {
        return ref(self._p.?._path orelse py.Py_None());
    }

    /// `rt.varargs`: the arguments the function being run got beyond its
    /// parameters (a function kind with extra="keep"); () otherwise and at
    /// the top level.
    pub fn get_varargs(self: *const Runtime) ?*PyObject {
        // (the function's frame: the block scopes' frames are inside it)
        var fo = self._frame;
        while (fo) |o| {
            const f = objects.asFrame(o);
            if (f.varargs) |v| return ref(v);
            if (f.scope == NONE or !self.data().hasFrame(f.scope)) break;
            fo = f.parent;
        }
        return py.c.PyTuple_New(0);
    }

    /// What the host passed to the run (None without)
    pub fn get_context(self: *const Runtime) ?*PyObject {
        return ref(self._context orelse py.Py_None());
    }

    /// raise rt.Return(value)
    pub fn get_Return(_: *const Runtime) ?*PyObject {
        return ref(types.Return);
    }

    /// raise rt.Break()
    pub fn get_Break(_: *const Runtime) ?*PyObject {
        return ref(types.Break);
    }

    /// raise rt.Continue()
    pub fn get_Continue(_: *const Runtime) ?*PyObject {
        return ref(types.Continue);
    }

    /// raise rt.Throw(value, message=None)
    pub fn get_Throw(_: *const Runtime) ?*PyObject {
        return ref(types.Throw);
    }

    // ------------------------------------------------------------------
    // Errors
    // ------------------------------------------------------------------

    /// Raise a runtime error at a node; null.
    fn fail(self: *Runtime, idx: u32, comptime fmt: []const u8, args: anytype) ?*PyObject {
        var buf: [1024]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..];
        return self.raiseError(idx, msg, "runtime");
    }

    /// After a call into Python failed at `idx`: control flow and zrun
    /// errors go on as they are; a Python exception becomes a runtime error
    /// at the node. Null.
    fn raised(self: *Runtime, idx: u32) ?*PyObject {
        if (py.c.PyErr_Occurred() == null) return null;
        if (types.pendingControl() != .none or py.c.PyErr_ExceptionMatches(types.Error) != 0) return null;
        if (py.c.PyErr_ExceptionMatches(types.Throw) != 0) {
            self.recordThrow(idx);
            return null;
        }
        const msg = pythonMessage() orelse return null;
        defer py.Py_DecRef(msg);
        const text_msg = ph.utf8(msg, "message") orelse return null;
        return self.raiseError(idx, text_msg, "runtime");
    }

    /// A rt.Throw going by a semantic for the first time: where it was
    /// raised, kept with it (as the zrun.Error it is if nothing catches it).
    fn recordThrow(self: *Runtime, idx: u32) void {
        var t: ?*PyObject = null;
        var v: ?*PyObject = null;
        var tb: ?*PyObject = null;
        py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        defer py.c.PyErr_Restore(t, v, tb);
        const exc = v orelse return;
        if (py.c.PyObject_HasAttrString(exc, "_zrun_error") == 1) return;
        const message = throwMessage(exc) orelse {
            py.c.PyErr_Clear();
            return;
        };
        defer py.Py_DecRef(message);
        const text_msg = ph.utf8(message, "message") orelse {
            py.c.PyErr_Clear();
            return;
        };
        const err = self.makeError(idx, text_msg, "runtime") orelse {
            py.c.PyErr_Clear();
            return;
        };
        defer py.Py_DecRef(err);
        if (py.c.PyObject_SetAttrString(exc, "_zrun_error", err) != 0) py.c.PyErr_Clear();
    }

    /// A Throw's message: its own, or str() of its value.
    fn throwMessage(exc: *PyObject) ?*PyObject {
        const m = py.c.PyObject_GetAttrString(exc, "message") orelse return null;
        if (m != py.Py_None()) return py.c.PyObject_Str(m);
        py.Py_DecRef(m);
        const val = py.c.PyObject_GetAttrString(exc, "value") orelse return null;
        defer py.Py_DecRef(val);
        return py.c.PyObject_Str(val);
    }

    fn raiseError(self: *Runtime, idx: u32, message: []const u8, code: []const u8) ?*PyObject {
        const exc = self.makeError(idx, message, code) orelse return null;
        defer py.Py_DecRef(exc);
        py.c.PyErr_SetObject(types.Error, exc);
        return null;
    }

    /// A zrun.Error at a node, with the calls being run.
    fn makeError(self: *Runtime, idx: u32, message: []const u8, code: []const u8) ?*PyObject {
        var entries: std.ArrayListUnmanaged(StackEntry) = .empty;
        defer entries.deinit(allocator);
        var i = self._calls.items.len;
        while (i > 0) {
            i -= 1;
            const f = objects.asFrame(self._calls.items[i]);
            const fname = ph.utf8(f.name.?, "name") orelse return null;
            entries.append(allocator, .{ .name = fname, .call = f.call }) catch return null;
        }
        return errorObject(self._p.?, idx, message, code, entries.items);
    }

    /// A frame of an error's stack: the function, the node calling it
    const StackEntry = struct { name: []const u8, call: u32 };

    /// A zrun.Error: its diagnostic at the node, the stack of calls (innermost
    /// first), the rendered text as its message. (Both modes make theirs here.)
    fn errorObject(p: *Program, idx: u32, message: []const u8, code: []const u8, entries: []const StackEntry) ?*PyObject {
        const d = p.ctx().data;
        const at = if (idx < d.nodes.len) d.nodes[idx] else tree_mod.FlatNode{ .text_start = 0, .text_end = 0, .subtree_size = 0, .meta = 0 };
        const diag = diagnostic("error", code, message, at.text_start, at.text_end, d) orelse return null;
        defer py.Py_DecRef(diag);
        const stack = py.c.PyList_New(0) orelse return null;
        defer py.Py_DecRef(stack);
        for (entries) |e| {
            const call_node = if (e.call < d.nodes.len) d.nodes[e.call] else at;
            var buf: [256]u8 = undefined;
            const note = std.fmt.bufPrint(&buf, "in {s}()", .{e.name}) catch "in a call";
            const nd = diagnostic("note", "call", note, call_node.text_start, call_node.text_end, d) orelse return null;
            defer py.Py_DecRef(nd);
            const name = ph.newString(e.name) orelse return null;
            defer py.Py_DecRef(name);
            const pair = py.c.PyTuple_Pack(2, name, nd) orelse return null;
            defer py.Py_DecRef(pair);
            if (py.c.PyList_Append(stack, pair) != 0) return null;
        }
        const rendered = renderError(p, diag, stack) orelse return null;
        defer py.Py_DecRef(rendered);
        const exc = py.c.PyObject_CallFunctionObjArgs(types.Error, rendered, @as(?*PyObject, null)) orelse return null;
        if (py.c.PyObject_SetAttrString(exc, "diagnostic", diag) != 0 or py.c.PyObject_SetAttrString(exc, "stack", stack) != 0) {
            py.Py_DecRef(exc);
            return null;
        }
        return exc;
    }

    /// The error as a compiler prints it: the diagnostic with the source, then
    /// each call that led there.
    fn renderError(p: *Program, diag: *PyObject, stack: *PyObject) ?*PyObject {
        const path = p._path orelse py.Py_None();
        const main = (if (path == py.Py_None())
            py.c.PyObject_CallMethod(diag, "render", "(O)", p._source.?)
        else
            py.c.PyObject_CallMethod(diag, "render", "(OO)", p._source.?, path)) orelse return null;
        defer py.Py_DecRef(main);
        const lines = py.c.PyList_New(0) orelse return null;
        defer py.Py_DecRef(lines);
        if (py.c.PyList_Append(lines, main) != 0) return null;
        const where: []const u8 = if (path == py.Py_None()) "<program>" else ph.utf8(path, "path") orelse return null;
        const n: usize = @intCast(py.c.PyList_Size(stack));
        for (0..n) |i| {
            const pair = py.c.PyList_GetItem(stack, @intCast(i)).?;
            const fname = ph.utf8(py.c.PyTuple_GetItem(pair, 0).?, "name") orelse return null;
            const nd = py.c.PyTuple_GetItem(pair, 1).?;
            const line = attrInt(nd, "line") orelse return null;
            const col = attrInt(nd, "column") orelse return null;
            var buf: [1024]u8 = undefined;
            const text_line = std.fmt.bufPrint(&buf, "  in {s}(), called at {s}:{d}:{d}", .{ fname, where, line, col }) catch continue;
            const s = ph.newString(text_line) orelse return null;
            defer py.Py_DecRef(s);
            if (py.c.PyList_Append(lines, s) != 0) return null;
        }
        const nl = ph.newString("\n") orelse return null;
        defer py.Py_DecRef(nl);
        return py.c.PyUnicode_Join(nl, lines);
    }

    pub const __doc__: [*:0]const u8 = "rt: what semantics use to run the program. eval(x) / exec(x) run a child node (or each of a list); load(name) / store(name, value) read and write the variable a name node refers to; function(node) makes a function, call(f, args) calls one (or a host function); loop(body) runs a loop's body once and says whether to go on; raise rt.Return(value) / rt.Break() / rt.Continue(); error(node, message) stops the program with a runtime error.";
    pub const eval__doc__: [*:0]const u8 = "The value of an expression node (a list: of each); a value that isn't a node is its own value.";
    pub const eval__params__ = "x";
    pub const exec__doc__: [*:0]const u8 = "Run a statement node (a list: each in order).";
    pub const exec__params__ = "x";
    pub const loop__doc__: [*:0]const u8 = "Run a loop's body once: False if it broke out of the loop (rt.Break), True otherwise (rt.Continue included).";
    pub const loop__params__ = "body";
    pub const load__doc__: [*:0]const u8 = "The value of the variable a name node refers to (zrules' symbol); a builtin's is the host function of its name.";
    pub const load__params__ = "name";
    pub const store__doc__: [*:0]const u8 = "Set the variable a name node refers to.";
    pub const store__params__ = "name, value";
    pub const function__doc__: [*:0]const u8 = "A function made from a node of a function kind (Language.function), seeing the variables of where it's made.";
    pub const function__params__ = "node";
    pub const call__doc__: [*:0]const u8 = "Call a function of the program, or a host function, with a list of arguments.";
    pub const call__params__ = "f, args";
    pub const error__doc__: [*:0]const u8 = "Stop the program with a runtime error at the node.";
    pub const kind__doc__: [*:0]const u8 = "A node's kind (also when it has a field named kind).";
    pub const kind__params__ = "node";
    pub const text__doc__: [*:0]const u8 = "A node's text.";
    pub const text__params__ = "node";
    pub const span__doc__: [*:0]const u8 = "A node's (start, end) byte offsets.";
    pub const span__params__ = "node";
};

/// The message of the Python exception being raised, as the program's
/// runtime error says it (cleared). Null on failure.
fn pythonMessage() ?*PyObject {
    var t: ?*PyObject = null;
    var v: ?*PyObject = null;
    var tb: ?*PyObject = null;
    py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    defer inline for (.{ t, v, tb }) |o| {
        if (o) |x| py.Py_DecRef(x);
    };
    const typ = t orelse return ph.newString("error");
    const val = v orelse return ph.newString("error");
    const is = struct {
        fn f(exc_type: *PyObject, base: *PyObject) bool {
            return py.c.PyErr_GivenExceptionMatches(exc_type, base) != 0;
        }
    }.f;
    if (is(typ, types.IntegerOverflow)) return ph.newString("integer overflow");
    if (is(typ, py.PyExc_ZeroDivisionError())) return ph.newString("division by zero");
    if (is(typ, py.c.PyExc_RecursionError)) return ph.newString("call stack too deep");
    if (is(typ, py.PyExc_KeyError())) {
        const args = py.c.PyObject_GetAttrString(val, "args") orelse return null;
        defer py.Py_DecRef(args);
        if (py.c.PyTuple_Size(args) < 1) return ph.newString("key not found");
        return py.c.PyUnicode_FromFormat("key not found: %R", py.c.PyTuple_GetItem(args, 0).?);
    }
    if (is(typ, py.PyExc_TypeError()) or is(typ, py.PyExc_IndexError()) or is(typ, py.PyExc_ValueError()) or is(typ, py.PyExc_AttributeError()))
        return py.c.PyObject_Str(val);
    const name = py.c.PyObject_GetAttrString(typ, "__name__") orelse return null;
    defer py.Py_DecRef(name);
    return py.c.PyUnicode_FromFormat("%U: %S", name, val);
}

fn attrInt(o: *PyObject, name: [*:0]const u8) ?i64 {
    const v = py.c.PyObject_GetAttrString(o, name) orelse return null;
    defer py.Py_DecRef(v);
    return py.c.PyLong_AsLongLong(v);
}

fn diagnostic(severity: []const u8, code: []const u8, message: []const u8, start: u32, end_: u32, d: *const program_mod.Data) ?*PyObject {
    const lc = d.lineCol(start);
    return py.c.PyObject_CallFunction(objects.Diagnostic, "s#s#s#(II)II", severity.ptr, @as(py.Py_ssize_t, @intCast(severity.len)), code.ptr, @as(py.Py_ssize_t, @intCast(code.len)), message.ptr, @as(py.Py_ssize_t, @intCast(message.len)), @as(c_uint, start), @as(c_uint, end_), @as(c_uint, lc.line), @as(c_uint, lc.col));
}

fn typeName(o: *PyObject) []const u8 {
    if (o == py.Py_None()) return "None";
    if (py.PyBool_Check(o)) return "bool";
    if (py.PyLong_Check(o)) return "int";
    if (py.PyFloat_Check(o)) return "float";
    if (py.PyUnicode_Check(o)) return "str";
    if (py.PyList_Check(o)) return "list";
    if (py.PyTuple_Check(o)) return "tuple";
    if (py.PyDict_Check(o)) return "dict";
    return "object";
}

/// A tuple with each item wrapped (ints made I64).
/// (first, *rest) as a new tuple.
fn prepend(first: *PyObject, rest: *PyObject) ?*PyObject {
    const n = py.c.PyTuple_Size(rest);
    const out = py.c.PyTuple_New(n + 1) orelse return null;
    _ = py.c.PyTuple_SetItem(out, 0, ref(first));
    for (0..@intCast(n)) |i| _ = py.c.PyTuple_SetItem(out, @intCast(i + 1), ref(py.c.PyTuple_GetItem(rest, @intCast(i)).?));
    return out;
}

fn wrapAll(args: *PyObject) ?*PyObject {
    const n = py.c.PyTuple_Size(args);
    const out = py.c.PyTuple_New(n) orelse return null;
    for (0..@intCast(n)) |i| {
        const v = types.wrap(py.c.PyTuple_GetItem(args, @intCast(i)).?) orelse {
            py.Py_DecRef(out);
            return null;
        };
        _ = py.c.PyTuple_SetItem(out, @intCast(i), v);
    }
    return out;
}

// ============================================================================
// Module
// ============================================================================

fn version() []const u8 {
    return @import("build_options").version;
}

fn moduleInit(module: *PyObject) callconv(.c) c_int {
    if (types.init(module) != 0) return -1;
    objects.init(module) catch return -1;
    bridge.init(module) catch return -1;
    @import("proxies.zig").init(module) catch return -1;
    name_program = ph.newString("<program>") orelse return -1;
    CallerType = py.c.PyType_FromSpec(&caller_spec) orelse return -1;
    // (compiled code words Python's errors as the reference mode does)
    helpers.pythonMessage = &pythonMessage;
    return 0;
}

pub const Module = pyoz.module(.{
    .name = "zrun",
    .doc = "zrun - execution for languages defined with zgram and checked with zrules: semantics written as Python functions, run natively.",
    .funcs = &.{
        pyoz.func("version", version, "Return the zrun version string"),
    },
    .classes = &.{
        pyoz.class("Language", Language),
        pyoz.class("Registrar", Registrar),
        pyoz.class("Program", Program),
        pyoz.class("Runtime", Runtime),
    },
    .module_init = moduleInit,
});

// Required: forces analysis of all pub decls so PyInit_ is exported.
comptime {
    for (@typeInfo(@This()).@"struct".decls) |decl| {
        _ = @field(@This(), decl.name);
    }
}
