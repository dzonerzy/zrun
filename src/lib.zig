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
const ztypes = @import("types.zig");
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
const bytes_mod = @import("bytes.zig");
const gil = @import("gil.zig");
const pool = @import("pool.zig");
const gc = @import("gc.zig");
const aot = @import("aot.zig");
const cache = @import("cache.zig");
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
    /// Calls of a call site in compiled code before code of its own is
    /// compiled for it (what it knows of its arguments)
    _hot_calls: i64 = 1000,
    /// Language(strict=True): every semantic compiled, the code calling
    /// nothing in Python (what would is a CompileError; at run time, the
    /// run's error: StrictError); errors still Python's exceptions
    _strict: bool = false,
    /// What values the nodes of the language's types evaluate to
    /// (types()): a dict by type name, or a function of the type's text
    _types: ?*PyObject = null,
    /// The semantics read by the compiler's front, by function object
    /// (each holds a reference to its function); those marked
    /// native=False aren't here
    _read: std.AutoHashMapUnmanaged(*PyObject, *front.Function) = .empty,
    /// Semantics the compiler couldn't compile: compiled programs run them
    /// as Python (bridge.zig); learned as programs are compiled
    _python: driver.PythonSet = .empty,
    /// List and dict literals of the semantics compiled programs build at
    /// run time (they escape where known ones can't follow)
    _escaping: std.AutoHashMapUnmanaged(compile_mod.EscapeKey, void) = .empty,

    pub fn __new__(args: pyoz.Args(struct { parser: *PyObject, rules: ?*PyObject = null, max_depth: i64 = 1000, hot_calls: i64 = 1000, strict: bool = false })) ?Language {
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
        if (v.hot_calls < 1) {
            ph.raise(py.PyExc_ValueError(), "hot_calls must be at least 1", .{});
            return lang.fail();
        }
        lang._hot_calls = v.hot_calls;
        lang._strict = v.strict;
        return lang;
    }

    fn fail(self: *Language) ?Language {
        if (py.c.PyErr_Occurred() == null) _ = py.c.PyErr_NoMemory();
        self.release();
        return null;
    }

    fn release(self: *Language) void {
        inline for (.{ "_parser", "_rules", "_evals", "_execs", "_hosts", "_types" }) |f| {
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

    /// `lang.types(mapping)`: what the nodes of each of the language's
    /// types (zrules' types(): their texts, 'int', 'list[int]', 'fn(int) ->
    /// int', a struct's name) evaluate to, as Python types: int, float,
    /// bool, str, type(None), list, tuple, dict, zrun.Function, a record
    /// class (a dataclass or one with __slots__). A dict by name ('int', or
    /// a generic's name: 'list' for 'list[int]', 'fn' for a function's
    /// type), or a function of the type's text returning one (or None: not
    /// known). Compiled code knows the kind of a node's value from its type
    /// then: no checks of its kind after the one where it's made (a value
    /// not of the kind declared is an error there).
    pub fn types(self: *Language, mapping: *PyObject) ?*PyObject {
        if (!py.PyDict_Check(mapping) and !py.PyCallable_Check(mapping)) {
            ph.raise(py.PyExc_TypeError(), "types() takes a dict of type names, or a function of a type's text", .{});
            return null;
        }
        if (self._types) |o| py.Py_DecRef(o);
        self._types = ref(mapping);
        return none();
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

    pub fn __del__(self: *Language) void {
        const saved = ph.SavedError.save();
        defer saved.restore();
        self.release();
    }

    /// For Python's cycle collector: a language is in a cycle through its
    /// semantics (a closure, or the module, naming it). The objects it
    /// owns: its fields', and the functions of the semantics read
    /// (_read's); the other tables (_eval_of, _python...) borrow.
    pub fn __traverse__(self: *Language, visitor: pyoz.GCVisitor) c_int {
        inline for (.{ "_parser", "_rules", "_evals", "_execs", "_hosts", "_types" }) |f| {
            const r = visitor.call(@field(self, f));
            if (r != 0) return r;
        }
        var it = self._read.valueIterator();
        while (it.next()) |f| {
            const r = visitor.call(f.*.py_function);
            if (r != 0) return r;
        }
        return 0;
    }

    pub fn __clear__(self: *Language) void {
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
            if (self._strict) {
                ph.raise(ztypes.CompileError, "strict: a semantic marked native=False (it would run in Python)", .{});
                return false;
            }
            if (!self.markPython(func, "native=False")) return false;
        } else if (!self.readSemantic(func)) {
            if (py.c.PyErr_ExceptionMatches(ztypes.CompileError) == 0) return false;
            // (strict: why it can't be compiled is the error)
            if (self._strict) return false;
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

    /// `lang.native_host(name, capsule)`: a native library's function (a
    /// "zrun.native.v1" capsule, native.zig) as the host function `name`:
    /// compiled code calls it directly, without Python. The callable made
    /// for it (zrun.NativeHost).
    pub fn native_host(self: *Language, name: *PyObject, capsule: *PyObject) ?*PyObject {
        if (!py.PyUnicode_Check(name)) {
            ph.raise(py.PyExc_TypeError(), "a host function's name must be a str", .{});
            return null;
        }
        const h = @import("native.zig").make(name, capsule) orelse return null;
        if (!self.addHost(name, h)) {
            py.Py_DecRef(h);
            return null;
        }
        return h;
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
        if (!prog.setup(self, tree, null)) {
            prog.release();
            return null;
        }
        return prog;
    }

    /// `lang.session()`: a Session: programs run one after another, each
    /// seeing what the ones before it defined (a REPL's).
    pub fn session(self: *Language) ?Session {
        const programs = py.c.PyList_New(0) orelse return null;
        const known = py.c.PyDict_New() orelse {
            py.Py_DecRef(programs);
            return null;
        };
        return .{ ._lang = ref(Module.selfObject(Language, self)), ._programs = programs, ._names = known };
    }

    /// `lang.repl(prompt="> ", more="... ", show=None)`: an interactive
    /// session. Each entry is read with input() (more lines while what it
    /// writes isn't finished; an empty line ends it anyway) and run; the
    /// value of an expression is shown with show(value) (by default
    /// printed, unless None). Errors are printed and the session goes on.
    /// End of input (Ctrl-D) ends it; Ctrl-C drops the entry being written,
    /// or stops the one running.
    pub fn repl(self: *Language, args: pyoz.Args(struct { prompt: ?*PyObject = null, more: ?*PyObject = null, show: ?*PyObject = null })) ?*PyObject {
        const builtins = py.c.PyImport_ImportModule("builtins") orelse return null;
        defer py.Py_DecRef(builtins);
        const input = py.c.PyObject_GetAttrString(builtins, "input") orelse return null;
        defer py.Py_DecRef(input);
        const print = py.c.PyObject_GetAttrString(builtins, "print") orelse return null;
        defer py.Py_DecRef(print);
        const prompt = if (optional(args.value.prompt)) |p| ref(p) else ph.newString("> ") orelse return null;
        defer py.Py_DecRef(prompt);
        const more = if (optional(args.value.more)) |m| ref(m) else ph.newString("... ") orelse return null;
        defer py.Py_DecRef(more);
        const show = optional(args.value.show) orelse print;
        const s_obj = Module.toPy(Session, self.session() orelse return null) orelse return null;
        defer py.Py_DecRef(s_obj);
        const s = unwrap(Session, s_obj).?;
        while (true) {
            const entry = readEntry(s, input, prompt, more) orelse {
                if (py.c.PyErr_ExceptionMatches(py.PyExc_EOFError()) != 0) {
                    py.c.PyErr_Clear();
                    // (the line the prompt was on, ended)
                    const r = py.c.PyObject_CallNoArgs(print) orelse return null;
                    py.Py_DecRef(r);
                    return none();
                }
                if (!printError(print)) return null;
                continue;
            };
            defer py.Py_DecRef(entry);
            const value = s.run(entry) orelse {
                if (!printError(print)) return null;
                continue;
            };
            defer py.Py_DecRef(value);
            if (value == py.Py_None()) continue;
            const r = py.c.PyObject_CallFunctionObjArgs(show, value, @as(?*PyObject, null)) orelse {
                if (!printError(print)) return null;
                continue;
            };
            py.Py_DecRef(r);
        }
    }

    /// `lang.compile(source, output, path=None)`: the program compiled, saved
    /// as a compiled module at `output` (program.save()); the Program.
    pub fn compile(self: *Language, args: pyoz.Args(struct { source: *PyObject, output: *PyObject, path: ?*PyObject = null })) ?Program {
        const v = args.value;
        var prog = self.load(.{ .value = .{ .source = v.source, .path = v.path } }) orelse return null;
        const r = prog.save(v.output) orelse {
            prog.release();
            return null;
        };
        py.Py_DecRef(r);
        return prog;
    }

    /// `lang.load_compiled(path)`: a program saved as a compiled module
    /// (program.save(), lang.compile()), loaded without compiling it again.
    /// Refused if it was made for another definition of the language,
    /// another zrun or another CPU.
    pub fn load_compiled(self: *Language, path: *PyObject) ?Program {
        const p = ph.utf8(path, "path") orelse return null;
        const ca = std.heap.c_allocator;
        var file = aot.read(ca, p) catch |e| {
            switch (e) {
                error.Io => ph.raise(py.PyExc_OSError(), "can't read the compiled module {s}", .{p}),
                error.OutOfMemory => _ = py.c.PyErr_NoMemory(),
                else => ph.raise(py.PyExc_ValueError(), "{s} isn't a compiled module of zrun's (or it's cut short)", .{p}),
            }
            return null;
        };
        defer file.deinit(ca);
        if (!std.mem.eql(u8, file.zrun, aot.version)) {
            ph.raise(py.PyExc_ValueError(), "{s} was compiled by zrun {s}, this is zrun {s}: compile it again", .{ p, file.zrun, aot.version });
            return null;
        }
        const def = self.definitionHash() orelse return null;
        // (LLVM's, for the salt: zgram's)
        _ = @import("jit.zig").get() orelse return null;
        if (!std.mem.eql(u8, &file.definition, &def)) {
            ph.raise(py.PyExc_ValueError(), "{s} was compiled for another definition of the language (its grammar, semantics or host functions changed since): compile it again", .{p});
            return null;
        }
        if (!std.mem.eql(u8, &file.salt, cache.salt())) {
            ph.raise(py.PyExc_ValueError(), "{s} was compiled for another CPU (or another LLVM): compile it again", .{p});
            return null;
        }
        // (its objects given to the cache's look-ups: owned there from now)
        for (file.objects) |*o| {
            cache.give(o.key, o.bytes);
            o.bytes = &.{};
        }
        const source = ph.newString(file.source) orelse return null;
        defer py.Py_DecRef(source);
        const prog_path = if (file.path.len > 0) ph.newString(file.path) orelse return null else null;
        defer if (prog_path) |x| py.Py_DecRef(x);
        return self.load(.{ .value = .{ .source = source, .path = prog_path } });
    }

    /// The hash of the language's definition (aot.zig): what a compiled
    /// module of it is made for.
    fn definitionHash(self: *Language) ?[32]u8 {
        const S = struct {
            var func: ?*PyObject = null;
        };
        if (S.func == null) {
            const ns = compile_mod.runPython(aot.definition_source) orelse return null;
            defer py.Py_DecRef(ns);
            const f = py.c.PyDict_GetItemString(ns, "definition") orelse return null;
            py.Py_IncRef(f);
            S.func = f;
        }
        // (the rest as text: the function kinds, the limits)
        var text: std.ArrayListUnmanaged(u8) = .empty;
        defer text.deinit(allocator);
        for (self._functions, 0..) |spec, i| if (spec) |s| {
            text.print(allocator, "{d}:{any};", .{ i, s }) catch {
                _ = py.c.PyErr_NoMemory();
                return null;
            };
        };
        text.print(allocator, "max_depth={d};hot_calls={d};strict={}", .{ self._max_depth, self._hot_calls, self._strict }) catch {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        const rest = py.c.PyTuple_New(2) orelse return null;
        defer py.Py_DecRef(rest);
        _ = py.c.PyTuple_SetItem(rest, 0, ph.newString(text.items) orelse return null);
        _ = py.c.PyTuple_SetItem(rest, 1, ref(self._types orelse py.Py_None()));
        const tables = py.c.PyTuple_New(2) orelse return null;
        defer py.Py_DecRef(tables);
        _ = py.c.PyTuple_SetItem(tables, 0, ref(self._evals.?));
        _ = py.c.PyTuple_SetItem(tables, 1, ref(self._execs.?));
        const call_args = py.c.PyTuple_New(4) orelse return null;
        defer py.Py_DecRef(call_args);
        _ = py.c.PyTuple_SetItem(call_args, 0, ref(self._parser.?));
        _ = py.c.PyTuple_SetItem(call_args, 1, ref(tables));
        _ = py.c.PyTuple_SetItem(call_args, 2, ref(self._hosts.?));
        _ = py.c.PyTuple_SetItem(call_args, 3, ref(rest));
        const r = py.c.PyObject_CallObject(S.func.?, call_args) orelse return null;
        defer py.Py_DecRef(r);
        var out: [32]u8 = undefined;
        var buf: [*c]u8 = undefined;
        var len: py.c.Py_ssize_t = 0;
        if (py.c.PyBytes_AsStringAndSize(r, &buf, &len) != 0 or len != 32) {
            if (py.c.PyErr_Occurred() == null) ph.raise(py.PyExc_RuntimeError(), "the language's definition hash isn't 32 bytes", .{});
            return null;
        }
        @memcpy(&out, buf[0..32]);
        return out;
    }

    pub const __doc__: [*:0]const u8 = "Language(parser, rules=None, *, max_depth=1000, hot_calls=1000, strict=False): a language to run programs of: its zgram parser, its zrules rules (for names and their variables), and its semantics (eval, exec, function, host). load(source) parses and checks a program. strict=True: compiled code calling nothing in Python (what would is a CompileError, or a StrictError as it runs).";
    pub const eval__doc__: [*:0]const u8 = "@lang.eval(kind): the semantics of an expression kind (a -> class or rule name, or a list of them): fn(node, rt) -> value.";
    pub const eval__params__ = "kind";
    pub const exec__doc__: [*:0]const u8 = "@lang.exec(kind): the semantics of a statement kind: fn(node, rt).";
    pub const exec__params__ = "kind";
    pub const function__doc__: [*:0]const u8 = "function(kind, params='params', body='body', name='name', hoist=True): nodes of kind define functions, with those labels for their parameters, body and name.";
    pub const host__doc__: [*:0]const u8 = "@lang.host, @lang.host('name') or lang.host('name', fn): a Python function the program calls through the builtin of that name.";
    pub const load__doc__: [*:0]const u8 = "load(source, path=None): parse and check a program; a Program. Raises zrun.LoadError listing its errors.";
    pub const types__doc__: [*:0]const u8 = "types(mapping): what values of the language's types (zrules' types(): 'int', 'list[int]', a struct's name) are, as Python types (int, float, bool, str, type(None), list, tuple, dict, zrun.Function, a record class): a dict by type name (or a generic's name: 'list' for 'list[int]'), or a function of the type's text. Compiled code then knows a node's kind from its type: checked once where it's made, unboxed after.";
    pub const types__params__ = "mapping";
    pub const native_host__doc__: [*:0]const u8 = "native_host(name, capsule): a native library's function (a 'zrun.native.v1' capsule) as the host function `name`, called by compiled code directly, without Python. Returns the zrun.NativeHost made for it.";
    pub const native_host__params__ = "name, capsule";
    pub const python_semantics__doc__: [*:0]const u8 = "python_semantics(): the semantics compiled programs run as Python, {name: why}: those marked native=False and those the compiler couldn't compile (where, and what): what to rewrite for speed.";
    pub const compile__doc__: [*:0]const u8 = "compile(source, output, path=None): load the program, compile it and save it as a compiled module at `output` (Program.save()); returns the Program.";
    pub const load_compiled__doc__: [*:0]const u8 = "load_compiled(path): a program saved as a compiled module, loaded without compiling it again. Raises ValueError if it was made for another definition of the language, another zrun or another CPU.";
    pub const load_compiled__params__ = "path";
    pub const ir__doc__: [*:0]const u8 = "ir(fn): a semantic (or any function) as the compiler's front reads it, as text (for tests, and to see what gets compiled).";
    pub const ir__params__ = "fn";
    pub const session__doc__: [*:0]const u8 = "session(): a Session, to run programs one after another, each seeing the variables and functions the ones before it defined (a REPL's).";
    pub const repl__doc__: [*:0]const u8 = "repl(prompt='> ', more='... ', show=None): an interactive session on input(): more lines while an entry isn't finished (an empty line ends it anyway); the value of an expression shown with show(value) (default: printed, unless None); errors printed and the session going on. Ctrl-D ends it, Ctrl-C drops or stops the entry.";
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
    const exc = py.c.PyObject_CallFunctionObjArgs(ztypes.CompileError, text, @as(?*PyObject, null)) orelse return;
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
    py.c.PyErr_SetObject(ztypes.CompileError, exc);
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

/// A REPL's entry: lines read with input() (`prompt` first, `more` after)
/// until what they write is finished, or a line is empty (blank lines
/// before it skipped); a new str, or null with the exception (EOFError at
/// the end of input).
fn readEntry(s: *Session, input: *PyObject, prompt: *PyObject, more: *PyObject) ?*PyObject {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(allocator);
    var first = true;
    while (true) {
        const line_obj = py.c.PyObject_CallFunctionObjArgs(input, if (first) prompt else more, @as(?*PyObject, null)) orelse return null;
        defer py.Py_DecRef(line_obj);
        const line = ph.utf8(line_obj, "line") orelse return null;
        const blank = std.mem.trim(u8, line, " \t\r\n").len == 0;
        if (first and blank) continue;
        buf.appendSlice(allocator, line) catch return py.c.PyErr_NoMemory();
        buf.append(allocator, '\n') catch return py.c.PyErr_NoMemory();
        const text = ph.newString(buf.items) orelse return null;
        if (blank) return text;
        const unfinished = s.unfinished(text) orelse {
            py.Py_DecRef(text);
            return null;
        };
        if (!unfinished) return text;
        py.Py_DecRef(text);
        first = false;
    }
}

/// A REPL's error, printed to sys.stderr (cleared): zrun's errors as
/// they read, others as "Type: message". False (the exception kept) for
/// SystemExit, which ends the REPL.
fn printError(print: *PyObject) bool {
    if (py.c.PyErr_ExceptionMatches(py.PyExc_SystemExit()) != 0) return false;
    var t: ?*PyObject = null;
    var v: ?*PyObject = null;
    var tb: ?*PyObject = null;
    py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
    defer inline for (.{ t, v, tb }) |o| if (o) |x| py.Py_DecRef(x);
    const exc = v orelse return true;
    const text = py.c.PyObject_Str(exc) orelse return false;
    defer py.Py_DecRef(text);
    const ours = py.c.PyErr_GivenExceptionMatches(exc, ztypes.Error) != 0 or py.c.PyErr_GivenExceptionMatches(exc, ztypes.LoadError) != 0;
    const line = if (ours) ref(text) else blk: {
        const name = ph.attr(@ptrCast(@alignCast(ph.typeOf(exc))), "__name__") orelse return false;
        defer py.Py_DecRef(name);
        break :blk py.c.PyUnicode_FromFormat("%U: %U", name, text) orelse return false;
    };
    defer py.Py_DecRef(line);
    const args = py.c.PyTuple_Pack(1, line) orelse return false;
    defer py.Py_DecRef(args);
    const kwargs = py.c.PyDict_New() orelse return false;
    defer py.Py_DecRef(kwargs);
    if (py.c.PySys_GetObject("stderr")) |err| if (py.c.PyDict_SetItemString(kwargs, "file", err) != 0) return false;
    const r = py.c.PyObject_Call(print, args, kwargs) orelse return false;
    py.Py_DecRef(r);
    return true;
}

/// rules.analyze(tree, builtins=builtins): a new Analysis, or null with
/// an exception.
fn analyze(rules: *PyObject, tree: *PyObject, builtins: ?*PyObject) ?*PyObject {
    const method = py.c.PyObject_GetAttrString(rules, "analyze") orelse return null;
    defer py.Py_DecRef(method);
    const args = py.c.PyTuple_Pack(1, tree) orelse return null;
    defer py.Py_DecRef(args);
    const b = builtins orelse return py.c.PyObject_Call(method, args, null);
    const kwargs = py.c.PyDict_New() orelse return null;
    defer py.Py_DecRef(kwargs);
    if (py.c.PyDict_SetItemString(kwargs, "builtins", b) != 0) return null;
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
    /// The program being compiled optimized, in the background (ensureCompiled)
    _pending: ?*driver.Pending = null,
    /// Code it ran before its optimized code took over: kept (values its
    /// runs made may still refer to it)
    _retired: std.ArrayListUnmanaged(*driver.Compiled) = .empty,
    /// mode="auto" found it can't be compiled: run as Python
    _uncompilable: bool = false,
    /// The last run with report=True: where its compiled code went through
    /// Python, {what: times} (report())
    _crossings: ?*PyObject = null,
    /// The Python type Language.types() says each node's values are
    /// (checked in the reference mode), by node: worked out once
    _declared: std.AutoHashMapUnmanaged(u32, ?*PyObject) = .empty,
    /// The compiled program's variables, kept for program.call() after its
    /// top level ran (the first call's)
    _cglobals: ?*value_mod.Frame = null,
    /// The mode it last ran in (program.call()'s, unless said)
    _ran_mode: ?Mode = null,
    /// Its variables and module state made immortal (shared by calls
    /// without the GIL, on several threads at once: freezeShared)
    _frozen: bool = false,
    /// The times its compiled runs and calls took the GIL back
    _gil_taken: u64 = 0,
    /// The session it's an entry of (borrowed: the session keeps it, and
    /// clears this when it goes), or null
    _session: ?*Session = null,
    /// A session's entry (checked with the names the entries before it
    /// defined: its code is its session's, not shared)
    _entry: bool = false,

    fn release(self: *Program) void {
        if (self._cglobals) |g| value_mod.decrefFrame(g);
        self._cglobals = null;
        if (self._compiled) |c| c.drop();
        self._compiled = null;
        // (the optimized code being made: the program's next load's)
        if (self._pending) |p| {
            const key = if (self.seed()) |what| self.shareKey(&what) else blk: {
                py.c.PyErr_Clear();
                break :blk null;
            };
            if (key) |k| p.orphan(&k, .{ self._state, self._lang, self._path }) else p.abandon();
        }
        self._pending = null;
        for (self._retired.items) |c| c.drop();
        self._retired.deinit(allocator);
        self._retired = .empty;
        self._declared.deinit(allocator);
        self._declared = .empty;
        inline for (.{ "_state", "_lang", "_source", "_path", "_crossings" }) |f| {
            if (@field(self, f)) |o| py.Py_DecRef(o);
            @field(self, f) = null;
        }
    }

    pub fn __del__(self: *Program) void {
        // (the error it's freed on the way out of, if any, kept)
        const saved = ph.SavedError.save();
        defer saved.restore();
        self.release();
    }

    fn state(self: *const Program) *objects.StateObject {
        return objects.asState(self._state.?);
    }

    fn ctx(self: *const Program) *objects.Context {
        return self.state().ctx.?;
    }

    /// Check the tree (syntax errors, the rules) and build the native data.
    /// `builtins`: names defined outside the program (a session's earlier
    /// entries'), a list of str, or null.
    fn setup(self: *Program, lang: *Language, tree_obj: *PyObject, builtins: ?*PyObject) bool {
        defer py.Py_DecRef(tree_obj);
        // The diagnostics: the rules' (syntax errors included), or the
        // tree's syntax errors
        var analysis: ?*PyObject = null;
        defer if (analysis) |a| py.Py_DecRef(a);
        var diagnostics: *PyObject = undefined;
        if (lang._rules) |rules| {
            analysis = analyze(rules, tree_obj, builtins) orelse return false;
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
        const exc = py.c.PyObject_CallFunctionObjArgs(ztypes.LoadError, message, @as(?*PyObject, null)) orelse return false;
        defer py.Py_DecRef(exc);
        if (py.c.PyObject_SetAttrString(exc, "diagnostics", diags) != 0) return false;
        py.c.PyErr_SetObject(ztypes.LoadError, exc);
        return false;
    }

    fn language(self: *const Program) *Language {
        return unwrap(Language, self._lang.?).?;
    }

    /// `program.run(mode="python", report=False)`: run the program from its
    /// start, its semantics as Python ("python"), compiled to native code
    /// ("compiled"), or compiled when its optimized code is at hand
    /// ("auto": as Python while it's compiled in the background, compiled
    /// from the run after it's done; compiled at once when the cache has
    /// it). "compiled": compiled fast the first time if the cache hasn't
    /// the optimized code, which runs from the run after it's made in the
    /// background (zrun.configure(tiers=False): optimized at once). Raises
    /// zrun.Error on a runtime error, the same in every mode. report=True:
    /// where the compiled code goes through Python is counted (report(); a
    /// little slower).
    pub fn run(self: *Program, args: pyoz.Args(struct { mode: ?*PyObject = null, report: bool = false })) ?*PyObject {
        const mode = parseMode(args.value.mode, .python) orelse return null;
        const compiled = switch (mode) {
            .python => false,
            .compiled => true,
            .auto => self.optimizedAtHand() orelse return null,
        };
        self._ran_mode = mode;
        if (!compiled) return onBigStack(runHere, .{self}, self.language()._max_depth);
        // (auto: the optimized code's at hand, not waited for)
        if (!self.ensureCompiled(if (mode == .auto) .optimized else .any)) return null;
        const reporting = args.value.report;
        helpers.collecting = reporting;
        defer if (reporting) {
            helpers.collecting = false;
            // (the run's error, if any, stays the one raised)
            var t: ?*PyObject = null;
            var v: ?*PyObject = null;
            var tb: ?*PyObject = null;
            py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
            if (self._crossings) |o| py.Py_DecRef(o);
            self._crossings = helpers.takeStats();
            if (self._crossings == null) py.c.PyErr_Clear();
            py.c.PyErr_Restore(t, v, tb);
        };
        // (on this thread when it can: as program.call())
        const max_depth = self.language()._max_depth;
        if (!self._compiled.?.compiler.uses_python) if (stackLow()) |low| {
            if (@frameAddress() > low + @as(usize, max_depth) * stack_per_call + stack_spare)
                return self.withCompiled(null, low + stack_spare, runMain, {});
        };
        return onBigStack(runCompiled, .{self}, max_depth);
    }

    const Mode = enum { python, compiled, auto };

    /// A mode given (none: `default`); null with an exception.
    fn parseMode(given: ?*PyObject, default: Mode) ?Mode {
        const m = optional(given) orelse return default;
        const s = ph.utf8(m, "mode") orelse return null;
        return std.meta.stringToEnum(Mode, s) orelse {
            ph.raise(py.PyExc_ValueError(), "mode must be 'python', 'compiled' or 'auto', not '{s}'", .{s});
            return null;
        };
    }

    /// `program.save(path)`: the program as a compiled module (aot.zig):
    /// its source and the objects of the code compiled for it so far (the
    /// program compiled first if it isn't: save after running it, the code
    /// compiled as it ran goes in too). lang.load_compiled(path) loads it.
    pub fn save(self: *Program, path: *PyObject) ?*PyObject {
        const p = ph.utf8(path, "path") orelse return null;
        if (!self.ensureCompiled(.optimized)) return null;
        // (the optimized code of what was compiled fast as it ran: in the
        // cache once the background's done)
        driver.waitJobs();
        const def = self.language().definitionHash() orelse return null;
        const c = self._compiled.?;
        var objs: std.ArrayListUnmanaged(aot.Object) = .empty;
        defer {
            for (objs.items) |o| allocator.free(o.bytes);
            objs.deinit(allocator);
        }
        for (c.keys.items) |key| {
            for (objs.items) |o| {
                if (std.mem.eql(u8, &o.key, &key)) break;
            } else {
                const bytes = c.objectOf(key) orelse {
                    ph.raise(py.PyExc_RuntimeError(), "the compiled code of the program isn't at hand any more (the cache cleared?): load the program again, then save it", .{});
                    return null;
                };
                objs.append(allocator, .{ .key = key, .bytes = bytes }) catch {
                    allocator.free(bytes);
                    return py.c.PyErr_NoMemory();
                };
            }
        }
        const source = ph.utf8(self._source.?, "source") orelse return null;
        const prog_path = if (self._path) |x| ph.utf8(x, "path") orelse return null else "";
        aot.write(allocator, p, def, prog_path, source, objs.items) catch |e| {
            if (e == error.OutOfMemory) return py.c.PyErr_NoMemory();
            ph.raise(py.PyExc_OSError(), "can't write the compiled module {s}", .{p});
            return null;
        };
        return none();
    }

    /// `program.report()`: what to look at to make the compiled program
    /// faster, as data. {"python_crossings": {what: times} (where compiled
    /// code went through Python in the last run with report=True),
    /// "module_state": {name: "native" or why Python keeps it} (the
    /// module-level tables and records the semantics use), "cache":
    /// {"loaded": n, "compiled": n} (modules of compiled code), "code":
    /// "fast", "optimized" or None (the code compiled runs run: compiled
    /// fast while the optimized code is made, or none yet), "optimizing":
    /// whether it's being made in the background, "gil_taken":
    /// n (the times compiled runs, calls and map()'s took the GIL back to
    /// touch Python: calls that didn't run in parallel), "speculated":
    /// {"line N": kinds} (functions given a typed entry for the kinds their
    /// arguments have been)}. Which semantics run as Python:
    /// Language.python_semantics().
    pub fn report(self: *Program) ?*PyObject {
        const out = py.c.PyDict_New() orelse return null;
        const crossings = if (self._crossings) |o| ref(o) else py.c.PyDict_New() orelse return null;
        defer py.Py_DecRef(crossings);
        if (py.c.PyDict_SetItemString(out, "python_crossings", crossings) != 0) return null;
        const module_state = py.c.PyDict_New() orelse return null;
        defer py.Py_DecRef(module_state);
        const cache_d = py.c.PyDict_New() orelse return null;
        defer py.Py_DecRef(cache_d);
        if (self._compiled) |c| {
            var it = c.compiler.module_state.iterator();
            while (it.next()) |e| {
                const v = ph.newString(e.value_ptr.*) orelse return null;
                defer py.Py_DecRef(v);
                const k = ph.newString(e.key_ptr.*) orelse return null;
                defer py.Py_DecRef(k);
                if (py.c.PyDict_SetItem(module_state, k, v) != 0) return null;
            }
        }
        inline for (.{ .{ "loaded", "cache_loaded" }, .{ "compiled", "cache_kept" } }) |p| {
            const n = py.c.PyLong_FromUnsignedLongLong(if (self._compiled) |c| @field(c, p[1]) else 0) orelse return null;
            defer py.Py_DecRef(n);
            if (py.c.PyDict_SetItemString(cache_d, p[0], n) != 0) return null;
        }
        if (py.c.PyDict_SetItemString(out, "module_state", module_state) != 0) return null;
        if (py.c.PyDict_SetItemString(out, "cache", cache_d) != 0) return null;
        // (the code compiled runs run: compiled fast or optimized; whether
        // the optimized code is being made)
        const code = if (self._compiled) |c| (ph.newString(if (c.opt == 0) "fast" else "optimized") orelse return null) else ref(py.Py_None());
        defer py.Py_DecRef(code);
        if (py.c.PyDict_SetItemString(out, "code", code) != 0) return null;
        if (py.c.PyDict_SetItemString(out, "optimizing", if (self._pending != null) py.Py_True() else py.Py_False()) != 0) return null;
        const taken = py.c.PyLong_FromUnsignedLongLong(self._gil_taken) orelse return null;
        defer py.Py_DecRef(taken);
        if (py.c.PyDict_SetItemString(out, "gil_taken", taken) != 0) return null;
        // (the functions given a typed entry for what their arguments have
        // been: "line N": their kinds)
        const speculated = py.c.PyDict_New() orelse return null;
        defer py.Py_DecRef(speculated);
        if (self._compiled) |c| {
            var it = c.compiler.speculations.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.*.code == 0) continue;
                const fnode = e.key_ptr.*;
                const shapes = (c.compiler.typed_params.get(fnode) orelse continue) orelse continue;
                var buf: std.ArrayListUnmanaged(u8) = .empty;
                defer buf.deinit(allocator);
                for (shapes, 0..) |s, i| {
                    if (i > 0) buf.appendSlice(allocator, ", ") catch return py.c.PyErr_NoMemory();
                    buf.appendSlice(allocator, @tagName(s)) catch return py.c.PyErr_NoMemory();
                }
                const where = c.compiler.data.lineCol(c.compiler.data.nodes[fnode].text_start);
                const key = std.fmt.allocPrint(allocator, "line {d}", .{where.line}) catch return py.c.PyErr_NoMemory();
                defer allocator.free(key);
                const k = ph.newString(key) orelse return null;
                defer py.Py_DecRef(k);
                const v = ph.newString(buf.items) orelse return null;
                defer py.Py_DecRef(v);
                if (py.c.PyDict_SetItem(speculated, k, v) != 0) return null;
            }
        }
        if (py.c.PyDict_SetItemString(out, "speculated", speculated) != 0) return null;
        // "python_functions": {qualified name: why} of the Python functions
        // compiled code called that couldn't be compiled (they run as
        // Python; the process's, every program's)
        const funcs = py.c.PyDict_New() orelse return null;
        defer py.Py_DecRef(funcs);
        var it = driver.uncompiled.iterator();
        while (it.next()) |e| {
            const name = ph.attr(e.key_ptr.*, "__qualname__") orelse return null;
            defer py.Py_DecRef(name);
            const why = ph.newString(e.value_ptr.*) orelse return null;
            defer py.Py_DecRef(why);
            if (py.c.PyDict_SetItem(funcs, name, why) != 0) return null;
        }
        if (py.c.PyDict_SetItemString(out, "python_functions", funcs) != 0) return null;
        return out;
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
            .hot_calls = lang._hot_calls,
            .types = lang._types,
            .strict = lang._strict,
        };
    }

    const Want = enum {
        /// a run's: code compiled fast will do while the optimized code is
        /// made in the background (zrun.configure(tiers=True))
        any,
        /// calls', map()'s, save()'s: the optimized code (their state is
        /// the code's: it stays), waited for if it's being made
        optimized,
    };

    /// The program compiled (self._compiled); false with an exception. The
    /// optimized code takes over once it's made, unless calls have state
    /// in the code they ran (program.call()'s variables).
    fn ensureCompiled(self: *Program, want: Want) bool {
        if (self._pending) |p| if (self._cglobals == null and (want == .optimized or p.ready())) {
            if (!self.adoptPending()) return false;
        };
        if (self._compiled) |c| if (c.opt != 0 or want == .any or self._cglobals != null) return true;
        const what = self.seed() orelse return false;
        // (the program loaded before: its optimized code, or the optimized
        // code it was having made)
        const key = self.shareKey(&what);
        if (key != null and self._pending == null) {
            if (driver.sharedCode(&key.?, 2)) |c| {
                self.retire();
                self._compiled = c;
                return true;
            }
            if (driver.takeOrphan(&key.?)) |p| {
                self._pending = p;
                if (want == .optimized or p.ready()) return self.adoptPending();
            }
        }
        const lang = self.language();
        if (want == .any and driver.tiers) {
            // (nothing compiled yet: the optimized code made in the
            // background, compiled fast meanwhile, unless it's at hand)
            if (self._pending == null) {
                self._pending = driver.compileInBackground(self.ctx().data, self.langView(), &lang._python, ztypes.CompileError, &what) orelse return false;
                if (self._pending.?.ready()) return self.adoptPending();
            }
            if (key) |k| if (driver.sharedCode(&k, 0)) |c| {
                self._compiled = c;
                return true;
            };
            self._compiled = driver.compileProgram(self.ctx().data, self.langView(), &lang._python, ztypes.CompileError, &what, 0) orelse return false;
            if (key) |k| self.share(self._compiled.?, &k);
            return true;
        }
        const c = driver.compileProgram(self.ctx().data, self.langView(), &lang._python, ztypes.CompileError, &what, 2) orelse return false;
        self.retire();
        self._compiled = c;
        if (key) |k| self.share(c, &k);
        return true;
    }

    /// What the program's code is shared as (driver.share): its seed, its
    /// language (the one object: its semantics' functions are the code's),
    /// its path (the code's messages say it); null: not shared (a session's
    /// entry).
    fn shareKey(self: *Program, what: *const [32]u8) ?driver.ShareKey {
        if (self._entry) return null;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(what);
        h.update(std.mem.asBytes(&@intFromPtr(self._lang.?)));
        if (self._path) |p| if (p != py.Py_None()) {
            h.update(ph.utf8(p, "path") orelse {
                py.c.PyErr_Clear();
                return null;
            });
        };
        var out: driver.ShareKey = undefined;
        h.final(&out);
        return out;
    }

    /// Its optimized code shared with the program's later loads.
    fn share(self: *Program, c: *driver.Compiled, key: *const driver.ShareKey) void {
        driver.share(c, key, .{ self._state, self._lang, self._path });
    }

    /// The optimized code made in the background, the program's from now
    /// (waited for if it's still being made); false with an exception.
    fn adoptPending(self: *Program) bool {
        const p = self._pending.?;
        self._pending = null;
        const c = p.finish() orelse return false;
        self.retire();
        self._compiled = c;
        if (self.seed()) |what| {
            if (self.shareKey(&what)) |key| self.share(c, &key);
        } else py.c.PyErr_Clear();
        return true;
    }

    /// The code the program ran so far kept aside (values may refer to it).
    fn retire(self: *Program) void {
        const c = self._compiled orelse return;
        self._compiled = null;
        self._retired.append(allocator, c) catch {
            // (kept for good, then)
        };
    }

    /// mode="auto": whether the optimized code is at hand (made, or the
    /// code that has call state), its making started in the background
    /// if it wasn't; null with an exception. A program that can't be
    /// compiled runs as Python.
    fn optimizedAtHand(self: *Program) ?bool {
        if (self._uncompilable) return false;
        if (self._compiled) |c| if (c.opt != 0 or self._cglobals != null) return true;
        if (self._pending) |p| return p.ready();
        const what = self.seed() orelse return null;
        if (self.shareKey(&what)) |key| {
            if (driver.isShared(&key)) return true;
            if (driver.takeOrphan(&key)) |p| {
                self._pending = p;
                return p.ready();
            }
        }
        self._pending = driver.compileInBackground(self.ctx().data, self.langView(), &self.language()._python, ztypes.CompileError, &what) orelse {
            if (py.c.PyErr_ExceptionMatches(ztypes.CompileError) == 0) return null;
            py.c.PyErr_Clear();
            self._uncompilable = true;
            return false;
        };
        return self._pending.?.ready();
    }

    /// What the program is, for its compiled code's names (driver.zig's
    /// prefixFor): its language's definition and its source, hashed.
    fn seed(self: *Program) ?[32]u8 {
        const def = self.language().definitionHash() orelse return null;
        const source = ph.utf8(self._source.?, "source") orelse return null;
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(&def);
        h.update(source);
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }

    /// `program.native_objects()`: the program compiled ahead of time to run
    /// without Python (a strict language's), as object files to link with
    /// zrun's runtime (libzrun_rt.a): a list of bytes (zrun.build_native
    /// links them).
    pub fn native_objects(self: *Program, args: pyoz.Args(struct { prune: bool = false, left_out: ?*PyObject = null, setup: ?*PyObject = null })) ?*PyObject {
        const prune = args.value.prune;
        const left_out = optional(args.value.left_out);
        const setup_fn = optional(args.value.setup);
        if (left_out) |l| if (!py.PyList_Check(l)) {
            ph.raise(py.PyExc_TypeError(), "left_out must be a list", .{});
            return null;
        };
        if (!self.language()._strict) {
            ph.raise(ztypes.CompileError, "a program runs without Python only in a strict language: zrun.Language(..., strict=True)", .{});
            return null;
        }
        const s = self.seed() orelse return null;
        const path: []const u8 = if (self._path) |p| (if (p == py.Py_None()) "<program>" else ph.utf8(p, "path") orelse return null) else "<program>";
        const b = driver.buildStandalone(self.ctx().data, self.langView(), &self.language()._python, ztypes.CompileError, &s, 2, @intCast(self.language()._max_depth), path, prune, setup_fn) orelse return null;
        defer {
            b.deinit();
            std.heap.c_allocator.destroy(b);
        }
        if (left_out) |l| for (b.left_out.items) |name| {
            const s_name = ph.newString(name) orelse return null;
            defer py.Py_DecRef(s_name);
            if (py.c.PyList_Append(l, s_name) != 0) return null;
        };
        const list = py.c.PyList_New(0) orelse return null;
        for (b.objects.items) |o| {
            const bytes = py.c.PyBytes_FromStringAndSize(o.ptr, @intCast(o.len)) orelse {
                py.Py_DecRef(list);
                return null;
            };
            defer py.Py_DecRef(bytes);
            if (py.c.PyList_Append(list, bytes) != 0) {
                py.Py_DecRef(list);
                return null;
            }
        }
        return list;
    }

    /// `program.compiled_ir()`: the LLVM IR the program compiles to (before
    /// LLVM optimizes it), as text.
    pub fn compiled_ir(self: *Program) ?*PyObject {
        return driver.irText(self.ctx().data, self.langView(), &self.language()._python, ztypes.CompileError);
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

    fn makeErrorObject(raw: *anyopaque, idx: u32, message: []const u8) ?*PyObject {
        const self: *Program = @ptrCast(@alignCast(raw));
        return Runtime.errorObject(self, idx, message, "runtime", &.{});
    }

    fn makeNodeObject(raw: *anyopaque, idx: u32) ?*PyObject {
        const self: *Program = @ptrCast(@alignCast(raw));
        return objects.newNode(self._state.?, self.ctx(), idx);
    }

    fn runCompiled(self: *Program) ?*PyObject {
        return self.withCompiled(null, 0, runMain, {});
    }

    fn runMain(self: *Program, ectx: *helpers.Ctx, _: void) ?*PyObject {
        const c = self._compiled.?;
        const globals = helpers.zr_frame_new(null, c.globals) orelse {
            _ = py.c.PyErr_NoMemory();
            return null;
        };
        defer {
            value_mod.decrefFrame(globals);
            // (the run's cycles: the globals and the functions in them)
            _ = gc.collectHere();
        }
        // (the GIL released, as for program.call(): the code takes it back
        // if it touches Python)
        const takes = gil.takes();
        const strict = self.language()._strict;
        if (strict) gil.strictBegin();
        const ok = gil.without(runMainCode, .{ c, ectx, globals });
        const entered = if (strict) gil.strictEnd() else null;
        self._gil_taken += gil.takes() - takes;
        if (entered) |e| return self.strictError(e);
        if (ok) return none();
        return self.compiledError(ectx);
    }

    /// zrun.StrictError: a strict language's code went into Python as it
    /// ran (where: the program's node, what it called, zrun's code); null.
    fn strictError(self: *Program, e: gil.Entry) ?*PyObject {
        var where: std.ArrayListUnmanaged(u8) = .empty;
        defer where.deinit(allocator);
        const d = self.ctx().data;
        if (e.node) |n| if (n < d.nodes.len) {
            const at = d.lineCol(d.nodes[n].text_start);
            const path: []const u8 = if (self._path) |p| (if (p == py.Py_None()) "<program>" else ph.utf8(p, "path") orelse "<program>") else "<program>";
            where.print(allocator, "{s}:{d}:{d}: ", .{ path, at.line, at.col }) catch {};
        };
        py.c.PyErr_Clear();
        const what: []const u8 = if (e.calledName()) |name| name else "";
        // (a Python function that couldn't be compiled: why)
        const why = driver.uncompiledReason(e.callee);
        ph.raise(ztypes.StrictError, "strict: {s}the compiled code went into Python while it ran{s}{s}{s}{s}{s} ({s}, {s}:{d} in zrun)", .{
            where.items,
            if (what.len == 0) "" else if (e.phrase) ": " else ", calling ",
            what,
            if (what.len > 0 and !e.phrase) "()" else "",
            if (why != null) ", which isn't compiled: " else "",
            why orelse "",
            e.at.fn_name,
            e.at.file,
            e.at.line,
        });
        return null;
    }

    fn runMainCode(c: *driver.Compiled, ectx: *helpers.Ctx, globals: *value_mod.Frame) bool {
        return c.main(ectx, globals);
    }

    /// Run `body` with the context compiled code runs with (the bridge's
    /// link, rt.context: `context`; the native stack's end: Ctx.stack_low).
    fn withCompiled(self: *Program, context: ?*PyObject, stack_low: usize, comptime body: anytype, extra: anytype) ?*PyObject {
        const c = self._compiled.?;
        const st = self.state();
        const link = bridge.Link{
            .program = self,
            .compiled = c,
            .data = self.ctx().data,
            .hosts = self.language()._hosts.?,
            .analysis = st.analysis,
            .path = self._path,
            .context = context,
            .semantic = &linkSemantic,
            .error_object = &linkErrorObject,
        };
        var ectx = helpers.Ctx{
            .node_maker = .{ .ctx = self, .make_fn = &makeNodeObject, .owner = Module.selfObject(Program, self), .error_fn = &makeErrorObject },
            .objects = &c.compiler.objects,
            .program = c.id,
            .max_depth = self.language()._max_depth,
            .stack_low = stack_low,
            .link = @constCast(&link),
        };
        defer ectx.deinit();
        // (the run rt values reaching Python belong to)
        const outer = bridge.current;
        bridge.current = &ectx;
        defer bridge.current = outer;
        // (semantics run as Python recurse through Python: room for
        // max_depth calls, as in the reference mode)
        const saved_limit = py.c.Py_GetRecursionLimit();
        const want: c_int = @intCast(@min(@as(u64, ectx.max_depth) * 40 + 1000, std.math.maxInt(c_int)));
        if (want > saved_limit) py.c.Py_SetRecursionLimit(want);
        defer py.c.Py_SetRecursionLimit(saved_limit);
        return body(self, &ectx, extra);
    }

    /// `program.call(name, *args)` compiled: the program's top level run
    /// the first time (its variables kept for the calls after), then the
    /// function the name holds there called with the arguments.
    const CallArgs = struct { name: []const u8, args: *PyObject };

    /// The function a name holds at the program's top level (its top level
    /// run the first time, its variables kept), compiled; null with an
    /// exception.
    fn entryFunction(self: *Program, ectx: *helpers.Ctx, name: []const u8) ?value_mod.Value {
        const c = self._compiled.?;
        const data = self.ctx().data;
        const slot = for (data.syms, 0..) |s, i| {
            if (s.builtin or !std.mem.eql(u8, s.name, name) or data.homeOf(@intCast(i)) != NONE) continue;
            break c.compiler.slot_of.get(@intCast(i)).?;
        } else {
            ph.raise(py.PyExc_KeyError(), "the program defines no function '{s}'", .{name});
            return null;
        };
        if (self._cglobals == null) {
            const globals = helpers.zr_frame_new(null, c.globals) orelse {
                _ = py.c.PyErr_NoMemory();
                return null;
            };
            if (!c.main(ectx, globals)) {
                value_mod.decrefFrame(globals);
                _ = self.compiledError(ectx);
                return null;
            }
            self._cglobals = globals;
        }
        const fv = self._cglobals.?.slots()[slot];
        if (fv.tag == helpers.UNSET) {
            ph.raise(py.PyExc_KeyError(), "the program defines no function '{s}'", .{name});
            return null;
        }
        if (!self.freezeShared()) return null;
        return fv;
    }

    /// The program's variables and module state made immortal, once: calls
    /// run without the GIL, on several threads at once, share them, their
    /// counts not raced (they live as long as the program does).
    fn freezeShared(self: *Program) bool {
        if (self._frozen) return true;
        var fz = value_mod.Freezer{};
        defer fz.deinit();
        fz.frame(self._cglobals.?) catch {
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        for (self._compiled.?.compiler.adopted.items) |v| fz.value(v) catch {
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        self._frozen = true;
        return true;
    }

    fn callFunction(ectx: *helpers.Ctx, fv: value_mod.Value, vals: []const value_mod.Value, out: *value_mod.Value) bool {
        return helpers.zr_call(ectx, 0, fv.tag, fv.bits, vals.ptr, vals.len, null, out);
    }

    fn callCompiledIn(self: *Program, ectx: *helpers.Ctx, a: CallArgs) ?*PyObject {
        const fv = self.entryFunction(ectx, a.name) orelse return null;
        // (the arguments as values: the call borrows them)
        const n: usize = @intCast(py.c.PyTuple_Size(a.args));
        const vals = allocator.alloc(value_mod.Value, n) catch return py.c.PyErr_NoMemory();
        defer allocator.free(vals);
        var made: usize = 0;
        defer for (vals[0..made]) |v| value_mod.decref(v);
        for (vals, 0..) |*v, i| {
            v.* = value_mod.fromPython(py.c.PyTuple_GetItem(a.args, @intCast(i)).?) orelse return null;
            made += 1;
        }
        var out = value_mod.Value.none_v;
        // (the GIL released: Python threads run meanwhile, and calls on
        // them; the code takes it back if it touches Python)
        const takes = gil.takes();
        const strict = self.language()._strict;
        if (strict) gil.strictBegin();
        const ok = gil.without(callFunction, .{ ectx, fv, vals, &out });
        const entered = if (strict) gil.strictEnd() else null;
        self._gil_taken += gil.takes() - takes;
        if (entered) |e| {
            if (ok) value_mod.decref(out);
            return self.strictError(e);
        }
        if (!ok) return self.compiledError(ectx);
        defer value_mod.decref(out);
        return value_mod.toPython(out, ectx.node_maker);
    }

    const MapArgs = struct { name: []const u8, items: *PyObject, threads: usize };

    /// `program.map(name, items, threads=None, context=None)` (get_map):
    /// on `threads` native threads (none said: one per CPU). An item
    /// failing: the error of the first one failing in the items' order, as
    /// one thread calling them in order raises it (the others run all the
    /// same). The calls share the program's variables and module state:
    /// they mustn't change them.
    fn mapEntry(self: *Program, name: *PyObject, items: *PyObject, threads_obj: ?*PyObject, context: ?*PyObject) ?*PyObject {
        if (!self.ensureCompiled(.optimized)) return null;
        const wanted = ph.utf8(name, "name") orelse return null;
        var threads: usize = std.Thread.getCpuCount() catch 1;
        if (threads_obj) |t| {
            const n = py.c.PyLong_AsLongLong(t);
            if (n == -1 and py.c.PyErr_Occurred() != null) return null;
            if (n < 1) {
                ph.raise(py.PyExc_ValueError(), "map() needs at least one thread, not {d}", .{n});
                return null;
            }
            threads = @intCast(n);
        }
        // (the top level, run the first time, on a stack as deep as a run's;
        // its allocator lists not this thread's: the workers' results it
        // frees would stay in them)
        const map_args = MapArgs{ .name = wanted, .items = items, .threads = threads };
        return onBigStackLending(withCompiled, .{ self, context, @as(usize, 0), mapIn, map_args }, self.language()._max_depth, false);
    }

    fn mapIn(self: *Program, ectx: *helpers.Ctx, a: MapArgs) ?*PyObject {
        const fv = self.entryFunction(ectx, a.name) orelse return null;
        const seq = py.c.PySequence_Fast(a.items, "map()'s items must be a sequence") orelse return null;
        defer py.Py_DecRef(seq);
        const n: usize = @intCast(py.c.PySequence_Size(seq));
        const ca = std.heap.c_allocator;
        // The items' arguments as values, made here (with the GIL): the
        // calls borrow them
        const args = ca.alloc([]value_mod.Value, n) catch return py.c.PyErr_NoMemory();
        var made: usize = 0;
        defer {
            for (args[0..made]) |xs| {
                for (xs) |v| value_mod.decref(v);
                ca.free(xs);
            }
            ca.free(args);
        }
        for (0..n) |i| {
            const item = py.c.PySequence_GetItem(seq, @intCast(i)) orelse return null;
            defer py.Py_DecRef(item);
            const is_tuple = ph.typeOf(item) == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyTuple_Type"))));
            const k: usize = if (is_tuple) @intCast(py.c.PyTuple_Size(item)) else 1;
            const xs = ca.alloc(value_mod.Value, k) catch return py.c.PyErr_NoMemory();
            var got: usize = 0;
            for (xs, 0..) |*slot, j| {
                const x = if (is_tuple) py.c.PyTuple_GetItem(item, @intCast(j)).? else item;
                const conv = if (bytes_mod.isData(x)) bytes_mod.of(x) else ref(x);
                const v = if (conv) |o| blk: {
                    defer py.Py_DecRef(o);
                    break :blk value_mod.fromPython(o);
                } else null;
                slot.* = v orelse {
                    for (xs[0..got]) |w| value_mod.decref(w);
                    ca.free(xs);
                    return null;
                };
                got += 1;
            }
            args[i] = xs;
            made += 1;
        }
        const results = ca.alloc(value_mod.Value, n) catch return py.c.PyErr_NoMemory();
        defer ca.free(results);
        @memset(results, value_mod.Value.none_v);
        defer for (results) |v| value_mod.decref(v);
        var job = MapJob{ .fv = fv, .args = args, .results = results, .template = ectx, .strict = self.language()._strict };
        defer if (job.failed) |f| {
            f.deinit();
            ca.destroy(f);
        };
        // The threads (as deep a stack as a run's), waited for without the
        // GIL
        const count = @max(@min(a.threads, n), 1);
        const stack: usize = @min(@as(usize, ectx.max_depth) * 64 * 1024 + 16 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
        const handles = ca.alloc(std.Thread, count) catch return py.c.PyErr_NoMemory();
        defer ca.free(handles);
        var spawned: usize = 0;
        for (handles) |*h| {
            h.* = std.Thread.spawn(.{ .stack_size = stack }, MapJob.work, .{&job}) catch break;
            spawned += 1;
        }
        gil.without(MapJob.join, .{handles[0..spawned]});
        self._gil_taken += job.gil_takes.load(.monotonic);
        if (spawned == 0) return py.c.PyErr_NoMemory();
        if (job.entered) |e| return self.strictError(e);
        if (job.failed) |f| return self.compiledError(f);
        const list = py.c.PyList_New(@intCast(n)) orelse return null;
        for (results, 0..) |v, i| {
            const o = value_mod.toPython(v, ectx.node_maker) orelse {
                py.Py_DecRef(list);
                return null;
            };
            _ = py.c.PyList_SetItem(list, @intCast(i), o);
        }
        return list;
    }

    /// program.map()'s calls: each thread takes the next item until none is
    /// left, with a context of its own; the first failing item's (in the
    /// items' order) kept.
    const MapJob = struct {
        fv: value_mod.Value,
        args: []const []value_mod.Value,
        results: []value_mod.Value,
        template: *const helpers.Ctx,
        next: std.atomic.Value(usize) = .init(0),
        lock: std.atomic.Mutex = .unlocked,
        failed_at: usize = std.math.maxInt(usize),
        failed: ?*helpers.Ctx = null,
        /// The times the threads took the GIL
        gil_takes: std.atomic.Value(u64) = .init(0),
        /// A strict language's: where a thread's code first went into
        /// Python (under `lock`)
        strict: bool = false,
        entered: ?gil.Entry = null,

        fn fresh(job: *const MapJob) helpers.Ctx {
            const t = job.template;
            return .{ .node_maker = t.node_maker, .objects = t.objects, .program = t.program, .max_depth = t.max_depth, .link = t.link };
        }

        fn work(job: *MapJob) void {
            gil.worker();
            if (job.strict) gil.strictBegin();
            defer if (job.strict) if (gil.strictEnd()) |at| {
                while (!job.lock.tryLock()) std.atomic.spinLoopHint();
                defer job.lock.unlock();
                if (job.entered == null) job.entered = at;
            };
            var own = job.fresh();
            bridge.current = &own;
            while (true) {
                const i = job.next.fetchAdd(1, .monotonic);
                if (i >= job.args.len) break;
                var out = value_mod.Value.none_v;
                if (helpers.zr_call(&own, 0, job.fv.tag, job.fv.bits, job.args[i].ptr, job.args[i].len, null, &out)) {
                    job.results[i] = out;
                } else job.keep(i, &own);
                gil.done();
            }
            // (its calls' cycles, while their thread's here)
            _ = gc.collectParallel();
            own.deinit();
            gil.done();
            _ = job.gil_takes.fetchAdd(gil.takes(), .monotonic);
        }

        /// A call failed: its context kept if it's the first failing item
        /// so far, the thread going on with a new one.
        fn keep(job: *MapJob, i: usize, own: *helpers.Ctx) void {
            while (!job.lock.tryLock()) std.atomic.spinLoopHint();
            defer job.lock.unlock();
            if (i < job.failed_at) if (std.heap.c_allocator.create(helpers.Ctx)) |kept| {
                if (job.failed) |old| {
                    old.deinit();
                    std.heap.c_allocator.destroy(old);
                }
                kept.* = own.*;
                job.failed = kept;
                job.failed_at = i;
                own.* = job.fresh();
                return;
            } else |_| {};
            own.deinit();
            own.* = job.fresh();
        }

        fn join(handles: []std.Thread) void {
            for (handles) |h| h.join();
        }
    };

    /// A compiled run's or call's error, raised as the reference mode
    /// raises it; null.
    fn compiledError(self: *Program, ectx: *helpers.Ctx) ?*PyObject {
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
        py.c.PyErr_SetObject(ztypes.Error, exc);
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

    /// A session's entry run (as Python): as runHere, the value of its last
    /// statement if that's an expression (a new reference), else None.
    fn runEntry(self: *Program) ?*PyObject {
        var rt = Runtime.begin(self) orelse return null;
        defer rt.end();
        const frame = objects.newFrame(NONE, NONE, null, name_program orelse return null) orelse return null;
        rt.self()._frame = frame;
        const st = self.state();
        if (st.globals) |g| py.Py_DecRef(g);
        st.globals = ref(frame);
        const r = rt.self();
        if (!r.hoist(NONE)) return uncaught();
        return r.execRoot() orelse uncaught();
    }

    /// A run ended by an exception: a rt.Throw nothing caught becomes the
    /// zrun.Error it is (made where it was raised); null.
    fn uncaught() ?*PyObject {
        if (py.c.PyErr_ExceptionMatches(ztypes.Throw) == 0) return null;
        var t: ?*PyObject = null;
        var v: ?*PyObject = null;
        var tb: ?*PyObject = null;
        py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
        const err = if (v) |exc| py.c.PyObject_GetAttrString(exc, "_zrun_error") else null;
        if (err) |e| {
            inline for (.{ t, v, tb }) |o| if (o) |x| py.Py_DecRef(x);
            py.c.PyErr_SetObject(ztypes.Error, e);
            py.Py_DecRef(e);
        } else {
            py.c.PyErr_Clear();
            py.c.PyErr_Restore(t, v, tb);
        }
        return null;
    }

    /// `program.call(name, *args, mode=None, context=None)`: call a
    /// function the program defines at its top level (running the program
    /// first if it hasn't run), in a mode ("python", "compiled" or "auto",
    /// as run()'s, but compiled is always the optimized code, waited for if
    /// it's being made; none said: the one it last ran in, "compiled" if it
    /// hasn't run), with rt.context the context.
    fn callEntry(self: *Program, name: *PyObject, given: *PyObject, mode_obj: ?*PyObject, context: ?*PyObject) ?*PyObject {
        // (data, any kind of it: a zrun.Bytes over its memory, in every mode)
        const args = py.c.PyTuple_New(py.c.PyTuple_Size(given)) orelse return null;
        defer py.Py_DecRef(args);
        for (0..@intCast(py.c.PyTuple_Size(given))) |i| {
            const x = py.c.PyTuple_GetItem(given, @intCast(i)).?;
            const v = if (bytes_mod.isData(x)) bytes_mod.of(x) orelse return null else ref(x);
            _ = py.c.PyTuple_SetItem(args, @intCast(i), v);
        }
        const mode = parseMode(mode_obj, self._ran_mode orelse .compiled) orelse return null;
        const compiled = switch (mode) {
            .python => false,
            .compiled => true,
            .auto => self.optimizedAtHand() orelse return null,
        };
        if (compiled) {
            if (!self.ensureCompiled(.optimized)) return null;
            const wanted = ph.utf8(name, "name") orelse return null;
            const call_args = CallArgs{ .name = wanted, .args = args };
            // (on this thread when its stack has room for the deepest calls
            // and no semantic runs as Python, recursing through Python:
            // no thread made for the call)
            const max_depth = self.language()._max_depth;
            if (!self._compiled.?.compiler.uses_python) if (stackLow()) |low| {
                const here = @frameAddress();
                if (here > low + @as(usize, max_depth) * stack_per_call + stack_spare)
                    return self.withCompiled(context, low + stack_spare, callCompiledIn, call_args);
            };
            return onBigStack(withCompiled, .{ self, context, @as(usize, 0), callCompiledIn, call_args }, max_depth);
        }
        return onBigStack(callHere, .{ self, name, args, context }, self.language()._max_depth);
    }

    fn callHere(self: *Program, name: *PyObject, args: *PyObject, context: ?*PyObject) ?*PyObject {
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
            if (context) |x| rt.self()._context = ref(x);
            return rt.self().callValue(f, args, null) orelse uncaught();
        }
        ph.raise(py.PyExc_KeyError(), "the program defines no function '{s}'", .{wanted});
        return null;
    }

    /// `program.call(name, *args)`: call a function the program defines
    /// at its top level (running the program first if it hasn't run)
    pub fn get_call(self: *const Program) ?*PyObject {
        return self.caller(false);
    }

    /// `program.map(name, items, threads=None, context=None)`: the function
    /// `name` called with each item (a tuple: its arguments; anything else:
    /// the one argument), compiled, on native threads at once, without the
    /// GIL nor Python between items: the results, in the items' order
    pub fn get_map(self: *const Program) ?*PyObject {
        return self.caller(true);
    }

    fn caller(self: *const Program, map: bool) ?*PyObject {
        const alloc: py.c.allocfunc = @ptrCast(py.c.PyType_GetSlot(@ptrCast(CallerType), py.c.Py_tp_alloc));
        const o = alloc.?(@ptrCast(CallerType), 0) orelse return null;
        const c: *CallerObject = @ptrCast(@alignCast(o));
        c.program = ref(Module.selfObject(Program, self));
        c.map = map;
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
    pub const run__doc__: [*:0]const u8 = "run(mode='python', report=False): run the program from its start. mode: 'python' (semantics run as Python: the reference), 'compiled' (native code; compiled fast first if the cache hasn't the optimized code, which is made in the background: zrun.configure(tiers=)), 'auto' (as Python until the optimized code is at hand, compiled from then). Raises zrun.Error on a runtime error, the same in every mode. report=True: count where compiled code goes through Python (report()).";
    pub const call__doc__: [*:0]const u8 = "call(name, *args, mode=None, context=None): call a function the program defines at its top level (its top level run first, the first time): an engine's entry point, called many times. Compiled calls release the GIL. mode: as run()'s (default: the one it last ran in, else 'compiled'); context: what rt.context is.";
    pub const map__doc__: [*:0]const u8 = "map(name, items, threads=None, context=None): call the function `name` with each item (a tuple: its arguments) on native threads at once (default: one per CPU), without the GIL or Python between items: the results, in order. The first failing item's error is raised, as one thread calling them in order would. The calls share the program's variables: they mustn't change them.";
    pub const save__doc__: [*:0]const u8 = "save(path): the program as a compiled module: its source and the code compiled for it so far (save after running it: what was compiled as it ran goes in too). Language.load_compiled(path) loads it.";
    pub const save__params__ = "path";
    pub const report__doc__: [*:0]const u8 = "report(): what to look at to make the compiled program faster: python_crossings (where compiled code went through Python, in the last run with report=True), module_state, cache (modules loaded / compiled), code ('fast', 'optimized' or None), optimizing, gil_taken, speculated (typed entries made for functions' argument kinds).";
    pub const compiled_ir__doc__: [*:0]const u8 = "compiled_ir(): the LLVM IR the program compiles to (before LLVM optimizes it), as text.";
    pub const native_objects__doc__: [*:0]const u8 = "native_objects(prune=False, left_out=None): the program compiled ahead of time to run without Python (a strict language's: zrun.CompileError otherwise), as object files to link with zrun's runtime, libzrun_rt.a: a list of bytes. zrun.build_native() links them. prune: compile only the Python functions held in the language's tables under a name the program can make (its words, the strs the code uses): a smaller, faster build; a function left out that's reached anyway stops the program with an error naming it. left_out: a list, the names of the functions left out appended. setup: a Python function compiled with the program, called with the program's path and its arguments (a list of strs) before its top level runs.";
    pub const source__doc__: [*:0]const u8 = "The program's source.";
    pub const tree__doc__: [*:0]const u8 = "The zgram Tree.";
    pub const analysis__doc__: [*:0]const u8 = "The zrules Analysis (None without rules).";
    pub const diagnostics__doc__: [*:0]const u8 = "The warnings found loading it.";
    pub const root__doc__: [*:0]const u8 = "The root node.";
};

var name_program: ?*PyObject = null;

/// Run `f(args)` on a thread of its own with a stack for `max_depth`
/// calls of the language: semantics recurse through Python and native
/// frames, more than a default stack holds (8 MB on Linux, 1 MB on
/// Windows). The calling thread waits without the GIL; the result and any
/// exception come back to it.
fn onBigStack(comptime f: anytype, args: anytype, max_depth: u32) ?*PyObject {
    return onBigStackLending(f, args, max_depth, true);
}

/// onBigStack(), the values' allocator's lists of this thread (waiting) the
/// new one's while it runs when `lend` (theirs are slower to reach: pool.zig)
fn onBigStackLending(comptime f: anytype, args: anytype, max_depth: u32, lend: bool) ?*PyObject {
    const Ctx = struct {
        args: @TypeOf(args),
        owner: usize,
        result: ?*PyObject = null,
        t: ?*PyObject = null,
        v: ?*PyObject = null,
        tb: ?*PyObject = null,

        fn work(c: *@This()) void {
            const lent = pool.borrowHome(c.owner);
            defer if (lent) pool.restoreHome(c.owner);
            const g = py.c.PyGILState_Ensure();
            c.result = @call(.auto, f, c.args);
            if (c.result == null) py.c.PyErr_Fetch(@ptrCast(&c.t), @ptrCast(&c.v), @ptrCast(&c.tb));
            py.c.PyGILState_Release(g);
        }
    };
    var ctx = Ctx{ .args = args, .owner = if (lend) pool.threadPointer() else 0 };
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

extern "c" fn pthread_getattr_np(thread: std.c.pthread_t, attr: *std.c.pthread_attr_t) c_int;
extern "c" fn pthread_attr_getstack(attr: *const std.c.pthread_attr_t, addr: *?*anyopaque, size: *usize) c_int;
extern "c" fn pthread_attr_destroy(attr: *std.c.pthread_attr_t) c_int;

/// Where this thread's stack ends (its lowest address), or null: not known
/// here (another system than Linux).
fn stackLow() ?usize {
    const S = struct {
        threadlocal var low: usize = 0;
    };
    if (S.low != 0) return S.low;
    if (@import("builtin").os.tag != .linux) return null;
    var attr: std.c.pthread_attr_t = undefined;
    if (pthread_getattr_np(std.c.pthread_self(), &attr) != 0) return null;
    defer _ = pthread_attr_destroy(&attr);
    var addr: ?*anyopaque = null;
    var size: usize = 0;
    if (pthread_attr_getstack(&attr, &addr, &size) != 0) return null;
    S.low = @intFromPtr(addr);
    return S.low;
}

/// The stack per language call compiled code is given running on the
/// caller's thread (well over what a call takes; zr_call refuses one going
/// past the end anyway), and what's kept spare at the end
const stack_per_call = 4096;
const stack_spare = 256 * 1024;

/// `program.call`: a callable taking `(name, *args)` (PyOZ methods take
/// fixed arguments, and its types can't get methods added: the property
/// returns this)
const CallerObject = extern struct {
    ob_base: py.c.PyObject,
    program: ?*PyObject,
    /// program.map rather than program.call
    map: bool,
};

var CallerType: *PyObject = undefined;

fn callerCall(self_obj: ?*PyObject, args: ?*PyObject, kwargs: ?*PyObject) callconv(.c) ?*PyObject {
    const c: *CallerObject = @ptrCast(@alignCast(self_obj.?));
    var mode: ?*PyObject = null;
    var context: ?*PyObject = null;
    var threads: ?*PyObject = null;
    if (kwargs) |kw| {
        var pos: py.c.Py_ssize_t = 0;
        var k: ?*PyObject = null;
        var v: ?*PyObject = null;
        while (py.c.PyDict_Next(kw, &pos, @ptrCast(&k), @ptrCast(&v)) != 0) {
            const key = ph.utf8(k.?, "keyword") orelse return null;
            if (std.mem.eql(u8, key, "mode") and !c.map) {
                if (v.? != py.Py_None()) mode = v;
            } else if (std.mem.eql(u8, key, "threads") and c.map) {
                if (v.? != py.Py_None()) threads = v;
            } else if (std.mem.eql(u8, key, "context")) {
                if (v.? != py.Py_None()) context = v;
            } else {
                ph.raise(py.PyExc_TypeError(), "{s}() got an unexpected keyword argument '{s}'", .{ if (c.map) "map" else "call", key });
                return null;
            }
        }
    }
    const n = py.c.PyTuple_Size(args);
    const self = unwrap(Program, c.program.?) orelse return null;
    if (c.map) {
        if (n != 2) {
            ph.raise(py.PyExc_TypeError(), "map(name, items, threads=None, context=None) takes a name and items", .{});
            return null;
        }
        return self.mapEntry(py.c.PyTuple_GetItem(args, 0).?, py.c.PyTuple_GetItem(args, 1).?, threads, context);
    }
    if (n < 1) {
        ph.raise(py.PyExc_TypeError(), "call(name, *args) needs the function's name", .{});
        return null;
    }
    const rest = py.c.PyTuple_GetSlice(args, 1, n) orelse return null;
    defer py.Py_DecRef(rest);
    return self.callEntry(py.c.PyTuple_GetItem(args, 0).?, rest, mode, context);
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
// Session: a REPL's programs
// ============================================================================

/// `lang.session()`: programs run one after another as the entries of a
/// session (a REPL's), each seeing what the entries before it defined at
/// their top level: the same variables (assigning one changes it for the
/// earlier entry's code too) and functions (run in their own program). A
/// name defined again is the new entry's from then on. Entries run as
/// Python (the reference mode): each runs once.
pub const Session = struct {
    _lang: ?*PyObject = null,
    /// The entries run so far (Programs): their functions run in them
    _programs: ?*PyObject = null,
    /// Each name defined at an entry's top level -> (the entry's frame, the
    /// name's symbol there), the last entry defining it
    _names: ?*PyObject = null,

    /// Where a session's variable lives
    pub const Place = struct { frame: *objects.FrameObject, sym: u32 };

    fn place(self: *Session, name: *PyObject) ?Place {
        const t = py.c.PyDict_GetItem(self._names.?, name) orelse return null;
        const frame = py.c.PyTuple_GetItem(t, 0).?;
        const sym = py.c.PyLong_AsUnsignedLong(py.c.PyTuple_GetItem(t, 1).?);
        return .{ .frame = objects.asFrame(frame), .sym = @intCast(sym) };
    }

    /// The entry whose State `state_obj` is, or null.
    fn programOf(self: *Session, state_obj: *PyObject) ?*Program {
        const list = self._programs.?;
        for (0..@intCast(py.c.PyList_Size(list))) |i| {
            const p = unwrap(Program, py.c.PyList_GetItem(list, @intCast(i)).?).?;
            if (p._state == state_obj) return p;
        }
        return null;
    }

    pub fn __del__(self: *Session) void {
        // (an entry someone else keeps doesn't look at the session any more)
        if (self._programs) |list| {
            for (0..@intCast(py.c.PyList_Size(list))) |i| unwrap(Program, py.c.PyList_GetItem(list, @intCast(i)).?).?._session = null;
        }
        inline for (.{ "_lang", "_programs", "_names" }) |f| {
            if (@field(self, f)) |o| py.Py_DecRef(o);
            @field(self, f) = null;
        }
    }

    /// `session.run(source)`: run an entry: load it (zrun.LoadError with
    /// its errors) and run it as Python (zrun.Error on a runtime error).
    /// The value of its last statement if that's an expression, else
    /// None. What an entry defined before an error is kept, as a REPL
    /// keeps it.
    pub fn run(self: *Session, source: *PyObject) ?*PyObject {
        if (!py.PyUnicode_Check(source)) {
            ph.raise(py.PyExc_TypeError(), "source must be a str", .{});
            return null;
        }
        const lang = unwrap(Language, self._lang.?).?;
        const tree = parseTree(lang._parser.?, source) orelse return null;
        const known = py.c.PyDict_Keys(self._names.?) orelse {
            py.Py_DecRef(tree);
            return null;
        };
        defer py.Py_DecRef(known);
        var prog = Program{ ._entry = true };
        prog._lang = ref(self._lang.?);
        prog._source = ref(source);
        if (!prog.setup(lang, tree, known)) {
            prog.release();
            return null;
        }
        const obj = Module.toPy(Program, prog) orelse {
            prog.release();
            return null;
        };
        defer py.Py_DecRef(obj);
        if (py.c.PyList_Append(self._programs.?, obj) != 0) return null;
        const p = unwrap(Program, obj).?;
        p._session = self;
        p._ran_mode = .python;
        const result = onBigStack(Program.runEntry, .{p}, lang._max_depth);
        if (!self.keep(p)) {
            if (result) |r| py.Py_DecRef(r);
            return null;
        }
        return result;
    }

    /// The variables an entry defined at its top level (those with a
    /// value), the session's from now.
    fn keep(self: *Session, p: *Program) bool {
        const g = p.state().globals orelse return true;
        const frame = objects.asFrame(g);
        const d = p.ctx().data;
        for (d.syms, 0..) |s, i| {
            if (s.builtin or d.homeOf(@intCast(i)) != NONE) continue;
            if (frame.slots.?.get(@intCast(i)) == null) continue;
            const key = ph.newString(s.name) orelse return false;
            defer py.Py_DecRef(key);
            const t = py.c.Py_BuildValue("(OI)", g, @as(c_uint, @intCast(i))) orelse return false;
            defer py.Py_DecRef(t);
            if (py.c.PyDict_SetItem(self._names.?, key, t) != 0) return false;
        }
        return true;
    }

    pub const __doc__: [*:0]const u8 = "A session (Language.session()): run(source) runs an entry, which sees the variables and functions the entries before it defined (the same variables); names() lists them. Entries run as Python.";
    pub const run__doc__: [*:0]const u8 = "Run an entry: its last statement's value if that's an expression, else None. Raises zrun.LoadError or zrun.Error; what it defined before an error is kept.";
    pub const run__params__ = "source";
    pub const names__doc__: [*:0]const u8 = "The names the entries defined so far, sorted.";

    /// `session.names()`: the names the entries defined so far, sorted.
    pub fn names(self: *Session) ?*PyObject {
        const keys = py.c.PyDict_Keys(self._names.?) orelse return null;
        if (py.c.PyList_Sort(keys) != 0) {
            py.Py_DecRef(keys);
            return null;
        }
        return keys;
    }

    /// Whether `text` ends before what it's writing does (a REPL asks for
    /// more lines): its syntax error, if any, is at its end.
    fn unfinished(self: *Session, text: *PyObject) ?bool {
        const lang = unwrap(Language, self._lang.?).?;
        const r = py.c.PyObject_CallMethod(lang._parser.?, "parse", "(O)", text) orelse {
            // (taken, to look at it: put back unless it's a syntax error)
            var t: ?*PyObject = null;
            var v: ?*PyObject = null;
            var tb: ?*PyObject = null;
            py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
            py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&v), @ptrCast(&tb));
            const zgram = py.c.PyImport_ImportModule("zgram") orelse {
                inline for (.{ t, v, tb }) |o| if (o) |x| py.Py_DecRef(x);
                return null;
            };
            defer py.Py_DecRef(zgram);
            const parse_error = py.c.PyObject_GetAttrString(zgram, "ParseError") orelse {
                inline for (.{ t, v, tb }) |o| if (o) |x| py.Py_DecRef(x);
                return null;
            };
            defer py.Py_DecRef(parse_error);
            if (v == null or py.c.PyErr_GivenExceptionMatches(v, parse_error) == 0) {
                py.c.PyErr_Restore(t, v, tb);
                return null;
            }
            defer inline for (.{ t, v, tb }) |o| if (o) |x| py.Py_DecRef(x);
            const off_obj = py.c.PyObject_GetAttrString(v.?, "offset") orelse return null;
            defer py.Py_DecRef(off_obj);
            const off = py.c.PyLong_AsLongLong(off_obj);
            if (off == -1 and py.c.PyErr_Occurred() != null) return null;
            // (bytes: the end of the text, past its trailing blanks)
            const s = ph.utf8(text, "text") orelse return null;
            const end = std.mem.trimEnd(u8, s, " \t\r\n").len;
            return off >= end;
        };
        py.Py_DecRef(r);
        return false;
    }
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
        return ztypes.wrap(x);
    }

    fn evalNode(self: *Runtime, idx: u32) ?*PyObject {
        if (!self.data().hasFrame(idx)) return self.checkKind(idx, self.evalHere(idx) orelse return null);
        const saved = self.enterScope(idx) orelse return null;
        defer self.leaveScope(saved);
        return self.checkKind(idx, self.evalHere(idx) orelse return null);
    }

    /// A node's value checked against what Language.types() says its
    /// type's values are, as compiled code checks it: the value (taken),
    /// or null with the error.
    fn checkKind(self: *Runtime, idx: u32, v: *PyObject) ?*PyObject {
        const mapping = self._lang.?._types orelse return v;
        const p = self._p.?;
        const analysis = p.state().analysis orelse return v;
        const e = p._declared.getOrPut(allocator, idx) catch {
            py.Py_DecRef(v);
            return py.c.PyErr_NoMemory();
        };
        if (!e.found_existing) {
            e.value_ptr.* = compile_mod.declaredClass(mapping, analysis, idx) catch {
                _ = p._declared.remove(idx);
                py.Py_DecRef(v);
                return null;
            };
        }
        const cls = e.value_ptr.* orelse return v;
        const declared = compile_mod.Compiler.kindOfClass(cls) orelse return v;
        const t = ph.typeOf(v);
        const ok = switch (declared.shape) {
            // (an int of 64 bits, or an I64: compiled code's ints)
            .int => blk: {
                if (@as(*PyObject, @ptrCast(@alignCast(t))) == ztypes.I64) break :blk true;
                if (t != @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyLong_Type"))))) break :blk false;
                var overflow: c_int = 0;
                _ = py.c.PyLong_AsLongLongAndOverflow(v, &overflow);
                break :blk overflow == 0;
            },
            .none => v == py.Py_None(),
            .function => objects.asFunction(v) != null,
            else => @as(*PyObject, @ptrCast(@alignCast(t))) == cls,
        };
        if (ok) return v;
        py.Py_DecRef(v);
        const type_text = py.c.PyObject_CallMethod(analysis, "type_of", "I", @as(c_uint, idx)) orelse return null;
        defer py.Py_DecRef(type_text);
        return self.fail(idx, "the value isn't what types() says the type {s} is", .{ph.utf8(type_text, "type") orelse "?"});
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
                return ztypes.wrapOwned(r) orelse self.raised(idx);
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

    /// The root run as execNode(0) runs it, its value (a REPL shows it):
    /// what the root's exec semantics return (a language's chunk giving
    /// what a `return` at its top level returned), or, the root having no
    /// semantics of its own, its last statement's if that's an expression.
    /// A new reference (None otherwise), or null with the exception.
    fn execRoot(self: *Runtime) ?*PyObject {
        const lang = self._lang.?;
        lang.resolve();
        const d = self.data();
        const root_rule = d.rule(0);
        if (root_rule < lang._exec_of.len) if (lang._exec_of[root_rule]) |f| {
            var saved: ?*PyObject = null;
            if (d.hasFrame(0)) saved = self.enterScope(0) orelse return null;
            defer if (saved) |s| self.leaveScope(s);
            const prev = self._at;
            self._at = 0;
            defer self._at = prev;
            const n = self.node(0) orelse return null;
            defer py.Py_DecRef(n);
            const r = py.c.PyObject_CallFunctionObjArgs(f, n, self.obj(), @as(?*PyObject, null)) orelse return self.raised(0);
            return ztypes.wrapOwned(r) orelse self.raised(0);
        };
        const own = struct {
            fn has(l: *Language, rid: u32) bool {
                return (rid < l._exec_of.len and l._exec_of[rid] != null) or
                    (rid < l._functions.len and l._functions[rid] != null) or
                    (rid < l._eval_of.len and l._eval_of[rid] != null);
            }
        }.has;
        if (own(lang, d.rule(0))) return if (self.execNode(0)) none() else null;
        var saved: ?*PyObject = null;
        if (d.hasFrame(0)) saved = self.enterScope(0) orelse return null;
        defer if (saved) |s| self.leaveScope(s);
        const values = objects.childValues(self.stateObj(), self.ctx(), 0) orelse return null;
        defer py.Py_DecRef(values);
        // (the last statement, if it's an expression: its rule has eval
        // semantics and no exec ones, or it's a name or a literal)
        // (a literal's value is the statement itself: its own value)
        const n: isize = if (py.PyList_Check(values)) py.c.PyList_Size(values) else -1;
        const last: ?*PyObject = if (n > 0) blk: {
            const item = py.c.PyList_GetItem(values, n - 1).?;
            if (objects.asNode(item)) |nd| {
                if (nd.ctx != self.ctx()) break :blk null;
                break :blk if (self.isExpression(nd.idx) orelse return null) item else null;
            }
            break :blk if (py.PyList_Check(item) or py.PyTuple_Check(item)) null else item;
        } else null;
        const target = last orelse return if (self.execObj(values)) none() else null;
        const before = py.c.PyList_GetSlice(values, 0, n - 1) orelse return null;
        defer py.Py_DecRef(before);
        if (!self.execObj(before)) return null;
        return self.evalObj(target);
    }

    /// Whether a statement node is an expression run for its value: its
    /// rule has eval semantics and no exec ones (nor is a function kind),
    /// or it's a name, or a literal (a value of zgram's action). Null with
    /// an exception.
    fn isExpression(self: *Runtime, idx: u32) ?bool {
        const lang = self._lang.?;
        const d = self.data();
        const rid = d.rule(idx);
        if (rid < lang._exec_of.len and lang._exec_of[rid] != null) return false;
        if (rid < lang._functions.len and lang._functions[rid] != null) return false;
        if (rid < lang._eval_of.len and lang._eval_of[rid] != null) return true;
        if (d.symbolIndex(idx) != null) return true;
        const values = objects.childValues(self.stateObj(), self.ctx(), idx) orelse return null;
        defer py.Py_DecRef(values);
        return py.c.PyList_Size(values) == 1 and objects.asNode(py.c.PyList_GetItem(values, 0).?) == null;
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
        switch (ztypes.pendingControl()) {
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
            // (a variable an earlier entry of the session defined)
            if (self.sessionVariable(key)) |at| {
                const v = at.frame.slots.?.get(at.sym) orelse return self.fail(idx, "'{s}' has no value yet", .{sym.name});
                return self.checkKind(idx, ref(v));
            }
            const h = py.c.PyDict_GetItem(self._lang.?._hosts.?, key) orelse
                return self.fail(idx, "no host function for the builtin '{s}'", .{sym.name});
            return ref(h);
        }
        const f = self.frameFor(si, idx) orelse return null;
        const v = f.slots.?.get(si) orelse return self.fail(idx, "'{s}' has no value yet", .{sym.name});
        return self.checkKind(idx, ref(v));
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
        var slot = si;
        const f = if (d.syms[si].builtin) blk: {
            const key = ph.newString(d.syms[si].name) orelse return false;
            defer py.Py_DecRef(key);
            // (a variable an earlier entry of the session defined)
            const at = self.sessionVariable(key) orelse {
                _ = self.fail(idx, "can't assign to the builtin '{s}'", .{d.syms[si].name});
                return false;
            };
            slot = at.sym;
            break :blk at.frame;
        } else self.frameFor(si, idx) orelse return false;
        const v = ztypes.wrap(value) orelse {
            _ = self.raised(idx);
            return false;
        };
        defer py.Py_DecRef(v);
        objects.setSlot(f, slot, v) catch {
            _ = py.c.PyErr_NoMemory();
            return false;
        };
        return true;
    }

    /// Where a variable of an earlier entry of the program's session lives
    /// (`name`, a builtin here), or null (none: not a session's, or not
    /// defined by its entries).
    fn sessionVariable(self: *Runtime, name: *PyObject) ?Session.Place {
        const s = self._p.?._session orelse return null;
        return s.place(name);
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

    /// `rt.tail_call(f, args, receiver=None)`: return what calling f with
    /// the arguments returns, the function being run left first (its frame
    /// gone: a chain of tail calls takes no more depth). Raises
    /// zrun.TailCall, carrying the call, for the call running the function
    /// to make (callValue); at the top level, no function to leave: the
    /// call made here and its result returned (rt.Return).
    pub fn tail_call(self: *Runtime, a: pyoz.Args(struct { f: *PyObject, args: *PyObject, receiver: ?*PyObject = null })) ?*PyObject {
        const v = a.value;
        const tuple = py.c.PySequence_Tuple(v.args) orelse return null;
        defer py.Py_DecRef(tuple);
        const recv = optional(v.receiver);
        if (self._calls.items.len == 0) {
            const r = self.callValue(v.f, tuple, recv) orelse return null;
            defer py.Py_DecRef(r);
            const exc = py.c.PyObject_CallFunctionObjArgs(ztypes.Return, r, @as(?*PyObject, null)) orelse return null;
            defer py.Py_DecRef(exc);
            py.c.PyErr_SetObject(ztypes.Return, exc);
            return null;
        }
        const exc = py.c.PyObject_CallFunctionObjArgs(ztypes.TailCall, v.f, tuple, recv orelse py.Py_None(), @as(?*PyObject, null)) orelse return null;
        defer py.Py_DecRef(exc);
        py.c.PyErr_SetObject(ztypes.TailCall, exc);
        return null;
    }

    /// Call `f` (a function of the program or a host function); a function
    /// ending with rt.tail_call: the call it left made here, at the same
    /// depth, its call site this one, and so on.
    fn callValue(self: *Runtime, f: *PyObject, args: *PyObject, receiver: ?*PyObject) ?*PyObject {
        const at = self._at;
        var cur_f = ref(f);
        var cur_args = ref(args);
        var cur_recv: ?*PyObject = if (receiver) |r| ref(r) else null;
        while (true) {
            const r = self.callValueOnce(cur_f, cur_args, cur_recv);
            py.Py_DecRef(cur_f);
            py.Py_DecRef(cur_args);
            if (cur_recv) |x| py.Py_DecRef(x);
            if (r != null or ztypes.pendingControl() != .tail) return r;
            // (the call the exception carries: (f, args, receiver))
            var t: ?*PyObject = null;
            var e: ?*PyObject = null;
            var tb: ?*PyObject = null;
            py.c.PyErr_Fetch(@ptrCast(&t), @ptrCast(&e), @ptrCast(&tb));
            py.c.PyErr_NormalizeException(@ptrCast(&t), @ptrCast(&e), @ptrCast(&tb));
            defer inline for (.{ t, e, tb }) |o| {
                if (o) |held| py.Py_DecRef(held);
            };
            const exc_args = py.c.PyObject_GetAttrString(e orelse return null, "args") orelse return null;
            defer py.Py_DecRef(exc_args);
            cur_f = ref(py.c.PyTuple_GetItem(exc_args, 0) orelse return null);
            cur_args = ref(py.c.PyTuple_GetItem(exc_args, 1).?);
            const rv = py.c.PyTuple_GetItem(exc_args, 2).?;
            cur_recv = if (rv == py.Py_None()) null else ref(rv);
            self._at = at;
        }
    }

    fn callValueOnce(self: *Runtime, f: *PyObject, args: *PyObject, receiver: ?*PyObject) ?*PyObject {
        if (objects.asFunction(f)) |fo| return self.callFunction(fo, args, receiver);
        if (py.PyCallable_Check(f)) {
            const all = if (receiver) |r| prepend(r, args) orelse return null else ref(args);
            defer py.Py_DecRef(all);
            const wrapped = wrapAll(all) orelse return null;
            defer py.Py_DecRef(wrapped);
            const r = py.c.PyObject_CallObject(f, wrapped) orelse return self.hostFailed(f);
            return ztypes.wrapOwned(r) orelse self.raised(self._at);
        }
        const tname = typeName(f);
        return self.fail(self._at, "'{s}' value is not callable", .{tname});
    }

    fn hostFailed(self: *Runtime, f: *PyObject) ?*PyObject {
        if (ztypes.pendingControl() != .none or py.c.PyErr_ExceptionMatches(ztypes.Error) != 0) return null;
        // (a host function can throw an error of the language too)
        if (py.c.PyErr_ExceptionMatches(ztypes.Throw) != 0) {
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
        if (fo.state != self.stateObj()) {
            // (an earlier entry of the session's: called in its program)
            const s = self._p.?._session orelse return self.fail(self._at, "a function of another program can't be called here", .{});
            const other = s.programOf(fo.state.?) orelse return self.fail(self._at, "a function of another program can't be called here", .{});
            const h = Runtime.begin(other) orelse return null;
            defer h.end();
            const r = h.self();
            r._depth = self._depth;
            r._at = fo.node;
            if (self._context) |c| r._context = ref(c);
            return r.callFunction(fo, args, receiver);
        }
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
        if (ztypes.pendingControl() == .ret) return ztypes.takeReturn();
        if (ztypes.pendingControl() == .brk or ztypes.pendingControl() == .cont) {
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

    /// `rt.u8(data, i)`: the int at offset i of data (a zrun.Bytes, or any
    /// data), bounds checked; `rt.i8`, `rt.u16le`... `rt.i64be` the same
    /// for their widths, signs and byte orders.
    pub fn @"u8"(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.u8, buf, i);
    }
    pub fn @"i8"(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.i8, buf, i);
    }
    pub fn u16le(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.u16le, buf, i);
    }
    pub fn u16be(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.u16be, buf, i);
    }
    pub fn i16le(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.i16le, buf, i);
    }
    pub fn i16be(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.i16be, buf, i);
    }
    pub fn u32le(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.u32le, buf, i);
    }
    pub fn u32be(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.u32be, buf, i);
    }
    pub fn i32le(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.i32le, buf, i);
    }
    pub fn i32be(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.i32be, buf, i);
    }
    pub fn u64le(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.u64le, buf, i);
    }
    pub fn u64be(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.u64be, buf, i);
    }
    pub fn i64le(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.i64le, buf, i);
    }
    pub fn i64be(_: *Runtime, buf: *PyObject, i: *PyObject) ?*PyObject {
        return bytes_mod.read(.i64be, buf, i);
    }

    /// `rt.wrapping_add(a, b)`, `rt.wrapping_sub`, `rt.wrapping_mul`: 64-bit
    /// arithmetic wrapping around; `rt.wrapping_shl(a, n)`,
    /// `rt.wrapping_shr`, `rt.wrapping_ushr`: 64-bit shifts by n modulo 64
    /// (wrapping.zig).
    pub fn wrapping_add(self: *Runtime, a: *PyObject, b: *PyObject) ?*PyObject {
        _ = self;
        return @import("wrapping.zig").ofPython(.add, a, b);
    }

    pub fn wrapping_sub(self: *Runtime, a: *PyObject, b: *PyObject) ?*PyObject {
        _ = self;
        return @import("wrapping.zig").ofPython(.sub, a, b);
    }

    pub fn wrapping_mul(self: *Runtime, a: *PyObject, b: *PyObject) ?*PyObject {
        _ = self;
        return @import("wrapping.zig").ofPython(.mul, a, b);
    }

    pub fn wrapping_shl(self: *Runtime, a: *PyObject, n: *PyObject) ?*PyObject {
        _ = self;
        return @import("wrapping.zig").ofPython(.shl, a, n);
    }

    pub fn wrapping_shr(self: *Runtime, a: *PyObject, n: *PyObject) ?*PyObject {
        _ = self;
        return @import("wrapping.zig").ofPython(.shr, a, n);
    }

    pub fn wrapping_ushr(self: *Runtime, a: *PyObject, n: *PyObject) ?*PyObject {
        _ = self;
        return @import("wrapping.zig").ofPython(.ushr, a, n);
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
        return ref(ztypes.Return);
    }

    /// raise rt.Break()
    pub fn get_Break(_: *const Runtime) ?*PyObject {
        return ref(ztypes.Break);
    }

    /// raise rt.Continue()
    pub fn get_Continue(_: *const Runtime) ?*PyObject {
        return ref(ztypes.Continue);
    }

    /// raise rt.Throw(value, message=None)
    pub fn get_Throw(_: *const Runtime) ?*PyObject {
        return ref(ztypes.Throw);
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
        if (ztypes.pendingControl() != .none or py.c.PyErr_ExceptionMatches(ztypes.Error) != 0) return null;
        if (py.c.PyErr_ExceptionMatches(ztypes.Throw) != 0) {
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
        py.c.PyErr_SetObject(ztypes.Error, exc);
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
        const exc = py.c.PyObject_CallFunctionObjArgs(ztypes.Error, rendered, @as(?*PyObject, null)) orelse return null;
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
    pub const tail_call__doc__: [*:0]const u8 = "Return what calling f with the arguments returns, the function being run left first (its frame given up: a chain of tail calls takes no depth). Never returns. At the top level, a call whose result is returned.";
    pub const tail_call__params__ = "f, args";
    pub const error__doc__: [*:0]const u8 = "Stop the program with a runtime error at the node.";
    pub const kind__doc__: [*:0]const u8 = "A node's kind (also when it has a field named kind).";
    pub const kind__params__ = "node";
    pub const text__doc__: [*:0]const u8 = "A node's text.";
    pub const text__params__ = "node";
    pub const span__doc__: [*:0]const u8 = "A node's (start, end) byte offsets.";
    pub const span__params__ = "node";
    pub const Return__doc__: [*:0]const u8 = "zrun.Return: raise rt.Return(value) to return from the function being run.";
    pub const Break__doc__: [*:0]const u8 = "zrun.Break: raise rt.Break() to leave the loop rt.loop() is running.";
    pub const Continue__doc__: [*:0]const u8 = "zrun.Continue: raise rt.Continue() to go on with the loop's next iteration.";
    pub const Throw__doc__: [*:0]const u8 = "zrun.Throw: raise rt.Throw(value, message=None) for an error of the language carrying a value of it, caught by semantics (except rt.Throw as e: e.value).";
    pub const receiver__doc__: [*:0]const u8 = "What the method being run was called on (rt.call(f, args, receiver=x)), through the frames functions were made in; None outside a method.";
    pub const varargs__doc__: [*:0]const u8 = "The arguments the function being run got beyond its parameters (a function kind with extra='keep'), a tuple; () otherwise.";
    pub const context__doc__: [*:0]const u8 = "What the host passed to the run or call (context=), or None.";
    pub const path__doc__: [*:0]const u8 = "The name the program was loaded under (Language.load(source, path)), or None.";
    pub const fresh__doc__: [*:0]const u8 = "From here, the variables of the block scope `node` (being run) are new ones, closures made so far keeping theirs: a loop's variable new each time round (Lua's for, JavaScript's for (let ...)).";
    pub const fresh__params__ = "node";
    pub const scope__doc__: [*:0]const u8 = "The scope node the name `node` refers to is defined in (a function, a struct...), or None for builtins and the global scope.";
    pub const scope__params__ = "node";
    pub const symbol__doc__: [*:0]const u8 = "zrules' Symbol for the name `node` defines or uses, or None.";
    pub const symbol__params__ = "node";
    pub const type_of__doc__: [*:0]const u8 = "The type zrules' types() rule gave a node, as text ('int', 'list[float]', 'Point?'), or None.";
    pub const type_of__params__ = "node";
    pub const node_at__doc__: [*:0]const u8 = "The node at an index of the tree (symbols refer to nodes by index).";
    pub const node_at__params__ = "index";
    pub const wrapping_add__doc__: [*:0]const u8 = "a + b in 64 bits, wrapping around (signed).";
    pub const wrapping_add__params__ = "a, b";
    pub const wrapping_sub__doc__: [*:0]const u8 = "a - b in 64 bits, wrapping around (signed).";
    pub const wrapping_sub__params__ = "a, b";
    pub const wrapping_mul__doc__: [*:0]const u8 = "a * b in 64 bits, wrapping around (signed).";
    pub const wrapping_mul__params__ = "a, b";
    pub const wrapping_shl__doc__: [*:0]const u8 = "a << n in 64 bits, the bits shifted out lost; n modulo 64.";
    pub const wrapping_shl__params__ = "a, n";
    pub const wrapping_shr__doc__: [*:0]const u8 = "a >> n in 64 bits, the sign kept (arithmetic shift); n modulo 64.";
    pub const wrapping_shr__params__ = "a, n";
    pub const wrapping_ushr__doc__: [*:0]const u8 = "a >> n in 64 bits with zeros shifted in (a as unsigned: logical shift); n modulo 64.";
    pub const wrapping_ushr__params__ = "a, n";
    pub const u8__doc__: [*:0]const u8 = "The unsigned byte at offset i of data (a zrun.Bytes, bytes, or any buffer); IndexError past the end.";
    pub const u8__params__ = "data, i";
    pub const i8__doc__: [*:0]const u8 = "The signed byte at offset i of data; IndexError past the end.";
    pub const i8__params__ = "data, i";
    pub const u16le__doc__: [*:0]const u8 = "The unsigned 16-bit little-endian int at offset i of data; IndexError past the end.";
    pub const u16le__params__ = "data, i";
    pub const u16be__doc__: [*:0]const u8 = "The unsigned 16-bit big-endian int at offset i of data; IndexError past the end.";
    pub const u16be__params__ = "data, i";
    pub const i16le__doc__: [*:0]const u8 = "The signed 16-bit little-endian int at offset i of data; IndexError past the end.";
    pub const i16le__params__ = "data, i";
    pub const i16be__doc__: [*:0]const u8 = "The signed 16-bit big-endian int at offset i of data; IndexError past the end.";
    pub const i16be__params__ = "data, i";
    pub const u32le__doc__: [*:0]const u8 = "The unsigned 32-bit little-endian int at offset i of data; IndexError past the end.";
    pub const u32le__params__ = "data, i";
    pub const u32be__doc__: [*:0]const u8 = "The unsigned 32-bit big-endian int at offset i of data; IndexError past the end.";
    pub const u32be__params__ = "data, i";
    pub const i32le__doc__: [*:0]const u8 = "The signed 32-bit little-endian int at offset i of data; IndexError past the end.";
    pub const i32le__params__ = "data, i";
    pub const i32be__doc__: [*:0]const u8 = "The signed 32-bit big-endian int at offset i of data; IndexError past the end.";
    pub const i32be__params__ = "data, i";
    pub const u64le__doc__: [*:0]const u8 = "The unsigned 64-bit little-endian int at offset i of data; IndexError past the end.";
    pub const u64le__params__ = "data, i";
    pub const u64be__doc__: [*:0]const u8 = "The unsigned 64-bit big-endian int at offset i of data; IndexError past the end.";
    pub const u64be__params__ = "data, i";
    pub const i64le__doc__: [*:0]const u8 = "The signed 64-bit little-endian int at offset i of data; IndexError past the end.";
    pub const i64le__params__ = "data, i";
    pub const i64be__doc__: [*:0]const u8 = "The signed 64-bit big-endian int at offset i of data; IndexError past the end.";
    pub const i64be__params__ = "data, i";
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
    if (is(typ, ztypes.IntegerOverflow)) return ph.newString("integer overflow");
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
        const v = ztypes.wrap(py.c.PyTuple_GetItem(args, @intCast(i)).?) orelse {
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

/// zrun.collect(): this thread's cycle collection (gc.zig), now.
fn collect() i64 {
    return @intCast(gc.collectHere());
}

/// zrun._blocks(): the values' blocks allocated and not freed (for the
/// tests: compiled code gives back what it takes).
fn blocks() i64 {
    return pool.inUse();
}

/// zrun.configure(cache=None, cache_size=None, perf_map=None, tiers=None):
/// process-wide settings.
fn configure(args: pyoz.Args(struct { cache: ?*PyObject = null, cache_size: ?*PyObject = null, perf_map: ?*PyObject = null, tiers: ?*PyObject = null })) ?*PyObject {
    const v = args.value;
    if (optional(v.cache_size)) |s| {
        const n = py.c.PyLong_AsLongLong(s);
        if (n == -1 and py.c.PyErr_Occurred() != null) {
            py.c.PyErr_Clear();
            ph.raise(py.PyExc_TypeError(), "cache_size must be an int (bytes; 0: no limit)", .{});
            return null;
        }
        if (n < 0) {
            ph.raise(py.PyExc_ValueError(), "cache_size must be 0 (no limit) or more bytes, not {d}", .{n});
            return null;
        }
        @import("cache.zig").limit = @intCast(n);
    }
    if (optional(v.cache)) |c| {
        const s: @import("cache.zig").Setting = if (c == py.Py_True()) .default else if (c == py.Py_False()) .off else blk: {
            const path = ph.utf8(c, "cache") orelse {
                py.c.PyErr_Clear();
                ph.raise(py.PyExc_TypeError(), "cache must be True, False or a directory (str)", .{});
                return null;
            };
            break :blk .{ .dir = path };
        };
        @import("cache.zig").set(s) catch return py.c.PyErr_NoMemory();
    }
    if (optional(v.perf_map)) |p| {
        const r = py.c.PyObject_IsTrue(p);
        if (r < 0) return null;
        driver.perf_map = r == 1;
    }
    if (optional(v.tiers)) |t| {
        const r = py.c.PyObject_IsTrue(t);
        if (r < 0) return null;
        driver.tiers = r == 1;
    }
    return none();
}

/// zrun.clear_cache(): the cache's objects deleted.
fn clearCache() ?*PyObject {
    @import("cache.zig").trim(0);
    return none();
}

/// `@zrun.comptime`: the function marked (returned as it is: the reference
/// mode calls it as ever); compiled code given values known when compiling
/// calls it then, its result a constant of the code.
fn comptimeMark(f: *PyObject) ?*PyObject {
    if (!py.PyCallable_Check(f)) {
        ph.raise(py.PyExc_TypeError(), "zrun.comptime takes a function", .{});
        return null;
    }
    if (py.c.PyObject_SetAttrString(f, compile_mod.comptime_attr, py.Py_True()) != 0) {
        py.c.PyErr_Clear();
        ph.raise(py.PyExc_TypeError(), "zrun.comptime takes a function it can mark (a def or a lambda, not a builtin)", .{});
        return null;
    }
    return ref(f);
}

/// The builder of executables (exe/build.py), run once when first asked for.
var exe_builder: ?*PyObject = null;

/// zrun.build_executable(language, source, output, target=None, path=None,
/// python=None): exe/build.py's, given the launcher's source.
fn buildExecutable(args: pyoz.Args(struct { language: *PyObject, source: *PyObject, output: *PyObject, target: ?*PyObject = null, path: ?*PyObject = null, python: ?*PyObject = null, setup: ?*PyObject = null })) ?*PyObject {
    const v = args.value;
    const ns = exe_builder orelse blk: {
        const n = @import("compile.zig").runPython(@embedFile("exe/build.py")) orelse return null;
        exe_builder = n;
        break :blk n;
    };
    const build = py.c.PyDict_GetItemString(ns, "build_executable") orelse return null;
    const launcher = ph.newString(@embedFile("exe/launcher.zig")) orelse return null;
    defer py.Py_DecRef(launcher);
    const pos = py.c.PyTuple_New(3) orelse return null;
    defer py.Py_DecRef(pos);
    for ([_]*PyObject{ v.language, v.source, v.output }, 0..) |o, i| {
        _ = py.c.PyTuple_SetItem(pos, @intCast(i), ref(o));
    }
    const kw = py.c.PyDict_New() orelse return null;
    defer py.Py_DecRef(kw);
    if (py.c.PyDict_SetItemString(kw, "launcher_source", launcher) != 0) return null;
    inline for (.{ "target", "path", "python", "setup" }) |name| {
        if (optional(@field(v, name))) |o| {
            if (py.c.PyDict_SetItemString(kw, name, o) != 0) return null;
        }
    }
    return py.c.PyObject_Call(build, pos, kw);
}

/// zrun's runtime for standalone programs (libzrun_rt.a), in Linux builds
const rt_archive: []const u8 = if (@import("build_options").has_rt) @embedFile("zrun_rt_archive") else "";

/// zrun.build_native(language, source, output, path=None): exe/build.py's,
/// given the runtime to link with.
fn buildNative(args: pyoz.Args(struct { language: *PyObject, source: *PyObject, output: *PyObject, path: ?*PyObject = null, prune: ?*PyObject = null, left_out: ?*PyObject = null, setup: ?*PyObject = null })) ?*PyObject {
    const v = args.value;
    const ns = exe_builder orelse blk: {
        const n = @import("compile.zig").runPython(@embedFile("exe/build.py")) orelse return null;
        exe_builder = n;
        break :blk n;
    };
    const build = py.c.PyDict_GetItemString(ns, "build_native") orelse return null;
    const runtime = py.c.PyBytes_FromStringAndSize(rt_archive.ptr, @intCast(rt_archive.len)) orelse return null;
    defer py.Py_DecRef(runtime);
    const pos = py.c.PyTuple_New(3) orelse return null;
    defer py.Py_DecRef(pos);
    for ([_]*PyObject{ v.language, v.source, v.output }, 0..) |o, i| {
        _ = py.c.PyTuple_SetItem(pos, @intCast(i), ref(o));
    }
    const kw = py.c.PyDict_New() orelse return null;
    defer py.Py_DecRef(kw);
    if (py.c.PyDict_SetItemString(kw, "runtime", runtime) != 0) return null;
    if (optional(v.path)) |o| if (py.c.PyDict_SetItemString(kw, "path", o) != 0) return null;
    if (optional(v.prune)) |o| if (py.c.PyDict_SetItemString(kw, "prune", o) != 0) return null;
    if (optional(v.left_out)) |o| if (py.c.PyDict_SetItemString(kw, "left_out", o) != 0) return null;
    if (optional(v.setup)) |o| if (py.c.PyDict_SetItemString(kw, "setup", o) != 0) return null;
    return py.c.PyObject_Call(build, pos, kw);
}

fn moduleInit(module: *PyObject) callconv(.c) c_int {
    ph.initVersion() catch return -1;
    if (ztypes.init(module) != 0) return -1;
    objects.init(module) catch return -1;
    bridge.init(module) catch return -1;
    @import("proxies.zig").init(module) catch return -1;
    @import("bytes.zig").init(module) catch return -1;
    @import("native.zig").init(module) catch return -1;
    helpers.init() catch return -1;
    @import("set.zig").init();
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
        pyoz.func("_blocks", blocks, "The values' blocks allocated and not freed (for tests)"),
        pyoz.func("collect", collect, "collect(): free the compiled code's values that only reference one another (reference cycles); how many were freed. Runs by itself as values are made, and at the end of a run."),
        pyoz.func("clear_cache", clearCache, "clear_cache(): delete the compiled code kept in the cache."),
        pyoz.func("comptime", comptimeMark, "@zrun.comptime: a function whose result depends only on its arguments (its author says so: any Python, the subset or not). Compiled code calling it with values known when compiling (the program's tree's, constants, other such results) calls it then, once for those values in the process: its result is a constant of the code (lists, dicts and tuples in it native, read-only). With values known only at run time it's called as any function is. The reference mode calls it as ever."),
        pyoz.kwfunc("build_executable", buildExecutable, "build_executable(language, source, output, target=None, path=None, python=None, setup=None): one executable file running the program: a Python runtime, zrun and its packages, the language's module (and the modules beside it), the program and its compiled code. language: the Language, or 'module:attribute'; source: the program's text or its file; target: 'x86_64-linux' or 'x86_64-windows' (default: this machine's); python: '3.10' ... '3.14' (default: this one's); setup: 'module:function', a function of a module beside the language's called with the program's path and its arguments before it runs (Lua's `arg`).Needs the ziglang package (pip install zrun-py[exe]); downloads the runtime (python-build-standalone's) once. Returns the executable's path."),
        pyoz.kwfunc("build_native", buildNative, "build_native(language, source, output, path=None, prune=False, left_out=None, setup=None): a standalone program: a strict language's program compiled ahead of time, all of it, and linked with zrun's runtime into one executable with no Python in it (Linux, for this machine). language: the Language, or 'module:attribute'; source: the program's text or its file; path: the name its errors give it; prune: only the library functions the program can name compiled (smaller, faster to build; one reached anyway stops the program with an error naming it); left_out: a list, the names of those left out appended; setup: a function (or 'module:function', a function of a module beside the language's), compiled too, called with the program's path and its arguments (a list of strs) before it runs: 'lua:set_args' makes Lua's `arg`. Needs the ziglang package (pip install zrun-py[exe]). Its runtime errors are written as the reference mode words them, exiting with 1. Returns the executable's path."),
        pyoz.kwfunc("configure", configure, "configure(cache=None, cache_size=None, perf_map=None, tiers=None): process-wide settings (those not given stay). cache: True (the platform's place for caches: %LOCALAPPDATA%\\zrun\\Cache on Windows, ~/Library/Caches/zrun on macOS, $XDG_CACHE_HOME/zrun or ~/.cache/zrun elsewhere), False (no cache), or a directory; cache_size: the most the cache takes, in bytes (default 1 GiB; 0: no limit): past it, the least recently used compiled code is deleted, down to 80% of it; perf_map: name compiled functions for Linux's perf (/tmp/perf-<pid>.map); tiers: True (default: a program run compiled whose optimized code isn't cached is compiled fast first, optimized in the background, the optimized code running from the run after it's done) or False (optimized at once)."),
    },
    .classes = &.{
        pyoz.class("Language", Language),
        pyoz.class("Registrar", Registrar),
        pyoz.class("Program", Program),
        pyoz.class("Runtime", Runtime),
        pyoz.class("Session", Session),
    },
    .module_init = moduleInit,
});

// Required: forces analysis of all pub decls so PyInit_ is exported.
comptime {
    for (@typeInfo(@This()).@"struct".decls) |decl| {
        _ = @field(@This(), decl.name);
    }
}
