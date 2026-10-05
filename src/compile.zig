//! The compiler: a program's semantics, partially evaluated for its tree,
//! built as LLVM IR in memory (ir.zig, LLVM's C API) and JIT-compiled by
//! zgram's LLVM.
//!
//! Each semantic is run at compile time with its node as a known value:
//! whatever depends only on the tree (fields, kinds, texts, `node.op ==
//! "+"`, loops over a node's children) is decided here, and what depends on
//! the program's data becomes code. `rt.eval(child)` inlines the child's
//! semantic; `rt.load` / `rt.store` are the variable's slot; `raise
//! rt.Return(v)` is a jump. A language function becomes an LLVM function,
//! the top level another.
//!
//! Values known here are `SVal`s; values only known at run time are
//! `Dyn`: a tag and 64 bits in SSA registers, with the shape known of them
//! (an int, a str...), which lets int and float arithmetic be inlined.
//! Dynamic values are owned references: whoever ends up with one keeps it
//! or drops it.

const std = @import("std");
const ph = @import("pyhelp.zig");
const py = ph.py;
const PyObject = ph.PyObject;
const front = @import("front.zig");
const ir = @import("ir.zig");
const jit_c = @import("jit.zig").c;
const L = @import("jit.zig").f;
const program_mod = @import("program.zig");
const grammar_mod = @import("grammar.zig");
const helpers = @import("helpers.zig");
const value = @import("value.zig");
const objects_mod = @import("objects.zig");
const types_mod = @import("types.zig");
const adopt_mod = @import("adopt.zig");
const Value = value.Value;

const Allocator = std.mem.Allocator;
const NONE = program_mod.NONE;
const FunctionSpec = program_mod.FunctionSpec;

/// What the compiler needs of the language
pub const LangView = struct {
    grammar: *const grammar_mod.Grammar,
    eval_of: []const ?*PyObject,
    exec_of: []const ?*PyObject,
    functions: []const ?FunctionSpec,
    /// The semantics, and the helpers they call, as the front read them
    /// (the language's: helpers are added as they're read)
    read: *std.AutoHashMapUnmanaged(*PyObject, *front.Function),
    hosts: *PyObject,
    /// The State object of the program (its tree, for scalar field values)
    tree: *PyObject,
    analysis: ?*PyObject,
    /// The name the program was loaded under (rt.path), if any
    path: ?[]const u8 = null,
    /// Semantics run as Python (they couldn't be compiled): called from
    /// the compiled code with an rt over its frames (bridge.zig)
    python: *const PythonSet,
    /// List and dict literals built at run time (they escape where a known
    /// one can't follow: learned while compiling), by the code they're in
    /// (Gen.unit: elsewhere, the same literal may stay known)
    escaping: *std.AutoHashMapUnmanaged(EscapeKey, void),
    /// Calls of a call site before its helper is compiled for it (Site;
    /// Language(hot_calls=...))
    hot_calls: i64,
    /// Language.types(): a dict by type name or a function of a type's text
    types: ?*PyObject = null,
};

pub const EscapeKey = struct { origin: *const front.Expr, unit: u64 };

/// Semantics run as Python, by their function: why (the compiler's
/// message, owned by c_allocator)
pub const PythonSet = std.AutoHashMapUnmanaged(*PyObject, []const u8);

pub const Which = enum(u32) { eval = 0, exec = 1 };

/// A node's semantic
pub const Semantic = union(enum) {
    none,
    compiled: *const front.Function,
    python: *PyObject,
};

/// A compile failure: what and where (a source line of a semantic, or a
/// node of the program)
pub const Failure = struct {
    message: std.ArrayListUnmanaged(u8) = .empty,
};

pub const Error = error{ OutOfMemory, Python, Unsupported };

// ======================================================================
// Values at compile time
// ======================================================================

pub const Shape = enum { any, none, bool, int, float, str, list, tuple, dict, record, function, node };

/// A value only known at run time: its two words (LLVM values: computed,
/// or constants)
pub const Dyn = struct {
    tag: ir.Value,
    bits: ir.Value,
    shape: Shape,
    /// A record's type, when known (its type's kind: Language.types())
    rtype: ?*value.RecordType = null,
    /// The language function it most likely is (a variable a function's
    /// definition binds): called directly, behind a check (directCall)
    func: u32 = NONE,
    /// A variable's value read without a reference of its own (Gen.loadVar):
    /// a slot (i64) saying, as the code runs, whether it's borrowed (0, the
    /// variable's reference keeps it), owned (1: a reference taken since)
    /// or given up (2). Null: an ordinary value, owned.
    state: ir.Value = null,

    fn heapish(self: Dyn) bool {
        return switch (self.shape) {
            .none, .bool, .int, .float, .node => false,
            else => true,
        };
    }
};

pub const RtMethod = enum { eval, exec, loop, load, store, function, call, @"error", kind, text, span, scope, symbol, type_of, node_at, fresh, Return, Break, Continue };

/// A list known at compile time (its items may be dynamic): mutable, with
/// identity (aliases see changes)
pub const SList = struct {
    items: std.ArrayListUnmanaged(SVal) = .empty,
    /// The literal (or comprehension) that made it, if one did: built at
    /// run time instead once it's known to escape where a known one can't
    /// follow (Compiler.escaping)
    origin: ?*const front.Expr = null,
    /// A module's table only read (frozenTable): the object it is
    frozen: ?*PyObject = null,
};

/// A dict known at compile time: keys known (scalars), values maybe not
pub const SDict = struct {
    keys: std.ArrayListUnmanaged(SVal) = .empty,
    values: std.ArrayListUnmanaged(SVal) = .empty,
    origin: ?*const front.Expr = null,
    /// A module's table only read (frozenTable): the object it is
    frozen: ?*PyObject = null,

    fn find(self: *const SDict, key: SVal) ?usize {
        for (self.keys.items, 0..) |k, i| if (sameKey(k, key)) return i;
        return null;
    }
};

/// Equal keys known when compiling (Python's equality for scalars).
fn sameKey(a: SVal, b: SVal) bool {
    if (intOf(a)) |x| {
        if (intOf(b)) |y| return x == y;
        return b == .float and intEqualsFloat(x, b.float);
    }
    return switch (a) {
        .str => |s| b == .str and std.mem.eql(u8, s, b.str),
        .float => |x| (b == .float and b.float == x) or (if (intOf(b)) |y| intEqualsFloat(y, x) else false),
        .none => b == .none,
        .node => |n| b == .node and b.node == n,
        // (big ints)
        .py => |o| b == .py and (o == b.py or py.c.PyObject_RichCompareBool(o, b.py, py.c.Py_EQ) == 1),
        else => false,
    };
}

/// An int (or a bool) known when compiling, either kind.
fn intOf(v: SVal) ?i64 {
    return switch (v) {
        .int, .pint => |n| n,
        .bool => |b| @intFromBool(b),
        else => null,
    };
}

/// An int and a float equal exactly (as Python compares them).
fn intEqualsFloat(i: i64, f: f64) bool {
    if (f != @trunc(f) or !(@abs(f) < 9.3e18)) return false;
    if (f >= 9223372036854775807.0 or f < -9223372036854775808.0) return false;
    return @as(i64, @intFromFloat(f)) == i;
}

pub const SVal = union(enum) {
    none,
    bool: bool,
    /// An int of the program (zrun.I64 in the reference mode: what rt.eval,
    /// rt.load, fields... give): its arithmetic is checked, an overflow an
    /// error
    int: i64,
    /// A plain int a semantic's code makes (a literal, len(), a count...):
    /// as Python's, its arithmetic overflowing 64 bits makes a big int (a
    /// .py int); rt makes it an `int` where the reference mode makes an I64
    pint: i64,
    float: f64,
    str: []const u8,
    node: u32,
    list: *SList,
    tuple: []const SVal,
    dict: *SDict,
    rt,
    rt_method: RtMethod,
    /// A Python object known at compile time (a module-level name)
    py: *PyObject,
    /// raise rt.Return(v) / rt.Break() / rt.Continue(): what was raised
    control: struct { kind: RtMethod, value: ?*const SVal },
    /// `obj.name` for a method call
    method: struct { recv: *const SVal, name: []const u8 },
    dyn: Dyn,

    fn isStatic(self: SVal) bool {
        return self != .dyn;
    }
};

// ======================================================================
// Python objects known when compiling
// ======================================================================

/// Python's types compiled code tells objects apart by (the types
/// module's), fetched once
const PyTypes = struct {
    function: *PyObject,
    method: *PyObject,
    stable: *PyObject,
};

var py_types: ?PyTypes = null;

fn pyTypes() error{Python}!PyTypes {
    if (py_types) |t| return t;
    const src =
        \\import types as _t
        \\function = _t.FunctionType
        \\method = _t.MethodType
        \\stable = (type, _t.ModuleType, _t.FunctionType, _t.BuiltinFunctionType, _t.MethodType,
        \\    _t.MethodDescriptorType, _t.WrapperDescriptorType, _t.ClassMethodDescriptorType,
        \\    _t.MethodWrapperType, staticmethod, classmethod, property, frozenset, bytes, range,
        \\    complex, type(None), type(Ellipsis), type(NotImplemented))
    ;
    const ns = runPython(src) orelse return error.Python;
    defer py.Py_DecRef(ns);
    var t: PyTypes = undefined;
    inline for (@typeInfo(PyTypes).@"struct".fields) |fd| {
        const o = py.c.PyDict_GetItemString(ns, fd.name) orelse return error.Python;
        // (kept for the process: types live as long)
        py.Py_IncRef(o);
        @field(t, fd.name) = o;
    }
    py_types = t;
    return t;
}

/// types.FunctionType (null if it couldn't be had).
pub fn pyFunctionType() ?*PyObject {
    const t = pyTypes() catch {
        py.c.PyErr_Clear();
        return null;
    };
    return t.function;
}

/// types.MethodType (null if it couldn't be had).
pub fn pyMethodType() ?*PyObject {
    const t = pyTypes() catch {
        py.c.PyErr_Clear();
        return null;
    };
    return t.method;
}

/// Python source run in a fresh namespace: the namespace (a new reference),
/// or null with the exception.
fn runPython(src: [:0]const u8) ?*PyObject {
    const ns = py.c.PyDict_New() orelse return null;
    const builtins = py.c.PyEval_GetBuiltins() orelse return null;
    if (py.c.PyDict_SetItemString(ns, "__builtins__", builtins) != 0) {
        py.Py_DecRef(ns);
        return null;
    }
    const code = py.c.Py_CompileString(src, "<zrun>", py.c.Py_file_input) orelse {
        py.Py_DecRef(ns);
        return null;
    };
    defer py.Py_DecRef(code);
    const r = py.c.PyEval_EvalCode(code, ns, ns) orelse {
        py.Py_DecRef(ns);
        return null;
    };
    py.Py_DecRef(r);
    return ns;
}

/// A Python object whose attributes and truth are decided when compiling:
/// a module, a class, a function, a builtin, an immutable object (what
/// semantics refer to by name). Anything else (an instance, a list, a
/// dict...) may change after: read at run time.
fn stablePy(o: *PyObject) error{Python}!bool {
    const r = py.c.PyObject_IsInstance(o, (try pyTypes()).stable);
    if (r < 0) return error.Python;
    return r == 1;
}

fn isInstanceOf(o: *PyObject, t: *PyObject) error{Python}!bool {
    const r = py.c.PyObject_IsInstance(o, t);
    if (r < 0) return error.Python;
    return r == 1;
}

/// The module-level names of a module some function of it assigns
/// (`global x`): read when the code runs, not when compiling. Learned
/// once per module (again if names were added since).
fn reboundGlobals(globals: *PyObject) error{Python}!*PyObject {
    return (try moduleScan(globals)).names;
}

/// The module-level tables (dicts, lists, tuples) of a module its code
/// only reads (subscripts, `in`, iteration, get/items/keys/values...): as
/// good as constants, known when compiling.
fn frozenGlobals(globals: *PyObject) error{Python}!*PyObject {
    return (try moduleScan(globals)).frozen;
}

fn moduleScan(globals: *PyObject) error{Python}!ScanEntry {
    const n = py.c.PyDict_Size(globals);
    if (rebound_cache.get(globals)) |e| if (e.len == n) return e;
    try scanners();
    const pair = py.c.PyObject_CallFunctionObjArgs(rebound_scanner.?, globals, @as(?*PyObject, null)) orelse return error.Python;
    defer py.Py_DecRef(pair);
    const entry = ScanEntry{ .len = n, .names = py.c.PyTuple_GetItem(pair, 0).?, .frozen = py.c.PyTuple_GetItem(pair, 1).? };
    py.Py_IncRef(entry.names);
    py.Py_IncRef(entry.frozen);
    if (rebound_cache.fetchRemove(globals)) |old| {
        py.Py_DecRef(old.value.names);
        py.Py_DecRef(old.value.frozen);
    } else py.Py_IncRef(globals);
    rebound_cache.put(std.heap.c_allocator, globals, entry) catch {
        py.Py_DecRef(entry.names);
        py.Py_DecRef(entry.frozen);
        py.Py_DecRef(globals);
        _ = py.c.PyErr_NoMemory();
        return error.Python;
    };
    return entry;
}

/// Whether a Python function is pure: what it returns depends only on its
/// arguments, and it changes nothing outside it (it reads its parameters
/// and locals, builtins like len() and int(), exceptions, constants and
/// tables only read of its module, `math`, other pure functions; it
/// changes only what it made). Given constants, its result is known when
/// compiling (Gen.foldCall). Worked out once per function.
fn pureFunction(o: *PyObject) error{Python}!bool {
    try scanners();
    const r = py.c.PyObject_CallFunctionObjArgs(pure_checker.?, o, @as(?*PyObject, null)) orelse return error.Python;
    defer py.Py_DecRef(r);
    return r == py.Py_True();
}

/// Whether a Python function only reads its positional parameter `i`
/// (indexes it, iterates it, `in` it, len() of it...: never changes, keeps,
/// returns or gives it away). Worked out once per function.
fn readonlyParam(o: *PyObject, i: usize) error{Python}!bool {
    try scanners();
    const r = py.c.PyObject_CallFunctionObjArgs(readonly_checker.?, o, @as(?*PyObject, null)) orelse return error.Python;
    defer py.Py_DecRef(r);
    if (i >= py.c.PyTuple_Size(r)) return false;
    return py.c.PyTuple_GetItem(r, @intCast(i)).? == py.Py_True();
}

/// The Python code reading modules (moduleScan, pureFunction), made once.
fn scanners() error{Python}!void {
    if (rebound_scanner == null) {
        const src =
            \\import ast, dis, inspect, sys, types
            \\SAFE_METHODS = {"get", "items", "keys", "values", "count", "index", "copy"}
            \\SAFE_CALLS = {"len", "sorted", "list", "tuple", "set", "frozenset", "min", "max", "sum", "any",
            \\              "all", "enumerate", "zip", "iter", "reversed", "dict", "str", "repr", "bool"}
            \\def frozen(g):
            \\    mod = sys.modules.get(g.get("__name__"))
            \\    if mod is None or getattr(mod, "__dict__", None) is not g:
            \\        return frozenset()
            \\    try:
            \\        tree = ast.parse(inspect.getsource(mod))
            \\    except Exception:
            \\        return frozenset()
            \\    cands = {k for k, v in g.items() if type(v) in (dict, list, tuple, frozenset)}
            \\    parents = {}
            \\    for node in ast.walk(tree):
            \\        for ch in ast.iter_child_nodes(node):
            \\            parents[ch] = node
            \\    bad = set()
            \\    for node in ast.walk(tree):
            \\        if not isinstance(node, ast.Name) or node.id not in cands:
            \\            continue
            \\        p = parents.get(node)
            \\        if isinstance(node.ctx, ast.Store):
            \\            # (its definition, at the module's top level)
            \\            if not (isinstance(p, (ast.Assign, ast.AnnAssign)) and parents.get(p) is tree):
            \\                bad.add(node.id)
            \\            continue
            \\        if isinstance(node.ctx, ast.Del):
            \\            bad.add(node.id)
            \\            continue
            \\        ok = (isinstance(p, ast.Subscript) and p.value is node and isinstance(p.ctx, ast.Load)) \
            \\            or (isinstance(p, ast.Compare) and node in p.comparators and all(isinstance(o, (ast.In, ast.NotIn)) for o in p.ops)) \
            \\            or (isinstance(p, (ast.For, ast.comprehension)) and p.iter is node) \
            \\            or (isinstance(p, ast.Attribute) and p.value is node and p.attr in SAFE_METHODS
            \\                and isinstance(parents.get(p), ast.Call) and parents[p].func is p) \
            \\            or (isinstance(p, ast.Call) and node in p.args and isinstance(p.func, ast.Name) and p.func.id in SAFE_CALLS)
            \\        if not ok:
            \\            bad.add(node.id)
            \\    return frozenset(n for n in cands if n not in bad)
            \\def scan(g):
            \\    out = set()
            \\    seen = set()
            \\    def code(c):
            \\        if id(c) in seen:
            \\            return
            \\        seen.add(id(c))
            \\        for ins in dis.get_instructions(c):
            \\            if ins.opname in ("STORE_GLOBAL", "DELETE_GLOBAL"):
            \\                out.add(ins.argval)
            \\        for k in c.co_consts:
            \\            if isinstance(k, types.CodeType):
            \\                code(k)
            \\    def visit(v, depth):
            \\        if isinstance(v, types.FunctionType):
            \\            if v.__globals__ is g:
            \\                code(v.__code__)
            \\        elif isinstance(v, (staticmethod, classmethod)):
            \\            visit(v.__func__, depth)
            \\        elif isinstance(v, property):
            \\            for f in (v.fget, v.fset, v.fdel):
            \\                if f is not None:
            \\                    visit(f, depth)
            \\        elif isinstance(v, type) and depth < 4 and v.__module__ == g.get("__name__"):
            \\            for x in list(vars(v).values()):
            \\                visit(x, depth + 1)
            \\    for v in list(g.values()):
            \\        visit(v, 0)
            \\    return (frozenset(out), frozen(g) - out)
            \\import builtins, textwrap
            \\SAFE_BUILTINS = {"int", "float", "str", "len", "chr", "ord", "abs", "min", "max", "bool", "tuple", "list",
            \\                 "dict", "set", "frozenset", "isinstance", "range", "enumerate", "zip", "sorted", "reversed",
            \\                 "sum", "any", "all", "repr", "hex", "oct", "bin", "divmod", "round", "pow", "type", "hash",
            \\                 "format", "iter", "next", "map", "filter", "None", "True", "False"}
            \\SAFE_MODULES = {"math", "string"}
            \\IMMUTABLE = (int, float, complex, str, bytes, bool, type(None), frozenset)
            \\pure_cache = {}
            \\frozen_cache = {}
            \\def constant(v):
            \\    if type(v) is tuple:
            \\        return all(constant(x) for x in v)
            \\    return type(v) in IMMUTABLE
            \\def frozen_of(g):
            \\    k = id(g)
            \\    if k not in frozen_cache:
            \\        frozen_cache[k] = (g, scan(g)[1])
            \\    return frozen_cache[k][1]
            \\def pure(fn):
            \\    if fn in pure_cache:
            \\        return pure_cache[fn]
            \\    # (one calling itself, or a cycle: each checked on its own)
            \\    pure_cache[fn] = True
            \\    try:
            \\        ok = check(fn)
            \\    except Exception:
            \\        ok = False
            \\    pure_cache[fn] = ok
            \\    return ok
            \\def check(fn):
            \\    if not isinstance(fn, types.FunctionType) or fn.__closure__ or fn.__kwdefaults__:
            \\        return False
            \\    if fn.__defaults__ and not constant(fn.__defaults__):
            \\        return False
            \\    fdef = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
            \\    if not isinstance(fdef, ast.FunctionDef):
            \\        return False
            \\    a = fdef.args
            \\    params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            \\    params |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
            \\    local = set(params)
            \\    for n in ast.walk(fdef):
            \\        if n is not fdef and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            \\            return False
            \\        if isinstance(n, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await)):
            \\            return False
            \\        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            \\            local.add(n.id)
            \\        if isinstance(n, ast.ExceptHandler) and n.name:
            \\            local.add(n.name)
            \\    g = fn.__globals__
            \\    fz = frozen_of(g)
            \\    for n in ast.walk(fdef):
            \\        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in local:
            \\            if n.id in g:
            \\                v = g[n.id]
            \\                if constant(v) or n.id in fz:
            \\                    continue
            \\                if isinstance(v, types.ModuleType) and v.__name__ in SAFE_MODULES:
            \\                    continue
            \\                if isinstance(v, type) and issubclass(v, BaseException):
            \\                    continue
            \\                if isinstance(v, types.FunctionType) and pure(v):
            \\                    continue
            \\                return False
            \\            b = getattr(builtins, n.id, None)
            \\            if n.id in SAFE_BUILTINS or (isinstance(b, type) and issubclass(b, BaseException)):
            \\                continue
            \\            return False
            \\        # (changing only what it made)
            \\        if isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, (ast.Store, ast.Del)):
            \\            base = n.value
            \\            while isinstance(base, (ast.Attribute, ast.Subscript)):
            \\                base = base.value
            \\            if not (isinstance(base, ast.Name) and base.id in local and base.id not in params):
            \\                return False
            \\    return True
            \\readonly_cache = {}
            \\def readonly(fn):
            \\    # Which positional parameters the function only reads (as frozen()
            \\    # tells a table only read): a list given there can't be changed
            \\    # or kept by it
            \\    if fn in readonly_cache:
            \\        return readonly_cache[fn]
            \\    out = ()
            \\    try:
            \\        fdef = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
            \\        names = [x.arg for x in fdef.args.posonlyargs + fdef.args.args]
            \\        parents = {}
            \\        for node in ast.walk(fdef):
            \\            for ch in ast.iter_child_nodes(node):
            \\                parents[ch] = node
            \\        bad = set()
            \\        for node in ast.walk(fdef):
            \\            if not isinstance(node, ast.Name) or node.id not in names:
            \\                continue
            \\            p = parents.get(node)
            \\            if not isinstance(node.ctx, ast.Load):
            \\                bad.add(node.id)
            \\                continue
            \\            ok = (isinstance(p, ast.Subscript) and p.value is node and isinstance(p.ctx, ast.Load)) \
            \\                or (isinstance(p, ast.Compare) and node in p.comparators and all(isinstance(o, (ast.In, ast.NotIn)) for o in p.ops)) \
            \\                or (isinstance(p, (ast.For, ast.comprehension)) and p.iter is node) \
            \\                or (isinstance(p, ast.Attribute) and p.value is node and p.attr in SAFE_METHODS - {"copy"}
            \\                    and isinstance(parents.get(p), ast.Call) and parents[p].func is p) \
            \\                or (isinstance(p, ast.Call) and node in p.args and isinstance(p.func, ast.Name) and p.func.id in ("len", "isinstance", "bool"))
            \\            if not ok:
            \\                bad.add(node.id)
            \\        out = tuple(n not in bad for n in names)
            \\    except Exception:
            \\        pass
            \\    readonly_cache[fn] = out
            \\    return out
        ;
        const ns = runPython(src) orelse return error.Python;
        defer py.Py_DecRef(ns);
        const f = py.c.PyDict_GetItemString(ns, "scan") orelse return error.Python;
        const p = py.c.PyDict_GetItemString(ns, "pure") orelse return error.Python;
        const r = py.c.PyDict_GetItemString(ns, "readonly") orelse return error.Python;
        py.Py_IncRef(f);
        py.Py_IncRef(p);
        py.Py_IncRef(r);
        rebound_scanner = f;
        pure_checker = p;
        readonly_checker = r;
    }
}

const ScanEntry = struct { len: isize, names: *PyObject, frozen: *PyObject };

/// The record type of a class whose objects compiled code makes natively
/// (records): a dataclass, or a plain class with __slots__ (all the way
/// up, single inheritance) and no special methods but __init__, __repr__
/// and __str__ (what native records do is then what Python does). Made
/// once per class, for the process (records outlive programs); null for
/// another class (its objects are Python's).
pub fn recordOf(cls: *PyObject) Error!?*value.RecordType {
    if (record_types.get(cls)) |t| return t;
    if (not_records.contains(cls)) return null;
    if (record_describer == null) {
        const src =
            \\import dataclasses
            \\OK = {"__module__", "__qualname__", "__doc__", "__slots__", "__init__", "__repr__", "__str__",
            \\      "__annotations__", "__match_args__", "__firstlineno__", "__static_attributes__"}
            \\def describe(cls):
            \\    if type(cls) is not type or cls.__mro__[-1] is not object or len(cls.__bases__) != 1:
            \\        return None
            \\    base = cls.__bases__[0]
            \\    if dataclasses.is_dataclass(cls):
            \\        p = cls.__dataclass_params__
            \\        if p.order or "__post_init__" in dir(cls):
            \\            return None
            \\        names = tuple(f.name for f in dataclasses.fields(cls))
            \\        return (names, False, bool(p.eq), bool(p.frozen), base if dataclasses.is_dataclass(base) else None)
            \\    out = []
            \\    for k in reversed(cls.__mro__[:-1]):
            \\        d = k.__dict__
            \\        if "__slots__" not in d:
            \\            return None
            \\        s = d["__slots__"]
            \\        for n in ((s,) if isinstance(s, str) else tuple(s)):
            \\            if n in ("__dict__", "__weakref__"):
            \\                return None
            \\            if n.startswith("__") and not n.endswith("__"):
            \\                n = "_" + k.__name__.lstrip("_") + n
            \\            out.append(n)
            \\        for n in d:
            \\            if n.startswith("__") and n.endswith("__") and n not in OK:
            \\                return None
            \\    return (tuple(out), True, False, False, None if base is object else base)
        ;
        const ns = runPython(src) orelse return error.Python;
        defer py.Py_DecRef(ns);
        const f = py.c.PyDict_GetItemString(ns, "describe") orelse return error.Python;
        py.Py_IncRef(f);
        record_describer = f;
    }
    const gpa = std.heap.c_allocator;
    const d = py.c.PyObject_CallFunctionObjArgs(record_describer.?, cls, @as(?*PyObject, null)) orelse return error.Python;
    defer py.Py_DecRef(d);
    if (d == py.Py_None()) {
        try not_records.put(gpa, cls, {});
        py.Py_IncRef(cls);
        return null;
    }
    const field_names = py.c.PyTuple_GetItem(d, 0).?;
    const base_cls = py.c.PyTuple_GetItem(d, 4).?;
    const base = if (base_cls == py.Py_None()) null else try recordOf(base_cls) orelse {
        try not_records.put(gpa, cls, {});
        py.Py_IncRef(cls);
        return null;
    };
    const n: usize = @intCast(py.c.PyTuple_Size(field_names));
    const names = try gpa.alloc([]const u8, n);
    for (names, 0..) |*slot, i| slot.* = try gpa.dupe(u8, ph.utf8(py.c.PyTuple_GetItem(field_names, @intCast(i)).?, "field") orelse return error.Python);
    const qual = ph.attr(cls, "__name__") orelse return error.Python;
    defer py.Py_DecRef(qual);
    const t = try gpa.create(value.RecordType);
    t.* = .{
        .name = try gpa.dupe(u8, ph.utf8(qual, "name") orelse return error.Python),
        .fields = names,
        .py_class = cls,
        .slots = py.c.PyTuple_GetItem(d, 1).? == py.Py_True(),
        .value_eq = py.c.PyTuple_GetItem(d, 2).? == py.Py_True(),
        .frozen = py.c.PyTuple_GetItem(d, 3).? == py.Py_True(),
        .base = base,
    };
    // (the class is kept: its records may live as long as the process)
    py.Py_IncRef(cls);
    try record_types.put(gpa, cls, t);
    return t;
}

var record_describer: ?*PyObject = null;
var record_types: std.AutoHashMapUnmanaged(*PyObject, *value.RecordType) = .empty;
var not_records: std.AutoHashMapUnmanaged(*PyObject, void) = .empty;

var rebound_scanner: ?*PyObject = null;
var pure_checker: ?*PyObject = null;

/// The Python type Language.types() says node `idx`'s values are (from its
/// type's text, zrules'), or null: none said. (Borrowed: types live long.)
pub fn declaredClass(mapping: *PyObject, analysis: *PyObject, idx: u32) error{Python}!?*PyObject {
    const text = py.c.PyObject_CallMethod(analysis, "type_of", "I", @as(c_uint, idx)) orelse {
        py.c.PyErr_Clear();
        return null;
    };
    defer py.Py_DecRef(text);
    if (text == py.Py_None()) return null;
    if (py.PyDict_Check(mapping)) {
        if (py.c.PyDict_GetItem(mapping, text)) |o| return o;
        // (a generic's name: 'list' for 'list[int]', 'fn' for a function's
        // type)
        const s = ph.utf8(text, "type") orelse return error.Python;
        const end = std.mem.indexOfAny(u8, s, "[(") orelse return null;
        const head = ph.newString(s[0..end]) orelse return error.Python;
        defer py.Py_DecRef(head);
        return py.c.PyDict_GetItem(mapping, head);
    }
    const r = py.c.PyObject_CallFunctionObjArgs(mapping, text, @as(?*PyObject, null)) orelse return error.Python;
    py.Py_DecRef(r);
    return if (r == py.Py_None()) null else r;
}
/// Modules' tables only read, native (Gen.frozenConst), by the table
var frozen_natives: std.AutoHashMapUnmanaged(*PyObject, Value) = .empty;
var readonly_checker: ?*PyObject = null;
/// (the module dicts are kept: modules live as long)
var rebound_cache: std.AutoHashMapUnmanaged(*PyObject, ScanEntry) = .empty;

// ======================================================================
// The compiler
// ======================================================================

/// In a helper's code out of line, the node its errors are reported at:
/// its caller's, given at run time
const AT_PARAM: u32 = NONE - 2;
/// In a helper's code out of line, the scope of the frames it runs in: its
/// caller's, given at run time
const OWNER_PARAM: u32 = NONE - 3;

/// A helper compiled out of line, once for the calls like it: the
/// arguments it's made for (rt, Python objects, bools: its code depends on
/// them most), the others (`.dyn` here: nodes, strs, values) given at run
/// time
const HelperSpec = struct {
    func: *const front.Function,
    args: []const SVal,
    name: [:0]const u8,
    /// The semantic it runs for (run as Python if the helper can't be
    /// compiled)
    semantic: ?*PyObject,
    /// A closure's code: the function called is the last argument, its
    /// variables (cells) read from it when the code runs
    closure: bool = false,
    /// A specialization (Site): the arguments the calls give in the array,
    /// as the generic code's (it knows some of them: those it skips)
    layout: ?[]const bool = null,

    fn matches(self: *const HelperSpec, func: *const front.Function, args: []const SVal) bool {
        if (self.func != func or self.args.len != args.len or self.layout != null) return false;
        for (self.args, args) |x, y| if (!sameSpec(x, y)) return false;
        return true;
    }

    /// The same specialization: the same function, known arguments, array.
    fn sameSpecial(self: *const HelperSpec, func: *const front.Function, args: []const SVal, layout: []const bool) bool {
        if (self.func != func or self.args.len != args.len) return false;
        const l = self.layout orelse return false;
        if (!std.mem.eql(bool, l, layout)) return false;
        for (self.args, args) |x, y| if (!sameSpec(x, y)) return false;
        return true;
    }

    /// The same argument to compile for (exactly: 1 and 1.0 aren't).
    fn sameSpec(x: SVal, y: SVal) bool {
        if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
        return switch (x) {
            .dyn, .none, .rt => true,
            .bool => |b| b == y.bool,
            .node => |n| n == y.node,
            .str => |s| std.mem.eql(u8, s, y.str),
            .py => |o| o == y.py,
            .int => |n| n == y.int,
            .pint => |n| n == y.pint,
            .float => |v| @as(u64, @bitCast(v)) == @as(u64, @bitCast(y.float)),
            .tuple => |t| t.len == y.tuple.len and for (t, y.tuple) |a, b| {
                if (!sameSpec(a, b)) break false;
            } else true,
            .list => |l| l.items.items.len == y.list.items.items.len and for (l.items.items, y.list.items.items) |a, b| {
                if (!sameSpec(a, b)) break false;
            } else true,
            else => false,
        };
    }
};

/// A call site of a helper out of line knowing more of its arguments than
/// the helper's generic code does (a node, a str...): the calls counted,
/// and once it's hot (`hot_calls`), the helper compiled for them
/// (Compiled.specialize), called from then on. `hot` is what the code
/// reads: the specialized code's address (0: none yet), the calls so far
/// (negative: never to specialize).
pub const Site = struct {
    hot: *Hot,
    func: *const front.Function,
    /// The arguments as the site knows them (.dyn: given at run time)
    args: []const SVal,
    /// Which are in the array the calls give (the generic code's)
    layout: []const bool,
    semantic: ?*PyObject,

    pub const Hot = extern struct { code: u64 = 0, count: i64 = 0 };
};

/// The most specializations a program makes (compiling costs)
pub const max_specialized = 256;
/// The biggest specialization compiled (blocks of code, the helpers it
/// needs with it)
const max_specialized_blocks = 1500;

/// Where a language function keeps its variables
const Layout = struct {
    /// The symbols living in it, by slot
    syms: std.ArrayListUnmanaged(u32) = .empty,
    /// In a heap frame (functions are made in it), or on the stack
    heap: bool = false,
};

pub const Compiler = struct {
    a: Allocator,
    data: *program_mod.Data,
    lang: LangView,
    m: ir.Module,
    failure: *Failure,
    /// Python objects the code refers to (owned references: the compiled
    /// program keeps them)
    objects: std.ArrayListUnmanaged(*PyObject) = .empty,
    /// Per function node (NONE: the top level): its variables' layout
    layouts: std.AutoHashMapUnmanaged(u32, *Layout) = .empty,
    /// Slot of each symbol in its layout
    slot_of: std.AutoHashMapUnmanaged(u32, u32) = .empty,
    /// Language functions compiled or to compile
    compiled_fns: std.AutoHashMapUnmanaged(u32, void) = .empty,
    queue: std.ArrayListUnmanaged(u32) = .empty,
    /// Checked i64 arithmetic (LLVM's intrinsics)
    sadd: ir.Fn = undefined,
    ssub: ir.Fn = undefined,
    smul: ir.Fn = undefined,
    /// The innermost semantic that couldn't be compiled (the driver runs it
    /// as Python and compiles again)
    failed_semantic: ?*PyObject = null,
    /// The code calls semantics run as Python (zr_py_semantic)
    uses_python: bool = false,
    /// Thunks made so far (their names' numbers)
    thunks: u32 = 0,
    /// The language functions this module compiles (forgotten if it fails:
    /// they aren't in the JIT then)
    new_fns: std.ArrayListUnmanaged(u32) = .empty,
    /// Every frame on the heap (code run through the bridge sees the
    /// variables through the frames): set by the driver when `need_frames`
    /// was found
    force_heap: bool = false,
    need_frames: bool = false,
    /// Every layout is on the heap (those made later too)
    all_heap: bool = false,
    /// A literal was marked to be built at run time: compile again
    need_retry: bool = false,
    /// Helpers compiled out of line (one already being run inline, called
    /// again: a recursive one), each a specialization; and those still to
    /// generate. `helpers_kept`: how many the JIT has (a failed module's
    /// are forgotten).
    helper_fns: std.ArrayListUnmanaged(*HelperSpec) = .empty,
    helper_queue: std.ArrayListUnmanaged(*HelperSpec) = .empty,
    helpers_kept: usize = 0,
    /// Record fields by module and name (Gen.fieldCandidates)
    field_cands: std.StringHashMapUnmanaged([]const Gen.FieldCandidate) = .empty,
    /// Call sites of helpers out of line that may get code of their own
    /// (Site), by number; specializations made so far
    sites: std.ArrayListUnmanaged(*Site) = .empty,
    specialized: usize = 0,
    /// Blocks of the helpers generated so far (a specialization's size)
    helper_blocks: usize = 0,
    /// Names given to addresses so far (ir.Module.ptrConst)
    ksyms: u32 = 0,
    /// The module-level tables and records the semantics use: "native", or
    /// why Python keeps them (adopt.zig; Program.report())
    module_state: std.StringArrayHashMapUnmanaged([]const u8) = .empty,
    /// Nodes' kinds of values, worked out (kindOfNode)
    kinds: std.AutoHashMapUnmanaged(u32, ?Kind) = .empty,
    /// Symbols always set where they're read (Gen.alwaysSet), worked out
    always_set: std.AutoHashMapUnmanaged(u32, bool) = .empty,
    /// Functions' typed entries (Gen.typedParams), worked out
    typed_params: std.AutoHashMapUnmanaged(u32, ?[]const Shape) = .empty,
    /// By node: its kind's str (kindTable); by label, its child (fieldTable)
    kind_table: ?[]u64 = null,
    owner_table: ?[]u64 = null,
    field_tables: std.AutoHashMapUnmanaged(u8, []u64) = .empty,

    /// In a field table: a node whose label isn't one child or none (a
    /// list of them, a value an action makes): zr_getattr's
    pub const field_other: i64 = @as(i64, NONE) - 1;

    /// The kind of value the nodes of a type evaluate to (Language.types()):
    /// its shape (and a record's type)
    pub const Kind = struct { shape: Shape, rtype: ?*value.RecordType = null };

    /// The kind of value node `idx` evaluates to, from its type (zrules'
    /// types(), what Language.types() says its values are), or null: not
    /// known. Worked out once per node.
    pub fn kindOfNode(self: *Compiler, idx: u32) Error!?Kind {
        const mapping = self.lang.types orelse return null;
        const analysis = self.lang.analysis orelse return null;
        if (self.kinds.get(idx)) |k| return k;
        const k = kindFrom(mapping, analysis, idx) catch |e| return e;
        try self.kinds.put(self.a, idx, k);
        return k;
    }

    fn kindFrom(mapping: *PyObject, analysis: *PyObject, idx: u32) error{Python}!?Kind {
        const cls = try declaredClass(mapping, analysis, idx) orelse return null;
        return kindOfClass(cls);
    }

    /// The kind of a Python type's values, for the types whose values are
    /// native in compiled code wherever they come from (scalars, the
    /// language's functions); null for any other (a list, a record a host
    /// function made, stays a Python object: its kind not one tag).
    pub fn kindOfClass(cls: *PyObject) ?Kind {
        const is = struct {
            fn t(x: *PyObject, comptime name: [:0]const u8) bool {
                return x == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject(name))));
            }
        };
        if (is.t(cls, "PyLong_Type")) return .{ .shape = .int };
        if (is.t(cls, "PyFloat_Type")) return .{ .shape = .float };
        if (is.t(cls, "PyBool_Type")) return .{ .shape = .bool };
        if (is.t(cls, "PyUnicode_Type")) return .{ .shape = .str };
        if (cls == @as(*PyObject, @ptrCast(@alignCast(ph.typeOf(py.Py_None()))))) return .{ .shape = .none };
        if (cls == objects_mod.FunctionType) return .{ .shape = .function };
        return null;
    }

    /// What became of module state named `name` (the last word on it).
    pub fn noteState(self: *Compiler, name: []const u8, what: []const u8) !void {
        const e = try self.module_state.getOrPut(self.a, name);
        if (!e.found_existing) e.key_ptr.* = try self.a.dupe(u8, name);
        e.value_ptr.* = what;
    }

    /// Each node's owner (the scope whose frame its code runs in: ownerOf),
    /// by node: what code out of line walks frames by (zr_frame_of).
    pub fn ownerTable(self: *Compiler) Error![]u64 {
        if (self.owner_table) |t| return t;
        const t = try self.a.alloc(u64, self.data.nodes.len);
        for (t, 0..) |*slot, i| slot.* = self.ownerOf(@intCast(i));
        self.owner_table = t;
        return t;
    }

    /// Each node's kind as a str (immortal: value.literal), by node: what
    /// node.kind of a node only known at run time reads.
    pub fn kindTable(self: *Compiler) Error![]u64 {
        if (self.kind_table) |t| return t;
        const d = self.data;
        const t = try self.a.alloc(u64, d.nodes.len);
        for (t, 0..) |*slot, i| {
            const s = value.literal(d.grammar.kind_names[d.rule(@intCast(i))]) orelse return error.OutOfMemory;
            slot.* = @intFromPtr(s);
        }
        self.kind_table = t;
        return t;
    }

    /// Each node's child of a label, by node (NONE: none, field_other:
    /// several or a value an action makes): what node.<label> of a node
    /// only known at run time reads (as bridge.nodeAttr gives it).
    pub fn fieldTable(self: *Compiler, field: u8) Error![]u64 {
        if (self.field_tables.get(field)) |t| return t;
        const d = self.data;
        const t = try self.a.alloc(u64, d.nodes.len);
        for (t, 0..) |*slot, i| {
            const idx: u32 = @intCast(i);
            const label = d.grammar.labelOf(d.rule(idx), field);
            var found: u64 = NONE;
            var count: usize = 0;
            var other = label != null and label.?.many;
            var ch = idx + 1;
            const stop = d.end(idx);
            while (ch < stop and !other) : (ch = d.end(ch)) {
                if (d.nodes[ch].fieldId() != field) continue;
                const crid = d.rule(ch);
                const action: grammar_mod.Action = if (crid < d.grammar.actions.len) d.grammar.actions[crid] else .none;
                switch (action) {
                    .none, .class => {
                        count += 1;
                        found = ch;
                    },
                    .drop => {},
                    else => other = true,
                }
            }
            slot.* = if (other or count > 1) @bitCast(field_other) else found;
        }
        try self.field_tables.put(self.a, field, t);
        return t;
    }

    /// Whether every function's (and block's) variables are in frames.
    pub fn allHeap(self: *const Compiler) bool {
        var it = self.layouts.valueIterator();
        while (it.next()) |l| if (!l.*.heap) return false;
        return true;
    }

    pub fn init(a: Allocator, data: *program_mod.Data, lang: LangView, prefix: []const u8, failure: *Failure) Compiler {
        return .{ .a = a, .data = data, .lang = lang, .m = ir.Module.init(a, prefix), .failure = failure };
    }

    /// Release what isn't in the arena (nothing now: the helpers read are
    /// the language's).
    pub fn deinit(self: *Compiler, gpa: Allocator) void {
        _ = self;
        _ = gpa;
    }

    /// A new module for more code of the program (the compiler's state,
    /// its layouts and functions, stays: what it compiled before is in
    /// the JIT, called by name).
    pub fn newModule(self: *Compiler) !void {
        self.m.deinit();
        self.m = ir.Module.init(self.a, self.m.prefix);
        self.new_fns.clearRetainingCapacity();
        self.helpers_kept = self.helper_fns.items.len;
        try self.declareRuntime();
    }

    /// The module being made failed: the functions it was to compile are
    /// still to compile.
    pub fn forgetModule(self: *Compiler) void {
        for (self.new_fns.items) |f| _ = self.compiled_fns.remove(f);
        self.new_fns.clearRetainingCapacity();
        self.queue.clearRetainingCapacity();
        self.helper_fns.shrinkRetainingCapacity(self.helpers_kept);
        self.helper_queue.clearRetainingCapacity();
    }

    /// Generate what the code generated so far calls: language functions,
    /// helpers out of line.
    pub fn drainQueues(self: *Compiler) Error!void {
        while (true) {
            if (self.queue.pop()) |f| {
                try self.genFunction(f);
            } else if (self.helper_queue.pop()) |h| {
                try self.genHelper(h);
            } else return;
        }
    }

    /// A helper's code out of line: `i32 <name>(ctx, frame, args, at,
    /// owner, receiver, varargs, out)`, a status as a thunk's (its result
    /// in out), the run-time arguments in `args`, run in its caller's
    /// frames (`frame`, of the scope `owner`: rt.eval there), its errors
    /// reported at node `at`, the caller's receiver and varargs given.
    fn genHelper(self: *Compiler, h: *HelperSpec) Error!void {
        const fun = try self.helperFn(h.name);
        var g = Gen{ .c = self, .f = ir.Function.init(&self.m, fun.v), .fnode = NONE, .layout = try self.layoutOf(NONE), .thunk = true, .detached = true, .helper_semantic = h.semantic, .unit = std.hash.Wyhash.hash(1, h.name), .specialized = h.layout != null };
        // (one that can't be compiled: its semantic runs as Python)
        errdefer if (self.failed_semantic == null) {
            self.failed_semantic = h.semantic;
        };
        g.ctx = g.f.param(0);
        g.frame = g.f.param(1);
        g.at_param = g.f.param(3);
        g.owner_param = g.f.param(4);
        g.recv_slot = g.f.param(5);
        g.varargs_slot = g.f.param(6);
        g.out_param = g.f.param(7);
        g.out = try g.f.alloca(self.m.t.val);
        g.err_label = try g.f.label("error");
        g.ret_label = try g.f.label("return");
        // (its arguments: the known ones, the others from the array, a
        // reference each)
        const args = try self.a.alloc(SVal, h.args.len);
        var j: usize = 0;
        for (h.args, 0..) |x, i| {
            // (a specialization: the array is the generic code's, with
            // what it knows skipped)
            const in_array = if (h.layout) |l| l[i] else x == .dyn;
            const at = j;
            if (in_array) j += 1;
            if (x == .list) {
                // (a specialization's list of known items: a copy of its
                // own, the helper only reads it)
                const l = try self.a.create(SList);
                l.* = .{};
                try l.items.appendSlice(self.a, x.list.items.items);
                args[i] = .{ .list = l };
                continue;
            }
            if (x != .dyn) {
                args[i] = x;
                continue;
            }
            const d = try g.loadSlot(g.elem(g.f.param(2), at), .any);
            // (a closure's function, last: borrowed, its cells read)
            if (h.closure and i == h.args.len - 1) {
                g.closure_fn = d;
                g.closure_root = h.func;
                continue;
            }
            try g.increfDyn(d);
            args[i] = .{ .dyn = d };
        }
        const params = try g.withDefaults(h.func, if (h.closure) args[0 .. args.len - 1] else args);
        const v = try g.materialize(try g.runFunction(h.func, AT_PARAM, params), AT_PARAM);
        try g.storeSlot(g.out_param, v);
        try g.f.ret(self.m.k32(1));
        try g.f.block(g.err_label);
        try g.f.ret(self.m.k32(0));
        g.f.finish();
        self.helper_blocks += g.f.next_label;
    }

    /// A Big for a constant (immortal: freed with the program).
    pub fn bigConst(_: *Compiler, v: i128) !*value.Big {
        // (the process's: values made of it may outlive the program)
        return value.bigLiteral(v) orelse error.OutOfMemory;
    }

    /// A Python function as the front reads it (once: the language keeps
    /// it).
    pub fn readFunction(self: *Compiler, o: *PyObject) Error!*const front.Function {
        if (self.lang.read.get(o)) |f| return f;
        var failure = front.Failure{};
        const f = front.read(std.heap.c_allocator, o, &failure) catch |e| switch (e) {
            error.Unsupported => return self.unsupported("{s}", .{failure.text()}),
            else => |x| return x,
        };
        self.lang.read.put(std.heap.c_allocator, o, f) catch {
            f.destroy(std.heap.c_allocator);
            return error.OutOfMemory;
        };
        return f;
    }

    /// The code of a Python function compiled code calls at run time
    /// (a library function of the language...), for `nargs` arguments
    /// (those in `rt_mask`: rt values, the call's frames): a helper's code
    /// out of line, in a module of its own. Its name.
    pub fn compileCalled(self: *Compiler, o: *PyObject, nargs: usize, rt_mask: u64, closure: bool) Error![:0]const u8 {
        try self.newModule();
        const func = try self.readFunction(o);
        if (nargs < func.required or nargs > func.param_count) return self.unsupported("{s}() takes {d} to {d} arguments, called with {d}", .{ func.name, func.required, func.param_count, nargs });
        // (a closure: the function called given last, its variables read
        // from it)
        const key = try self.a.alloc(SVal, nargs + @intFromBool(closure));
        for (key, 0..) |*slot, i| slot.* = if (i < nargs and rt_mask & (@as(u64, 1) << @intCast(i)) != 0) .rt else .{ .dyn = undefined };
        _ = try self.objectIndex(o);
        const h = try self.a.create(HelperSpec);
        h.* = .{ .func = func, .args = key, .name = try std.fmt.allocPrintSentinel(self.a, "{s}_c{d}", .{ self.m.prefix, self.helper_fns.items.len }, 0), .semantic = null, .closure = closure };
        try self.helper_fns.append(self.a, h);
        try self.helper_queue.append(self.a, h);
        try self.drainQueues();
        return h.name;
    }

    /// The specialization a site's calls are, made before (another site
    /// like it): its name, or null.
    pub fn specializedBefore(self: *Compiler, site: usize) ?[:0]const u8 {
        const s = self.sites.items[site];
        for (self.helper_fns.items[0..self.helpers_kept]) |h| if (h.sameSpecial(s.func, s.args, s.layout)) return h.name;
        return null;
    }

    /// A site's helper compiled for what the site knows of its arguments
    /// (Site), in a module of its own: its name.
    pub fn compileSpecialized(self: *Compiler, site: usize) Error![:0]const u8 {
        const s = self.sites.items[site];
        try self.newModule();
        self.specialized += 1;
        const blocks0 = self.helper_blocks;
        const h = try self.a.create(HelperSpec);
        h.* = .{ .func = s.func, .args = s.args, .name = try std.fmt.allocPrintSentinel(self.a, "{s}_s{d}", .{ self.m.prefix, self.helper_fns.items.len }, 0), .semantic = s.semantic, .layout = s.layout };
        try self.helper_fns.append(self.a, h);
        try self.helper_queue.append(self.a, h);
        try self.drainQueues();
        // (code too big to be worth compiling: LLVM's time grows with it,
        // the generic code serves)
        const blocks = self.helper_blocks - blocks0;
        if (blocks > max_specialized_blocks) return self.unsupported("{d} blocks: too big to specialize", .{blocks});
        return h.name;
    }

    /// A helper's code out of line, declared in this module.
    fn helperFn(self: *Compiler, name: []const u8) !ir.Fn {
        const t = self.m.t;
        return self.m.function(name, t.i32, &.{ t.ptr, t.ptr, t.ptr, t.i32, t.i32, t.ptr, t.ptr, t.ptr }, true);
    }

    /// Whether any of the language's semantics run as Python.
    fn anyPython(self: *const Compiler) bool {
        for ([_][]const ?*PyObject{ self.lang.eval_of, self.lang.exec_of }) |table| {
            for (table) |fo| {
                const o = fo orelse continue;
                if (self.lang.python.contains(o) or !self.lang.read.contains(o)) return true;
            }
        }
        return false;
    }

    /// A thunk: a function running one node's eval or exec in the frames
    /// of a semantic run as Python (`owner`: the node whose frame it is),
    /// `i32 <name>(ctx, frame, out)` returning a status (0 error, 1 done,
    /// 2 Return, 3 Break, 4 Continue; a value in out). Its name.
    pub fn compileThunk(self: *Compiler, idx: u32, which: Which, owner: u32) Error![:0]const u8 {
        try self.newModule();
        self.thunks += 1;
        const name = try std.fmt.allocPrintSentinel(self.a, "{s}_t{d}", .{ self.m.prefix, self.thunks }, 0);
        const t = self.m.t;
        const fun = try self.m.function(name, t.i32, &.{ t.ptr, t.ptr, t.ptr }, true);
        // (the function around the code: the owner's, through the scopes)
        var fnode = owner;
        while (fnode != NONE and !self.isFunctionNode(fnode)) fnode = self.ownerOf(fnode);
        var g = Gen{ .c = self, .f = ir.Function.init(&self.m, fun.v), .fnode = fnode, .layout = try self.layoutOf(fnode), .thunk = true, .unit = std.hash.Wyhash.hash(2, std.mem.asBytes(&[3]u32{ idx, @intFromEnum(which), owner })) };
        g.ctx = g.f.param(0);
        g.out_param = g.f.param(2);
        try g.thunkPrologue(owner, g.f.param(1));
        if (which == .eval) {
            const v = try g.materialize(try g.evalNode(idx), idx);
            try g.storeSlot(g.out_param, v);
        } else try g.execNode(idx);
        try g.f.ret(self.m.k32(1));
        try g.f.block(g.err_label);
        try g.f.ret(self.m.k32(0));
        g.f.finish();
        try self.drainQueues();
        return name;
    }

    fn unsupported(self: *Compiler, comptime fmt: []const u8, args: anytype) Error {
        self.failure.message.clearRetainingCapacity();
        self.failure.message.print(self.a, fmt, args) catch {};
        return error.Unsupported;
    }

    /// Unsupported at a semantic's line.
    fn unsupportedAt(self: *Compiler, f: *const front.Function, pos: front.Pos, comptime fmt: []const u8, args: anytype) Error {
        self.failure.message.clearRetainingCapacity();
        self.failure.message.print(self.a, "{s}:{d}:{d}: in {s}(): ", .{ f.file, pos.line, pos.col, f.name }) catch {};
        self.failure.message.print(self.a, fmt, args) catch {};
        return error.Unsupported;
    }

    /// The whole program: the top level as `@<prefix>_main`, each language
    /// function reached as `@<prefix>_f<node>`.
    pub fn compileProgram(self: *Compiler) Error!void {
        try self.declareRuntime();
        try self.computeLayouts();
        try self.genFunction(NONE);
        try self.drainQueues();
    }

    /// The runtime's helpers (helpers.zig), by name and signature: the
    /// result then the parameters, one letter each (v void, b i1, i i32,
    /// l i64, p ptr).
    const runtime_decls = [_][2][]const u8{
        .{ "zr_incref", "vll" },
        .{ "zr_decref", "vll" },
        .{ "zr_free", "vll" },
        .{ "zr_fail", "bpip" },
        .{ "zr_unset", "bpip" },
        .{ "zr_overflow", "bpi" },
        .{ "zr_binary", "bpiillllp" },
        .{ "zr_compare", "bpiillllp" },
        .{ "zr_unary", "bpiillp" },
        .{ "zr_truthy", "bll" },
        .{ "zr_function", "bpppiplp" },
        .{ "zr_call", "bpillplpp" },
        .{ "zr_specialize", "vpl" },
        .{ "zr_frame_of", "ppiip" },
        .{ "zr_object", "vplp" },
        .{ "zr_frame_new", "ppl" },
        .{ "zr_frame_release", "vp" },
        .{ "zr_list", "bpiplp" },
        .{ "zr_tuple", "bpiplp" },
        .{ "zr_dict", "bpipplp" },
        .{ "zr_record", "bpippp" },
        .{ "zr_is_record", "bllp" },
        .{ "zr_getattr", "bpillpp" },
        .{ "zr_setattr", "bpillpll" },
        .{ "zr_getitem", "bpillllp" },
        .{ "zr_setitem", "bpillllll" },
        .{ "zr_items", "bpillp" },
        .{ "zr_list_len", "lll" },
        .{ "zr_list_at", "vlllp" },
        .{ "zr_append", "bpillll" },
        .{ "zr_extend_items", "bpillpl" },
        .{ "zr_call_method", "bpillpplp" },
        .{ "zr_call_python", "bpilplp" },
        .{ "zr_global", "bpilpp" },
        .{ "zr_record_new", "bpipp" },
        .{ "zr_isinstance", "bpilllp" },
        .{ "zr_runtime", "bpipip" },
        .{ "zr_call_seq", "bpillllpp" },
        .{ "zr_raise", "bpill" },
        .{ "zr_slice", "bpillllllllp" },
        .{ "zr_exc_matches", "bpl" },
        .{ "zr_exc_catch", "bpip" },
        .{ "zr_type", "vllp" },
        .{ "zr_cell", "bpilllp" },
        .{ "zr_builtin", "bpiilllp" },
        .{ "zr_range", "bpiplp" },
        .{ "zr_is_type", "blli" },
        .{ "zr_format", "bpillipp" },
        .{ "zr_concat", "bpiplp" },
        .{ "zr_unpack", "bpilllp" },
        .{ "zr_varargs", "bpipllp" },
        .{ "zr_py_semantic", "ipiipip" },
        .{ "zr_run_value", "ipiillpip" },
    };

    fn letterType(self: *Compiler, l: u8) ir.Type {
        const t = self.m.t;
        return switch (l) {
            'v' => t.void,
            'b' => t.i1,
            'i' => t.i32,
            'l' => t.i64,
            'p' => t.ptr,
            else => unreachable,
        };
    }

    fn declareRuntime(self: *Compiler) !void {
        // (the program's names for addresses, across its modules)
        self.m.counter = &self.ksyms;
        for (runtime_decls) |d| {
            var params: [12]ir.Type = undefined;
            for (d[1][1..], 0..) |l, i| params[i] = self.letterType(l);
            try self.m.declare(d[0], self.letterType(d[1][0]), params[0 .. d[1].len - 1]);
        }
        const t = self.m.t;
        const i64s = [_]ir.Type{t.i64};
        self.sadd = self.m.intrinsic("llvm.sadd.with.overflow", &i64s);
        self.ssub = self.m.intrinsic("llvm.ssub.with.overflow", &i64s);
        self.smul = self.m.intrinsic("llvm.smul.with.overflow", &i64s);
        // Reference counts, inline: the runtime's objects (tags 4 to 9)
        // counted here, unless immortal; Python objects (10) by the runtime
        try self.refcountFunction("zr_inc", "zr_incref", false);
        try self.refcountFunction("zr_dec", "zr_decref", true);
    }

    /// zr_inc / zr_dec(tag, bits): a count changed inline (always inlined).
    fn refcountFunction(self: *Compiler, name: []const u8, python_helper: []const u8, dec: bool) !void {
        const m = &self.m;
        const t = m.t;
        const fun = try m.function(name, t.void, &.{ t.i64, t.i64 }, false);
        m.alwaysInline(fun);
        var f = ir.Function.init(m, fun.v);
        const tag = f.param(0);
        const bits = f.param(1);
        const counted = try f.label("counted");
        const change = try f.label("change");
        const other = try f.label("other");
        const python = try f.label("python");
        const done = try f.label("done");
        // (counted: str..function (4-9), a Big (13))
        const k = f.sub(tag, m.k64(4));
        try f.condBr(f.or_(f.icmp(jit_c.LLVMIntULT, k, m.k64(6)), f.icmp(jit_c.LLVMIntEQ, tag, m.k64(@intFromEnum(value.Tag.big)))), counted, other);
        try f.block(counted);
        const p = f.intToPtr(bits);
        const rc = f.load(t.i64, p);
        try f.condBr(f.icmp(jit_c.LLVMIntULT, rc, m.k64(@bitCast(value.IMMORTAL))), change, done);
        try f.block(change);
        if (dec) {
            const rc1 = f.sub(rc, m.k64(1));
            f.store(rc1, p);
            const free = try f.label("free");
            try f.condBr(f.icmp(jit_c.LLVMIntEQ, rc1, m.k64(0)), free, done);
            try f.block(free);
            _ = f.callName("zr_free", &.{ tag, bits });
        } else {
            f.store(f.add(rc, m.k64(1)), p);
        }
        try f.br(done);
        try f.block(other);
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, tag, m.k64(10)), python, done);
        try f.block(python);
        _ = f.callName(python_helper, &.{ tag, bits });
        try f.br(done);
        try f.block(done);
        try f.retVoid();
        f.finish();
    }

    /// Each symbol's slot, in the function it lives in; which functions
    /// keep their variables in a heap frame (those functions are made in).
    fn computeLayouts(self: *Compiler) !void {
        const d = self.data;
        for (d.syms, 0..) |s, i| {
            if (s.builtin) continue;
            const home = d.homeOf(@intCast(i));
            const layout = try self.layoutOf(home);
            try self.slot_of.put(self.a, @intCast(i), @intCast(layout.syms.items.len));
            try layout.syms.append(self.a, @intCast(i));
        }
        // A function with a function inside it: heap (the inner one's env)
        for (d.nodes, 0..) |n, i| {
            if (!self.isFunctionNode(@intCast(i))) continue;
            _ = n;
            const outer = self.enclosingFunction(@intCast(i));
            const layout = try self.layoutOf(outer);
            layout.heap = true;
        }
        (try self.layoutOf(NONE)).heap = true;
        // Block scopes with frames of their own: on the heap (closures keep
        // them), and what's around them up to their function (a frame's
        // parent is the frame around it)
        var it = d.frame_scopes.keyIterator();
        while (it.next()) |scope| {
            (try self.layoutOf(scope.*)).heap = true;
            var up = self.ownerOf(scope.*);
            while (true) : (up = self.ownerOf(up)) {
                (try self.layoutOf(up)).heap = true;
                if (up == NONE or self.isFunctionNode(up)) break;
            }
        }
        // Semantics run as Python (and nodes run through the bridge) see the
        // variables through the frames: every function's on the heap
        if (self.force_heap or self.anyPython()) {
            for (d.nodes, 0..) |_, i| {
                if (self.isFunctionNode(@intCast(i))) (try self.layoutOf(@intCast(i))).heap = true;
            }
            var lit = self.layouts.valueIterator();
            while (lit.next()) |l| l.*.heap = true;
            self.all_heap = true;
        }
    }

    /// The node whose frame the code of a node runs in: the innermost
    /// function node or block scope with frames around it (NONE: the
    /// program's).
    pub fn ownerOf(self: *const Compiler, idx: u32) u32 {
        var n = self.data.parents[idx];
        while (n != NONE) : (n = self.data.parents[n]) {
            if (self.isFunctionNode(n) or self.data.hasFrame(n)) return n;
        }
        return NONE;
    }

    /// A function's variables (and its block scopes') in frames: one not
    /// compiled yet, whose code needs its frame.
    pub fn heapFunction(self: *Compiler, fnode: u32) !void {
        (try self.layoutOf(fnode)).heap = true;
        var it = self.layouts.iterator();
        while (it.next()) |e| {
            var up = e.key_ptr.*;
            while (up != NONE and !self.isFunctionNode(up)) up = self.ownerOf(up);
            if (up == fnode) e.value_ptr.*.heap = true;
        }
    }

    pub fn layoutOf(self: *Compiler, fnode: u32) !*Layout {
        const entry = try self.layouts.getOrPut(self.a, fnode);
        if (!entry.found_existing) {
            entry.value_ptr.* = try self.a.create(Layout);
            // (one made after the layouts were worked out: as the rest, all
            // on the heap if they are)
            entry.value_ptr.*.* = .{ .heap = self.all_heap };
        }
        return entry.value_ptr.*;
    }

    pub fn isFunctionNode(self: *const Compiler, idx: u32) bool {
        const rid = self.data.rule(idx);
        return rid < self.lang.functions.len and self.lang.functions[rid] != null;
    }

    /// The function node around a node (NONE: the top level).
    pub fn enclosingFunction(self: *const Compiler, idx: u32) u32 {
        var n = self.data.parents[idx];
        while (n != NONE) : (n = self.data.parents[n]) {
            if (self.isFunctionNode(n)) return n;
        }
        return NONE;
    }

    /// The LLVM function of the top level (NONE: `<prefix>_main(ctx,
    /// globals)`, the entry point) or of a language function
    /// (`<prefix>_f<node>(ctx, env, args, nargs, recv, result)`).
    fn llvmFunction(self: *Compiler, fnode: u32) !ir.Fn {
        const t = self.m.t;
        if (fnode == NONE) {
            const name = try std.fmt.allocPrint(self.a, "{s}_main", .{self.m.prefix});
            return self.m.function(name, t.i1, &.{ t.ptr, t.ptr }, true);
        }
        // (external: modules compiled later, thunks, call them by name)
        if (fnode & TYPED != 0) {
            // (`<prefix>_t<node>(ctx, env, recv, result, params...)`: its
            // parameters plain values, of the kinds declared)
            const n = self.typed_params.get(fnode & ~TYPED).?.?.len;
            const params = try self.a.alloc(ir.Type, 4 + n);
            @memcpy(params[0..4], &[_]ir.Type{ t.ptr, t.ptr, t.ptr, t.ptr });
            @memset(params[4..], t.i64);
            const name = try std.fmt.allocPrint(self.a, "{s}_t{d}", .{ self.m.prefix, fnode & ~TYPED });
            return self.m.function(name, t.i1, params, true);
        }
        const name = try std.fmt.allocPrint(self.a, "{s}_f{d}", .{ self.m.prefix, fnode });
        return self.m.function(name, t.i1, &.{ t.ptr, t.ptr, t.ptr, t.i64, t.ptr, t.ptr }, true);
    }

    /// A function node's typed entry, in the queue and the compiled
    /// functions: the node with this bit.
    pub const TYPED: u32 = 1 << 30;

    /// The code of a language function (`fnode | TYPED`: its typed entry),
    /// compiled (queued if it isn't yet).
    pub fn functionCode(self: *Compiler, fnode: u32) !ir.Value {
        if (!self.compiled_fns.contains(fnode)) {
            try self.compiled_fns.put(self.a, fnode, {});
            try self.new_fns.append(self.a, fnode);
            try self.queue.append(self.a, fnode);
        }
        return (try self.llvmFunction(fnode)).v;
    }

    /// The record type of a class, if its objects are records (recordOf).
    fn recordType(self: *Compiler, cls: *PyObject) Error!?*value.RecordType {
        const t = try recordOf(cls) orelse return null;
        _ = try self.objectIndex(cls);
        return t;
    }

    /// A Python object the code refers to: its index (a new reference kept).
    pub fn objectIndex(self: *Compiler, o: *PyObject) !usize {
        for (self.objects.items, 0..) |x, i| if (x == o) return i;
        py.Py_IncRef(o);
        try self.objects.append(self.a, o);
        return self.objects.items.len - 1;
    }

    /// Generate one function: the top level (NONE) or a language function.
    pub fn genFunction(self: *Compiler, key: u32) Error!void {
        const fun = try self.llvmFunction(key);
        const fnode = if (key == NONE) key else key & ~TYPED;
        var g = Gen{ .c = self, .f = ir.Function.init(&self.m, fun.v), .fnode = fnode, .layout = try self.layoutOf(fnode), .unit = std.hash.Wyhash.hash(3, std.mem.asBytes(&key)) };
        g.ctx = g.f.param(0);
        if (fnode == NONE) {
            g.globals = g.f.param(1);
        } else if (key & TYPED != 0) {
            g.typed = true;
            g.env = g.f.param(1);
            g.args = self.m.nullPtr();
            g.nargs = g.k(@intCast(self.typed_params.get(fnode).?.?.len));
            g.recv = g.f.param(2);
            g.result = g.f.param(3);
        } else {
            g.env = g.f.param(1);
            g.args = g.f.param(2);
            g.nargs = g.f.param(3);
            g.recv = g.f.param(4);
            g.result = g.f.param(5);
        }
        try g.prologue();
        if (fnode == NONE) {
            try g.hoist(NONE);
            try g.execNode(0);
        } else {
            const spec = self.specOf(fnode).?;
            try g.hoist(fnode);
            try g.bindParams(fnode, spec);
            // the body: a field, or the child of a rule
            const body = if (spec.body != 0)
                try g.fieldValue(fnode, spec.body)
            else if (program_mod.childOfRule(self.data, fnode, spec.body_rule)) |b|
                try g.nodeValue(b)
            else
                SVal.none;
            try g.execValue(body);
        }
        try g.epilogue();
        g.f.finish();
    }

    /// The semantic a node runs (its eval or exec): compiled, run as Python
    /// (native=False, or one that couldn't be compiled), or none.
    pub fn semanticOf(self: *const Compiler, idx: u32, which: Which) Semantic {
        const rid = self.data.rule(idx);
        const table = if (which == .eval) self.lang.eval_of else self.lang.exec_of;
        if (rid >= table.len) return .none;
        const fobj = table[rid] orelse return .none;
        if (self.lang.python.contains(fobj)) return .{ .python = fobj };
        if (self.lang.read.get(fobj)) |f| return .{ .compiled = f };
        return .{ .python = fobj };
    }

    pub fn specOf(self: *const Compiler, idx: u32) ?FunctionSpec {
        const rid = self.data.rule(idx);
        return if (rid < self.lang.functions.len) self.lang.functions[rid] else null;
    }
};

// ======================================================================
// Generating a function
// ======================================================================

/// A semantic running inline: its function, node, locals
const Inst = struct {
    func: *const front.Function,
    /// The node it runs for (errors are reported at it)
    node: u32,
    locals: []Local,
    /// Its result: known (returned outside run-time control flow) or in a
    /// slot (returned from inside it)
    result: ?SVal = null,
    result_slot: ?ir.Value = null,
    exit_label: ir.Block,
    /// Run-time control flow depth within it (if, while...)
    dyn_depth: u32 = 0,
    /// The run-time loops around it when it started (one opened since: its
    /// code runs more than once)
    loop_level0: u32 = 0,
    /// Try bodies it's in (a return there goes through their finally; a
    /// raise there lands in their handlers, the rest isn't dead)
    in_try: u32 = 0,
    /// Python loop targets of the semantic itself (break / continue)
    loops: std.ArrayListUnmanaged(PyLoop) = .empty,
    /// It returned (outside run-time control flow): the rest is dead
    done: bool = false,
    /// Slots of values it holds for a while (a run-time loop's items):
    /// None but while held, released with its locals
    temps: std.ArrayListUnmanaged(ir.Value) = .empty,
};

const Local = union(enum) {
    unset,
    static: SVal,
    /// In a stack slot (an alloca of {i64, i64}), with the shape last stored
    slot: struct { ptr: ir.Value, shape: Shape },
};

/// An rt.loop's targets, and how many semantics ran when it started (those
/// above are left by a Break / Continue)
const LoopTarget = struct { brk: ir.Block, cont: ir.Block, depth: usize, scope_depth: usize = 0, tries: usize };

/// A semantic's own loop (break / continue): its targets, and the try
/// statements around it (jumping out leaves those inside)
const PyLoop = struct { brk: ir.Block, cont: ir.Block, tries: usize };

/// A try statement being compiled
const TryFrame = struct {
    inst: *Inst,
    /// The semantics, scopes, run when it started
    depth: usize,
    scope_depth: usize,
    /// In its body: errors go to its handlers (else: in its handlers,
    /// else or finally, only its finally runs on the way out)
    catching: bool = true,
    finally: []const front.Stmt,
    /// Where errors went before it
    outer_err: ir.Block,
    /// Each handler's code, and the exception it caught (a slot: None when
    /// it caught a jump)
    handler_blocks: []ir.Block,
    caught: []ir.Value,
    /// The handler catching rt.Return / rt.Break / rt.Continue (jumps in
    /// compiled code; exceptions caught by an `except` in Python), if any
    catch_return: ?usize = null,
    catch_break: ?usize = null,
    catch_continue: ?usize = null,

    fn catches(self: *const TryFrame, kind: RtMethod) ?usize {
        if (!self.catching) return null;
        return switch (kind) {
            .Return => self.catch_return,
            .Break => self.catch_break,
            .Continue => self.catch_continue,
            else => null,
        };
    }
};

const Gen = struct {
    c: *Compiler,
    f: ir.Function,
    fnode: u32,
    layout: *Layout,
    /// The function's parameters: the context; the globals (the top
    /// level), or the frame around, the arguments, the receiver and the
    /// result slot (a language function)
    ctx: ir.Value = null,
    globals: ir.Value = null,
    env: ir.Value = null,
    args: ir.Value = null,
    recv: ir.Value = null,
    result: ir.Value = null,
    /// The frame pointer (a heap frame) or null (stack slots)
    frame: ?ir.Value = null,
    /// Stack slots of the variables, by layout slot (stack layouts)
    var_slots: std.ArrayListUnmanaged(ir.Value) = .empty,
    /// The argument count (a language function)
    nargs: ir.Value = null,
    /// Variables' values read borrowed (loadVar), by variable
    borrows: std.ArrayListUnmanaged(struct { sym: u32, state: ir.Value, kept: ir.Value }) = .empty,
    /// A function's typed entry: its parameters are the LLVM function's,
    /// plain values of their declared kinds (Gen.typedParams)
    typed: bool = false,
    /// The receiver and the extra arguments of the function being run
    /// (rt.receiver, rt.varargs): two hidden slots of its frame, or stack
    /// slots
    recv_slot: ir.Value = null,
    varargs_slot: ir.Value = null,
    /// Scratch space for helpers' results
    out: ir.Value = null,
    /// Where errors go (return false), and returns (rt.Return)
    err_label: ir.Block = null,
    ret_label: ir.Block = null,
    /// rt.loop targets, innermost last
    loops: std.ArrayListUnmanaged(LoopTarget) = .empty,
    /// Try statements being compiled, innermost last
    tries: std.ArrayListUnmanaged(*TryFrame) = .empty,
    /// The exceptions of the handlers being compiled (slots), innermost
    /// last: what a bare `raise` raises again
    caught: std.ArrayListUnmanaged(ir.Value) = .empty,
    /// Semantics running inline, innermost last
    insts: std.ArrayListUnmanaged(*Inst) = .empty,
    /// Loops being compiled (of the semantics or rt.loop): code in one
    /// runs many times
    loop_level: u32 = 0,
    /// Block scopes with frames being run here, innermost last: their
    /// frame pointers' slots
    scopes: std.ArrayListUnmanaged(struct { scope: u32, slot: ir.Value }) = .empty,
    /// A thunk (one node's eval or exec, called by the bridge for a
    /// semantic run as Python): it returns a status, its value in
    /// `out_param`; the first `base_scopes` scopes are the caller's
    thunk: bool = false,
    /// A helper's code out of line: the semantic it runs for, and the node
    /// its errors are reported at (a parameter: AT_PARAM stands for it)
    helper_semantic: ?*PyObject = null,
    at_param: ir.Value = null,
    /// A helper's code out of line runs in its caller's frames, whatever
    /// they are (`detached`): the scope they're of is a parameter
    /// (OWNER_PARAM stands for it); its receiver and varargs are given
    /// (pointers, null for none)
    detached: bool = false,
    owner_param: ir.Value = null,
    /// A closure's code: the function called (its cells), and the function
    /// it's the code of
    closure_fn: ?Dyn = null,
    closure_root: ?*const front.Function = null,
    out_param: ir.Value = null,
    base_scopes: usize = 0,
    /// While loops being unrolled around the code
    unrolling: u32 = 0,
    /// A helper's code made for a call site that ran often (Site): hot
    specialized: bool = false,
    /// What code this is (a language function's, a helper's, a thunk's: the
    /// same each time it's compiled again): literals escaping here are
    /// built at run time here only (EscapeKey)
    unit: u64 = 0,

    /// Code that runs many times (a function's, a loop's): reference
    /// counts inline. (Elsewhere, calls: each inline one is blocks for
    /// LLVM to compile.)
    fn hot(self: *const Gen) bool {
        return self.fnode != NONE or self.loop_level > 0 or self.specialized;
    }

    /// A value's count up (or down): inline in hot code; else a call,
    /// made only for a counted tag (most values aren't: ints, None,
    /// bools, nodes; one branch instead of a call).
    fn refcount(self: *Gen, dec: bool, tag: ir.Value, bits: ir.Value) Error!void {
        if (self.hot()) {
            _ = self.call(if (dec) "zr_dec" else "zr_inc", &.{ tag, bits });
            return;
        }
        const f = &self.f;
        const m = &self.c.m;
        // (counted: 4 to 10, 13; as bits of a mask: tags under 16)
        const mask: u64 = 0x7F0 | (1 << @intFromEnum(value.Tag.big));
        const bit = f.and_(f.lshr(m.k64(mask), f.and_(tag, m.k64(15))), m.k64(1));
        const counted = f.and_(f.icmp(jit_c.LLVMIntULT, tag, m.k64(16)), f.icmp(jit_c.LLVMIntNE, bit, m.k64(0)));
        const yes = try f.label("rc");
        const done = try f.label("rc_done");
        try f.condBr(counted, yes, done);
        try f.block(yes);
        _ = self.call(if (dec) "zr_decref" else "zr_incref", &.{ tag, bits });
        try f.br(done);
        try f.block(done);
    }

    fn a(self: *Gen) Allocator {
        return self.c.a;
    }

    /// An i64 constant.
    fn k(self: *Gen, n: i64) ir.Value {
        return self.c.m.k64(n);
    }

    /// An i32 constant (node indexes, operator codes).
    fn k32(self: *Gen, n: u32) ir.Value {
        // (in a helper's code out of line: the node errors are reported
        // at is its caller's, a parameter)
        if (n == AT_PARAM) return self.at_param.?;
        if (n == OWNER_PARAM) return self.owner_param.?;
        return self.c.m.k32(n);
    }

    /// A runtime helper's call.
    fn call(self: *Gen, name: []const u8, args: []const ir.Value) ir.Value {
        return self.f.callName(name, args);
    }

    /// A helper that can fail: called, then to the error exit on false.
    fn callCheck(self: *Gen, name: []const u8, args: []const ir.Value) Error!void {
        try self.check(self.call(name, args));
    }

    /// A slot of n values (in the entry block).
    fn valueSlots(self: *Gen, n: usize) Error!ir.Value {
        const t = self.c.m.t;
        return self.f.alloca(L("LLVMArrayType2")(t.val, @max(n, 1)));
    }

    /// &arr[i] of an array of values.
    fn elem(self: *Gen, arr: ir.Value, i: usize) ir.Value {
        return self.f.at(self.c.m.t.val, arr, self.k(@intCast(i)));
    }

    /// A thunk's start: the frames of the code it's called from (`base`, the
    /// frame of `owner`), the scopes' and the function's, worked out.
    fn thunkPrologue(self: *Gen, owner: u32, base: ir.Value) Error!void {
        const c = self.c;
        const f = &self.f;
        const t = c.m.t;
        self.out = try f.alloca(t.val);
        self.err_label = try f.label("error");
        self.ret_label = try f.label("return");
        // The block scopes between the owner and the function, innermost
        // first, their frames up through the parents
        var chain: std.ArrayListUnmanaged(u32) = .empty;
        var at = owner;
        while (at != self.fnode) : (at = c.ownerOf(at)) try chain.append(self.a(), at);
        var frame = base;
        const slots = try self.a().alloc(ir.Value, chain.items.len);
        for (chain.items, 0..) |_, i| {
            slots[i] = try f.alloca(t.ptr);
            f.store(frame, slots[i]);
            frame = f.load(t.ptr, f.offset(frame, 16));
        }
        var i = chain.items.len;
        while (i > 0) {
            i -= 1;
            try self.scopes.append(self.a(), .{ .scope = chain.items[i], .slot = slots[i] });
        }
        self.base_scopes = chain.items.len;
        self.frame = frame;
        if (self.fnode == NONE) {
            self.globals = frame;
        } else {
            self.env = f.load(t.ptr, f.offset(frame, 16));
            const n = self.layout.syms.items.len;
            self.recv_slot = f.offset(frame, 32 + 16 * @as(i64, @intCast(n)));
            self.varargs_slot = f.offset(frame, 32 + 16 * @as(i64, @intCast(n + 1)));
        }
    }

    fn prologue(self: *Gen) Error!void {
        const f = &self.f;
        const t = self.c.m.t;
        self.out = try f.alloca(t.val);
        self.err_label = try f.label("error");
        self.ret_label = try f.label("return");
        const n = self.layout.syms.items.len;
        if (self.fnode == NONE) {
            self.frame = self.globals;
        } else if (self.layout.heap) {
            // (and two hidden slots: the receiver, the extra arguments)
            self.frame = self.call("zr_frame_new", &.{ self.env, self.k(@intCast(n + 2)) });
            self.recv_slot = f.offset(self.frame.?, 32 + 16 * @as(i64, @intCast(n)));
            self.varargs_slot = f.offset(self.frame.?, 32 + 16 * @as(i64, @intCast(n + 1)));
        } else {
            for (0..n) |_| try self.var_slots.append(self.a(), try f.alloca(t.val));
            // (unset until assigned; the entry block runs once)
            for (self.var_slots.items) |slot| f.entryStore(self.k(@bitCast(helpers.UNSET)), slot);
            self.recv_slot = try self.noneSlot();
            self.varargs_slot = try self.noneSlot();
            try self.var_slots.append(self.a(), self.recv_slot);
            try self.var_slots.append(self.a(), self.varargs_slot);
        }
        if (self.fnode != NONE) {
            // (None until a Return says otherwise)
            try self.storeSlot(self.result, .{ .tag = self.k(0), .bits = self.k(0), .shape = .none });
            try self.storeReceiver();
            const spec = self.c.specOf(self.fnode).?;
            if (spec.extra == .keep) {
                const nparams = self.paramNodes(self.fnode, spec).len;
                try self.callCheck("zr_varargs", &.{ self.ctx, self.k32(self.fnode), self.args, self.nargs, self.k(@intCast(nparams)), self.out });
                try self.storeSlot(self.varargs_slot, try self.loadOut(.tuple));
            }
        }
    }

    /// The function's receiver (rt.receiver), in its slot: the call's, or
    /// (a call without one) the receiver of the function it was made in,
    /// as the frames around it are searched in the reference mode.
    fn storeReceiver(self: *Gen) Error!void {
        const f = &self.f;
        const c = self.c;
        const has = f.icmp(jit_c.LLVMIntNE, self.recv, c.m.nullPtr());
        const given = try f.label("recv_given");
        const outer = try f.label("recv_outer");
        const join = try f.label("recv_set");
        try f.condBr(has, given, outer);
        try f.block(given);
        const v = try self.loadSlot(self.recv, .any);
        try self.increfDyn(v);
        try self.storeSlot(self.recv_slot, v);
        try f.br(join);
        try f.block(outer);
        // (the nearest function around, through the frames)
        var at = c.ownerOf(self.fnode);
        while (at != NONE and !c.isFunctionNode(at)) at = c.ownerOf(at);
        if (at != NONE) {
            const frame = try self.frameOf(at, "the receiver of the function around isn't reachable");
            const n = (try c.layoutOf(at)).syms.items.len;
            const ov = try self.loadSlot(f.offset(frame, 32 + 16 * @as(i64, @intCast(n))), .any);
            // (an unset slot holds None here: the hidden slots start unset)
            const unset = f.icmp(jit_c.LLVMIntEQ, ov.tag, self.k(@bitCast(helpers.UNSET)));
            const tag = f.select(unset, self.k(0), ov.tag);
            const bits = f.select(unset, self.k(0), ov.bits);
            const d = Dyn{ .tag = tag, .bits = bits, .shape = .any };
            try self.increfDyn(d);
            try self.storeSlot(self.recv_slot, d);
        } else try self.storeSlot(self.recv_slot, self.noneDyn());
        try f.br(join);
        try f.block(join);
    }

    fn epilogue(self: *Gen) Error!void {
        const f = &self.f;
        try f.br(self.ret_label);
        try f.block(self.ret_label);
        try self.releaseFrame();
        try f.ret(self.c.m.k1(true));
        try f.block(self.err_label);
        try self.releaseFrame();
        try f.ret(self.c.m.k1(false));
    }

    /// Drop the function's variables (its heap frame, or its stack slots'
    /// values).
    fn releaseFrame(self: *Gen) Error!void {
        if (self.fnode == NONE) return;
        if (self.frame) |fr| {
            _ = self.call("zr_frame_release", &.{fr});
            return;
        }
        for (self.var_slots.items) |slot| {
            const v = try self.loadSlot(slot, .any);
            // (an unset slot holds no reference: decrefs ignore its tag)
            try self.refcount(true, v.tag, v.bits);
        }
    }

    // ------------------------------------------------------------------
    // Errors
    // ------------------------------------------------------------------

    /// After a helper that can fail: on false, to the error exit.
    fn check(self: *Gen, ok: ir.Value) Error!void {
        const cont = try self.f.label("ok");
        try self.f.condBr(ok, cont, self.err_label);
        try self.f.block(cont);
    }

    /// A runtime error at a node with a fixed message: to the error exit.
    fn failAt(self: *Gen, node: u32, message: []const u8) Error!void {
        const s = try self.c.m.string(message);
        _ = self.call("zr_fail", &.{ self.ctx, self.k32(node), s });
        try self.f.br(self.err_label);
        // (the code after it is unreachable: a fresh block keeps it valid)
        try self.f.block(try self.f.label("dead"));
    }

    // ------------------------------------------------------------------
    // Values: making them dynamic, dropping them
    // ------------------------------------------------------------------

    fn dyn(tag: ir.Value, bits: ir.Value, shape: Shape) SVal {
        return .{ .dyn = .{ .tag = tag, .bits = bits, .shape = shape } };
    }

    /// A value of a tag and constant bits.
    /// A module's table only read (frozenTable) at run time: made native
    /// once, for the process (immortal: the module's table never changes),
    /// its address a constant.
    fn frozenConst(self: *Gen, o: *PyObject, shape: Shape) Error!Dyn {
        const v = frozen_natives.get(o) orelse blk: {
            const v = try nativeCopy(o);
            v.ptr().rc = value.IMMORTAL;
            // (the table kept: its address is the key)
            py.Py_IncRef(o);
            try frozen_natives.put(std.heap.c_allocator, o, v);
            break :blk v;
        };
        return .{ .tag = self.k(@intCast(v.tag)), .bits = self.c.m.addrInt(v.bits), .shape = shape };
    }

    /// A Python list or dict as a native one (its items as values are:
    /// lists and dicts in it Python's).
    fn nativeCopy(o: *PyObject) Error!Value {
        if (py.PyList_Check(o)) {
            const n: usize = @intCast(py.c.PyList_Size(o));
            const l = value.newList(n) orelse return error.OutOfMemory;
            for (0..n) |i| {
                const x = value.fromPython(py.c.PyList_GetItem(o, @intCast(i)).?) orelse return error.Python;
                if (!value.listPush(l, x)) return error.OutOfMemory;
            }
            return Value.obj(.list, &l.head);
        }
        const d = value.newDict() orelse return error.OutOfMemory;
        var pos: py.Py_ssize_t = 0;
        var key: ?*PyObject = null;
        var val: ?*PyObject = null;
        while (py.c.PyDict_Next(o, &pos, @ptrCast(&key), @ptrCast(&val)) != 0) {
            const kv = value.fromPython(key.?) orelse return error.Python;
            defer value.decref(kv);
            const x = value.fromPython(val.?) orelse return error.Python;
            defer value.decref(x);
            if (!value.dictSet(d, kv, x)) return error.OutOfMemory;
        }
        return Value.obj(.dict, &d.head);
    }

    fn konst(self: *Gen, tag: i64, bits: i64, shape: Shape) Dyn {
        return .{ .tag = self.k(tag), .bits = self.k(bits), .shape = shape };
    }

    /// None at run time.
    fn noneDyn(self: *Gen) Dyn {
        return self.konst(0, 0, .none);
    }

    /// A value's run-time form (an owned reference).
    fn materialize(self: *Gen, v: SVal, at: u32) Error!Dyn {
        return switch (v) {
            // (a borrowed one: owned, an ordinary value from here on)
            .dyn => |d| blk: {
                if (d.state == null) break :blk d;
                try self.owned(d);
                var out = d;
                out.state = null;
                break :blk out;
            },
            .none => self.noneDyn(),
            .bool => |b| self.konst(1, @intFromBool(b), .bool),
            .int => |n| self.konst(2, n, .int),
            .pint => |n| self.konst(@intCast(value.PINT_TAG), n, .int),
            .float => |x| self.konst(3, @bitCast(x), .float),
            .str => |s| .{ .tag = self.k(4), .bits = self.f.ptrToInt(try self.c.m.string(s)), .shape = .str },
            .node => |n| self.konst(11, n, .node),
            // (a big int within 128 bits: a Big, made once for the program)
            .py => |o| if (ph.typeOf(o) == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyLong_Type")))) and value.bigOf(o) != null) blk: {
                const b = try self.c.bigConst(value.bigOf(o).?);
                break :blk .{ .tag = self.k(@intFromEnum(value.Tag.big)), .bits = self.c.m.addrInt(@intFromPtr(b)), .shape = .any };
            } else blk: {
                const idx = try self.c.objectIndex(o);
                _ = self.call("zr_object", &.{ self.ctx, self.k(@intCast(idx)), self.out });
                break :blk try self.loadOut(.any);
            },
            .list => |l| blk: {
                if (l.frozen) |o| break :blk try self.frozenConst(o, .list);
                const d = try self.buildSequence("zr_list", l.items.items, at, .list);
                try self.promote(v, d);
                break :blk d;
            },
            .tuple => |t| self.buildSequence("zr_tuple", t, at, .tuple),
            .dict => |x| blk: {
                if (x.frozen) |o| break :blk try self.frozenConst(o, .dict);
                const d = try self.buildDict(x, at);
                try self.promote(v, d);
                break :blk d;
            },
            // (handed over: an rt value of the frames here, their scope in
            // the tag's upper word)
            .rt => blk: {
                const c = self.c;
                if (!c.allHeap()) {
                    c.need_frames = true;
                    return c.unsupported("rt is handed over here: the program needs its variables in frames", .{});
                }
                const f = &self.f;
                const owner = f.zext64(self.k32(self.currentOwner()));
                const tag = f.or_(self.k(@intFromEnum(value.Tag.rt)), f.shl(owner, self.k(32)));
                break :blk Dyn{ .tag = tag, .bits = f.ptrToInt(try self.currentFrame()), .shape = .any };
            },
            else => self.c.unsupported("a {s} can't be kept in a variable or passed as a value (node {d})", .{ @tagName(v), at }),
        };
    }

    /// A list or dict a literal (or a comprehension) made: known when
    /// compiling, unless it's one that escapes where a known one can't
    /// follow (then built at run time).
    fn literal(self: *Gen, v: SVal, e: *const front.Expr, at: u32) Error!SVal {
        switch (v) {
            .list => |l| l.origin = e,
            .dict => |d| d.origin = e,
            else => return v,
        }
        if (!self.c.lang.escaping.contains(.{ .origin = e, .unit = self.unit })) return v;
        return .{ .dyn = try self.materialize(v, at) };
    }

    /// A known list or dict made a run-time one (it escapes: given to code
    /// that sees it as an object, which may change it): the variables
    /// referring to it refer to the run-time one from here (each with a
    /// reference of its own), so both see the same object. Inside run-time
    /// control flow that can't follow every path: its literal is marked to
    /// be built at run time, and the program compiled again.
    fn promote(self: *Gen, old: SVal, d: Dyn) Error!void {
        const ptr = containerPtr(old) orelse return;
        if (!self.aliased(ptr)) return;
        if (self.inFlow() and !self.straightFor(ptr)) {
            // (one no literal made: a copy, for reading (one changed is
            // refused: materializeToChange))
            const origin = originOf(old) orelse return;
            try self.c.lang.escaping.put(std.heap.c_allocator, .{ .origin = origin, .unit = self.unit }, {});
            self.c.need_retry = true;
            return self.c.unsupported("a list or dict escapes inside run-time control flow: compiled again, made at run time", .{});
        }
        for (self.insts.items) |inst| {
            for (inst.locals) |*l| switch (l.*) {
                .static => |*sv| try self.replaceRefs(sv, ptr, d, 4),
                else => {},
            };
        }
    }

    /// A value about to be changed at run time, as materialize makes it: a
    /// known list or dict no literal made, that variables refer to, can't
    /// follow inside run-time control flow.
    fn materializeToChange(self: *Gen, v: SVal, at: u32) Error!Dyn {
        if (containerPtr(v)) |ptr| if (originOf(v) == null and self.aliased(ptr) and self.inFlow() and !self.straightFor(ptr))
            return self.c.unsupported("a list or dict made when compiling (not by a literal) changed inside run-time control flow isn't compiled yet", .{});
        return self.materialize(v, at);
    }

    /// Whether the variables referring to a known container are all of
    /// semantics running straight on from here (no run-time control flow
    /// of their own, no return from inside one, no loop since they
    /// started): run-time control flow around them only, so every path
    /// their code is on from here is this one (the variables can refer to
    /// the run-time object from here).
    fn straightFor(self: *Gen, ptr: *const anyopaque) bool {
        for (self.insts.items) |inst| {
            const refers = for (inst.locals) |l| switch (l) {
                .static => |sv| if (refersTo(sv, ptr, 4)) break true,
                else => {},
            } else false;
            if (!refers) continue;
            if (inst.dyn_depth > 0 or inst.result_slot != null or inst.loop_level0 != self.loop_level) return false;
        }
        return true;
    }

    fn containerPtr(v: SVal) ?*const anyopaque {
        return switch (v) {
            .list => |l| l,
            .dict => |x| x,
            else => null,
        };
    }

    fn originOf(v: SVal) ?*const front.Expr {
        return switch (v) {
            .list => |l| l.origin,
            .dict => |x| x.origin,
            else => null,
        };
    }

    /// A semantic's variable refers to the known container.
    fn aliased(self: *Gen, ptr: *const anyopaque) bool {
        for (self.insts.items) |inst| {
            for (inst.locals) |l| switch (l) {
                .static => |sv| if (refersTo(sv, ptr, 4)) return true,
                else => {},
            };
        }
        return false;
    }

    /// Inside run-time control flow (after a return from inside it too: the
    /// rest runs on some paths only).
    fn inFlow(self: *Gen) bool {
        if (self.loop_level > 0) return true;
        for (self.insts.items) |inst| if (inst.dyn_depth > 0 or inst.result_slot != null) return true;
        return false;
    }

    fn refersTo(v: SVal, ptr: *const anyopaque, depth: u32) bool {
        if (depth == 0) return false;
        return switch (v) {
            .list => |l| @as(*const anyopaque, l) == ptr or for (l.items.items) |x| {
                if (refersTo(x, ptr, depth - 1)) break true;
            } else false,
            .dict => |x| @as(*const anyopaque, x) == ptr or for (x.values.items) |y| {
                if (refersTo(y, ptr, depth - 1)) break true;
            } else false,
            .tuple => |t| for (t) |x| {
                if (refersTo(x, ptr, depth - 1)) break true;
            } else false,
            else => false,
        };
    }

    /// The references to a known container in `v` (and the known
    /// containers in it) replaced by the run-time one.
    fn replaceRefs(self: *Gen, v: *SVal, ptr: *const anyopaque, d: Dyn, depth: u32) Error!void {
        if (depth == 0) return;
        const is_it = switch (v.*) {
            .list => |l| @as(*const anyopaque, l) == ptr,
            .dict => |x| @as(*const anyopaque, x) == ptr,
            else => false,
        };
        if (is_it) {
            try self.increfDyn(d);
            v.* = .{ .dyn = d };
            return;
        }
        switch (v.*) {
            .list => |l| for (l.items.items) |*x| try self.replaceRefs(x, ptr, d, depth - 1),
            .dict => |x| for (x.values.items) |*y| try self.replaceRefs(y, ptr, d, depth - 1),
            .tuple => |t| for (@constCast(t)) |*x| try self.replaceRefs(x, ptr, d, depth - 1),
            else => {},
        }
    }

    /// A stack array of values (each materialized: owned references).
    fn valueArray(self: *Gen, items: []const SVal, at: u32) Error!ir.Value {
        const arr = try self.valueSlots(items.len);
        for (items, 0..) |item, i| {
            const d = try self.materialize(item, at);
            try self.storeSlot(self.elem(arr, i), d);
        }
        return arr;
    }

    /// A run-time list or tuple of known items (they're taken).
    fn buildSequence(self: *Gen, helper: []const u8, items: []const SVal, at: u32, shape: Shape) Error!Dyn {
        const arr = try self.valueArray(items, at);
        try self.callCheck(helper, &.{ self.ctx, self.k32(at), arr, self.k(@intCast(items.len)), self.out });
        return self.loadOut(shape);
    }

    fn buildDict(self: *Gen, d: *SDict, at: u32) Error!Dyn {
        const keys = try self.valueArray(d.keys.items, at);
        const vals = try self.valueArray(d.values.items, at);
        const ok = self.call("zr_dict", &.{ self.ctx, self.k32(at), keys, vals, self.k(@intCast(d.keys.items.len)), self.out });
        // (zr_dict borrowed them)
        try self.dropArray(keys, d.keys.items.len);
        try self.dropArray(vals, d.values.items.len);
        try self.check(ok);
        return self.loadOut(.dict);
    }

    fn dropArray(self: *Gen, arr: ir.Value, n: usize) Error!void {
        for (0..n) |i| {
            const v = try self.loadSlot(self.elem(arr, i), .any);
            try self.refcount(true, v.tag, v.bits);
        }
    }

    /// The helpers' result slot, read (an owned reference).
    fn loadOut(self: *Gen, shape: Shape) Error!Dyn {
        return self.loadSlot(self.out, shape);
    }

    /// Give up a dynamic value (decref it; a borrowed one: given up).
    fn drop(self: *Gen, v: SVal) Error!void {
        switch (v) {
            .dyn => |d| if (d.heapish()) {
                if (d.state) |state| {
                    const f = &self.f;
                    const is_owned = try f.label("borrow_owned");
                    const given = try f.label("borrow_given");
                    const done = try f.label("borrow_dropped");
                    try f.condBr(f.icmp(jit_c.LLVMIntEQ, f.load(self.c.m.t.i64, state), self.k(1)), is_owned, given);
                    try f.block(is_owned);
                    try self.refcount(true, d.tag, d.bits);
                    try f.br(done);
                    try f.block(given);
                    f.store(self.k(2), state);
                    try f.br(done);
                    try f.block(done);
                } else try self.refcount(true, d.tag, d.bits);
            },
            else => {},
        }
    }

    fn increfDyn(self: *Gen, d: Dyn) Error!void {
        if (!d.heapish()) return;
        try self.owned(d);
        try self.refcount(false, d.tag, d.bits);
    }

    /// A borrowed value made owned (its reference taken, if it hasn't
    /// been): what the code may keep, whatever happens to the variable.
    fn owned(self: *Gen, d: Dyn) Error!void {
        const state = d.state orelse return;
        const f = &self.f;
        const take = try f.label("borrow_take");
        const done = try f.label("borrow_taken");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, f.load(self.c.m.t.i64, state), self.k(0)), take, done);
        try f.block(take);
        try self.refcount(false, d.tag, d.bits);
        f.store(self.k(1), state);
        try f.br(done);
        try f.block(done);
    }

    /// A value to read and give up (drop), not keep: a variable's value
    /// borrowed stays borrowed.
    fn borrowed(self: *Gen, v: SVal, at: u32) Error!Dyn {
        return if (v == .dyn) v.dyn else self.materialize(v, at);
    }

    /// A slot ({i64, i64}) of the stack: its pointer.
    fn storeSlot(self: *Gen, slot: ir.Value, d: Dyn) Error!void {
        const f = &self.f;
        const t = self.c.m.t;
        // (a borrowed value kept in memory: owned)
        try self.owned(d);
        f.store(d.tag, slot);
        f.store(d.bits, f.field(t.val, slot, 1));
    }

    fn loadSlot(self: *Gen, slot: ir.Value, shape: Shape) Error!Dyn {
        const f = &self.f;
        const t = self.c.m.t;
        const tag = f.load(t.i64, slot);
        const bits = f.load(t.i64, f.field(t.val, slot, 1));
        return .{ .tag = tag, .bits = bits, .shape = shape };
    }

    // ------------------------------------------------------------------
    // Truth
    // ------------------------------------------------------------------

    const Truth = union(enum) { known: bool, dyn: ir.Value };

    fn truth(self: *Gen, v: SVal, at: u32) Error!Truth {
        switch (v) {
            .none => return .{ .known = false },
            .bool => |b| return .{ .known = b },
            .int, .pint => |n| return .{ .known = n != 0 },
            .float => |x| return .{ .known = x != 0 },
            .str => |s| return .{ .known = s.len != 0 },
            .list => |l| return .{ .known = l.items.items.len != 0 },
            .tuple => |t| return .{ .known = t.len != 0 },
            .dict => |d| return .{ .known = d.keys.items.len != 0 },
            .node, .rt, .rt_method, .method, .control => return .{ .known = true },
            .py => |o| {
                // (an object that may change, a list...: when the code runs)
                if (!try stablePy(o)) return self.truth(.{ .dyn = try self.materialize(v, at) }, at);
                const r = py.c.PyObject_IsTrue(o);
                if (r < 0) return error.Python;
                return .{ .known = r == 1 };
            },
            .dyn => |d| {
                const f = &self.f;
                const m = &self.c.m;
                const t = switch (d.shape) {
                    .bool, .int => f.icmp(jit_c.LLVMIntNE, d.bits, self.k(0)),
                    .none => m.k1(false),
                    .function => m.k1(true),
                    .float => f.fcmp(jit_c.LLVMRealUNE, f.bitcast(d.bits, m.t.f64), L("LLVMConstNull")(m.t.f64)),
                    // (its length: in the same place for each)
                    .str, .list, .tuple, .dict => f.icmp(jit_c.LLVMIntNE, f.load(m.t.i64, f.offset(f.intToPtr(d.bits), @offsetOf(value.List, "len"))), self.k(0)),
                    else => self.call("zr_truthy", &.{ d.tag, d.bits }),
                };
                try self.drop(v);
                return .{ .dyn = t };
            },
        }
    }

    // ------------------------------------------------------------------
    // Variables of the language
    // ------------------------------------------------------------------

    /// The slot pointer of a symbol's variable, from here: in this
    /// function's frame or stack, a block scope's frame, or a frame around
    /// this function.
    fn varSlot(self: *Gen, sym: u32) Error!ir.Value {
        const c = self.c;
        const f = &self.f;
        const home = c.data.homeOf(sym);
        const slot = c.slot_of.get(sym).?;
        if (home == self.fnode and self.frame == null) return self.var_slots.items[slot];
        const frame = try self.frameOf(home, "a variable of another function isn't reachable from here");
        return f.offset(frame, 32 + 16 * @as(i64, slot));
    }

    /// The frame of `owner` (a function node, a block scope with frames,
    /// or NONE for the program) seen from here: a block scope running in
    /// this function, this function's, or one around it through the
    /// frames' parents.
    fn frameOf(self: *Gen, owner: u32, comptime unreachable_msg: []const u8) Error!ir.Value {
        const c = self.c;
        const f = &self.f;
        var i = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            if (self.scopes.items[i].scope == owner) return f.load(c.m.t.ptr, self.scopes.items[i].slot);
        }
        // (a helper's code out of line runs in its caller's frames, of a
        // scope it's given: the frames up to `owner`'s walked when it runs)
        if (self.detached) {
            const owners = try c.ownerTable();
            return self.call("zr_frame_of", &.{ self.frame.?, self.owner_param, c.m.k32(owner), c.m.ptrConst(@intFromPtr(owners.ptr)) });
        }
        if (owner == self.fnode) return self.frame orelse c.unsupported(unreachable_msg ++ " (node {d})", .{owner});
        if (self.fnode == NONE) return c.unsupported(unreachable_msg ++ " (node {d})", .{owner});
        // (env is the frame the function was made in)
        var frame: ir.Value = self.env;
        var at = c.ownerOf(self.fnode);
        while (at != owner) {
            if (at == NONE) return c.unsupported(unreachable_msg ++ " (node {d})", .{owner});
            frame = f.load(c.m.t.ptr, f.offset(frame, 16));
            at = c.ownerOf(at);
        }
        return frame;
    }

    /// The frame code runs in here: the innermost block scope's, or the
    /// function's.
    fn currentFrame(self: *Gen) Error!ir.Value {
        if (self.scopes.items.len > 0) return self.f.load(self.c.m.t.ptr, self.scopes.items[self.scopes.items.len - 1].slot);
        return self.frame orelse {
            // (the code here needs its frame: compiled again with every
            // function's variables in frames)
            self.c.need_frames = true;
            return self.c.unsupported("code here needs the frame of a function without one (node {d}): compiled again with frames", .{self.fnode});
        };
    }

    /// A block scope with frames of its own, entered: a new frame (in a
    /// slot: rt.fresh replaces it), its hoisted functions defined.
    fn enterScope(self: *Gen, idx: u32) Error!void {
        const c = self.c;
        const parent = try self.frameOf(c.ownerOf(idx), "a block scope's frame isn't reachable from here");
        const n = (try c.layoutOf(idx)).syms.items.len;
        const frame = self.call("zr_frame_new", &.{ parent, self.k(@intCast(n)) });
        const slot = try self.f.alloca(c.m.t.ptr);
        self.f.store(frame, slot);
        try self.scopes.append(self.a(), .{ .scope = idx, .slot = slot });
        try self.hoist(idx);
    }

    fn leaveScope(self: *Gen) Error!void {
        const s = self.scopes.pop().?;
        _ = self.call("zr_frame_release", &.{self.f.load(self.c.m.t.ptr, s.slot)});
    }

    /// Release the frames of the block scopes above `depth` (leaving them
    /// by a jump), keeping them active for the code after the jump.
    fn releaseScopesAbove(self: *Gen, depth: usize) void {
        var i = self.scopes.items.len;
        while (i > depth) {
            i -= 1;
            _ = self.call("zr_frame_release", &.{self.f.load(self.c.m.t.ptr, self.scopes.items[i].slot)});
        }
    }

    /// rt.fresh(node): the block scope's frame replaced by a new one.
    fn freshScope(self: *Gen, idx: u32) Error!void {
        const c = self.c;
        if (!c.data.hasFrame(idx)) return;
        var i = self.scopes.items.len;
        while (i > 0) {
            i -= 1;
            const s = self.scopes.items[i];
            if (s.scope != idx) continue;
            const old = self.f.load(c.m.t.ptr, s.slot);
            const parent = self.f.load(c.m.t.ptr, self.f.offset(old, 16));
            const n = (try c.layoutOf(idx)).syms.items.len;
            const frame = self.call("zr_frame_new", &.{ parent, self.k(@intCast(n)) });
            _ = self.call("zr_frame_release", &.{old});
            self.f.store(frame, s.slot);
            return;
        }
        return c.unsupported("rt.fresh(): node {d} isn't a scope being run", .{idx});
    }

    /// rt.load(name): the variable (an owned reference); a builtin's host
    /// function.
    fn loadVar(self: *Gen, name_node: u32) Error!SVal {
        const c = self.c;
        const d = c.data;
        const si = d.symbolIndex(name_node) orelse return self.notAVariable(name_node);
        const sym = d.syms[si];
        if (sym.builtin) {
            const key = ph.newString(sym.name) orelse return error.Python;
            defer py.Py_DecRef(key);
            const h = py.c.PyDict_GetItem(c.lang.hosts, key) orelse {
                try self.failAt(name_node, try std.fmt.allocPrint(self.a(), "no host function for the builtin '{s}'", .{sym.name}));
                return SVal.none;
            };
            return SVal{ .py = h };
        }
        const f = &self.f;
        const slot = try self.varSlot(si);
        const v = try self.loadSlot(slot, .any);
        // Unset: an error at the name (one always set: not checked)
        if (!self.alwaysSet(si)) {
            const is_unset = f.icmp(jit_c.LLVMIntEQ, v.tag, self.k(@bitCast(helpers.UNSET)));
            const bad = try f.label("unset");
            const good = try f.label("set");
            try f.condBr(is_unset, bad, good);
            try f.block(bad);
            const name = try c.m.string(sym.name);
            _ = self.call("zr_unset", &.{ self.ctx, self.k32(name_node), name });
            try f.br(self.err_label);
            try f.block(good);
        }
        // (a variable of a type whose values' kind is declared: known, its
        // reference counted as one of the kind)
        var r = try self.typedValue(name_node, .{ .dyn = v });
        // (an int a variable holds is an I64: stores make it one)
        if (r.dyn.shape == .int) r.dyn.tag = self.k(@intFromEnum(value.Tag.int));
        if (r.dyn.heapish() and try self.borrowable(si)) {
            // (borrowed: the variable's reference keeps it until the
            // variable's stored to; taken then, if it's still in use)
            // (given up where it wasn't read; the value kept for the
            // stores, which may be where it isn't known)
            const state = try f.alloca(self.c.m.t.i64);
            f.entryStore(self.k(2), state);
            f.store(self.k(0), state);
            const kept = try f.alloca(self.c.m.t.val);
            f.store(r.dyn.tag, kept);
            f.store(r.dyn.bits, f.field(self.c.m.t.val, kept, 1));
            r.dyn.state = state;
            try self.borrows.append(self.a(), .{ .sym = si, .state = state, .kept = kept });
        } else try self.increfDyn(r.dyn);
        // (one a function's definition binds: that function, most likely)
        if (r == .dyn and sym.node != NONE) {
            const p = d.parents[sym.node];
            if (p != NONE and c.isFunctionNode(p)) if (c.specOf(p)) |spec| if (program_mod.labelled(d, p, spec.name) == sym.node) {
                r.dyn.func = p;
            };
        }
        return r;
    }

    /// rt.store(name, value), taking the value.
    fn storeVar(self: *Gen, name_node: u32, value_: SVal) Error!void {
        const c = self.c;
        const d = c.data;
        const si = d.symbolIndex(name_node) orelse {
            _ = try self.notAVariable(name_node);
            return;
        };
        if (d.syms[si].builtin) {
            try self.failAt(name_node, try std.fmt.allocPrint(self.a(), "can't assign to the builtin '{s}'", .{d.syms[si].name}));
            return;
        }
        // (its values borrowed and still in use: owned first)
        for (self.borrows.items) |b| if (b.sym == si) {
            var bd = try self.loadSlot(b.kept, .any);
            bd.state = b.state;
            try self.owned(bd);
        };
        // (stored as rt.store does: an int an I64)
        const v = try self.materialize(try self.checkedAt(value_, name_node), name_node);
        const slot = try self.varSlot(si);
        const old = try self.loadSlot(slot, .any);
        try self.storeSlot(slot, v);
        try self.refcount(true, old.tag, old.bits);
    }

    /// Whether a variable's value can be read borrowed: one of this
    /// function's variables on the stack (only its own code stores to it:
    /// storeVar), read borrowed in few places so far (each store checks
    /// them).
    fn borrowable(self: *Gen, si: u32) Error!bool {
        if (self.frame != null or self.detached or self.thunk or self.helper_semantic != null or self.closure_root != null) return false;
        if (self.c.data.homeOf(si) != self.fnode) return false;
        var n: usize = 0;
        for (self.borrows.items) |b| {
            if (b.sym == si) n += 1;
        }
        return n < max_borrows;
    }

    const max_borrows = 16;

    fn notAVariable(self: *Gen, idx: u32) Error!SVal {
        const msg = if (self.c.lang.analysis == null)
            "variables need rules with a scopes() rule"
        else
            try std.fmt.allocPrint(self.a(), "'{s}' is not a variable", .{self.c.data.text(idx)});
        try self.failAt(idx, msg);
        return SVal.none;
    }

    // ------------------------------------------------------------------
    // Functions of the language
    // ------------------------------------------------------------------

    /// The frame a function made from `fnode` sees: the frame of the
    /// function around it (here, or through the frames).
    fn envFor(self: *Gen, fnode: u32) Error!ir.Value {
        return self.frameOf(self.c.ownerOf(fnode), "a function made outside the function or scope around it");
    }

    /// rt.function(node): a function value.
    fn makeFunction(self: *Gen, fnode: u32) Error!SVal {
        const c = self.c;
        const spec = c.specOf(fnode) orelse {
            try self.failAt(fnode, try std.fmt.allocPrint(self.a(), "{s} isn't a function kind (Language.function)", .{c.data.grammar.kind_names[c.data.rule(fnode)]}));
            return SVal.none;
        };
        const code = try c.functionCode(fnode);
        const env = try self.envFor(fnode);
        const name_text = if (spec.name != 0) if (program_mod.labelled(c.data, fnode, spec.name)) |nn| c.data.text(nn) else "<anonymous>" else "<anonymous>";
        const name = try c.m.string(name_text);
        const flags = helpers.FunctionFlags{
            .nparams = @intCast(self.paramNodes(fnode, spec).len),
            .missing_none = spec.missing == .none,
            .extra = switch (spec.extra) {
                .@"error" => .@"error",
                .drop => .drop,
                .keep => .keep,
            },
        };
        try self.callCheck("zr_function", &.{ self.ctx, code, env, self.k32(fnode), name, self.k(flags.word()), self.out });
        return .{ .dyn = try self.loadOut(.function) };
    }

    /// A function node's parameters: the nodes their values are stored
    /// under.
    fn paramNodes(self: *Gen, fnode: u32, spec: FunctionSpec) []const u32 {
        const c = self.c;
        const d = c.data;
        var out: std.ArrayListUnmanaged(u32) = .empty;
        if (spec.params == 0) return &.{};
        var ch = fnode + 1;
        const stop = d.end(fnode);
        while (ch < stop) : (ch = d.end(ch)) {
            if (d.nodes[ch].fieldId() != spec.params) continue;
            // the parameter itself if it defines a variable, else its name
            var target = ch;
            if (d.symbolIndex(ch)) |si| {
                if (d.syms[si].node != ch) target = nameChild(d, ch) orelse ch;
            } else target = nameChild(d, ch) orelse ch;
            out.append(self.a(), target) catch return &.{};
        }
        return out.items;
    }

    fn nameChild(d: *const program_mod.Data, n: u32) ?u32 {
        const field = d.grammar.field_ids.get("name") orelse return null;
        return program_mod.labelled(d, n, field);
    }

    fn bindParams(self: *Gen, fnode: u32, spec: FunctionSpec) Error!void {
        if (self.typed) {
            const shapes = (try self.typedParams(fnode)).?;
            for (self.paramNodes(fnode, spec), shapes, 0..) |p, s, i| {
                const tag: value.Tag = if (s == .int) .int else shapeTag(s);
                try self.storeVar(p, dyn(self.k(@intCast(@intFromEnum(tag))), self.f.param(@intCast(4 + i)), s));
            }
            return;
        }
        for (self.paramNodes(fnode, spec), 0..) |p, i| {
            const v = try self.loadSlot(self.elem(self.args, i), .any);
            try self.increfDyn(v);
            try self.storeVar(p, .{ .dyn = v });
        }
    }

    /// The kinds of a function's parameters, when it has a typed entry
    /// (called with them plain values, no argument list): it takes
    /// parameters only, each declared (Language.types()) an int, a float or
    /// a bool. Null: it hasn't.
    fn typedParams(self: *Gen, fnode: u32) Error!?[]const Shape {
        const c = self.c;
        if (c.typed_params.get(fnode)) |r| return r;
        const result = blk: {
            const spec = c.specOf(fnode) orelse break :blk null;
            if (spec.extra == .keep) break :blk null;
            const params = self.paramNodes(fnode, spec);
            if (params.len == 0) break :blk null;
            const shapes = try c.a.alloc(Shape, params.len);
            for (params, shapes) |p, *s| {
                const kind = try c.kindOfNode(p) orelse break :blk null;
                switch (kind.shape) {
                    .int, .float, .bool => s.* = kind.shape,
                    else => break :blk null,
                }
            }
            break :blk shapes;
        };
        try c.typed_params.put(c.a, fnode, result);
        return result;
    }

    /// Whether a variable always has a value where code reads it: a
    /// function's parameter (set when it's called), a function's name its
    /// scope hoists (set when the scope's entered), nothing unsets them.
    fn alwaysSet(self: *Gen, si: u32) bool {
        const c = self.c;
        const d = c.data;
        if (c.always_set.get(si)) |b| return b;
        const def = d.syms[si].node;
        const result = blk: {
            if (def == NONE) break :blk false;
            var at = d.parents[def];
            while (at != NONE and !c.isFunctionNode(at)) at = d.parents[at];
            if (at == NONE) break :blk false;
            const spec = c.specOf(at) orelse break :blk false;
            for (self.paramNodes(at, spec)) |p| if (p == def) break :blk true;
            if (d.parents[def] == at and program_mod.labelled(d, at, spec.name) == def) {
                var it = d.hoisted.valueIterator();
                while (it.next()) |list| for (list.items) |h| if (h == at) break :blk true;
            }
            break :blk false;
        };
        c.always_set.put(c.a, si, result) catch {};
        return result;
    }

    /// Define the hoisted functions of a scope just entered.
    fn hoist(self: *Gen, scope: u32) Error!void {
        const c = self.c;
        const list = c.data.hoisted.get(scope) orelse return;
        for (list.items) |fnode| {
            const spec = c.specOf(fnode).?;
            const name_node = program_mod.labelled(c.data, fnode, spec.name) orelse continue;
            const fv = try self.makeFunction(fnode);
            try self.storeVar(name_node, fv);
        }
    }

    // ------------------------------------------------------------------
    // Nodes: their fields, their semantics
    // ------------------------------------------------------------------

    /// A node's value as a field (what its action makes, or the node).
    fn nodeValue(self: *Gen, idx: u32) Error!SVal {
        const c = self.c;
        const d = c.data;
        const rid = d.rule(idx);
        const action: grammar_mod.Action = if (rid < d.grammar.actions.len) d.grammar.actions[rid] else .none;
        switch (action) {
            .none, .class => return .{ .node = idx },
            .str, .int, .float, .unquote => {
                // zgram's own conversion, as the reference mode
                const zn = py.c.PyObject_CallMethod(c.lang.tree, "node", "I", @as(c_uint, idx)) orelse return error.Python;
                defer py.Py_DecRef(zn);
                const o = py.c.PyObject_CallMethod(zn, "to_ast", null) orelse return error.Python;
                defer py.Py_DecRef(o);
                // (an int of the program: an I64)
                return self.checkedAt(try self.constant(o, idx), idx);
            },
            .true_ => return .{ .bool = true },
            .false_ => return .{ .bool = false },
            .none_ => return .none,
            .drop => return .none,
            .list, .tuple, .first, .dict => {
                const l = try self.a().create(SList);
                l.* = .{};
                var ch = idx + 1;
                const stop = d.end(idx);
                while (ch < stop) : (ch = d.end(ch)) {
                    const crid = d.rule(ch);
                    if (crid < d.grammar.actions.len and d.grammar.actions[crid] == .drop) continue;
                    try l.items.append(self.a(), try self.nodeValue(ch));
                }
                if (action == .tuple) return .{ .tuple = l.items.items };
                if (action == .first) return if (l.items.items.len > 0) l.items.items[0] else SVal.none;
                if (action == .dict) return c.unsupported("a -> dict field isn't supported in compiled code yet (node {d})", .{idx});
                return .{ .list = l };
            },
        }
    }

    /// A value where the reference mode makes ints I64 (what rt gives and
    /// takes): a plain int made an `int`; one beyond 64 bits an overflow
    /// error there, as I64() raises it.
    fn checkedAt(self: *Gen, v: SVal, at: u32) Error!SVal {
        switch (v) {
            .pint => |n| return .{ .int = n },
            .dyn => |d| return .{ .dyn = self.checkedDyn(d) },
            .py => |o| if (ph.typeOf(o) == @as(*py.c.PyTypeObject, @ptrCast(py.types.typeObject("PyLong_Type")))) {
                const where = if (self.insts.items.len > 0) self.insts.items[self.insts.items.len - 1].node else at;
                try self.failAt(where, "integer overflow");
                return .none;
            },
            else => {},
        }
        return v;
    }

    /// A module-level table the module only reads (frozenGlobals), known:
    /// a dict or list of its items as constants (a tuple: as constant()
    /// makes it).
    fn frozenTable(self: *Gen, o: *PyObject, at: u32) Error!SVal {
        const t = ph.typeOf(o);
        if (t == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyDict_Type"))))) {
            const d = try self.a().create(SDict);
            d.* = .{ .frozen = o };
            var pos: py.Py_ssize_t = 0;
            var key: ?*PyObject = null;
            var v: ?*PyObject = null;
            while (py.c.PyDict_Next(o, &pos, @ptrCast(&key), @ptrCast(&v)) != 0) {
                try d.keys.append(self.a(), try self.constant(key.?, at));
                try d.values.append(self.a(), try self.constant(v.?, at));
            }
            return .{ .dict = d };
        }
        if (t == @as(*py.c.PyTypeObject, @ptrCast(@alignCast(py.types.typeObject("PyList_Type"))))) {
            const l = try self.a().create(SList);
            l.* = .{ .frozen = o };
            const n: usize = @intCast(py.c.PyList_Size(o));
            for (0..n) |i| try l.items.append(self.a(), try self.constant(py.c.PyList_GetItem(o, @intCast(i)).?, at));
            return .{ .list = l };
        }
        return self.constant(o, at);
    }

    /// A run-time value as rt hands it over (a plain int made an I64: its
    /// tag).
    fn checkedDyn(self: *Gen, d: Dyn) Dyn {
        if (d.shape != .int and d.shape != .any) return d;
        const f = &self.f;
        const plain = f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intCast(value.PINT_TAG)));
        var out = d;
        out.tag = f.select(plain, self.k(2), d.tag);
        return out;
    }

    /// A Python constant as a value known here.
    fn constant(self: *Gen, o: *PyObject, at: u32) Error!SVal {
        if (o == py.Py_None()) return .none;
        if (o == py.Py_True() or o == py.Py_False()) return .{ .bool = o == py.Py_True() };
        const t: *PyObject = @ptrCast(@alignCast(ph.typeOf(o)));
        const is_i64 = t == types_mod.I64;
        // (an int: a plain one, or an I64, an int of the program; a subclass,
        // an IntEnum..., is itself)
        if (is_i64 or t == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject("PyLong_Type"))))) {
            var overflow: c_int = 0;
            const n = py.c.PyLong_AsLongLongAndOverflow(o, &overflow);
            // (beyond 64 bits: a big int, a Python object; an I64 never is)
            if (overflow != 0) {
                // (kept: it may be one just computed)
                _ = try self.c.objectIndex(o);
                return .{ .py = o };
            }
            return if (is_i64) .{ .int = n } else .{ .pint = n };
        }
        if (py.PyFloat_Check(o)) return .{ .float = py.c.PyFloat_AsDouble(o) };
        if (py.PyUnicode_Check(o)) {
            const s = ph.utf8(o, "str") orelse return error.Python;
            return .{ .str = try self.a().dupe(u8, s) };
        }
        if (py.PyTuple_Check(o)) {
            const n: usize = @intCast(py.c.PyTuple_Size(o));
            const items = try self.a().alloc(SVal, n);
            for (items, 0..) |*it, i| it.* = try self.constant(py.c.PyTuple_GetItem(o, @intCast(i)).?, at);
            return .{ .tuple = items };
        }
        // (kept: it may be one just made (a str method's bytes...), the
        // code refers to it)
        _ = try self.c.objectIndex(o);
        return .{ .py = o };
    }

    /// A field of a node: the labelled child's value, a list of them for a
    /// label that repeats, None if absent.
    fn fieldValue(self: *Gen, idx: u32, field: u8) Error!SVal {
        const d = self.c.data;
        const label = d.grammar.labelOf(d.rule(idx), field);
        const many = label != null and label.?.many;
        const l = try self.a().create(SList);
        l.* = .{};
        var ch = idx + 1;
        const stop = d.end(idx);
        while (ch < stop) : (ch = d.end(ch)) {
            if (d.nodes[ch].fieldId() != field) continue;
            const crid = d.rule(ch);
            if (crid < d.grammar.actions.len and d.grammar.actions[crid] == .drop) continue;
            try l.items.append(self.a(), try self.nodeValue(ch));
        }
        if (many or l.items.items.len > 1) return .{ .list = l };
        if (l.items.items.len == 1) return l.items.items[0];
        return .none;
    }

    /// The children's values (`node.children`).
    fn childValues(self: *Gen, idx: u32) Error!SVal {
        const d = self.c.data;
        const l = try self.a().create(SList);
        l.* = .{};
        var ch = idx + 1;
        const stop = d.end(idx);
        while (ch < stop) : (ch = d.end(ch)) {
            const crid = d.rule(ch);
            if (crid < d.grammar.actions.len and d.grammar.actions[crid] == .drop) continue;
            try l.items.append(self.a(), try self.nodeValue(ch));
        }
        return .{ .list = l };
    }

    /// The semantic of a node, read by the front (null: none registered).
    fn semanticOf(self: *Gen, idx: u32, which: Which) Error!Semantic {
        return self.c.semanticOf(idx, which);
    }

    /// rt.eval of a node (a block scope with frames: in a new one).
    fn evalNode(self: *Gen, idx: u32) Error!SVal {
        if (!self.c.data.hasFrame(idx)) return self.typedValue(idx, try self.evalHere(idx));
        try self.enterScope(idx);
        const v = try self.evalHere(idx);
        try self.leaveScope();
        return self.typedValue(idx, v);
    }

    /// A node's value of a type whose values' kind is declared
    /// (Language.types()): its kind checked here, once (anything else: the
    /// error the reference mode raises too), known from then on.
    fn typedValue(self: *Gen, idx: u32, v: SVal) Error!SVal {
        if (v == .dyn and v.dyn.shape != .any) return v;
        const kind = try self.c.kindOfNode(idx) orelse return v;
        const f = &self.f;
        // A value known here: checked now (one not of the kind: the error,
        // when the code gets here)
        if (v != .dyn) {
            if (!staticKind(kind.shape, v)) try self.kindError(idx);
            return v;
        }
        const d = v.dyn;
        const T = value.Tag;
        const ok = switch (kind.shape) {
            // (an int of either kind)
            .int => f.or_(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(T.int))), f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intCast(value.PINT_TAG)))),
            else => f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intCast(@intFromEnum(shapeTag(kind.shape))))),
        };
        const good = try f.label("kind_ok");
        const bad = try f.label("kind_wrong");
        try f.condBr(ok, good, bad);
        try f.block(bad);
        try self.kindError(idx);
        try f.block(good);
        var out = d;
        out.shape = kind.shape;
        out.rtype = kind.rtype;
        // (a kind of one tag: that tag, known)
        if (kind.shape != .int) out.tag = self.k(@intCast(@intFromEnum(shapeTag(kind.shape))));
        return .{ .dyn = out };
    }

    /// The error of a value not of its type's kind (as the reference mode
    /// words it), at node `idx`: to the error exit.
    fn kindError(self: *Gen, idx: u32) Error!void {
        const text = py.c.PyObject_CallMethod(self.c.lang.analysis.?, "type_of", "I", @as(c_uint, idx)) orelse return error.Python;
        defer py.Py_DecRef(text);
        const msg = try std.fmt.allocPrint(self.a(), "the value isn't what types() says the type {s} is", .{ph.utf8(text, "type") orelse "?"});
        _ = self.call("zr_fail", &.{ self.ctx, self.k32(idx), try self.c.m.string(msg) });
        try self.f.br(self.err_label);
    }

    /// Whether a value known when compiling is of a kind (as the reference
    /// mode's check takes it).
    fn staticKind(shape: Shape, v: SVal) bool {
        return switch (shape) {
            .int => v == .int or v == .pint,
            .float => v == .float,
            .bool => v == .bool,
            .str => v == .str,
            .none => v == .none,
            .function => v == .py and objects_mod.asFunction(v.py) != null,
            else => false,
        };
    }

    /// The tag of a shape's values (one tag only: not int's, record's).
    fn shapeTag(s: Shape) value.Tag {
        return switch (s) {
            .none => .none,
            .bool => .bool,
            .float => .float,
            .str => .str,
            .list => .list,
            .tuple => .tuple,
            .dict => .dict,
            .function => .function,
            .node => .node,
            else => unreachable,
        };
    }

    fn evalHere(self: *Gen, idx: u32) Error!SVal {
        switch (try self.semanticOf(idx, .eval)) {
            // (its value as rt.eval gives it: an int an I64)
            .compiled => |func| return self.checkedAt(try self.runSemantic(func, idx), idx),
            .python => return self.pySemantic(idx, .eval),
            .none => {},
        }
        // Defaults: a name's variable; an only child's value
        const d = self.c.data;
        if (d.symbolIndex(idx) != null) return self.loadVar(idx);
        const kids = try self.childValues(idx);
        if (kids.list.items.items.len == 1) return self.evalValue(kids.list.items.items[0]);
        try self.failAt(idx, try std.fmt.allocPrint(self.a(), "no semantics to evaluate {s}", .{d.grammar.kind_names[d.rule(idx)]}));
        return .none;
    }

    fn evalValue(self: *Gen, v: SVal) Error!SVal {
        switch (v) {
            .node => |n| return self.evalNode(n),
            .list => |l| {
                const out = try self.a().create(SList);
                out.* = .{};
                for (l.items.items) |item| try out.items.append(self.a(), try self.evalValue(item));
                return .{ .list = out };
            },
            .dyn => return self.runValue(0, v, self.atNode()),
            else => return self.checkedAt(v, self.atNode()),
        }
    }

    /// The node errors are reported at here (the innermost semantic's).
    fn atNode(self: *const Gen) u32 {
        if (self.insts.items.len > 0) return self.insts.items[self.insts.items.len - 1].node;
        return if (self.fnode == NONE) 0 else self.fnode;
    }

    /// rt.eval / rt.exec / rt.loop (which: 0, 1, 2) of a value only known
    /// at run time (a node picked at run time...): dispatched then, to the
    /// node's compiled code (a thunk) or its Python semantic.
    fn runValue(self: *Gen, which: u32, v: SVal, at: u32) Error!SVal {
        const c = self.c;
        const f = &self.f;
        // (the code it runs sees this code's variables through the frames)
        if (!c.allHeap()) {
            c.need_frames = true;
            return c.unsupported("a node only known at run time is run here: the program needs its variables in frames", .{});
        }
        c.uses_python = true;
        const d = try self.materialize(v, at);
        const owner = self.currentOwner();
        const slot = if (self.scopes.items.len > 0) self.scopes.items[self.scopes.items.len - 1].slot else blk: {
            const s = try f.alloca(c.m.t.ptr);
            f.store(try self.currentFrame(), s);
            break :blk s;
        };
        const status = self.call("zr_run_value", &.{ self.ctx, self.k32(which), self.k32(at), d.tag, d.bits, slot, self.k32(owner), self.out });
        try self.drop(.{ .dyn = d });
        try self.statusJumps(status, at);
        return .{ .dyn = try self.loadOut(if (which == 2) .bool else .any) };
    }

    /// rt.exec of a node (a block scope with frames: in a new one).
    fn execNode(self: *Gen, idx: u32) Error!void {
        if (!self.c.data.hasFrame(idx)) return self.execHere(idx);
        try self.enterScope(idx);
        try self.execHere(idx);
        try self.leaveScope();
    }

    fn execHere(self: *Gen, idx: u32) Error!void {
        const c = self.c;
        switch (try self.semanticOf(idx, .exec)) {
            .compiled => |func| {
                try self.drop(try self.runSemantic(func, idx));
                return;
            },
            .python => {
                try self.drop(try self.pySemantic(idx, .exec));
                return;
            },
            .none => {},
        }
        // Defaults: a function definition (unless hoisted), an expression
        // for its effect, the children in order
        if (c.specOf(idx)) |spec| {
            if (spec.hoist) return;
            const name_node = program_mod.labelled(c.data, idx, spec.name) orelse return;
            try self.storeVar(name_node, try self.makeFunction(idx));
            return;
        }
        switch (try self.semanticOf(idx, .eval)) {
            .compiled => |func| {
                try self.drop(try self.runSemantic(func, idx));
                return;
            },
            .python => {
                try self.drop(try self.pySemantic(idx, .eval));
                return;
            },
            .none => {},
        }
        try self.execValue(try self.childValues(idx));
    }

    fn execValue(self: *Gen, v: SVal) Error!void {
        switch (v) {
            .node => |n| try self.execNode(n),
            .list => |l| for (l.items.items) |item| try self.execValue(item),
            .tuple => |t| for (t) |item| try self.execValue(item),
            .dyn => try self.drop(try self.runValue(1, v, self.atNode())),
            else => {},
        }
    }

    // ------------------------------------------------------------------
    // Running a semantic inline
    // ------------------------------------------------------------------

    /// Run a semantic for a node (its params: the node, rt); its result. If
    /// it can't be compiled, the compiler learns which semantic (the
    /// innermost) to run as Python instead.
    fn runSemantic(self: *Gen, func: *const front.Function, idx: u32) Error!SVal {
        return self.runFunction(func, idx, &.{ .{ .node = idx }, .rt }) catch |e| {
            if (e == error.Unsupported and self.c.failed_semantic == null) self.c.failed_semantic = func.py_function;
            return e;
        };
    }

    /// A semantic run as Python, for a node: called through the bridge
    /// with an rt over this code's frames; what it did (zr_py_semantic's
    /// status) handled as the compiled code would: a value, an error, a
    /// Return, a Break, a Continue.
    fn pySemantic(self: *Gen, idx: u32, which: Which) Error!SVal {
        const c = self.c;
        const f = &self.f;
        c.uses_python = true;
        // (the frame the code here runs in, in a slot rt.fresh can replace)
        const owner = self.currentOwner();
        const slot = if (self.scopes.items.len > 0) self.scopes.items[self.scopes.items.len - 1].slot else blk: {
            const s = try f.alloca(c.m.t.ptr);
            f.store(try self.currentFrame(), s);
            break :blk s;
        };
        const status = self.call("zr_py_semantic", &.{ self.ctx, self.k32(@intFromEnum(which)), self.k32(idx), slot, self.k32(owner), self.out });
        try self.statusJumps(status, idx);
        return .{ .dyn = try self.loadOut(.any) };
    }

    /// The node whose frame the code here runs in: the innermost block
    /// scope being run, or the function (NONE: the program).
    fn currentOwner(self: *const Gen) u32 {
        if (self.scopes.items.len > 0) return self.scopes.items[self.scopes.items.len - 1].scope;
        if (self.detached) return OWNER_PARAM;
        return self.fnode;
    }

    /// After a call reporting a status (0 error, 1 done, 2 Return with the
    /// value in `out`, 3 Break, 4 Continue): each to where it goes.
    fn statusJumps(self: *Gen, status: ir.Value, at: u32) Error!void {
        const f = &self.f;
        const done = try f.label("done");
        const not_done = try f.label("not_done");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, status, self.c.m.k32(1)), done, not_done);
        try f.block(not_done);
        const err = try f.label("raised_error");
        const control = try f.label("control");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, status, self.c.m.k32(0)), err, control);
        try f.block(err);
        try f.br(self.err_label);
        try f.block(control);
        const is_return = try f.label("raised_return");
        const loop_ctl = try f.label("raised_loop");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, status, self.c.m.k32(2)), is_return, loop_ctl);
        try f.block(is_return);
        try self.returnWith(try self.loadOut(.any), at);
        try f.block(loop_ctl);
        const is_break = try f.label("raised_break");
        const is_cont = try f.label("raised_continue");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, status, self.c.m.k32(3)), is_break, is_cont);
        try f.block(is_break);
        try self.loopJump(.Break, at);
        try f.block(is_cont);
        try self.loopJump(.Continue, at);
        try f.block(done);
    }

    /// Leave by a Return with a value (an owned reference): the function's
    /// result; out of a thunk, its status.
    /// Leaving the try statements above `stop` by a jump (`ctl`: rt.Return,
    /// rt.Break, rt.Continue; null: a return, break or continue of the
    /// semantic's): each one's finally on the way, innermost first; or a
    /// handler catching it (as Python's `except` catches those): to it,
    /// true.
    fn leaveTries(self: *Gen, stop: usize, ctl: ?RtMethod, ret: ?Dyn) Error!bool {
        var i = self.tries.items.len;
        while (i > stop) {
            i -= 1;
            const fr = self.tries.items[i];
            if (ctl) |kind| if (fr.catches(kind)) |h| {
                // (the exception Python would have raised, for `as e`:
                // rt.Return(value), rt.Break(), rt.Continue(); borrowing
                // the value)
                const cls = controlClass(kind).?;
                const n: usize = if (kind == .Return) 1 else 0;
                const arr = try self.valueSlots(1);
                if (n == 1) try self.storeSlot(self.elem(arr, 0), ret orelse self.noneDyn());
                try self.callCheck("zr_call_python", &.{ self.ctx, self.k32(self.atNode()), self.k(@intCast(try self.c.objectIndex(cls))), arr, self.k(@intCast(n)), self.out });
                const exc = try self.loadOut(.any);
                try self.releaseAbove(fr.depth);
                self.releaseScopesAbove(fr.scope_depth);
                try self.dropTemp(fr.caught[h]);
                try self.storeSlot(fr.caught[h], exc);
                try self.f.br(fr.handler_blocks[h]);
                try self.f.block(try self.f.label("after_jump"));
                return true;
            };
            if (fr.finally.len > 0) try self.runFinally(i);
        }
        return false;
    }

    /// A try's finally where code leaves it by a jump: compiled there, the
    /// try (and those in it) not around it, its errors going on out.
    fn runFinally(self: *Gen, i: usize) Error!void {
        const fr = self.tries.items[i];
        const saved = try self.a().dupe(*TryFrame, self.tries.items);
        const saved_err = self.err_label;
        self.tries.shrinkRetainingCapacity(i);
        self.err_label = fr.outer_err;
        try self.stmts(fr.inst, fr.finally);
        self.err_label = saved_err;
        self.tries.clearRetainingCapacity();
        try self.tries.appendSlice(self.a(), saved);
    }

    fn returnWith(self: *Gen, d: Dyn, at: u32) Error!void {
        // (through the try statements here: caught, or their finally run)
        if (try self.leaveTries(0, .Return, d)) {
            try self.drop(.{ .dyn = d });
            return;
        }
        if (self.thunk) {
            try self.releaseAbove(0);
            self.releaseScopesAbove(self.base_scopes);
            try self.storeSlot(self.out_param, d);
            try self.f.ret(self.c.m.k32(2));
            return;
        }
        if (self.fnode == NONE) {
            try self.drop(.{ .dyn = d });
            try self.failAt(at, "return outside a function");
            return;
        }
        try self.releaseAbove(0);
        self.releaseScopesAbove(0);
        try self.storeSlot(self.result, d);
        try self.f.br(self.ret_label);
    }

    /// Leave by a Break or a Continue: to the rt.loop running here; out of a
    /// thunk, its status; else an error.
    fn loopJump(self: *Gen, kind: RtMethod, at: u32) Error!void {
        // (through the try statements inside the loop: caught, or their
        // finally run)
        const stop = if (self.loops.items.len > 0) self.loops.items[self.loops.items.len - 1].tries else 0;
        if (try self.leaveTries(stop, kind, null)) return;
        if (self.loops.items.len == 0) {
            if (self.thunk) {
                try self.releaseAbove(0);
                self.releaseScopesAbove(self.base_scopes);
                try self.f.ret(self.c.m.k32(if (kind == .Break) 3 else 4));
                return;
            }
            try self.failAt(at, "break or continue outside a loop");
            return;
        }
        const target = self.loops.items[self.loops.items.len - 1];
        try self.releaseAboveLoop(target);
        try self.f.br(if (kind == .Break) target.brk else target.cont);
    }

    /// Call a helper: inline if it's small; out of line (its code for these
    /// arguments, shared by the calls like it) if it's big, or recursive
    /// (being run inline already). (Inline everywhere, big helpers calling
    /// big helpers would make code without end.)
    fn callHelper(self: *Gen, func: *const front.Function, at: u32, given: []const SVal) Error!SVal {
        const args = try self.withDefaults(func, given);
        for (self.insts.items) |i| if (i.func == func) return self.outOfLine(func, at, args);
        // (out of line needs the variables in frames: without them, big
        // helpers stay inline)
        if (func.size > inline_size and self.c.allHeap()) {
            if (try self.foldCall(func, args)) |v| return v;
            return self.outOfLine(func, at, args);
        }
        return self.runFunction(func, at, args);
    }

    /// A pure helper (pureFunction) given constants (a literal's text...):
    /// its result now, Python running it once while compiling (an
    /// immutable one; null: not this call, it runs when the code does (its
    /// error too)).
    fn foldCall(self: *Gen, func: *const front.Function, args: []const SVal) Error!?SVal {
        for (args) |x| if (!allConstant(x)) return null;
        if (!try pureFunction(func.py_function)) return null;
        const tuple = py.c.PyTuple_New(@intCast(args.len)) orelse return error.Python;
        defer py.Py_DecRef(tuple);
        for (args, 0..) |x, i| _ = py.c.PyTuple_SetItem(tuple, @intCast(i), try self.pyOf(x));
        const r = py.c.PyObject_CallObject(func.py_function, tuple) orelse {
            py.c.PyErr_Clear();
            return null;
        };
        defer py.Py_DecRef(r);
        const v = try self.constant(r, self.atNode());
        if (!allConstant(v)) return null;
        return v;
    }

    /// A constant all through: a str, a number, None, a bool, a tuple of
    /// them.
    fn allConstant(x: SVal) bool {
        return switch (x) {
            .none, .bool, .int, .pint, .float, .str => true,
            .tuple => |t| for (t) |item| {
                if (!allConstant(item)) break false;
            } else true,
            else => false,
        };
    }

    /// A value given by pointer (an owned reference), `default` where the
    /// pointer is null.
    fn loadGiven(self: *Gen, ptr: ir.Value, default: SVal) Error!Dyn {
        const f = &self.f;
        const slot = try self.valSlot();
        const given = try f.label("given");
        const none = try f.label("not_given");
        const join = try f.label("got");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, f.ptrToInt(ptr), self.k(0)), none, given);
        try f.block(given);
        const v = try self.loadSlot(ptr, .any);
        try self.increfDyn(v);
        try self.storeSlot(slot, v);
        try f.br(join);
        try f.block(none);
        try self.storeSlot(slot, try self.materialize(default, AT_PARAM));
        try f.br(join);
        try f.block(join);
        return self.loadSlot(slot, .any);
    }

    /// A call's arguments, those not given (the last ones) its defaults
    /// (the function's __defaults__: made when it was defined, the same
    /// objects each call, as Python's).
    fn withDefaults(self: *Gen, func: *const front.Function, args: []const SVal) Error![]const SVal {
        if (args.len == func.param_count) return args;
        if (args.len < func.required or args.len > func.param_count)
            return self.c.unsupportedAt(func, .{ .line = func.first_line }, "{s}() called with {d} arguments, takes {d} to {d}", .{ func.name, args.len, func.required, func.param_count });
        const all = try self.a().alloc(SVal, func.param_count);
        @memcpy(all[0..args.len], args);
        for (args.len..func.param_count) |i| all[i] = try self.defaultOf(func, i);
        return all;
    }

    /// The default value of parameter `i` (one of the last ones).
    fn defaultOf(self: *Gen, func: *const front.Function, i: usize) Error!SVal {
        const defaults = ph.attr(func.py_function, "__defaults__") orelse return error.Python;
        defer py.Py_DecRef(defaults);
        if (defaults == py.Py_None()) return self.c.unsupportedAt(func, .{ .line = func.first_line }, "{s}()'s defaults changed", .{func.name});
        const n: usize = @intCast(py.c.PyTuple_Size(defaults));
        const first = func.param_count - n;
        if (i < first) return self.c.unsupportedAt(func, .{ .line = func.first_line }, "{s}() missing an argument", .{func.name});
        const o = py.c.PyTuple_GetItem(defaults, @intCast(i - first)).?;
        // (kept: the function's defaults may change)
        _ = try self.c.objectIndex(o);
        return self.constant(o, self.atNode());
    }

    /// The biggest helper run inline (expressions)
    const inline_size = 40;
    /// The longest range() known when compiling that's unrolled
    const max_unrolled = 16;
    /// The most iterations of a while loop with a known test unrolled
    const max_unrolled_while = 8;
    /// The most while loops unrolled in one another
    const max_unrolled_nesting = 4;

    /// While loops (their tests) found to go on past max_unrolled_while
    /// iterations with their test known: not unrolled again (the front's
    /// functions live as long as their language)
    var long_loops: std.AutoHashMapUnmanaged(*const front.Expr, void) = .empty;

    /// A loop body worth unrolling: a few statements, no loops in it.
    fn smallBody(body: []const front.Stmt) bool {
        var n: usize = 0;
        return countSmall(body, &n) and n <= 12;
    }

    fn countSmall(body: []const front.Stmt, n: *usize) bool {
        for (body) |s| {
            n.* += 1;
            switch (s.kind) {
                .while_, .for_, .try_ => return false,
                .if_ => |i| if (!countSmall(i.body, n) or !countSmall(i.else_, n)) return false,
                else => {},
            }
        }
        return true;
    }

    /// A call of a helper's code out of line (made for the arguments known
    /// here, the first time: HelperSpec); the arguments are taken.
    fn outOfLine(self: *Gen, func: *const front.Function, at: u32, args: []const SVal) Error!SVal {
        const c = self.c;
        // (its code sees the variables here through the frames)
        if (!c.allHeap()) {
            c.need_frames = true;
            return c.unsupported("a recursive helper is called here: the program needs its variables in frames", .{});
        }
        if (args.len != func.param_count) return c.unsupportedAt(func, .{ .line = func.first_line }, "called with {d} arguments, takes {d}", .{ args.len, func.param_count });
        // What it's made for: rt, Python objects, None, bools (its code
        // depends on them most, and they're few); nodes, strs and values
        // are given (one version serves every call like it)
        const key = try self.a().alloc(SVal, args.len);
        var given: usize = 0;
        for (args, key) |x, *slot| {
            slot.* = switch (x) {
                .rt, .py, .none, .bool => x,
                .rt_method, .control, .method => return c.unsupportedAt(func, .{ .line = func.first_line }, "a {s} can't be given to a helper compiled out of line", .{@tagName(x)}),
                else => .{ .dyn = undefined },
            };
            if (slot.* == .dyn) given += 1;
        }
        const spec = for (c.helper_fns.items) |h| {
            if (h.matches(func, key)) break h;
        } else blk: {
            const h = try c.a.create(HelperSpec);
            // (the semantic: the outermost one run here, or the one this
            // helper's code runs for)
            const semantic = self.helper_semantic orelse if (self.insts.items.len > 0) self.insts.items[0].func.py_function else null;
            h.* = .{ .func = func, .args = key, .name = try std.fmt.allocPrintSentinel(c.a, "{s}_h{d}", .{ c.m.prefix, c.helper_fns.items.len }, 0), .semantic = semantic };
            try c.helper_fns.append(c.a, h);
            try c.helper_queue.append(c.a, h);
            break :blk h;
        };
        // The arguments given, in an array (borrowed by it)
        const arr = try self.valueSlots(@max(given, 1));
        const ds = try self.a().alloc(Dyn, given);
        var j: usize = 0;
        for (args, key) |x, slot| {
            if (slot != .dyn) continue;
            ds[j] = try self.materialize(x, at);
            try self.storeSlot(self.elem(arr, j), ds[j]);
            j += 1;
        }
        // (the frames here, their scope; the receiver and varargs here)
        const null_ptr = c.m.nullPtr();
        const fn_here = !self.detached and self.fnode != NONE;
        const recv = if (self.detached or fn_here) self.recv_slot else null_ptr;
        const varargs = if (self.detached or (fn_here and c.specOf(self.fnode).?.extra == .keep)) self.varargs_slot else null_ptr;
        const fun = try c.helperFn(spec.name);
        const call_args = [_]ir.Value{ self.ctx, try self.currentFrame(), arr, self.k32(at), self.k32(self.currentOwner()), recv, varargs, self.out };
        const status = if (try self.siteOf(func, args, key, spec.semantic)) |site|
            try self.siteCall(site, fun, &call_args)
        else
            self.f.call(fun, &call_args);
        for (ds) |d| try self.drop(.{ .dyn = d });
        try self.statusJumps(status, at);
        return .{ .dyn = try self.loadOut(.any) };
    }

    /// A call site that knows more of a helper's arguments than its generic
    /// code (`key`) does: its Site (a number in the compiler's), or null.
    fn siteOf(self: *Gen, func: *const front.Function, args: []const SVal, key: []const SVal, semantic: ?*PyObject) Error!?usize {
        const c = self.c;
        const full = try c.a.alloc(SVal, args.len);
        const layout = try c.a.alloc(bool, args.len);
        var more = false;
        for (args, key, full, layout, 0..) |x, generic, *slot, *given, i| {
            given.* = generic == .dyn;
            slot.* = generic;
            if (generic != .dyn) continue;
            if (specializable(x)) {
                slot.* = x;
            } else if (x == .list) {
                // (a list of known items, to a parameter the helper only
                // reads: the items as they are now, what the call gives)
                for (x.list.items.items) |item| {
                    if (!specializable(item)) break;
                } else if (try readonlyParam(func.py_function, i)) {
                    const l = try c.a.create(SList);
                    l.* = .{};
                    try l.items.appendSlice(c.a, x.list.items.items);
                    slot.* = .{ .list = l };
                }
            }
            if (slot.* != .dyn) more = true;
        }
        if (!more) return null;
        const counts = try c.a.create(Site.Hot);
        counts.* = .{};
        const site = try c.a.create(Site);
        site.* = .{ .hot = counts, .func = func, .args = full, .layout = layout, .semantic = semantic };
        try c.sites.append(c.a, site);
        return c.sites.items.len - 1;
    }

    /// What a helper can be compiled for when a site knows it: a node, a
    /// str, a number, a tuple of those (constant: what a call site knows
    /// stays so).
    fn specializable(x: SVal) bool {
        return switch (x) {
            .node, .str, .int, .pint, .float => true,
            .tuple => |t| for (t) |item| {
                if (!specializable(item)) break false;
            } else true,
            else => false,
        };
    }

    /// The call of a site: its specialized code if it has some, else the
    /// generic code, counted (zr_specialize once it's hot). Its status.
    fn siteCall(self: *Gen, site: usize, generic: ir.Fn, args: []const ir.Value) Error!ir.Value {
        const f = &self.f;
        const t = self.c.m.t;
        const counts = self.c.m.ptrConst(@intFromPtr(self.c.sites.items[site].hot));
        const count_ptr = f.offset(counts, @offsetOf(Site.Hot, "count"));
        const special = try f.label("site_special");
        const counted = try f.label("site_generic");
        const now_hot = try f.label("site_hot");
        const call_generic = try f.label("site_call");
        const join = try f.label("site_done");
        const code = f.load(t.i64, counts);
        try f.condBr(f.icmp(jit_c.LLVMIntNE, code, self.k(0)), special, counted);
        try f.block(counted);
        const n = f.add(f.load(t.i64, count_ptr), self.k(1));
        f.store(n, count_ptr);
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, n, self.k(self.c.lang.hot_calls)), now_hot, call_generic);
        try f.block(now_hot);
        _ = self.call("zr_specialize", &.{ self.ctx, self.k(@intCast(site)) });
        try f.br(call_generic);
        try f.block(call_generic);
        const s1 = f.call(generic, args);
        const end1 = f.current;
        try f.br(join);
        try f.block(special);
        const s2 = f.call(.{ .v = f.intToPtr(code), .ty = generic.ty }, args);
        const end2 = f.current;
        try f.br(join);
        try f.block(join);
        return f.phi(t.i32, s1, end1, s2, end2);
    }

    /// Run a front function inline with arguments.
    fn runFunction(self: *Gen, func: *const front.Function, at: u32, args: []const SVal) Error!SVal {
        if (self.insts.items.len > 200) return self.c.unsupported("semantics nest more than 200 deep (recursive helpers aren't compiled yet)", .{});
        const locals = try self.a().alloc(Local, func.locals.len);
        @memset(locals, .unset);
        if (args.len != func.param_count) return self.c.unsupportedAt(func, .{ .line = func.first_line }, "called with {d} arguments, takes {d}", .{ args.len, func.param_count });
        for (args, 0..) |arg, i| locals[i] = .{ .static = arg };
        const inst = try self.a().create(Inst);
        inst.* = .{ .func = func, .node = at, .locals = locals, .exit_label = try self.f.label("ret"), .loop_level0 = self.loop_level };
        try self.insts.append(self.a(), inst);
        defer _ = self.insts.pop();

        try self.stmts(inst, func.body);
        // Falling off the end: None
        if (!inst.done) try self.setResult(inst, .none);
        // The exit: where returns from run-time control flow meet (each
        // released the locals it had)
        if (inst.result_slot) |slot| {
            try self.f.block(inst.exit_label);
            return .{ .dyn = try self.loadSlot(slot, .any) };
        }
        try self.releaseLocals(inst);
        return inst.result orelse .none;
    }

    /// The semantic's result: kept, or in its slot (and to its exit).
    fn setResult(self: *Gen, inst: *Inst, v: SVal) Error!void {
        if (inst.dyn_depth == 0 and inst.result_slot == null and inst.in_try == 0) {
            inst.result = v;
            inst.done = true;
            return;
        }
        const slot = inst.result_slot orelse blk: {
            const s = try self.valSlot();
            inst.result_slot = s;
            break :blk s;
        };
        try self.storeSlot(slot, try self.materialize(v, inst.node));
        // (the finally of the try statements it's in, first)
        var stop = self.tries.items.len;
        while (stop > 0 and self.tries.items[stop - 1].inst == inst) stop -= 1;
        _ = try self.leaveTries(stop, null, null);
        // (the locals as they are on this path: a value assigned after
        // a return inside run-time control flow isn't there on the others)
        try self.releaseLocals(inst);
        try self.f.br(inst.exit_label);
        if (inst.dyn_depth == 0 and inst.in_try == 0) {
            inst.done = true;
        } else try self.f.block(try self.f.label("after_return"));
    }

    /// Drop the semantic's locals that hold run-time values.
    fn releaseLocals(self: *Gen, inst: *Inst) Error!void {
        for (inst.locals) |l| switch (l) {
            // (None again: the semantic may run again, along another path)
            .slot => |s| try self.dropTemp(s.ptr),
            .static => |sv| try self.drop(sv),
            .unset => {},
        };
        for (inst.temps.items) |slot| try self.dropTemp(slot);
    }

    /// A slot holding None from the function's start (whichever path
    /// reaches a release of it, it holds a value).
    fn noneSlot(self: *Gen) Error!ir.Value {
        const slot = try self.valSlot();
        self.f.entryStore(L("LLVMConstNull")(self.c.m.t.val), slot);
        return slot;
    }

    /// A slot for a value (in the entry block).
    fn valSlot(self: *Gen) Error!ir.Value {
        return self.f.alloca(self.c.m.t.val);
    }

    /// A temporary slot of a semantic (None until used).
    fn tempSlot(self: *Gen, inst: *Inst) Error!ir.Value {
        const slot = try self.noneSlot();
        try inst.temps.append(self.a(), slot);
        return slot;
    }

    /// Give up a temporary slot's value (None again).
    fn dropTemp(self: *Gen, slot: ir.Value) Error!void {
        try self.drop(.{ .dyn = try self.loadSlot(slot, .any) });
        try self.storeSlot(slot, self.noneDyn());
    }

    /// Release the locals of the semantics above `depth` (leaving them by a
    /// jump: rt.Return, rt.Break).
    fn releaseAbove(self: *Gen, depth: usize) Error!void {
        var i = self.insts.items.len;
        while (i > depth) {
            i -= 1;
            try self.releaseLocals(self.insts.items[i]);
        }
    }

    // ------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------

    fn stmts(self: *Gen, inst: *Inst, body: []const front.Stmt) Error!void {
        for (body) |s| {
            if (inst.done) return;
            try self.stmt(inst, s);
        }
    }

    fn stmt(self: *Gen, inst: *Inst, s: front.Stmt) Error!void {
        switch (s.kind) {
            .pass => {},
            .expr => |e| try self.drop(try self.expr(inst, e)),
            .assign => |a_| {
                const v = try self.expr(inst, a_.value);
                for (a_.targets, 0..) |t, i| {
                    // (each target after the first gets its own reference)
                    if (i > 0) if (v == .dyn) try self.increfDyn(v.dyn);
                    try self.assign(inst, t, v, s.pos);
                }
            },
            .aug => |a_| try self.augAssign(inst, a_.target, a_.op, a_.value, s.pos),
            .if_ => |i| {
                const t = try self.truth(try self.expr(inst, i.test_), inst.node);
                switch (t) {
                    .known => |b| try self.stmts(inst, if (b) i.body else i.else_),
                    .dyn => |cond| {
                        try self.prepareDynamic(inst, &.{ i.body, i.else_ });
                        const yes = try self.f.label("then");
                        const no = try self.f.label("else");
                        const join = try self.f.label("endif");
                        try self.f.condBr(cond, yes, no);
                        inst.dyn_depth += 1;
                        try self.f.block(yes);
                        try self.stmts(inst, i.body);
                        try self.f.br(join);
                        try self.f.block(no);
                        try self.stmts(inst, i.else_);
                        try self.f.br(join);
                        inst.dyn_depth -= 1;
                        try self.f.block(join);
                    },
                }
            },
            .while_ => |w| {
                // A test known now: iterations unrolled while it stays
                // known (a few); then, or from the first one it isn't
                // (its value already: into the body), a loop at run time
                const exit = try self.f.label("endwhile");
                var entry: ?ir.Value = null;
                var n: usize = 0;
                // (a small body without loops, in few others unrolled (an
                // `if` of the language in another's block...): code
                // growing a little)
                const unroll = self.unrolling < max_unrolled_nesting and smallBody(w.body) and !long_loops.contains(w.test_);
                if (unroll) self.unrolling += 1;
                defer if (unroll) {
                    self.unrolling -= 1;
                };
                while (unroll and n < max_unrolled_while) : (n += 1) {
                    switch (try self.truth(try self.expr(inst, w.test_), inst.node)) {
                        .known => |b| {
                            if (!b) {
                                try self.stmts(inst, w.else_);
                                try self.f.br(exit);
                                try self.f.block(exit);
                                return;
                            }
                            const next = try self.f.label("next");
                            try inst.loops.append(self.a(), .{ .brk = exit, .cont = next, .tries = self.tries.items.len });
                            try self.stmts(inst, w.body);
                            _ = inst.loops.pop();
                            try self.f.block(next);
                            if (inst.done) {
                                try self.f.block(exit);
                                return;
                            }
                        },
                        .dyn => |cond| {
                            entry = cond;
                            break;
                        },
                    }
                }
                // (one going on past the iterations unrolled: a loop over
                // data, not unrolled again)
                if (unroll and entry == null) try long_loops.put(std.heap.c_allocator, w.test_, {});
                try self.prepareDynamic(inst, &.{w.body});
                const head = try self.f.label("while");
                const body = try self.f.label("body");
                const els = try self.f.label("whileelse");
                if (entry) |cond| try self.f.condBr(cond, body, els) else try self.f.br(head);
                try self.f.block(head);
                inst.dyn_depth += 1;
                self.loop_level += 1;
                defer self.loop_level -= 1;
                const t = try self.truth(try self.expr(inst, w.test_), inst.node);
                switch (t) {
                    .known => |b| try self.f.br(if (b) body else els),
                    .dyn => |cond| try self.f.condBr(cond, body, els),
                }
                try self.f.block(body);
                try inst.loops.append(self.a(), .{ .brk = exit, .cont = head, .tries = self.tries.items.len });
                try self.stmts(inst, w.body);
                _ = inst.loops.pop();
                try self.f.br(head);
                try self.f.block(els);
                try self.stmts(inst, w.else_);
                inst.dyn_depth -= 1;
                try self.f.br(exit);
                try self.f.block(exit);
            },
            .for_ => |fr| try self.forLoop(inst, fr.target, fr.iter, fr.body, fr.else_, s.pos),
            .return_ => |r| {
                const v = if (r) |e| try self.expr(inst, e) else SVal.none;
                try self.setResult(inst, v);
            },
            .raise_ => |r| {
                const v = if (r) |e| try self.expr(inst, e) else blk: {
                    // (bare: the exception being handled, again)
                    if (self.caught.items.len == 0) return self.c.unsupportedAt(inst.func, s.pos, "a bare `raise` outside an except can't be compiled", .{});
                    const d = try self.loadSlot(self.caught.items[self.caught.items.len - 1], .any);
                    try self.increfDyn(d);
                    break :blk SVal{ .dyn = d };
                };
                try self.raise(inst, v, s.pos);
            },
            .try_ => |t| try self.tryStmt(inst, t),
            .assert_ => |as| {
                const t = try self.truth(try self.expr(inst, as.test_), inst.node);
                const msg: []const u8 = if (as.msg) |m| blk: {
                    const mv = try self.expr(inst, m);
                    break :blk if (mv == .str) mv.str else return self.c.unsupportedAt(inst.func, s.pos, "an assert's message must be a string known when compiling", .{});
                } else "";
                const text = try std.fmt.allocPrint(self.a(), "AssertionError: {s}", .{msg});
                switch (t) {
                    .known => |b| if (!b) try self.failAt(inst.node, text),
                    .dyn => |cond| {
                        const bad = try self.f.label("assert_failed");
                        const good = try self.f.label("assert_ok");
                        try self.f.condBr(cond, good, bad);
                        try self.f.block(bad);
                        try self.failAt(inst.node, text);
                        try self.f.block(good);
                    },
                }
            },
            .break_, .continue_ => {
                if (inst.loops.items.len == 0) return self.c.unsupportedAt(inst.func, s.pos, "this break or continue can't be compiled", .{});
                const target = inst.loops.items[inst.loops.items.len - 1];
                _ = try self.leaveTries(target.tries, null, null);
                try self.f.br(if (s.kind == .break_) target.brk else target.cont);
                try self.f.block(try self.f.label("after_jump"));
            },
        }
    }

    /// try / except / else / finally. Errors in the body go to the handlers
    /// (each matched at run time against its classes: zr_exc_matches); a
    /// jump out (return, rt.Return...) runs the finally on the way, or is
    /// caught by a handler whose class covers it; the finally runs on every
    /// way out. (What the semantics run inside the body held when an error
    /// left them isn't released: a caught error leaks those.)
    fn tryStmt(self: *Gen, inst: *Inst, t: @FieldType(front.Stmt.Kind, "try_")) Error!void {
        const c = self.c;
        const f = &self.f;
        // (what the handlers, else and finally assign: slots, as every path
        // sees them; what the body assigns, if read where an error in the
        // body may land (anywhere but the body itself): a loop's known
        // items in the body stay known)
        const parts = try self.a().alloc([]const front.Stmt, t.handlers.len + 2);
        parts[0] = t.else_;
        parts[1] = t.finally;
        for (t.handlers, parts[2..]) |h, *p| p.* = h.body;
        try self.prepareDynamic(inst, parts);
        for (t.handlers) |h| if (h.name) |slot| try self.toSlot(inst, slot);
        var assigned = std.AutoHashMapUnmanaged(u32, void).empty;
        collectAssigned(t.body, &assigned, self.a()) catch return error.OutOfMemory;
        var read = std.AutoHashMapUnmanaged(u32, void).empty;
        collectReads(inst.func.body, t.body, &read, self.a()) catch return error.OutOfMemory;
        var it = assigned.keyIterator();
        while (it.next()) |slot| if (read.contains(slot.*)) try self.toSlot(inst, slot.*);

        const fr = try self.a().create(TryFrame);
        fr.* = .{
            .inst = inst,
            .depth = self.insts.items.len,
            .scope_depth = self.scopes.items.len,
            .finally = t.finally,
            .outer_err = self.err_label,
            .handler_blocks = try self.a().alloc(ir.Block, t.handlers.len),
            .caught = try self.a().alloc(ir.Value, t.handlers.len),
        };
        // The handlers' classes (known when compiling: an object index; null
        // for a bare except), and the jumps they catch
        const classes = try self.a().alloc(?usize, t.handlers.len);
        const types_ = @import("types.zig");
        for (t.handlers, 0..) |h, i| {
            fr.handler_blocks[i] = try f.label("handler");
            fr.caught[i] = try self.noneSlot();
            classes[i] = null;
            const te = h.type_ orelse {
                if (fr.catch_return == null) fr.catch_return = i;
                if (fr.catch_break == null) fr.catch_break = i;
                if (fr.catch_continue == null) fr.catch_continue = i;
                continue;
            };
            const cls: *PyObject = switch (try self.expr(inst, te)) {
                .py => |o| o,
                .rt_method => |m| controlClass(m) orelse return c.unsupportedAt(inst.func, h.pos, "an except's classes must be known when compiling", .{}),
                .tuple => |items| blk: {
                    const tuple = py.c.PyTuple_New(@intCast(items.len)) orelse return error.Python;
                    for (items, 0..) |x, j| {
                        const o = switch (x) {
                            .py => |o| o,
                            .rt_method => |m| controlClass(m),
                            else => null,
                        } orelse return c.unsupportedAt(inst.func, h.pos, "an except's classes must be known when compiling", .{});
                        py.Py_IncRef(o);
                        _ = py.c.PyTuple_SetItem(tuple, @intCast(j), o);
                    }
                    _ = try c.objectIndex(tuple);
                    py.Py_DecRef(tuple);
                    break :blk tuple;
                },
                else => return c.unsupportedAt(inst.func, h.pos, "an except's classes must be known when compiling", .{}),
            };
            classes[i] = try c.objectIndex(cls);
            // (rt.Return, rt.Break, rt.Continue are exceptions in Python)
            if (fr.catch_return == null and py.c.PyObject_IsSubclass(types_.Return, cls) == 1) fr.catch_return = i;
            if (fr.catch_break == null and py.c.PyObject_IsSubclass(types_.Break, cls) == 1) fr.catch_break = i;
            if (fr.catch_continue == null and py.c.PyObject_IsSubclass(types_.Continue, cls) == 1) fr.catch_continue = i;
            if (py.c.PyErr_Occurred() != null) return error.Python;
        }
        const catcher = try f.label("except");
        const handled = try f.label("handled");
        const done = try f.label("endtry");
        // (errors in the handlers and the else: the finally, then on)
        const fin_err = if (t.finally.len > 0) try f.label("finally_error") else fr.outer_err;

        // The body (run straight through): its errors to the handlers
        try self.tries.append(self.a(), fr);
        self.err_label = catcher;
        inst.in_try += 1;
        try self.stmts(inst, t.body);
        inst.in_try -= 1;
        // (a jump in it: the rest of it dead, not what follows it)
        inst.done = false;
        fr.catching = false;
        // The rest: on some paths only
        inst.dyn_depth += 1;
        defer inst.dyn_depth -= 1;
        self.err_label = fin_err;
        try self.stmts(inst, t.else_);
        try f.br(handled);

        // An error: the first handler matching it
        try f.block(catcher);
        for (t.handlers, 0..) |_, i| {
            const next = try f.label("next_handler");
            if (classes[i]) |idx| {
                const yes = try f.label("matched");
                try f.condBr(self.call("zr_exc_matches", &.{ self.ctx, self.k(@intCast(idx)) }), yes, next);
                try f.block(yes);
            }
            try self.callCheck("zr_exc_catch", &.{ self.ctx, self.k32(inst.node), self.out });
            try self.storeSlot(fr.caught[i], try self.loadOut(.any));
            try f.br(fr.handler_blocks[i]);
            try f.block(next);
        }
        // (none: the error goes on, after the finally)
        try f.br(fin_err);

        // The handlers (reached from an error, or a jump they catch)
        for (t.handlers, 0..) |h, i| {
            try f.block(fr.handler_blocks[i]);
            if (h.name) |slot| {
                const e = try self.loadSlot(fr.caught[i], .any);
                try self.increfDyn(e);
                try self.assign(inst, .{ .local = slot }, .{ .dyn = e }, h.pos);
            }
            try self.caught.append(self.a(), fr.caught[i]);
            try self.stmts(inst, h.body);
            _ = self.caught.pop();
            try self.dropTemp(fr.caught[i]);
            try f.br(handled);
        }
        _ = self.tries.pop();

        // Every way out: the finally
        self.err_label = fr.outer_err;
        try f.block(handled);
        try self.stmts(inst, t.finally);
        try f.br(done);
        if (t.finally.len > 0) {
            try f.block(fin_err);
            try self.stmts(inst, t.finally);
            try f.br(fr.outer_err);
        }
        try f.block(done);
    }

    /// rt.Return, rt.Break, rt.Continue as the exception classes they are.
    fn controlClass(m: RtMethod) ?*PyObject {
        const types_ = @import("types.zig");
        return switch (m) {
            .Return => types_.Return,
            .Break => types_.Break,
            .Continue => types_.Continue,
            else => null,
        };
    }

    /// Before run-time control flow (an if on a run-time value, a loop): the
    /// locals its statements assign become slots, so each path sees them.
    fn prepareDynamic(self: *Gen, inst: *Inst, bodies: []const []const front.Stmt) Error!void {
        var set = std.AutoHashMapUnmanaged(u32, void).empty;
        for (bodies) |b| collectAssigned(b, &set, self.a()) catch return error.OutOfMemory;
        var it = set.keyIterator();
        while (it.next()) |slot| try self.toSlot(inst, slot.*);
    }

    /// Make a local a stack slot (its current value in it).
    fn toSlot(self: *Gen, inst: *Inst, slot: u32) Error!void {
        switch (inst.locals[slot]) {
            .slot => return,
            .unset => {
                const p = try self.noneSlot();
                inst.locals[slot] = .{ .slot = .{ .ptr = p, .shape = .any } };
            },
            .static => |v| {
                const d = try self.materialize(v, inst.node);
                const p = try self.noneSlot();
                try self.storeSlot(p, d);
                inst.locals[slot] = .{ .slot = .{ .ptr = p, .shape = .any } };
            },
        }
    }

    fn assign(self: *Gen, inst: *Inst, t: front.Target, v: SVal, pos: front.Pos) Error!void {
        switch (t) {
            .local => |slot| switch (inst.locals[slot]) {
                .slot => |s| {
                    const d = try self.materialize(v, inst.node);
                    const old = try self.loadSlot(s.ptr, s.shape);
                    try self.storeSlot(s.ptr, d);
                    try self.drop(.{ .dyn = old });
                },
                .static => |old| {
                    if (inst.dyn_depth > 0) {
                        // (prepareDynamic makes such locals slots first)
                        try self.toSlot(inst, slot);
                        return self.assign(inst, t, v, pos);
                    }
                    try self.drop(old);
                    inst.locals[slot] = .{ .static = v };
                },
                .unset => {
                    if (inst.dyn_depth > 0) {
                        try self.toSlot(inst, slot);
                        return self.assign(inst, t, v, pos);
                    }
                    inst.locals[slot] = .{ .static = v };
                },
            },
            .tuple => |ts| {
                const known: ?[]const SVal = switch (v) {
                    .tuple => |x| x,
                    .list => |l| l.items.items,
                    else => null,
                };
                if (known) |items| if (items.len == ts.len) {
                    // (a known container keeps its items)
                    for (ts, items) |x, item| try self.assign(inst, x, try self.copyOf(item), pos);
                    return;
                };
                // At run time (and the errors, as Python words them)
                const d = try self.materialize(v, inst.node);
                const arr = try self.valueSlots(ts.len);
                const ok = self.call("zr_unpack", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, self.k(@intCast(ts.len)), arr });
                try self.drop(.{ .dyn = d });
                try self.check(ok);
                for (ts, 0..) |x, i| {
                    try self.assign(inst, x, .{ .dyn = try self.loadSlot(self.elem(arr, i), .any) }, pos);
                }
            },
            .attr => |x| try self.setAttr(inst, try self.expr(inst, x.obj), x.name, v, pos),
            .index => |x| {
                const obj = try self.expr(inst, x.obj);
                const key = try self.expr(inst, x.index);
                try self.setItem(inst, obj, key, v, pos);
            },
        }
    }

    /// obj.name = v (both taken)
    fn setAttr(self: *Gen, inst: *Inst, obj: SVal, name: []const u8, v: SVal, pos: front.Pos) Error!void {
        // (a Python object the module keeps, an instance: as Python does it
        // when the code runs; a module's or class's attributes are read
        // when compiling, they can't change)
        const py_instance = obj == .py and !try stablePy(obj.py);
        if (obj != .dyn and !py_instance) {
            try self.drop(v);
            return self.c.unsupportedAt(inst.func, pos, "assigning to an attribute of a {s} isn't compiled", .{@tagName(obj)});
        }
        const vd = try self.materialize(v, inst.node);
        const d = if (py_instance) try self.materialize(obj, inst.node) else obj.dyn;
        const f = &self.f;
        const t = self.c.m.t;
        // A field of a record of the module's classes: stored where it is
        // (its type checked); anything else by zr_setattr
        const cands = try self.fieldCandidates(inst, name, true);
        const join = try f.label("setfield_done");
        const generic = try f.label("setfield_generic");
        if (cands.len > 0) {
            const typed = try f.label("setfield_typed");
            try f.condBr(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(value.Tag.record))), typed, generic);
            try f.block(typed);
            const rt = self.recordTypeOf(d);
            for (cands) |cand| {
                const yes = try f.label("setfield_of");
                const no = try f.label("setfield_next");
                try f.condBr(f.icmp(jit_c.LLVMIntEQ, rt, self.c.m.addrInt(@intFromPtr(cand.rtype))), yes, no);
                try f.block(yes);
                const p = self.fieldPtr(d, cand.index);
                const old = Dyn{ .tag = f.load(t.i64, p), .bits = f.load(t.i64, f.offset(p, 8)), .shape = .any };
                try self.increfDyn(vd);
                try self.storeSlot(p, vd);
                // (an unset slot's old "value" isn't counted: the tag)
                try self.refcount(true, old.tag, old.bits);
                try f.br(join);
                try f.block(no);
            }
            try f.br(generic);
        } else try f.br(generic);
        try f.block(generic);
        const s = try self.c.m.string(name);
        const ok = self.call("zr_setattr", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, s, vd.tag, vd.bits });
        try self.check(ok);
        try f.br(join);
        try f.block(join);
        try self.drop(.{ .dyn = vd });
        try self.drop(obj);
    }

    /// obj[key] = v (all taken): a known container changed now (outside
    /// run-time control flow, at a known key), else zr_setitem (a known one
    /// made a run-time one first: materialize).
    fn setItem(self: *Gen, inst: *Inst, obj: SVal, key: SVal, v: SVal, pos: front.Pos) Error!void {
        _ = pos;
        switch (obj) {
            .list, .dict => if (inst.dyn_depth == 0 and key.isStatic() and isScalar(key)) {
                if (obj == .dict) return self.sdictSet(obj.dict, key, v);
                const items = obj.list.items.items;
                if (intOf(key)) |ki| {
                    const n: i64 = @intCast(items.len);
                    const i = if (ki < 0) ki + n else ki;
                    if (i >= 0 and i < n) {
                        try self.drop(items[@intCast(i)]);
                        items[@intCast(i)] = v;
                        return;
                    }
                }
                // (the error, at run time as Python raises it)
            },
            else => {},
        }
        const od = try self.materializeToChange(obj, inst.node);
        const kd = try self.materialize(key, inst.node);
        const vd = try self.materialize(v, inst.node);
        const ok = self.call("zr_setitem", &.{ self.ctx, self.k32(inst.node), od.tag, od.bits, kd.tag, kd.bits, vd.tag, vd.bits });
        try self.drop(.{ .dyn = vd });
        try self.drop(.{ .dyn = kd });
        try self.drop(.{ .dyn = od });
        try self.check(ok);
    }

    /// target op= value: the target's object (and key) evaluated once.
    fn augAssign(self: *Gen, inst: *Inst, t: front.Target, op: front.BinOp, value_e: *const front.Expr, pos: front.Pos) Error!void {
        switch (t) {
            .local => |slot| {
                const cur = try self.readLocal(inst, slot, pos);
                const rhs = try self.expr(inst, value_e);
                try self.assign(inst, t, try self.binary(inst, op, cur, rhs), pos);
            },
            .attr => |x| {
                const obj = try self.expr(inst, x.obj);
                const cur = try self.attr(inst, try self.copyOf(obj), x.name, pos);
                const rhs = try self.expr(inst, value_e);
                try self.setAttr(inst, obj, x.name, try self.binary(inst, op, cur, rhs), pos);
            },
            .index => |x| {
                const obj = try self.expr(inst, x.obj);
                const key = try self.expr(inst, x.index);
                const cur = try self.getItem(inst, try self.copyOf(obj), try self.copyOf(key));
                const rhs = try self.expr(inst, value_e);
                try self.setItem(inst, obj, key, try self.binary(inst, op, cur, rhs), pos);
            },
            .tuple => return self.c.unsupportedAt(inst.func, pos, "augmented assignment to a tuple", .{}),
        }
    }

    fn readLocal(self: *Gen, inst: *Inst, slot: u32, pos: front.Pos) Error!SVal {
        switch (inst.locals[slot]) {
            .unset => return self.c.unsupportedAt(inst.func, pos, "'{s}' may be read before it is assigned", .{inst.func.locals[slot]}),
            .static => |v| return self.copyOf(v),
            .slot => |s| {
                const d = try self.loadSlot(s.ptr, s.shape);
                try self.increfDyn(d);
                return .{ .dyn = d };
            },
        }
    }

    /// raise rt.Return(v) / rt.Break() / rt.Continue(): jumps; raise of
    /// anything else (rt.Throw(...), ValueError(...)): at run time.
    fn raise(self: *Gen, inst: *Inst, v: SVal, pos: front.Pos) Error!void {
        _ = pos;
        const ctl = switch (v) {
            .control => |x| x,
            else => {
                const d = try self.materialize(v, inst.node);
                _ = self.call("zr_raise", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits });
                try self.drop(.{ .dyn = d });
                try self.f.br(self.err_label);
                if (inst.dyn_depth == 0) inst.done = true;
                try self.f.block(try self.f.label("after_raise"));
                return;
            },
        };
        switch (ctl.kind) {
            .Return => {
                if (self.fnode == NONE and !self.thunk) {
                    try self.failAt(inst.node, "return outside a function");
                    return;
                }
                try self.returnWith(try self.materialize(if (ctl.value) |x| x.* else .none, inst.node), inst.node);
            },
            .Break, .Continue => try self.loopJump(ctl.kind, inst.node),
            else => unreachable,
        }
        if (inst.dyn_depth == 0) {
            inst.done = true;
            // (jumped: whatever follows is unreachable)
            try self.f.block(try self.f.label("after_raise"));
        } else try self.f.block(try self.f.label("after_raise"));
    }

    fn releaseAboveLoop(self: *Gen, target: LoopTarget) Error!void {
        // (the semantics run inside the loop's body, and its block scopes)
        try self.releaseAbove(target.depth);
        self.releaseScopesAbove(target.scope_depth);
    }

    fn forLoop(self: *Gen, inst: *Inst, target: front.Target, iter_e: *const front.Expr, body: []const front.Stmt, else_: []const front.Stmt, pos: front.Pos) Error!void {
        const items: []const SVal = switch (try self.iteration(inst, iter_e)) {
            .known => |x| x,
            .runtime => |it| return self.runtimeFor(inst, target, it, body, else_, pos),
        };
        // Known items: unrolled (break / continue jump within it)
        const exit = try self.f.label("endfor");
        var broke = false;
        for (items) |item| {
            const next = try self.f.label("next");
            if (item == .dyn) try self.increfDyn(item.dyn);
            try self.assign(inst, target, item, pos);
            try inst.loops.append(self.a(), .{ .brk = exit, .cont = next, .tries = self.tries.items.len });
            try self.stmts(inst, body);
            _ = inst.loops.pop();
            try self.f.block(next);
            if (inst.done) {
                broke = true;
                break;
            }
        }
        if (!broke) try self.stmts(inst, else_);
        try self.f.block(exit);
    }

    /// What a loop goes over: items known now, or lists at run time
    const Iteration = union(enum) { known: []const SVal, runtime: RtIter };

    /// Lists iterated at run time (held in temporary slots): one, or
    /// several in step (zip), or one with its indexes (enumerate)
    const RtIter = struct {
        kind: enum { plain, zip, enumerate, range },
        slots: []const ir.Value,
        /// The index slot (an i64); a range's: its next value
        index: ir.Value = null,
        /// A range's stop and step (i64 slots)
        stop: ir.Value = null,
        step: ir.Value = null,
    };

    /// The iteration of a loop's iterable; zip() and enumerate() of
    /// run-time values go in step over their arguments (no tuples made).
    fn iteration(self: *Gen, inst: *Inst, e: *const front.Expr) Error!Iteration {
        if (e.kind == .call and e.kind.call.keywords.len == 0 and e.kind.call.func.kind == .global) {
            const x = e.kind.call;
            const callee = try self.global(inst, x.func.kind.global, x.func.pos);
            if (callee == .py and (isBuiltin(callee.py, "zip") or (isBuiltin(callee.py, "enumerate") and x.args.len == 1))) {
                const args = try self.a().alloc(SVal, x.args.len);
                for (args, x.args) |*slot, ae| slot.* = try self.expr(inst, ae);
                const all_known = for (args) |v| {
                    if (v != .list and v != .tuple) break false;
                } else true;
                if (all_known or args.len == 0) return self.iterationOf(inst, (try self.builtinCall(inst, callee.py, args, e.pos)).?);
                const slots = try self.a().alloc(ir.Value, args.len);
                for (args, slots) |v, *slot| {
                    slot.* = try self.tempSlot(inst);
                    try self.storeSlot(slot.*, try self.itemsOf(inst, v));
                }
                return .{ .runtime = .{ .kind = if (isBuiltin(callee.py, "zip")) .zip else .enumerate, .slots = slots } };
            }
            // range() of run-time (or many) values: counted, no list made
            if (callee == .py and isBuiltin(callee.py, "range") and x.args.len >= 1 and x.args.len <= 3) {
                const args = try self.a().alloc(SVal, x.args.len);
                for (args, x.args) |*slot, ae| slot.* = try self.expr(inst, ae);
                // (known bounds, few values: unrolled)
                if (allScalar(args)) if (try self.builtinCall(inst, callee.py, args, e.pos)) |v| {
                    if (v == .list) return .{ .known = v.list.items.items };
                    try self.drop(v);
                };
                return self.rangeIteration(inst, args);
            }
        }
        return self.iterationOf(inst, try self.expr(inst, e));
    }

    /// range(args) counted at run time: the bounds checked (and their
    /// errors raised) by zr_range.
    fn rangeIteration(self: *Gen, inst: *Inst, args: []const SVal) Error!Iteration {
        const f = &self.f;
        const t = self.c.m.t;
        const arr = try self.valueArray(args, inst.node);
        const bounds = try f.alloca(L("LLVMArrayType2")(t.i64, 3));
        const ok = self.call("zr_range", &.{ self.ctx, self.k32(inst.node), arr, self.k(@intCast(args.len)), bounds });
        try self.dropArray(arr, args.len);
        try self.check(ok);
        var it = RtIter{ .kind = .range, .slots = &.{} };
        it.index = try f.alloca(t.i64);
        it.stop = try f.alloca(t.i64);
        it.step = try f.alloca(t.i64);
        for ([_]ir.Value{ it.index, it.stop, it.step }, 0..) |slot, i| f.store(f.load(t.i64, f.at(t.i64, bounds, self.k(@intCast(i)))), slot);
        return .{ .runtime = it };
    }

    fn iterationOf(self: *Gen, inst: *Inst, v: SVal) Error!Iteration {
        switch (v) {
            .list => |l| return .{ .known = l.items.items },
            .tuple => |t| return .{ .known = t },
            else => {
                const slots = try self.a().alloc(ir.Value, 1);
                slots[0] = try self.tempSlot(inst);
                try self.storeSlot(slots[0], try self.itemsOf(inst, v));
                return .{ .runtime = .{ .kind = .plain, .slots = slots } };
            },
        }
    }

    /// Before the loop: its index, 0.
    fn iterStart(self: *Gen, it: *RtIter) Error!void {
        // (a range: its first value there already)
        if (it.kind == .range) return;
        it.index = try self.f.alloca(self.c.m.t.i64);
        self.f.store(self.k(0), it.index);
    }

    /// The loop's head: whether there's an item at the index (the lengths
    /// read each time: a list growing in the loop is gone over, as Python
    /// does).
    fn iterHead(self: *Gen, it: RtIter) Error!ir.Value {
        const f = &self.f;
        const i = f.load(self.c.m.t.i64, it.index);
        if (it.kind == .range) {
            // (up to stop going up, down to it going down)
            const stop = f.load(self.c.m.t.i64, it.stop);
            const up = f.icmp(jit_c.LLVMIntSGT, f.load(self.c.m.t.i64, it.step), self.k(0));
            return f.select(up, f.icmp(jit_c.LLVMIntSLT, i, stop), f.icmp(jit_c.LLVMIntSGT, i, stop));
        }
        var more = self.c.m.k1(true);
        for (it.slots) |slot| {
            const l = try self.loadSlot(slot, .list);
            const n = self.call("zr_list_len", &.{ l.tag, l.bits });
            more = f.and_(more, f.icmp(jit_c.LLVMIntSLT, i, n));
        }
        return more;
    }

    /// The item at the index, into the loop's target.
    fn iterItem(self: *Gen, inst: *Inst, it: RtIter, target: front.Target, pos: front.Pos) Error!void {
        const f = &self.f;
        const i = f.load(self.c.m.t.i64, it.index);
        // (a range's values: plain ints)
        if (it.kind == .range) return self.assign(inst, target, .{ .dyn = .{ .tag = self.k(@intCast(value.PINT_TAG)), .bits = i, .shape = .int } }, pos);
        const parts = try self.a().alloc(Dyn, if (it.kind == .enumerate) 2 else it.slots.len);
        var n: usize = 0;
        if (it.kind == .enumerate) {
            parts[0] = .{ .tag = self.k(@intCast(value.PINT_TAG)), .bits = i, .shape = .int };
            n = 1;
        }
        for (it.slots) |slot| {
            const l = try self.loadSlot(slot, .list);
            _ = self.call("zr_list_at", &.{ l.tag, l.bits, i, self.out });
            parts[n] = try self.loadOut(.any);
            n += 1;
        }
        if (it.kind == .plain) return self.assign(inst, target, .{ .dyn = parts[0] }, pos);
        // (a, b) in zip(...): each straight into its name
        if (target == .tuple and target.tuple.len == parts.len) {
            for (target.tuple, parts) |t, d| try self.assign(inst, t, .{ .dyn = d }, pos);
            return;
        }
        const sv = try self.a().alloc(SVal, parts.len);
        for (sv, parts) |*s, d| s.* = .{ .dyn = d };
        try self.assign(inst, target, .{ .dyn = try self.buildSequence("zr_tuple", sv, inst.node, .tuple) }, pos);
    }

    fn iterStep(self: *Gen, it: RtIter) Error!void {
        const f = &self.f;
        const i = f.load(self.c.m.t.i64, it.index);
        if (it.kind == .range) {
            // (past the 64 bits: the end, as no value can be beyond stop)
            const pair = f.call(self.c.sadd, &.{ i, f.load(self.c.m.t.i64, it.step) });
            f.store(f.select(f.extract(pair, 1), f.load(self.c.m.t.i64, it.stop), f.extract(pair, 0)), it.index);
            return;
        }
        f.store(f.add(i, self.k(1)), it.index);
    }

    fn iterEnd(self: *Gen, it: RtIter) Error!void {
        for (it.slots) |slot| try self.dropTemp(slot);
    }

    /// Before a run-time loop: the variables its targets (and body) assign
    /// become slots, set before it so none is left unset when it doesn't run.
    fn slotTargets(self: *Gen, inst: *Inst, targets: []const front.Target) Error!void {
        var set = std.AutoHashMapUnmanaged(u32, void).empty;
        for (targets) |t| collectTarget(t, &set, self.a()) catch return error.OutOfMemory;
        var sit = set.keyIterator();
        while (sit.next()) |slot| try self.toSlot(inst, slot.*);
    }

    /// A for loop over run-time items.
    fn runtimeFor(self: *Gen, inst: *Inst, target: front.Target, it_arg: RtIter, body: []const front.Stmt, else_: []const front.Stmt, pos: front.Pos) Error!void {
        const f = &self.f;
        var it = it_arg;
        try self.slotTargets(inst, &.{target});
        try self.prepareDynamic(inst, &.{ body, else_ });
        try self.iterStart(&it);
        const head = try f.label("for");
        const step = try f.label("for_next");
        const loop_body = try f.label("for_body");
        const els = try f.label("for_else");
        const exit = try f.label("endfor");
        try f.br(head);
        try f.block(head);
        self.loop_level += 1;
        defer self.loop_level -= 1;
        try f.condBr(try self.iterHead(it), loop_body, els);
        try f.block(loop_body);
        inst.dyn_depth += 1;
        try self.iterItem(inst, it, target, pos);
        try inst.loops.append(self.a(), .{ .brk = exit, .cont = step, .tries = self.tries.items.len });
        try self.stmts(inst, body);
        _ = inst.loops.pop();
        try f.br(step);
        try f.block(step);
        try self.iterStep(it);
        try f.br(head);
        try f.block(els);
        try self.stmts(inst, else_);
        inst.dyn_depth -= 1;
        try f.br(exit);
        try f.block(exit);
        try self.iterEnd(it);
    }

    // ------------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------------

    fn expr(self: *Gen, inst: *Inst, e: *const front.Expr) Error!SVal {
        const c = self.c;
        switch (e.kind) {
            // (a literal of the semantic's: a plain int; a big one, Python's)
            .int => |n| return .{ .pint = n },
            .big => |o| {
                _ = try c.objectIndex(o);
                return .{ .py = o };
            },
            .float => |x| return .{ .float = x },
            .str => |s| return .{ .str = s },
            .bool => |b| return .{ .bool = b },
            .none => return .none,
            .local => |slot| return self.readLocal(inst, slot, e.pos),
            .global => |name| return self.global(inst, name, e.pos),
            .attr => |x| {
                const obj = try self.expr(inst, x.obj);
                return self.attr(inst, obj, x.name, e.pos);
            },
            .call => |x| return self.callExpr(inst, x.func, x.args, x.keywords, e.pos),
            .binary => |x| {
                const l = try self.expr(inst, x.left);
                const r = try self.expr(inst, x.right);
                return self.binary(inst, x.op, l, r);
            },
            .unary => |x| {
                const v = try self.expr(inst, x.operand);
                return self.unary(inst, x.op, v);
            },
            .compare => |x| {
                // type(a) is type(b), type(a) is C: their classes' keys
                if (x.ops.len == 1 and (x.ops[0] == .is or x.ops[0] == .is_not)) {
                    if (try self.typeIs(inst, x.ops[0], x.first, x.rest[0], e.pos)) |v| return v;
                }
                // a < b < c: each pair, and-ed (each operand once)
                var left = try self.expr(inst, x.first);
                if (x.ops.len == 1) {
                    const right = try self.expr(inst, x.rest[0]);
                    return self.compare(inst, x.ops[0], left, right);
                }
                var result: SVal = .{ .bool = true };
                for (x.ops, x.rest, 0..) |op, rest_e, i| {
                    const right = try self.expr(inst, rest_e);
                    if (i + 1 < x.ops.len and right == .dyn) try self.increfDyn(right.dyn);
                    const r = try self.compare(inst, op, left, right);
                    result = try self.andValues(inst, result, r);
                    left = right;
                }
                return result;
            },
            .and_, .or_ => |items| return self.boolOp(inst, e.kind == .and_, items),
            .cond => |x| {
                const t = try self.truth(try self.expr(inst, x.test_), inst.node);
                switch (t) {
                    .known => |b| return self.expr(inst, if (b) x.then else x.else_),
                    .dyn => |cond| return self.branchValue(inst, cond, x.then, x.else_),
                }
            },
            .list, .tuple => |items| {
                const l = try self.a().create(SList);
                l.* = .{ .origin = e };
                for (items) |item| try l.items.append(self.a(), try self.expr(inst, item));
                if (e.kind == .tuple) return .{ .tuple = l.items.items };
                return self.literal(.{ .list = l }, e, inst.node);
            },
            .list_comp, .gen_exp => |comp| return self.literal(try self.listComp(inst, comp, e.pos), e, inst.node),
            .dict => |x| {
                const d = try self.a().create(SDict);
                d.* = .{ .origin = e };
                var dynamic = false;
                const keys = try self.a().alloc(SVal, x.keys.len);
                const vals = try self.a().alloc(SVal, x.keys.len);
                for (x.keys, x.values, 0..) |ke, ve, i| {
                    keys[i] = try self.expr(inst, ke);
                    vals[i] = try self.expr(inst, ve);
                    if (!keys[i].isStatic() or !isScalar(keys[i])) dynamic = true;
                }
                if (!dynamic) {
                    for (keys, vals) |key, v| try self.sdictSet(d, key, v);
                    return self.literal(.{ .dict = d }, e, inst.node);
                }
                // Keys only known at run time: a run-time dict
                const ka = try self.valueArray(keys, inst.node);
                const va = try self.valueArray(vals, inst.node);
                const ok = self.call("zr_dict", &.{ self.ctx, self.k32(inst.node), ka, va, self.k(@intCast(keys.len)), self.out });
                try self.dropArray(ka, keys.len);
                try self.dropArray(va, vals.len);
                try self.check(ok);
                return .{ .dyn = try self.loadOut(.dict) };
            },
            .dict_comp => |comp| return self.dictComp(inst, comp.key, comp.value, comp.generators, e.pos),
            .index => |x| {
                const obj = try self.expr(inst, x.obj);
                const key = try self.expr(inst, x.index);
                return self.getItem(inst, obj, key);
            },
            .slice => |x| {
                const obj = try self.expr(inst, x.obj);
                const lo = if (x.lo) |v| try self.expr(inst, v) else SVal.none;
                const hi = if (x.hi) |v| try self.expr(inst, v) else SVal.none;
                const step = if (x.step) |v| try self.expr(inst, v) else SVal.none;
                // Known: Python's slice now (strings, known lists of scalars)
                if (isScalar(obj) and isScalar(lo) and isScalar(hi) and isScalar(step)) {
                    const o = try self.pyOf(obj);
                    defer py.Py_DecRef(o);
                    const sl = try self.sliceObject(lo, hi, step);
                    defer py.Py_DecRef(sl);
                    if (py.c.PyObject_GetItem(o, sl)) |r| {
                        defer py.Py_DecRef(r);
                        return self.constant(r, inst.node);
                    }
                    py.c.PyErr_Clear();
                }
                if (obj == .list and lo.isStatic() and hi.isStatic() and step.isStatic() and isScalar(lo) and isScalar(hi) and isScalar(step)) {
                    // A known list: the slice of its items
                    const n: isize = @intCast(obj.list.items.items.len);
                    const sl = try self.sliceObject(lo, hi, step);
                    defer py.Py_DecRef(sl);
                    var start: py.Py_ssize_t = 0;
                    var stop: py.Py_ssize_t = 0;
                    var stride: py.Py_ssize_t = 0;
                    if (py.c.PySlice_Unpack(sl, &start, &stop, &stride) == 0) {
                        const count = py.c.PySlice_AdjustIndices(n, &start, &stop, stride);
                        const out = try self.a().create(SList);
                        out.* = .{};
                        var i: isize = start;
                        for (0..@intCast(count)) |_| {
                            try out.items.append(self.a(), try self.copyOf(obj.list.items.items[@intCast(i)]));
                            i += stride;
                        }
                        return .{ .list = out };
                    }
                    py.c.PyErr_Clear();
                }
                // At run time: the slice natively (zr_slice), Python's for
                // other objects
                const ds = [_]Dyn{ try self.materialize(obj, inst.node), try self.materialize(lo, inst.node), try self.materialize(hi, inst.node), try self.materialize(step, inst.node) };
                const ok = self.call("zr_slice", &.{ self.ctx, self.k32(inst.node), ds[0].tag, ds[0].bits, ds[1].tag, ds[1].bits, ds[2].tag, ds[2].bits, ds[3].tag, ds[3].bits, self.out });
                for (ds) |d| try self.drop(.{ .dyn = d });
                try self.check(ok);
                return .{ .dyn = try self.loadOut(.any) };
            },
            .fstring => |parts| return self.fstring(inst, parts, e.pos),
        }
    }

    fn sdictSet(self: *Gen, d: *SDict, key: SVal, v: SVal) Error!void {
        if (d.find(key)) |i| {
            try self.drop(d.values.items[i]);
            d.values.items[i] = v;
            return;
        }
        try d.keys.append(self.a(), key);
        try d.values.append(self.a(), v);
    }

    /// slice(lo, hi, step) of known bounds (a new reference).
    fn sliceObject(self: *Gen, lo: SVal, hi: SVal, step: SVal) Error!*PyObject {
        const a_ = try self.pyOf(lo);
        defer py.Py_DecRef(a_);
        const b = try self.pyOf(hi);
        defer py.Py_DecRef(b);
        const s = try self.pyOf(step);
        defer py.Py_DecRef(s);
        return py.c.PySlice_New(a_, b, s) orelse error.Python;
    }

    /// obj[key]: known for known containers and keys; zr_getitem else.
    fn getItem(self: *Gen, inst: *Inst, obj: SVal, key: SVal) Error!SVal {
        if (key.isStatic() and isScalar(key)) {
            switch (obj) {
                .list => |l| if (intOf(key)) |ki| {
                    const n: i64 = @intCast(l.items.items.len);
                    const i = if (ki < 0) ki + n else ki;
                    if (i >= 0 and i < n) return self.copyOf(l.items.items[@intCast(i)]);
                },
                .tuple => |t| if (intOf(key)) |ki| {
                    const n: i64 = @intCast(t.len);
                    const i = if (ki < 0) ki + n else ki;
                    if (i >= 0 and i < n) return self.copyOf(t[@intCast(i)]);
                },
                .dict => |d| if (d.find(key)) |i| return self.copyOf(d.values.items[i]),
                .str => {
                    const o = try self.pyOf(obj);
                    defer py.Py_DecRef(o);
                    const ko = try self.pyOf(key);
                    defer py.Py_DecRef(ko);
                    if (py.c.PyObject_GetItem(o, ko)) |r| {
                        defer py.Py_DecRef(r);
                        return self.constant(r, inst.node);
                    }
                    py.c.PyErr_Clear();
                },
                else => {},
            }
        }
        // (read, given up: a variable's value borrowed)
        const od = try self.borrowed(obj, inst.node);
        const kd = try self.materialize(key, inst.node);
        if ((od.shape == .any or od.shape == .list or od.shape == .tuple) and canBeInt(kd)) return .{ .dyn = try self.indexInline(inst, od, kd) };
        return .{ .dyn = try self.getitemCall(inst, od, kd) };
    }

    /// len(v): a list's, tuple's, dict's length, a str's code points,
    /// inline; anything else by zr_builtin (a class's __len__, errors).
    fn lenInline(self: *Gen, inst: *Inst, len_obj: *PyObject, d: Dyn) Error!Dyn {
        const f = &self.f;
        const t = self.c.m.t;
        const T = value.Tag;
        const sized = try f.label("len_sized");
        const str = try f.label("len_str");
        const other = try f.label("len_other");
        const slow = try f.label("len_call");
        const join = try f.label("len_got");
        const tag = d.tag;
        const seq = f.or_(f.or_(f.icmp(jit_c.LLVMIntEQ, tag, self.k(@intFromEnum(T.list))), f.icmp(jit_c.LLVMIntEQ, tag, self.k(@intFromEnum(T.tuple)))), f.icmp(jit_c.LLVMIntEQ, tag, self.k(@intFromEnum(T.dict))));
        try f.condBr(seq, sized, other);
        try f.block(sized);
        // (their length at the same place)
        const n1 = f.load(t.i64, f.offset(f.intToPtr(d.bits), @offsetOf(value.List, "len")));
        try self.drop(.{ .dyn = d });
        const sized_end = f.current;
        try f.br(join);
        try f.block(other);
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, tag, self.k(@intFromEnum(T.str))), str, slow);
        try f.block(str);
        const n2 = f.load(t.i64, f.offset(f.intToPtr(d.bits), @offsetOf(value.Str, "chars")));
        try self.drop(.{ .dyn = d });
        const str_end = f.current;
        try f.br(join);
        try f.block(slow);
        const idx = try self.c.objectIndex(len_obj);
        const ok = self.call("zr_builtin", &.{ self.ctx, self.k32(inst.node), self.k32(@intFromEnum(helpers.Builtin.len)), self.k(@intCast(idx)), d.tag, d.bits, self.out });
        try self.drop(.{ .dyn = d });
        try self.check(ok);
        const g = try self.loadOut(.int);
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        const n = f.phiN(t.i64, &.{ n1, n2, g.bits }, &.{ sized_end, str_end, slow_end });
        const tg = f.phiN(t.i64, &.{ self.k(@intCast(value.PINT_TAG)), self.k(@intCast(value.PINT_TAG)), g.tag }, &.{ sized_end, str_end, slow_end });
        return .{ .tag = tg, .bits = n, .shape = .int };
    }

    fn getitemCall(self: *Gen, inst: *Inst, od: Dyn, kd: Dyn) Error!Dyn {
        const ok = self.call("zr_getitem", &.{ self.ctx, self.k32(inst.node), od.tag, od.bits, kd.tag, kd.bits, self.out });
        try self.drop(.{ .dyn = od });
        try self.drop(.{ .dyn = kd });
        try self.check(ok);
        return self.loadOut(.any);
    }

    /// v[i] of a list or tuple by an int in it (negative from its end):
    /// inline; anything else (and the errors) by zr_getitem.
    fn indexInline(self: *Gen, inst: *Inst, od: Dyn, kd: Dyn) Error!Dyn {
        const f = &self.f;
        const t = self.c.m.t;
        const T = value.Tag;
        const fast = try f.label("index_fast");
        const found = try f.label("index_found");
        const slow = try f.label("index_other");
        const join = try f.label("index_got");
        const is_list = f.icmp(jit_c.LLVMIntEQ, od.tag, self.k(@intFromEnum(T.list)));
        const is_tuple = f.icmp(jit_c.LLVMIntEQ, od.tag, self.k(@intFromEnum(T.tuple)));
        const key_int = f.or_(f.icmp(jit_c.LLVMIntEQ, kd.tag, self.k(@intFromEnum(T.int))), f.icmp(jit_c.LLVMIntEQ, kd.tag, self.k(@intCast(value.PINT_TAG))));
        try f.condBr(f.and_(f.or_(is_list, is_tuple), key_int), fast, slow);
        try f.block(fast);
        const p = f.intToPtr(od.bits);
        // (both: their length at the same place; a list's items through a
        // pointer, a tuple's after it)
        const len = f.load(t.i64, f.offset(p, @offsetOf(value.List, "len")));
        const of_list = try f.label("index_list");
        const of_tuple = try f.label("index_tuple");
        const have = try f.label("index_items");
        try f.condBr(is_list, of_list, of_tuple);
        try f.block(of_list);
        const list_items = f.load(t.ptr, f.offset(p, @offsetOf(value.List, "items")));
        try f.br(have);
        try f.block(of_tuple);
        const tuple_items = f.offset(p, @sizeOf(value.Tuple));
        try f.br(have);
        try f.block(have);
        const items = f.phi(t.ptr, list_items, of_list, tuple_items, of_tuple);
        const neg = f.icmp(jit_c.LLVMIntSLT, kd.bits, self.k(0));
        const i = f.select(neg, f.add(kd.bits, len), kd.bits);
        try f.condBr(f.icmp(jit_c.LLVMIntULT, i, len), found, slow);
        try f.block(found);
        const slot = f.at(t.val, items, i);
        const item = try self.loadSlot(slot, .any);
        try self.increfDyn(item);
        try self.drop(.{ .dyn = od });
        const found_end = f.current;
        try f.br(join);
        try f.block(slow);
        const g = try self.getitemCall(inst, od, kd);
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        return .{ .tag = f.phi(t.i64, item.tag, found_end, g.tag, slow_end), .bits = f.phi(t.i64, item.bits, found_end, g.bits, slow_end), .shape = .any };
    }

    /// An f-string: known pieces joined now; else each piece formatted at
    /// run time (Python's format()) and joined.
    fn fstring(self: *Gen, inst: *Inst, parts: []const front.FPart, pos: front.Pos) Error!SVal {
        const pieces = try self.a().alloc(SVal, parts.len);
        var all_known = true;
        for (parts, 0..) |p, i| switch (p) {
            .text => |t| pieces[i] = .{ .str = t },
            .value => |v| {
                const x = try self.expr(inst, v.expr);
                const spec = try self.fstring(inst, v.spec, pos);
                if (x.isStatic() and isScalar(x) and spec == .str) {
                    const o = try self.pyOf(x);
                    defer py.Py_DecRef(o);
                    var conv = o;
                    py.Py_IncRef(conv);
                    defer py.Py_DecRef(conv);
                    if (v.conversion != 0) {
                        py.Py_DecRef(conv);
                        conv = switch (v.conversion) {
                            'r' => py.c.PyObject_Repr(o),
                            'a' => py.c.PyObject_ASCII(o),
                            else => py.c.PyObject_Str(o),
                        } orelse return error.Python;
                    }
                    const s = ph.newString(spec.str) orelse return error.Python;
                    defer py.Py_DecRef(s);
                    const r = py.c.PyObject_Format(conv, s) orelse return error.Python;
                    defer py.Py_DecRef(r);
                    pieces[i] = try self.constant(r, inst.node);
                } else {
                    all_known = false;
                    const d = try self.materialize(x, inst.node);
                    const spec_s = try self.c.m.string(if (spec == .str) spec.str else return self.c.unsupportedAt(inst.func, pos, "a format spec only known at run time isn't compiled yet", .{}));
                    const ok = self.call("zr_format", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, self.k32(v.conversion), spec_s, self.out });
                    try self.drop(.{ .dyn = d });
                    try self.check(ok);
                    pieces[i] = .{ .dyn = try self.loadOut(.str) };
                }
            },
        };
        if (all_known) {
            var buf: std.ArrayListUnmanaged(u8) = .empty;
            for (pieces) |p| try buf.appendSlice(self.a(), p.str);
            return .{ .str = buf.items };
        }
        const arr = try self.valueArray(pieces, inst.node);
        const ok = self.call("zr_concat", &.{ self.ctx, self.k32(inst.node), arr, self.k(@intCast(pieces.len)), self.out });
        try self.dropArray(arr, pieces.len);
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.str) };
    }

    fn dictComp(self: *Gen, inst: *Inst, key_e: *const front.Expr, val_e: *const front.Expr, gens: []const front.Generator, pos: front.Pos) Error!SVal {
        // A run-time dict filled by the comprehension's loops (known
        // iterables unroll; run-time ones loop)
        const empty = try self.a().create(SDict);
        empty.* = .{};
        const d = try self.buildDict(empty, inst.node);
        const slot = try self.valSlot();
        try self.storeSlot(slot, d);
        var sink = Sink{ .kind = .dict_into, .list = undefined, .slot = slot, .key = key_e, .value = val_e };
        try self.compLoops(inst, gens, 0, pos, &sink);
        return .{ .dyn = try self.loadSlot(slot, .dict) };
    }

    /// A module-level name of the semantic: its value when compiling.
    fn global(self: *Gen, inst: *Inst, name: []const u8, pos: front.Pos) Error!SVal {
        const fobj = inst.func.py_function;
        const globals = ph.attr(fobj, "__globals__") orelse return error.Python;
        defer py.Py_DecRef(globals);
        const key = ph.newString(name) orelse return error.Python;
        defer py.Py_DecRef(key);
        // A captured variable first (a closure's code: read from the
        // function called), then the module, then builtins
        if (self.closure_fn) |cf| if (inst.func == self.closure_root.?) if (try freeVarIndex(fobj, name)) |idx| {
            try self.callCheck("zr_cell", &.{ self.ctx, self.k32(inst.node), cf.tag, cf.bits, self.k(@intCast(idx)), self.out });
            return .{ .dyn = try self.loadOut(.any) };
        };
        if (try closureValue(fobj, name)) |cell_value| return self.constant(cell_value, inst.node);
        // (one a function assigns: its value when the code runs)
        const rebound = try reboundGlobals(globals);
        if (py.c.PySequence_Contains(rebound, key) == 1) {
            const idx = try self.c.objectIndex(globals);
            const s = try self.c.m.string(name);
            try self.callCheck("zr_global", &.{ self.ctx, self.k32(inst.node), self.k(@intCast(idx)), s, self.out });
            return .{ .dyn = try self.loadOut(.any) };
        }
        if (py.c.PyDict_GetItem(globals, key)) |v| {
            // (a table only read: known, as a constant)
            if (py.c.PySequence_Contains(try frozenGlobals(globals), key) == 1) {
                try self.c.noteState(name, "constant (only read)");
                return self.frozenTable(v, inst.node);
            }
            // (a table or record kept there that semantics change: native,
            // shared with Python (adopt.zig), its address a constant)
            switch (try adopt_mod.adopt(self.c.a, v, globals)) {
                .native => |nv| {
                    try self.c.noteState(name, "native");
                    return .{ .dyn = .{
                        .tag = self.k(@intCast(nv.tag)),
                        .bits = self.c.m.addrInt(nv.bits),
                        .shape = switch (nv.kind()) {
                            .list => .list,
                            .dict => .dict,
                            else => .record,
                        },
                    } };
                },
                .refused => |why| try self.c.noteState(name, why),
                .not_state => {},
            }
            return self.constant(v, inst.node);
        }
        const builtins = py.c.PyImport_ImportModule("builtins") orelse return error.Python;
        defer py.Py_DecRef(builtins);
        if (py.c.PyObject_HasAttr(builtins, key) == 1) {
            const v = py.c.PyObject_GetAttr(builtins, key) orelse return error.Python;
            // (builtins live as long as the interpreter: borrowed is fine)
            py.Py_DecRef(v);
            return .{ .py = v };
        }
        return self.c.unsupportedAt(inst.func, pos, "name '{s}' is not defined", .{name});
    }

    /// The index of a function's free variable (its cell's), or null.
    fn freeVarIndex(fobj: *PyObject, name: []const u8) Error!?usize {
        const code = ph.attr(fobj, "__code__") orelse return error.Python;
        defer py.Py_DecRef(code);
        const freevars = ph.attr(code, "co_freevars") orelse return error.Python;
        defer py.Py_DecRef(freevars);
        const n: usize = @intCast(py.c.PyTuple_Size(freevars));
        for (0..n) |i| {
            const fv = ph.utf8(py.c.PyTuple_GetItem(freevars, @intCast(i)).?, "name") orelse return error.Python;
            if (std.mem.eql(u8, fv, name)) return i;
        }
        return null;
    }

    fn closureValue(fobj: *PyObject, name: []const u8) Error!?*PyObject {
        const code = ph.attr(fobj, "__code__") orelse return error.Python;
        defer py.Py_DecRef(code);
        const freevars = ph.attr(code, "co_freevars") orelse return error.Python;
        defer py.Py_DecRef(freevars);
        const closure = ph.attr(fobj, "__closure__") orelse return error.Python;
        defer py.Py_DecRef(closure);
        if (closure == py.Py_None()) return null;
        const n: usize = @intCast(py.c.PyTuple_Size(freevars));
        for (0..n) |i| {
            const fv = ph.utf8(py.c.PyTuple_GetItem(freevars, @intCast(i)).?, "name") orelse return error.Python;
            if (!std.mem.eql(u8, fv, name)) continue;
            const cell = py.c.PyTuple_GetItem(closure, @intCast(i)).?;
            const contents = ph.attr(cell, "cell_contents") orelse return error.Python;
            // (the cell keeps it alive)
            py.Py_DecRef(contents);
            return contents;
        }
        return null;
    }

    // ------------------------------------------------------------------
    // Attributes
    // ------------------------------------------------------------------

    fn attr(self: *Gen, inst: *Inst, obj: SVal, name: []const u8, pos: front.Pos) Error!SVal {
        const c = self.c;
        const eq = std.mem.eql;
        switch (obj) {
            .node => |idx| {
                const d = c.data;
                if (d.grammar.field_ids.get(name)) |field| return self.fieldValue(idx, field);
                const n = d.nodes[idx];
                const rid = n.ruleId();
                if (eq(u8, name, "kind")) return .{ .str = d.grammar.kind_names[rid] };
                if (eq(u8, name, "rule")) return .{ .str = d.grammar.rule_names[rid] };
                if (eq(u8, name, "text")) return .{ .str = d.text(idx) };
                // (plain ints, as the reference mode's Node gives them)
                if (eq(u8, name, "start")) return .{ .pint = n.text_start };
                if (eq(u8, name, "line")) return .{ .pint = d.lineCol(n.text_start).line };
                if (eq(u8, name, "column")) return .{ .pint = d.lineCol(n.text_start).col };
                if (eq(u8, name, "end")) return .{ .pint = n.text_end };
                if (eq(u8, name, "index")) return .{ .pint = idx };
                if (eq(u8, name, "span")) {
                    const items = try self.a().alloc(SVal, 2);
                    items[0] = .{ .pint = n.text_start };
                    items[1] = .{ .pint = n.text_end };
                    return .{ .tuple = items };
                }
                if (eq(u8, name, "children")) return self.childValues(idx);
                if (eq(u8, name, "parent")) {
                    const p = d.parents[idx];
                    return if (p == NONE) SVal.none else SVal{ .node = p };
                }
                return c.unsupportedAt(inst.func, pos, "{s} has no field '{s}'", .{ d.grammar.kind_names[rid], name });
            },
            .rt => {
                inline for (@typeInfo(RtMethod).@"enum".fields) |fd| {
                    if (eq(u8, name, fd.name)) return .{ .rt_method = @enumFromInt(fd.value) };
                }
                // (out of line: the caller's, given; none: None, ())
                if (self.detached and (eq(u8, name, "receiver") or eq(u8, name, "varargs"))) {
                    const is_recv = eq(u8, name, "receiver");
                    return .{ .dyn = try self.loadGiven(if (is_recv) self.recv_slot else self.varargs_slot, if (is_recv) .none else .{ .tuple = &.{} }) };
                }
                if (eq(u8, name, "receiver")) {
                    if (self.fnode == NONE) return .none;
                    // (the function's: its call's, or the one around's)
                    const v = try self.loadSlot(self.recv_slot, .any);
                    try self.increfDyn(v);
                    return .{ .dyn = v };
                }
                if (eq(u8, name, "varargs")) {
                    if (self.fnode == NONE or c.specOf(self.fnode).?.extra != .keep) return .{ .tuple = &.{} };
                    const v = try self.loadSlot(self.varargs_slot, .tuple);
                    try self.increfDyn(v);
                    return .{ .dyn = v };
                }
                if (eq(u8, name, "path")) {
                    const p = self.c.lang.path orelse return .none;
                    return .{ .str = p };
                }
                // (the class: raised, caught by Python)
                if (eq(u8, name, "Throw")) return .{ .py = @import("types.zig").Throw };
                return c.unsupportedAt(inst.func, pos, "rt has no '{s}' in compiled code", .{name});
            },
            .py => |o| {
                // (an object that may change: its attribute when the code
                // runs)
                if (!try stablePy(o)) return self.attr(inst, .{ .dyn = try self.materialize(obj, inst.node) }, name, pos);
                const key = ph.newString(name) orelse return error.Python;
                defer py.Py_DecRef(key);
                const v = py.c.PyObject_GetAttr(o, key) orelse return error.Python;
                // (kept alive by the object it belongs to, for constants: a
                // reference is kept with the compiled program when used)
                const sv = try self.constant(v, inst.node);
                if (sv == .py) _ = try c.objectIndex(v);
                py.Py_DecRef(v);
                return sv;
            },
            .dyn => |d| {
                // A field of a record of the module's classes: read where
                // it is (its type checked); anything else (a Python
                // object's attribute...) by zr_getattr
                const cands = try self.fieldCandidates(inst, name, false);
                if (cands.len > 0) return self.recordField(inst, d, name, cands);
                return .{ .dyn = try self.genericGetattr(inst, d, name) };
            },
            else => return c.unsupportedAt(inst.func, pos, "'{s}' of a {s} isn't compiled yet (only called, as a method)", .{ name, @tagName(obj) }),
        }
    }

    fn genericGetattr(self: *Gen, inst: *Inst, d: Dyn, name: []const u8) Error!Dyn {
        if (d.shape == .any or d.shape == .node) {
            const g = self.c.data.grammar;
            if (std.mem.eql(u8, name, "kind") and g.field_ids.get(name) == null) return self.nodeKind(inst, d, name);
            if (g.field_ids.get(name)) |field| return self.nodeField(inst, d, name, field);
        }
        return self.getattrCall(inst, d, name);
    }

    fn getattrCall(self: *Gen, inst: *Inst, d: Dyn, name: []const u8) Error!Dyn {
        const s = try self.c.m.string(name);
        const ok = self.call("zr_getattr", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, s, self.out });
        try self.drop(.{ .dyn = d });
        try self.check(ok);
        return self.loadOut(.any);
    }

    /// node.kind of a value that may be a node: the program's table of
    /// kinds (by node), inline; anything else by zr_getattr.
    fn nodeKind(self: *Gen, inst: *Inst, d: Dyn, name: []const u8) Error!Dyn {
        const f = &self.f;
        const t = self.c.m.t;
        const table = try self.c.kindTable();
        const is_node = try f.label("kind_node");
        const slow = try f.label("kind_other");
        const join = try f.label("kind_got");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(value.Tag.node))), is_node, slow);
        try f.block(is_node);
        // (an immortal str: no reference to take)
        const s = f.load(t.i64, f.at(t.i64, self.c.m.ptrConst(@intFromPtr(table.ptr)), d.bits));
        try f.br(join);
        try f.block(slow);
        const g = try self.getattrCall(inst, d, name);
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        return .{
            .tag = f.phi(t.i64, self.k(@intFromEnum(value.Tag.str)), is_node, g.tag, slow_end),
            .bits = f.phi(t.i64, s, is_node, g.bits, slow_end),
            .shape = .any,
        };
    }

    /// node.<label> of a value that may be a node: the program's table of
    /// that label's child (by node: one, or None), inline; a node whose
    /// label repeats (a list), or whose child an action makes a value of,
    /// and anything else by zr_getattr.
    fn nodeField(self: *Gen, inst: *Inst, d: Dyn, name: []const u8, field: u8) Error!Dyn {
        const f = &self.f;
        const t = self.c.m.t;
        const table = try self.c.fieldTable(field);
        const is_node = try f.label("field_node");
        const known = try f.label("field_known");
        const slow = try f.label("field_other");
        const join = try f.label("field_got");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(value.Tag.node))), is_node, slow);
        try f.block(is_node);
        const ch = f.load(t.i64, f.at(t.i64, self.c.m.ptrConst(@intFromPtr(table.ptr)), d.bits));
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, ch, self.k(Compiler.field_other)), slow, known);
        try f.block(known);
        const none = f.icmp(jit_c.LLVMIntEQ, ch, self.k(NONE));
        const tag = f.select(none, self.k(@intFromEnum(value.Tag.none)), self.k(@intFromEnum(value.Tag.node)));
        const bits = f.select(none, self.k(0), ch);
        try f.br(join);
        try f.block(slow);
        const g = try self.getattrCall(inst, d, name);
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        return .{
            .tag = f.phi(t.i64, tag, known, g.tag, slow_end),
            .bits = f.phi(t.i64, bits, known, g.bits, slow_end),
            .shape = .any,
        };
    }

    const FieldCandidate = struct { rtype: *value.RecordType, index: usize };

    /// The record classes of the semantic's module with a field `name`
    /// (not frozen ones, `for_store`), and where it is in them (made once
    /// per module and name).
    fn fieldCandidates(self: *Gen, inst: *Inst, name: []const u8, for_store: bool) Error![]const FieldCandidate {
        const c = self.c;
        const globals = ph.attr(inst.func.py_function, "__globals__") orelse return error.Python;
        defer py.Py_DecRef(globals);
        const key = try std.fmt.allocPrint(c.a, "{x}:{s}:{}", .{ @intFromPtr(globals), name, for_store });
        if (c.field_cands.get(key)) |cands| return cands;
        var out: std.ArrayListUnmanaged(FieldCandidate) = .empty;
        var pos: py.Py_ssize_t = 0;
        var gk: ?*PyObject = null;
        var v: ?*PyObject = null;
        while (py.c.PyDict_Next(globals, &pos, @ptrCast(&gk), @ptrCast(&v)) != 0) {
            if (!try isInstanceOf(v.?, @ptrCast(@alignCast(py.types.typeObject("PyType_Type"))))) continue;
            const rtype = try recordOf(v.?) orelse continue;
            if (for_store and rtype.frozen) continue;
            for (rtype.fields, 0..) |f, i| if (std.mem.eql(u8, f, name)) {
                try out.append(c.a, .{ .rtype = rtype, .index = i });
                break;
            };
        }
        try c.field_cands.put(c.a, key, out.items);
        return out.items;
    }

    /// Where field `index` of a record is (its tag word).
    fn fieldPtr(self: *Gen, d: Dyn, index: usize) ir.Value {
        return self.f.offset(self.f.intToPtr(d.bits), @intCast(@sizeOf(value.Record) + index * @sizeOf(value.Value)));
    }

    /// A record's type (the word after its header).
    fn recordTypeOf(self: *Gen, d: Dyn) ir.Value {
        return self.f.load(self.c.m.t.i64, self.f.offset(self.f.intToPtr(d.bits), @offsetOf(value.Record, "rtype")));
    }

    /// obj.name read where it is in a record of one of the candidates'
    /// types; anything else (or a slot never assigned: its error) by
    /// zr_getattr.
    fn recordField(self: *Gen, inst: *Inst, d: Dyn, name: []const u8, cands: []const FieldCandidate) Error!SVal {
        const f = &self.f;
        const t = self.c.m.t;
        const result = try self.valSlot();
        const join = try f.label("field_done");
        const generic = try f.label("field_generic");
        const is_record = f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(value.Tag.record)));
        const typed = try f.label("field_typed");
        try f.condBr(is_record, typed, generic);
        try f.block(typed);
        const rt = self.recordTypeOf(d);
        for (cands) |cand| {
            const yes = try f.label("field_of");
            const no = try f.label("field_next");
            try f.condBr(f.icmp(jit_c.LLVMIntEQ, rt, self.c.m.addrInt(@intFromPtr(cand.rtype))), yes, no);
            try f.block(yes);
            const p = self.fieldPtr(d, cand.index);
            const ftag = f.load(t.i64, p);
            const set = try f.label("field_set");
            try f.condBr(f.icmp(jit_c.LLVMIntEQ, ftag, self.k(@bitCast(value.UNSET_TAG))), generic, set);
            try f.block(set);
            const v = Dyn{ .tag = ftag, .bits = f.load(t.i64, f.offset(p, 8)), .shape = .any };
            try self.increfDyn(v);
            try self.storeSlot(result, v);
            // (the record: dropped, the field's taken)
            try self.drop(.{ .dyn = d });
            try f.br(join);
            try f.block(no);
        }
        try f.br(generic);
        try f.block(generic);
        try self.storeSlot(result, try self.genericGetattr(inst, d, name));
        try f.br(join);
        try f.block(join);
        return .{ .dyn = try self.loadSlot(result, .any) };
    }

    /// `obj.name(args)` on a value (not rt, a node or a module).
    fn methodCall(self: *Gen, inst: *Inst, obj: SVal, name: []const u8, args: []const SVal, pos: front.Pos) Error!SVal {
        const c = self.c;
        const eq = std.mem.eql;
        // A known list or dict changed while nothing runs at run time: now
        if (inst.dyn_depth == 0) {
            switch (obj) {
                .list => |l| if (eq(u8, name, "append") and args.len == 1) {
                    try l.items.append(self.a(), args[0]);
                    return .none;
                } else if (eq(u8, name, "extend") and args.len == 1 and args[0] == .list and args[0].list != l and args[0].list.frozen == null and !self.aliased(args[0].list)) {
                    // (by a list made here: its items, theirs now)
                    try l.items.appendSlice(self.a(), args[0].list.items.items);
                    return .none;
                },
                .dict => |d| if (eq(u8, name, "get") and args.len >= 1 and args.len <= 2 and args[0].isStatic()) {
                    if (d.find(args[0])) |i| return self.copyOf(d.values.items[i]);
                    return if (args.len == 2) args[1] else SVal.none;
                },
                else => {},
            }
        }
        // Known str methods on known arguments: Python's result now
        if (obj == .str and allScalar(args)) {
            const o = try self.pyOf(obj);
            defer py.Py_DecRef(o);
            const key = ph.newString(name) orelse return error.Python;
            defer py.Py_DecRef(key);
            const method = py.c.PyObject_GetAttr(o, key) orelse {
                py.c.PyErr_Clear();
                return c.unsupportedAt(inst.func, pos, "str has no method '{s}'", .{name});
            };
            defer py.Py_DecRef(method);
            const tuple = py.c.PyTuple_New(@intCast(args.len)) orelse return error.Python;
            defer py.Py_DecRef(tuple);
            for (args, 0..) |x, i| _ = py.c.PyTuple_SetItem(tuple, @intCast(i), try self.pyOf(x));
            if (py.c.PyObject_CallObject(method, tuple)) |r| {
                defer py.Py_DecRef(r);
                return self.constant(r, inst.node);
            }
            py.c.PyErr_Clear();
        }
        // At run time (a known container made a run-time one: the variables
        // referring to it see it change): append natively; the rest as
        // Python does it (on the object itself, through its proxy)
        const d = try self.materializeToChange(obj, inst.node);
        // extend() by a list made here, of items known one by one ([x]): each
        // pushed, no list made for them (one a variable refers to: made, as
        // its items are its)
        if (eq(u8, name, "extend") and args.len == 1 and args[0] == .list and args[0].list.frozen == null and !self.aliased(args[0].list)) {
            const items = args[0].list.items.items;
            const arr = try self.valueArray(items, inst.node);
            const ok = self.call("zr_extend_items", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, arr, self.k(@intCast(items.len)) });
            try self.dropArray(arr, items.len);
            try self.drop(.{ .dyn = d });
            try self.check(ok);
            return .none;
        }
        if (eq(u8, name, "append") and args.len == 1) {
            const x = try self.materialize(args[0], inst.node);
            const ok = self.call("zr_append", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, x.tag, x.bits });
            try self.drop(.{ .dyn = x });
            try self.drop(.{ .dyn = d });
            try self.check(ok);
            return .none;
        }
        // A record's method: its compiled code, for each record class of the
        // semantics' module having it (the record's type checked); else
        // as Python does it
        const cands = try self.methodCandidates(inst, name, args.len + 1);
        if (cands.len > 0 and c.allHeap()) return self.recordMethodCall(inst, d, name, args, cands);
        return self.genericMethodCall(inst, d, name, args);
    }

    fn genericMethodCall(self: *Gen, inst: *Inst, d: Dyn, name: []const u8, args: []const SVal) Error!SVal {
        const arr = try self.valueArray(args, inst.node);
        const s = try self.c.m.string(name);
        const ok = self.call("zr_call_method", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, s, arr, self.k(@intCast(args.len)), self.out });
        try self.dropArray(arr, args.len);
        try self.drop(.{ .dyn = d });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.any) };
    }

    const MethodCandidate = struct { rtype: *value.RecordType, func: *const front.Function };

    /// The record classes of the semantic's module (its globals) whose
    /// `name` is a method (a Python function) taking `nargs` (self
    /// included), with it read. (One taking others: Python's TypeError,
    /// through the generic call.)
    fn methodCandidates(self: *Gen, inst: *Inst, name: []const u8, nargs: usize) Error![]const MethodCandidate {
        const c = self.c;
        const globals = ph.attr(inst.func.py_function, "__globals__") orelse return error.Python;
        defer py.Py_DecRef(globals);
        var out: std.ArrayListUnmanaged(MethodCandidate) = .empty;
        const pt = try pyTypes();
        const key = ph.newString(name) orelse return error.Python;
        defer py.Py_DecRef(key);
        var pos: py.Py_ssize_t = 0;
        var gk: ?*PyObject = null;
        var v: ?*PyObject = null;
        while (py.c.PyDict_Next(globals, &pos, @ptrCast(&gk), @ptrCast(&v)) != 0) {
            if (!try isInstanceOf(v.?, @ptrCast(@alignCast(py.types.typeObject("PyType_Type"))))) continue;
            const rtype = try recordOf(v.?) orelse continue;
            const m = py.c.PyObject_GetAttr(v.?, key) orelse {
                py.c.PyErr_Clear();
                continue;
            };
            defer py.Py_DecRef(m);
            if (!try isInstanceOf(m, pt.function)) continue;
            // (one the front can't read: not a candidate, Python runs it)
            const func = self.helperFunction(m) catch |e| switch (e) {
                error.Unsupported => continue,
                else => return e,
            };
            if (nargs < func.required or nargs > func.param_count) continue;
            _ = try c.objectIndex(m);
            try out.append(self.a(), .{ .rtype = rtype, .func = func });
        }
        return out.items;
    }

    /// obj.name(args) checked against each candidate's record type: its
    /// method's compiled code (out of line); none: the generic call.
    fn recordMethodCall(self: *Gen, inst: *Inst, d: Dyn, name: []const u8, args: []const SVal, cands: []const MethodCandidate) Error!SVal {
        const f = &self.f;
        const t = self.c.m.t;
        // (the arguments run-time values first: each path takes them)
        const all = try self.a().alloc(SVal, args.len + 1);
        all[0] = .{ .dyn = d };
        for (args, all[1..]) |x, *slot| slot.* = .{ .dyn = try self.materialize(x, inst.node) };
        const result = try self.valSlot();
        const join = try f.label("method_done");
        const is_record = f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(value.Tag.record)));
        inst.dyn_depth += 1;
        defer inst.dyn_depth -= 1;
        for (cands) |cand| {
            const yes = try f.label("method_of");
            const no = try f.label("method_next");
            const of_type = try f.label("method_check");
            try f.condBr(is_record, of_type, no);
            try f.block(of_type);
            // (a record's type: the word after its header)
            const rt = f.load(t.i64, f.offset(f.intToPtr(d.bits), 16));
            try f.condBr(f.icmp(jit_c.LLVMIntEQ, rt, self.c.m.addrInt(@intFromPtr(cand.rtype))), yes, no);
            try f.block(yes);
            const r = try self.materialize(try self.outOfLine(cand.func, inst.node, try self.withDefaults(cand.func, all)), inst.node);
            try self.storeSlot(result, r);
            try f.br(join);
            try f.block(no);
        }
        const g = try self.materialize(try self.genericMethodCall(inst, d, name, all[1..]), inst.node);
        try self.storeSlot(result, g);
        try f.br(join);
        try f.block(join);
        return .{ .dyn = try self.loadSlot(result, .any) };
    }

    fn allScalar(items: []const SVal) bool {
        for (items) |x| if (!isScalar(x)) return false;
        return true;
    }

    /// A copy of a value read out of a known container (a reference of its
    /// own for a run-time one).
    fn copyOf(self: *Gen, v: SVal) Error!SVal {
        if (v != .dyn) return v;
        // (a variable's value borrowed: borrowed again)
        if (v.dyn.state) |state| for (self.borrows.items) |b| {
            if (b.state == state) {
                if (try self.borrowable(b.sym)) return .{ .dyn = try self.borrowAgain(v.dyn, b.sym) };
                break;
            }
        };
        try self.increfDyn(v.dyn);
        return v;
    }

    /// Another reference to a variable's value read borrowed: borrowed too
    /// while the variable still has it (the first is borrowed), its own
    /// reference taken if not.
    fn borrowAgain(self: *Gen, d: Dyn, si: u32) Error!Dyn {
        const f = &self.f;
        const t = self.c.m.t;
        const state = try f.alloca(t.i64);
        f.entryStore(self.k(2), state);
        const kept = try f.alloca(t.val);
        f.store(d.tag, kept);
        f.store(d.bits, f.field(t.val, kept, 1));
        const still = try f.label("borrow_again");
        const take = try f.label("borrow_own");
        const done = try f.label("borrow_copied");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, f.load(t.i64, d.state.?), self.k(0)), still, take);
        try f.block(still);
        f.store(self.k(0), state);
        try f.br(done);
        try f.block(take);
        try self.refcount(false, d.tag, d.bits);
        f.store(self.k(1), state);
        try f.br(done);
        try f.block(done);
        try self.borrows.append(self.a(), .{ .sym = si, .state = state, .kept = kept });
        var out = d;
        out.state = state;
        return out;
    }

    fn boxed(self: *Gen, v: SVal) Error!*const SVal {
        const p = try self.a().create(SVal);
        p.* = v;
        return p;
    }

    // ------------------------------------------------------------------
    // Calls
    // ------------------------------------------------------------------

    /// A call expression of a semantic.
    fn callExpr(self: *Gen, inst: *Inst, func_e: *const front.Expr, args_e: []const *const front.Expr, kws: []const front.Keyword, pos: front.Pos) Error!SVal {
        const c = self.c;
        // obj.name(...): a method of a value, or an attribute of rt, a node,
        // a module
        var callee: SVal = undefined;
        if (func_e.kind == .attr) {
            const obj = try self.expr(inst, func_e.kind.attr.obj);
            switch (obj) {
                .rt, .node, .py => callee = try self.attr(inst, obj, func_e.kind.attr.name, func_e.pos),
                else => {
                    if (kws.len != 0) return c.unsupportedAt(inst.func, pos, "keyword arguments to a method aren't compiled yet", .{});
                    const margs = try self.a().alloc(SVal, args_e.len);
                    for (margs, args_e) |*slot, ae| slot.* = try self.expr(inst, ae);
                    return self.methodCall(inst, obj, func_e.kind.attr.name, margs, pos);
                },
            }
        } else callee = try self.expr(inst, func_e);
        // all() / any() of a generator expression: up to the item deciding
        if (callee == .py and args_e.len == 1 and kws.len == 0 and args_e[0].kind == .gen_exp) {
            if (isBuiltin(callee.py, "all")) return self.allAny(inst, args_e[0].kind.gen_exp, false, pos);
            if (isBuiltin(callee.py, "any")) return self.allAny(inst, args_e[0].kind.gen_exp, true, pos);
        }
        // (arguments in order, as Python evaluates them)
        const args = try self.a().alloc(SVal, args_e.len);
        for (args, args_e) |*slot, ae| slot.* = try self.expr(inst, ae);
        var receiver: ?SVal = null;
        // A helper's keyword arguments: in their parameters' places
        if (kws.len > 0 and callee == .py and try isInstanceOf(callee.py, (try pyTypes()).function)) {
            const func = try self.helperFunction(callee.py);
            const all = try self.a().alloc(?SVal, func.param_count);
            @memset(all, null);
            if (args.len > func.param_count) return c.unsupportedAt(inst.func, pos, "{s}() takes {d} arguments", .{ func.name, func.param_count });
            for (args, 0..) |x, i| all[i] = x;
            for (kws) |kw| {
                const i = for (func.locals[0..func.param_count], 0..) |p, j| {
                    if (std.mem.eql(u8, p, kw.name)) break j;
                } else return c.unsupportedAt(inst.func, pos, "{s}() has no parameter {s}", .{ func.name, kw.name });
                if (all[i] != null) return c.unsupportedAt(inst.func, pos, "{s}() given {s} twice", .{ func.name, kw.name });
                all[i] = try self.expr(inst, kw.value);
            }
            const full = try self.a().alloc(SVal, func.param_count);
            for (all, full, 0..) |x, *slot, i| slot.* = x orelse (if (i < func.required) return c.unsupportedAt(inst.func, pos, "{s}() missing its argument {s}", .{ func.name, func.locals[i] }) else try self.defaultOf(func, i));
            return self.callHelper(func, inst.node, full);
        }
        for (kws) |kw| {
            if (callee == .rt_method and callee.rt_method == .call and std.mem.eql(u8, kw.name, "receiver")) {
                receiver = try self.expr(inst, kw.value);
            } else if (callee == .rt_method and callee.rt_method == .@"error" and std.mem.eql(u8, kw.name, "code")) {
                // (the code isn't kept by compiled errors yet: reported as runtime)
                try self.drop(try self.expr(inst, kw.value));
            } else return c.unsupportedAt(inst.func, pos, "the keyword argument {s}= isn't compiled", .{kw.name});
        }
        switch (callee) {
            .rt_method => |m| return self.rtCall(inst, m, args, receiver, pos),
            .py => |o| return self.pyCall(inst, o, args, pos),
            .dyn => {
                const l = try self.a().create(SList);
                l.* = .{};
                try l.items.appendSlice(self.a(), args);
                return self.dynCall(inst, callee, .{ .list = l }, null);
            },
            else => return c.unsupportedAt(inst.func, pos, "calling a {s} isn't compiled yet", .{@tagName(callee)}),
        }
    }

    fn rtCall(self: *Gen, inst: *Inst, m: RtMethod, args: []const SVal, receiver: ?SVal, pos: front.Pos) Error!SVal {
        const c = self.c;
        const want: usize = switch (m) {
            .eval, .exec, .loop, .load, .function, .kind, .text, .span, .scope, .symbol, .type_of, .node_at, .fresh => 1,
            .store, .call => 2,
            .@"error" => 2,
            .Return => if (args.len == 0) 0 else 1,
            .Break, .Continue => 0,
        };
        if (args.len != want) return c.unsupportedAt(inst.func, pos, "rt.{s}() takes {d} arguments here", .{ @tagName(m), want });
        // A node only known at run time: the method of an rt over the
        // frames here, then
        switch (m) {
            .load, .store, .function, .kind, .text, .span, .scope, .symbol, .type_of, .node_at => if (args[0] == .dyn) {
                const r = try self.materialize(.rt, inst.node);
                return self.methodCall(inst, .{ .dyn = r }, @tagName(m), args, pos);
            },
            else => {},
        }
        switch (m) {
            .eval => return self.evalValue(args[0]),
            .exec => {
                try self.execValue(args[0]);
                return .none;
            },
            .loop => return self.rtLoop(args[0]),
            .load => {
                const n = try self.nodeArg(inst, args[0], pos);
                return self.loadVar(n);
            },
            .store => {
                const n = try self.nodeArg(inst, args[0], pos);
                try self.storeVar(n, args[1]);
                return .none;
            },
            .function => return self.makeFunction(try self.nodeArg(inst, args[0], pos)),
            .fresh => {
                try self.freshScope(try self.nodeArg(inst, args[0], pos));
                return .none;
            },
            .call => return self.dynCall(inst, args[0], args[1], receiver),
            .@"error" => {
                const n = switch (args[0]) {
                    .node => |x| x,
                    .none => inst.node,
                    else => return c.unsupportedAt(inst.func, pos, "rt.error's node must be known when compiling", .{}),
                };
                const msg = switch (args[1]) {
                    .str => |s| s,
                    else => return c.unsupportedAt(inst.func, pos, "rt.error's message must be known when compiling (for now)", .{}),
                };
                try self.failAt(n, msg);
                return .none;
            },
            .kind => {
                const n = try self.nodeArg(inst, args[0], pos);
                return .{ .str = c.data.grammar.kind_names[c.data.rule(n)] };
            },
            .text => return .{ .str = c.data.text(try self.nodeArg(inst, args[0], pos)) },
            .span => {
                const n = c.data.nodes[try self.nodeArg(inst, args[0], pos)];
                const items = try self.a().alloc(SVal, 2);
                items[0] = .{ .pint = n.text_start };
                items[1] = .{ .pint = n.text_end };
                return .{ .tuple = items };
            },
            .scope => {
                const n = try self.nodeArg(inst, args[0], pos);
                const si = c.data.symbolIndex(n) orelse return .none;
                const s = c.data.syms[si].scope;
                return if (s == NONE or s >= c.data.nodes.len) SVal.none else SVal{ .node = s };
            },
            .symbol, .type_of => {
                const n = try self.nodeArg(inst, args[0], pos);
                const analysis = c.lang.analysis orelse return .none;
                const r = py.c.PyObject_CallMethod(analysis, if (m == .symbol) "resolve" else "type_of", "I", @as(c_uint, n)) orelse return error.Python;
                const sv = try self.constant(r, n);
                if (sv == .py) _ = try c.objectIndex(r);
                py.Py_DecRef(r);
                return sv;
            },
            .node_at => switch (args[0]) {
                .int, .pint => |i| return .{ .node = @intCast(i) },
                else => return c.unsupportedAt(inst.func, pos, "rt.node_at's index must be known when compiling", .{}),
            },
            .Return => return .{ .control = .{ .kind = .Return, .value = if (args.len == 1) try self.boxed(args[0]) else null } },
            .Break, .Continue => return .{ .control = .{ .kind = m, .value = null } },
        }
    }

    fn nodeArg(self: *Gen, inst: *Inst, v: SVal, pos: front.Pos) Error!u32 {
        return switch (v) {
            .node => |n| n,
            else => self.c.unsupportedAt(inst.func, pos, "a node only known at run time can't be used here", .{}),
        };
    }

    /// rt.loop(body): run the body; false if it broke out.
    fn rtLoop(self: *Gen, body: SVal) Error!SVal {
        const f = &self.f;
        const m = &self.c.m;
        const flag = try f.alloca(m.t.i1);
        f.store(m.k1(true), flag);
        const brk = try f.label("loop_break");
        const cont = try f.label("loop_continue");
        const done = try f.label("loop_done");
        try self.loops.append(self.a(), .{ .brk = brk, .cont = cont, .depth = self.insts.items.len, .scope_depth = self.scopes.items.len, .tries = self.tries.items.len });
        self.loop_level += 1;
        defer self.loop_level -= 1;
        try self.execValue(body);
        _ = self.loops.pop();
        try f.br(done);
        try f.block(brk);
        f.store(m.k1(false), flag);
        try f.br(done);
        try f.block(cont);
        try f.br(done);
        try f.block(done);
        return self.boolDyn(f.load(m.t.i1, flag));
    }

    /// A run-time bool from an i1.
    fn boolDyn(self: *Gen, b: ir.Value) SVal {
        return dyn(self.k(1), self.f.zext64(b), .bool);
    }

    /// Call a run-time function value (or a host function) with arguments.
    fn dynCall(self: *Gen, inst: *Inst, fv: SVal, args_v: SVal, receiver: ?SVal) Error!SVal {
        const items: []const SVal = switch (args_v) {
            .list => |l| l.items.items,
            .tuple => |t| t,
            else => return self.seqCall(inst, fv, args_v, receiver),
        };
        const fd = try self.materialize(fv, inst.node);
        // The arguments, in a stack array
        const n = items.len;
        const arr = try self.valueSlots(n);
        const ds = try self.a().alloc(Dyn, n);
        for (items, 0..) |item, i| {
            ds[i] = try self.materialize(item, inst.node);
            try self.storeSlot(self.elem(arr, i), ds[i]);
        }
        var recv_ptr = self.c.m.nullPtr();
        var recv_d: ?Dyn = null;
        if (receiver) |r| {
            recv_d = try self.materialize(r, inst.node);
            const p = try self.valSlot();
            try self.storeSlot(p, recv_d.?);
            recv_ptr = p;
        }
        const ok = if (fd.func != NONE and n == try self.paramCount(fd.func))
            try self.directCall(inst, fd, arr, ds, recv_ptr)
        else
            self.call("zr_call", &.{ self.ctx, self.k32(inst.node), fd.tag, fd.bits, arr, self.k(@intCast(n)), recv_ptr, self.out });
        // (the call borrowed them)
        for (ds) |d| try self.drop(.{ .dyn = d });
        if (recv_d) |r| try self.drop(.{ .dyn = r });
        try self.drop(.{ .dyn = fd });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.any) };
    }

    /// The parameters a language function takes.
    fn paramCount(self: *Gen, fnode: u32) Error!usize {
        const spec = self.c.specOf(fnode) orelse return std.math.maxInt(usize);
        return self.paramNodes(fnode, spec).len;
    }

    /// A call of a function value that is most likely language function
    /// `fd.func` given all its parameters: its code called directly, the
    /// language's call stack kept inline, when the value's code is that
    /// function's and the stack has room (arguments of the kinds its typed
    /// entry takes: that, given them plain); anything else by zr_call. The
    /// call's status (an i1).
    fn directCall(self: *Gen, inst: *Inst, fd: Dyn, arr: ir.Value, ds: []const Dyn, recv_ptr: ir.Value) Error!ir.Value {
        const f = &self.f;
        const m = &self.c.m;
        const t = m.t;
        const n = ds.len;
        const code = try self.c.functionCode(fd.func);
        const typed = if (try self.typedParams(fd.func)) |shapes| for (shapes, ds) |s, d| {
            if (d.shape != s) break false;
        } else true else false;
        const is_fn = try f.label("call_is_fn");
        const direct = try f.label("call_direct");
        const slow = try f.label("call_generic");
        const join = try f.label("call_done");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, fd.tag, self.k(@intFromEnum(value.Tag.function))), is_fn, slow);
        try f.block(is_fn);
        const fo = f.intToPtr(fd.bits);
        const its_code = f.load(t.i64, f.offset(fo, @offsetOf(value.Function, "code")));
        const ctx = self.ctx;
        const depth_p = f.offset(ctx, @offsetOf(helpers.Ctx, "depth"));
        const depth = f.load(t.i64, depth_p);
        const room = f.load(t.i64, f.offset(ctx, @offsetOf(helpers.Ctx, "calls_room")));
        // (the stack's room is max_depth: below it, a call is allowed)
        const same = f.icmp(jit_c.LLVMIntEQ, its_code, f.ptrToInt(code));
        try f.condBr(f.and_(same, f.icmp(jit_c.LLVMIntULT, depth, room)), direct, slow);
        try f.block(direct);
        const calls = f.load(t.ptr, f.offset(ctx, @offsetOf(helpers.Ctx, "calls")));
        const entry = f.offset(calls, 0);
        const at = f.at(L("LLVMArrayType2")(t.i8, @sizeOf(helpers.CallEntry)), entry, depth);
        f.store(f.load(t.ptr, f.offset(fo, @offsetOf(value.Function, "name"))), at);
        f.store(self.k32(inst.node), f.offset(at, @offsetOf(helpers.CallEntry, "node")));
        f.store(f.add(depth, self.k(1)), depth_p);
        const env = f.load(t.ptr, f.offset(fo, @offsetOf(value.Function, "env")));
        const st = if (typed) blk: {
            const entry_code = try self.c.functionCode(fd.func | Compiler.TYPED);
            const params = try self.a().alloc(ir.Value, 4 + n);
            @memcpy(params[0..4], &[_]ir.Value{ ctx, env, recv_ptr, self.out });
            for (params[4..], ds) |*p, d| p.* = d.bits;
            break :blk f.call(.{ .v = entry_code, .ty = (try self.c.llvmFunction(fd.func | Compiler.TYPED)).ty }, params);
        } else f.call(.{ .v = code, .ty = (try self.c.llvmFunction(fd.func)).ty }, &.{ ctx, env, arr, self.k(@intCast(n)), recv_ptr, self.out });
        f.store(depth, depth_p);
        // (its result as rt.call gives it: an int an I64)
        const out_tag = f.load(t.i64, self.out);
        f.store(f.select(f.icmp(jit_c.LLVMIntEQ, out_tag, self.k(@intCast(value.PINT_TAG))), self.k(@intFromEnum(value.Tag.int)), out_tag), self.out);
        const direct_end = f.current;
        try f.br(join);
        try f.block(slow);
        const st2 = self.call("zr_call", &.{ self.ctx, self.k32(inst.node), fd.tag, fd.bits, arr, self.k(@intCast(n)), recv_ptr, self.out });
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        return f.phi(t.i1, st, direct_end, st2, slow_end);
    }

    /// rt.call(f, args) with args only known at run time (all taken).
    fn seqCall(self: *Gen, inst: *Inst, fv: SVal, args_v: SVal, receiver: ?SVal) Error!SVal {
        const fd = try self.materialize(fv, inst.node);
        const ad = try self.materialize(args_v, inst.node);
        var recv_ptr = self.c.m.nullPtr();
        var recv_d: ?Dyn = null;
        if (receiver) |r| {
            recv_d = try self.materialize(r, inst.node);
            const p = try self.valSlot();
            try self.storeSlot(p, recv_d.?);
            recv_ptr = p;
        }
        const ok = self.call("zr_call_seq", &.{ self.ctx, self.k32(inst.node), fd.tag, fd.bits, ad.tag, ad.bits, recv_ptr, self.out });
        if (recv_d) |r| try self.drop(.{ .dyn = r });
        try self.drop(.{ .dyn = ad });
        try self.drop(.{ .dyn = fd });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.any) };
    }

    /// Calling a Python object known when compiling: a helper function of
    /// the semantics (compiled inline), or a builtin.
    fn pyCall(self: *Gen, inst: *Inst, o: *PyObject, args: []const SVal, pos: front.Pos) Error!SVal {
        const c = self.c;
        const pt = try pyTypes();
        // A bound method of a Python function: the function, its object
        // first
        if (try isInstanceOf(o, pt.method)) {
            const func = ph.attr(o, "__func__") orelse return error.Python;
            defer py.Py_DecRef(func);
            const recv = ph.attr(o, "__self__") orelse return error.Python;
            defer py.Py_DecRef(recv);
            if (try isInstanceOf(func, pt.function)) {
                // (both kept alive by the method, which the code keeps)
                _ = try c.objectIndex(o);
                const all = try self.a().alloc(SVal, args.len + 1);
                all[0] = try self.constant(recv, inst.node);
                @memcpy(all[1..], args);
                return self.pyCall(inst, func, all, pos);
            }
        }
        // A Python function of the semantics' module (a def: it has code):
        // compiled too
        if (try isInstanceOf(o, pt.function)) {
            const func = try self.helperFunction(o);
            return self.callHelper(func, inst.node, args);
        }
        // A class whose objects are records: made natively
        if (try isInstanceOf(o, @ptrCast(@alignCast(py.types.typeObject("PyType_Type"))))) if (try c.recordType(o)) |rtype| {
            if (!rtype.slots) {
                // (a dataclass: its fields, in order)
                if (args.len != rtype.fields.len) return c.unsupportedAt(inst.func, pos, "{s}() takes {d} fields, given {d} (keywords and defaults aren't compiled yet)", .{ rtype.name, rtype.fields.len, args.len });
                const arr = try self.valueArray(args, inst.node);
                try self.callCheck("zr_record", &.{ self.ctx, self.k32(inst.node), self.ptrConst(rtype), arr, self.out });
                return .{ .dyn = try self.loadOut(.record) };
            }
            // (a class with __slots__: its fields unset, then its __init__
            // on it, compiled)
            try self.callCheck("zr_record_new", &.{ self.ctx, self.k32(inst.node), self.ptrConst(rtype), self.out });
            const rec = try self.loadOut(.record);
            const init = ph.attr(o, "__init__") orelse return error.Python;
            defer py.Py_DecRef(init);
            if (try isInstanceOf(init, pt.function)) {
                try self.increfDyn(rec);
                const all = try self.a().alloc(SVal, args.len + 1);
                all[0] = .{ .dyn = rec };
                @memcpy(all[1..], args);
                _ = try c.objectIndex(init);
                try self.drop(try self.callHelper(try self.helperFunction(init), inst.node, all));
            } else if (args.len != 0) {
                for (args) |x| try self.drop(x);
                try self.drop(.{ .dyn = rec });
                try self.failAt(inst.node, try std.fmt.allocPrint(self.a(), "{s}() takes no arguments", .{rtype.name}));
                return .none;
            }
            return .{ .dyn = rec };
        };
        if (try self.builtinCall(inst, o, args, pos)) |v| return v;
        // Anything else: called as Python does (its arguments as Python
        // objects)
        return self.callPython(inst, o, args);
    }

    /// A pointer known when compiling, as an IR constant.
    fn ptrConst(self: *Gen, p: anytype) ir.Value {
        return self.c.m.ptrConst(@intFromPtr(p));
    }

    /// Call a Python object at run time with the arguments.
    fn callPython(self: *Gen, inst: *Inst, o: *PyObject, args: []const SVal) Error!SVal {
        const idx = try self.c.objectIndex(o);
        const arr = try self.valueArray(args, inst.node);
        const ok = self.call("zr_call_python", &.{ self.ctx, self.k32(inst.node), self.k(@intCast(idx)), arr, self.k(@intCast(args.len)), self.out });
        try self.dropArray(arr, args.len);
        try self.check(ok);
        return .{ .dyn = try self.loadOut(.any) };
    }

    fn isBuiltin(o: *PyObject, name: [*:0]const u8) bool {
        const builtins = py.c.PyImport_ImportModule("builtins") orelse {
            py.c.PyErr_Clear();
            return false;
        };
        defer py.Py_DecRef(builtins);
        const b = py.c.PyObject_GetAttrString(builtins, name) orelse {
            py.c.PyErr_Clear();
            return false;
        };
        defer py.Py_DecRef(b);
        return b == o;
    }

    /// The builtins compiled code knows; null for another callable.
    fn builtinCall(self: *Gen, inst: *Inst, o: *PyObject, args: []const SVal, pos: front.Pos) Error!?SVal {
        const c = self.c;
        // Known arguments: Python's result now (int("12"), len("ab"), ...)
        if (allScalar(args) and !isBuiltin(o, "print") and !isBuiltin(o, "input")) {
            const pure = [_][*:0]const u8{ "int", "float", "str", "bool", "len", "abs", "min", "max", "round", "repr", "ord", "chr", "hex", "oct", "bin", "divmod", "pow", "hash" };
            for (pure) |name| if (isBuiltin(o, name)) {
                const tuple = py.c.PyTuple_New(@intCast(args.len)) orelse return error.Python;
                defer py.Py_DecRef(tuple);
                for (args, 0..) |x, i| _ = py.c.PyTuple_SetItem(tuple, @intCast(i), try self.pyOf(x));
                if (py.c.PyObject_CallObject(o, tuple)) |r| {
                    defer py.Py_DecRef(r);
                    return try self.constant(r, inst.node);
                }
                py.c.PyErr_Clear();
                // (it fails when run: at run time, as Python)
                break;
            };
        }
        // int(), float(), len(), abs(), str(), bool() of a run-time value:
        // natively where it can be (zr_builtin)
        if (args.len == 1 and args[0] == .dyn and isBuiltin(o, "len")) return .{ .dyn = try self.lenInline(inst, o, args[0].dyn) };
        if (args.len == 1 and args[0] == .dyn) {
            inline for (@typeInfo(helpers.Builtin).@"enum".fields) |fd| {
                if (isBuiltin(o, fd.name)) {
                    const d = args[0].dyn;
                    const idx = try c.objectIndex(o);
                    const ok = self.call("zr_builtin", &.{ self.ctx, self.k32(inst.node), self.k32(fd.value), self.k(@intCast(idx)), d.tag, d.bits, self.out });
                    try self.drop(args[0]);
                    try self.check(ok);
                    // (len() is always an int, bool() a bool; the others
                    // may be anything a class's method made)
                    const shape: Shape = comptime if (std.mem.eql(u8, fd.name, "len")) .int else if (std.mem.eql(u8, fd.name, "bool")) .bool else .any;
                    return SVal{ .dyn = try self.loadOut(shape) };
                }
            }
        }
        // type(v): its class (known for a known value; else from its tag)
        if (isBuiltin(o, "type") and args.len == 1) switch (args[0]) {
            .dyn => |d| {
                _ = self.call("zr_type", &.{ d.tag, d.bits, self.out });
                try self.drop(args[0]);
                return SVal{ .dyn = try self.loadOut(.any) };
            },
            .rt, .rt_method, .control, .method => {},
            else => {
                const x = try self.pyOf(args[0]);
                defer py.Py_DecRef(x);
                const t: *PyObject = @ptrCast(@alignCast(ph.typeOf(x)));
                _ = try c.objectIndex(t);
                return SVal{ .py = t };
            },
        };
        if (isBuiltin(o, "isinstance")) {
            if (args.len != 2) return c.unsupportedAt(inst.func, pos, "isinstance() takes 2 arguments", .{});
            return try self.isInstance(inst, args[0], args[1], pos);
        }
        if (isBuiltin(o, "len")) {
            if (args.len == 1) switch (args[0]) {
                .list => |l| return SVal{ .pint = @intCast(l.items.items.len) },
                .tuple => |t| return SVal{ .pint = @intCast(t.len) },
                .dict => |d| return SVal{ .pint = @intCast(d.keys.items.len) },
                else => {},
            };
            return try self.callPython(inst, o, args);
        }
        if (isBuiltin(o, "zip") or isBuiltin(o, "enumerate")) {
            // Known sequences: the pairs now
            const is_zip = isBuiltin(o, "zip");
            var seqs: [8][]const SVal = undefined;
            var known = args.len <= seqs.len;
            if (known) for (args, 0..) |x, i| {
                seqs[i] = switch (x) {
                    .list => |l| l.items.items,
                    .tuple => |t| t,
                    else => blk: {
                        known = false;
                        break :blk &.{};
                    },
                };
            };
            if (known and (is_zip or args.len == 1)) {
                const out = try self.a().create(SList);
                out.* = .{};
                var n: usize = std.math.maxInt(usize);
                for (seqs[0..args.len]) |s| n = @min(n, s.len);
                if (args.len == 0) n = 0;
                for (0..n) |i| {
                    const items = try self.a().alloc(SVal, if (is_zip) args.len else 2);
                    if (is_zip) {
                        for (seqs[0..args.len], 0..) |s, j| items[j] = try self.copyOf(s[i]);
                    } else {
                        items[0] = .{ .pint = @intCast(i) };
                        items[1] = try self.copyOf(seqs[0][i]);
                    }
                    try out.items.append(self.a(), .{ .tuple = items });
                }
                return SVal{ .list = out };
            }
            // At run time: the items of each, paired here as a loop would
            return try self.zipRuntime(inst, is_zip, args, pos);
        }
        if (isBuiltin(o, "range")) {
            if (allScalar(args) and args.len >= 1 and args.len <= 3) {
                var lo: i128 = 0;
                var hi: i128 = 0;
                var step: i128 = 1;
                for (args) |x| if (x == .bool or intOf(x) == null) return null;
                if (args.len == 1) hi = intOf(args[0]).? else {
                    lo = intOf(args[0]).?;
                    hi = intOf(args[1]).?;
                    if (args.len == 3) step = intOf(args[2]).?;
                }
                if (step == 0) return null;
                const count: i128 = if (step > 0) @max(0, @divFloor(hi - lo + step - 1, step)) else @max(0, @divFloor(lo - hi - step - 1, -step));
                // (a few: unrolled where it's looped over; more, a loop at
                // run time: a body copied each time over would make code
                // without end)
                if (count <= max_unrolled) {
                    const out = try self.a().create(SList);
                    out.* = .{};
                    var i: i128 = 0;
                    // (plain ints, as range gives them)
                    while (i < count) : (i += 1) try out.items.append(self.a(), .{ .pint = @intCast(lo + i * step) });
                    return SVal{ .list = out };
                }
            }
            // (else Python's range object, a loop over it counted:
            // iteration())
            return null;
        }
        return null;
    }

    /// isinstance(v, T): known for known values; a tag check at run time.
    fn isInstance(self: *Gen, inst: *Inst, v: SVal, t: SVal, pos: front.Pos) Error!SVal {
        const c = self.c;
        const types_: []const SVal = switch (t) {
            .tuple => |x| x,
            else => &.{t},
        };
        var result: SVal = .{ .bool = false };
        for (types_, 0..) |ty, i| {
            const o = switch (ty) {
                .py => |x| x,
                else => return c.unsupportedAt(inst.func, pos, "isinstance()'s type must be known when compiling", .{}),
            };
            if (i + 1 < types_.len and v == .dyn) try self.increfDyn(v.dyn);
            const one = try self.isOne(inst, v, o, pos);
            result = if (i == 0) one else try self.orValues(inst, result, one);
        }
        return result;
    }

    fn isOne(self: *Gen, inst: *Inst, v: SVal, o: *PyObject, pos: front.Pos) Error!SVal {
        const c = self.c;
        if (!try isInstanceOf(o, @ptrCast(@alignCast(py.types.typeObject("PyType_Type")))))
            return c.unsupportedAt(inst.func, pos, "isinstance()'s second argument must be a class or a tuple of them", .{});
        // A Python object known when compiling: its class doesn't change
        if (v == .py) return .{ .bool = try isInstanceOf(v.py, o) };
        // A class whose objects are records: a record of it (or of a
        // subclass), or a Python object of it
        if (try c.recordType(o)) |rtype| {
            if (v.isStatic()) return .{ .bool = false };
            return self.isRecord(v.dyn, rtype);
        }
        if (o == objects_mod.FunctionType) {
            if (v.isStatic()) return .{ .bool = false };
            return self.isType(v.dyn, 8);
        }
        const codes = [_]struct { [*:0]const u8, u32 }{ .{ "int", 0 }, .{ "float", 1 }, .{ "str", 2 }, .{ "bool", 3 }, .{ "list", 4 }, .{ "tuple", 5 }, .{ "dict", 6 } };
        for (codes) |entry| if (isBuiltin(o, entry[0])) {
            if (v.isStatic()) return .{ .bool = switch (entry[1]) {
                0 => v == .int or v == .pint or v == .bool,
                1 => v == .float,
                2 => v == .str,
                3 => v == .bool,
                4 => v == .list,
                5 => v == .tuple,
                6 => v == .dict,
                else => false,
            } };
            return self.isType(v.dyn, entry[1]);
        };
        // Any other class: as Python answers it, the value as Python sees it
        const d = try self.materialize(v, inst.node);
        const idx = try c.objectIndex(o);
        try self.callCheck("zr_isinstance", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, self.k(@intCast(idx)), self.out });
        try self.drop(.{ .dyn = d });
        return .{ .dyn = try self.loadOut(.bool) };
    }

    /// isinstance() of a run-time value and a builtin type (`code`: as
    /// zr_is_type's): its tag, inline; a Python object's type by
    /// zr_is_type (subclasses...). The value dropped.
    fn isType(self: *Gen, d: Dyn, code: u32) Error!SVal {
        // (a value of a known kind: known)
        if (d.shape != .any) {
            const s = d.shape;
            const known = switch (code) {
                0 => s == .int or s == .bool,
                1 => s == .float,
                2 => s == .str,
                3 => s == .bool,
                4 => s == .list,
                5 => s == .tuple,
                6 => s == .dict,
                8 => s == .function,
                else => false,
            };
            try self.drop(.{ .dyn = d });
            return .{ .bool = known };
        }
        const f = &self.f;
        const T = value.Tag;
        const tags: []const u64 = switch (code) {
            // (int: an I64, a plain int, a bool, a Big)
            0 => &.{ @intFromEnum(T.int), value.PINT_TAG, @intFromEnum(T.bool), @intFromEnum(T.big) },
            1 => &.{@intFromEnum(T.float)},
            2 => &.{@intFromEnum(T.str)},
            3 => &.{@intFromEnum(T.bool)},
            4 => &.{@intFromEnum(T.list)},
            5 => &.{@intFromEnum(T.tuple)},
            6 => &.{@intFromEnum(T.dict)},
            8 => &.{@intFromEnum(T.function)},
            else => unreachable,
        };
        var native = f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intCast(tags[0])));
        for (tags[1..]) |t| native = f.or_(native, f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intCast(t))));
        const native64 = f.zext64(native);
        const host = try f.label("is_host");
        const join = try f.label("is_joined");
        const start = f.current;
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(T.host))), host, join);
        try f.block(host);
        const r = f.zext64(self.call("zr_is_type", &.{ d.tag, d.bits, self.k32(code) }));
        const host_end = f.current;
        try f.br(join);
        try f.block(join);
        const res = f.phi(self.c.m.t.i64, native64, start, r, host_end);
        try self.drop(.{ .dyn = d });
        return dyn(self.k(1), res, .bool);
    }

    /// isinstance() of a run-time value and a record class: a record of
    /// exactly it, inline; one of another type (a subclass?) or a Python
    /// object by zr_is_record; anything else isn't. The value dropped.
    fn isRecord(self: *Gen, d: Dyn, rtype: *const value.RecordType) Error!SVal {
        // (a value of a known kind: known)
        if (d.shape != .any and (d.shape != .record or d.rtype != null)) {
            const known = d.shape == .record and d.rtype.?.isA(rtype);
            try self.drop(.{ .dyn = d });
            return .{ .bool = known };
        }
        const f = &self.f;
        const t = self.c.m.t;
        const T = value.Tag;
        const maybe = try f.label("isrec_maybe");
        const rec = try f.label("isrec_record");
        const slow = try f.label("isrec_slow");
        const join = try f.label("isrec_joined");
        const is_rec = f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(T.record)));
        const start = f.current;
        try f.condBr(f.or_(is_rec, f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(T.host)))), maybe, join);
        try f.block(maybe);
        try f.condBr(is_rec, rec, slow);
        try f.block(rec);
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, self.recordTypeOf(d), self.c.m.addrInt(@intFromPtr(rtype))), join, slow);
        try f.block(slow);
        const r = f.zext64(self.call("zr_is_record", &.{ d.tag, d.bits, self.ptrConst(rtype) }));
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        const res = f.phiN(t.i64, &.{ self.k(0), self.k(1), r }, &.{ start, rec, slow_end });
        try self.drop(.{ .dyn = d });
        return dyn(self.k(1), res, .bool);
    }

    /// `type(a) is type(b)`, `type(a) is C` (`is not`): for values of the
    /// compiled code, a key for each one's class from its tag (a record's:
    /// its type), compared inline (no class objects made); a Python
    /// object's class by type() itself. Null: not that.
    fn typeIs(self: *Gen, inst: *Inst, op: front.CmpOp, l_e: *const front.Expr, r_e: *const front.Expr, pos: front.Pos) Error!?SVal {
        const f = &self.f;
        const t = self.c.m.t;
        var a_e = try self.typeArg(inst, l_e);
        var other = r_e;
        if (a_e == null) {
            a_e = try self.typeArg(inst, r_e);
            other = l_e;
        }
        const ae = a_e orelse return null;
        const type_obj = (try self.typeArgCallee(inst, if (other == r_e) l_e else r_e)).?;
        const be = try self.typeArg(inst, other);
        // (the other: type(b), or a class's name (nothing to run))
        var cls: ?*PyObject = null;
        var cls_key: ?ir.Value = null;
        if (be == null) {
            if (other.kind != .global) return null;
            const cv = try self.global(inst, other.kind.global, other.pos);
            if (cv != .py) return null;
            cls_key = try self.classKey(cv.py) orelse return null;
            cls = cv.py;
        }
        // (in Python's order: the left one first)
        var va: SVal = undefined;
        var vb: ?SVal = null;
        if (other == r_e) {
            va = try self.expr(inst, ae);
            if (be) |x| vb = try self.expr(inst, x);
        } else {
            if (be) |x| vb = try self.expr(inst, x);
            va = try self.expr(inst, ae);
        }
        // Known values: as type() and `is` do it
        if (va != .dyn or (vb != null and vb.? != .dyn)) {
            const ta = (try self.builtinCall(inst, type_obj, &.{va}, pos)).?;
            const tb = if (vb) |bv| (try self.builtinCall(inst, type_obj, &.{bv}, pos)).? else SVal{ .py = cls.? };
            return try self.compare(inst, op, ta, tb);
        }
        const ka = try self.typeKey(va.dyn);
        const kb = if (vb) |bv| try self.typeKey(bv.dyn) else TypeKey{ .key = cls_key.?, .host = self.c.m.k1(false) };
        const fast = try f.label("type_fast");
        const slow = try f.label("type_python");
        const join = try f.label("type_is");
        try f.condBr(f.or_(ka.host, kb.host), slow, fast);
        try f.block(fast);
        const same = f.icmp(if (op == .is) jit_c.LLVMIntEQ else jit_c.LLVMIntNE, ka.key, kb.key);
        const res_fast = f.zext64(same);
        try self.drop(va);
        if (vb) |bv| try self.drop(bv);
        const fast_end = f.current;
        try f.br(join);
        try f.block(slow);
        const ta = (try self.builtinCall(inst, type_obj, &.{va}, pos)).?;
        const tb = if (vb) |bv| (try self.builtinCall(inst, type_obj, &.{bv}, pos)).? else SVal{ .py = cls.? };
        const r = try self.materialize(try self.compare(inst, op, ta, tb), inst.node);
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        return dyn(self.k(1), f.phi(t.i64, res_fast, fast_end, r.bits, slow_end), .bool);
    }

    /// The argument of `type(x)` (the builtin, one argument), or null.
    fn typeArg(self: *Gen, inst: *Inst, e: *const front.Expr) Error!?*const front.Expr {
        _ = try self.typeArgCallee(inst, e) orelse return null;
        return e.kind.call.args[0];
    }

    fn typeArgCallee(self: *Gen, inst: *Inst, e: *const front.Expr) Error!?*PyObject {
        if (e.kind != .call) return null;
        const ce = e.kind.call;
        if (ce.args.len != 1 or ce.keywords.len != 0 or ce.func.kind != .global) return null;
        const callee = try self.global(inst, ce.func.kind.global, ce.func.pos);
        if (callee != .py or !isBuiltin(callee.py, "type")) return null;
        return callee.py;
    }

    const TypeKey = struct { key: ir.Value, host: ir.Value };

    /// A value's class as a key (typeIs): its tag (a Big's a plain int's,
    /// both `int`), a record's type; `host`: a Python object (its class
    /// Python's to say).
    fn typeKey(self: *Gen, d: Dyn) Error!TypeKey {
        const f = &self.f;
        const t = self.c.m.t;
        const T = value.Tag;
        // (a value of a known kind: its key known; an int's is its tag's,
        // an I64's or a plain int's)
        switch (d.shape) {
            .any, .int => {},
            .record => if (d.rtype) |rt| return .{ .key = self.c.m.addrInt(@intFromPtr(rt)), .host = self.c.m.k1(false) },
            else => return .{ .key = self.k(@intCast(@intFromEnum(shapeTag(d.shape)))), .host = self.c.m.k1(false) },
        }
        const low = f.and_(d.tag, self.k(0xffff_ffff));
        const key0 = f.select(f.icmp(jit_c.LLVMIntEQ, low, self.k(@intFromEnum(T.big))), self.k(@intCast(value.PINT_TAG)), low);
        const rec = try f.label("key_record");
        const done = try f.label("key_done");
        const start = f.current;
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, low, self.k(@intFromEnum(T.record))), rec, done);
        try f.block(rec);
        const kr = self.recordTypeOf(d);
        try f.br(done);
        try f.block(done);
        return .{ .key = f.phi(t.i64, key0, start, kr, rec), .host = f.icmp(jit_c.LLVMIntEQ, low, self.k(@intFromEnum(T.host))) };
    }

    /// The key (typeKey's) of a class known when compiling, or null: one
    /// only Python objects are of.
    fn classKey(self: *Gen, o: *PyObject) Error!?ir.Value {
        const T = value.Tag;
        const is = struct {
            fn t(x: *PyObject, comptime name: [:0]const u8) bool {
                return x == @as(*PyObject, @ptrCast(@alignCast(py.types.typeObject(name))));
            }
        };
        const tag: ?i64 = if (is.t(o, "PyLong_Type")) @intCast(value.PINT_TAG) else if (o == types_mod.I64) @intFromEnum(T.int) else if (is.t(o, "PyFloat_Type")) @intFromEnum(T.float) else if (is.t(o, "PyUnicode_Type")) @intFromEnum(T.str) else if (is.t(o, "PyBool_Type")) @intFromEnum(T.bool) else if (is.t(o, "PyList_Type")) @intFromEnum(T.list) else if (is.t(o, "PyTuple_Type")) @intFromEnum(T.tuple) else if (is.t(o, "PyDict_Type")) @intFromEnum(T.dict) else if (o == @as(*PyObject, @ptrCast(@alignCast(ph.typeOf(py.Py_None()))))) @intFromEnum(T.none) else if (o == objects_mod.FunctionType) @intFromEnum(T.function) else if (o == objects_mod.NodeType) @intFromEnum(T.node) else null;
        if (tag) |n| return self.k(n);
        if (try isInstanceOf(o, @ptrCast(@alignCast(py.types.typeObject("PyType_Type"))))) {
            if (try self.c.recordType(o)) |rt| return self.c.m.addrInt(@intFromPtr(rt));
        }
        return null;
    }

    fn orValues(self: *Gen, inst: *Inst, a_: SVal, b: SVal) Error!SVal {
        if (a_ == .bool and !a_.bool) return b;
        if (a_ == .bool and a_.bool) {
            try self.drop(b);
            return a_;
        }
        const x = self.truthValue(try self.truth(a_, inst.node));
        const y = self.truthValue(try self.truth(b, inst.node));
        return self.boolDyn(self.f.or_(x, y));
    }

    /// A truth as an i1.
    fn truthValue(self: *Gen, t: Truth) ir.Value {
        return switch (t) {
            .known => |b| self.c.m.k1(b),
            .dyn => |d| d,
        };
    }

    /// zip() / enumerate() of run-time values: a list of tuples built by a
    /// loop at run time.
    fn zipRuntime(self: *Gen, inst: *Inst, is_zip: bool, args: []const SVal, pos: front.Pos) Error!SVal {
        const c = self.c;
        const f = &self.f;
        if (!is_zip and args.len != 1) return c.unsupportedAt(inst.func, pos, "enumerate() with a start isn't compiled yet", .{});
        // Each argument's items as a list
        const lists = try self.a().alloc(Dyn, args.len);
        for (args, 0..) |x, i| lists[i] = try self.itemsOf(inst, x);
        // n = min of the lengths
        var n = self.k(0);
        for (lists, 0..) |l, i| {
            const len = self.call("zr_list_len", &.{ l.tag, l.bits });
            if (i == 0) n = len else {
                n = f.select(f.icmp(jit_c.LLVMIntSLT, len, n), len, n);
            }
        }
        // An empty list, then a tuple appended per index
        const empty = try self.buildSequence("zr_list", &.{}, inst.node, .list);
        const result = try self.valSlot();
        try self.storeSlot(result, empty);
        const i_slot = try f.alloca(c.m.t.i64);
        f.store(self.k(0), i_slot);
        const head = try f.label("zip");
        const body = try f.label("zip_body");
        const done = try f.label("zip_done");
        try f.br(head);
        try f.block(head);
        const i = f.load(c.m.t.i64, i_slot);
        try f.condBr(f.icmp(jit_c.LLVMIntSLT, i, n), body, done);
        try f.block(body);
        const width = if (is_zip) lists.len else 2;
        const arr = try self.valueSlots(width);
        if (is_zip) {
            for (lists, 0..) |l, j| _ = self.call("zr_list_at", &.{ l.tag, l.bits, i, self.elem(arr, j) });
        } else {
            try self.storeSlot(self.elem(arr, 0), .{ .tag = self.k(@intCast(value.PINT_TAG)), .bits = i, .shape = .int });
            _ = self.call("zr_list_at", &.{ lists[0].tag, lists[0].bits, i, self.elem(arr, 1) });
        }
        try self.callCheck("zr_tuple", &.{ self.ctx, self.k32(inst.node), arr, self.k(@intCast(width)), self.out });
        const tup = try self.loadOut(.tuple);
        const acc = try self.loadSlot(result, .list);
        const ok2 = self.call("zr_append", &.{ self.ctx, self.k32(inst.node), acc.tag, acc.bits, tup.tag, tup.bits });
        try self.drop(.{ .dyn = tup });
        try self.check(ok2);
        f.store(f.add(i, self.k(1)), i_slot);
        try f.br(head);
        try f.block(done);
        for (lists) |l| try self.drop(.{ .dyn = l });
        return .{ .dyn = try self.loadSlot(result, .list) };
    }

    /// The items of a value as a run-time list (an owned reference).
    fn itemsOf(self: *Gen, inst: *Inst, v: SVal) Error!Dyn {
        const d = try self.materialize(v, inst.node);
        if (d.shape == .list) return d;
        const ok = self.call("zr_items", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, self.out });
        try self.drop(.{ .dyn = d });
        try self.check(ok);
        return self.loadOut(.list);
    }

    /// A helper function (one a semantic calls), read once for the
    /// language (with the semantics read: the same front functions, so
    /// what's learned about their code, like literals that escape, holds).
    fn helperFunction(self: *Gen, o: *PyObject) Error!*const front.Function {
        return self.c.readFunction(o);
    }

    // ------------------------------------------------------------------
    // Operators
    // ------------------------------------------------------------------

    fn binary(self: *Gen, inst: *Inst, op: front.BinOp, l: SVal, r: SVal) Error!SVal {
        // Known both: computed now, as Python does (an error stays an error
        // at run time, where it would happen)
        if (l.isStatic() and r.isStatic() and isScalar(l) and isScalar(r)) {
            if (try self.staticBinary(inst, op, l, r)) |v| return v;
        }
        const f = &self.f;
        const ld = try self.materialize(l, inst.node);
        const rd = try self.materialize(r, inst.node);
        // Ints: inline, checked; for values that may be ints, behind a check
        // of their tags (anything else: the helper, off the fast path)
        if (canBeInt(ld) and canBeInt(rd) and (op == .add or op == .sub or op == .mul)) {
            const fast = try f.label("int");
            const slow = try f.label("generic");
            const join = try f.label("joined");
            try f.condBr(try self.intGuard(ld, rd), fast, slow);
            try f.block(fast);
            // (an I64 among them: an I64, I64 & plain being I64's tag)
            const tag = f.and_(ld.tag, rd.tag);
            const res = try self.checkedInt(inst, op, ld.bits, rd.bits, tag, slow);
            const fast_end = f.current;
            try f.br(join);
            try f.block(slow);
            const g = try self.binaryHelper(inst, op, ld, rd);
            const slow_end = f.current;
            try f.br(join);
            try f.block(join);
            const i64t = self.c.m.t.i64;
            const tag_v = f.phi(i64t, tag, fast_end, g.tag, slow_end);
            const bits = f.phi(i64t, res, fast_end, g.bits, slow_end);
            return dyn(tag_v, bits, .any);
        }
        return .{ .dyn = try self.binaryHelper(inst, op, ld, rd) };
    }

    fn canBeInt(d: Dyn) bool {
        return d.shape == .int or d.shape == .any;
    }

    /// Both values are ints, I64s or plain (an i1): their tags checked
    /// where not known.
    fn intGuard(self: *Gen, ld: Dyn, rd: Dyn) Error!ir.Value {
        const f = &self.f;
        const yes = self.c.m.k1(true);
        const not_plain = self.k(~@as(i64, @intCast(value.PLAIN)));
        const a_ = if (ld.shape == .int) yes else f.icmp(jit_c.LLVMIntEQ, f.and_(ld.tag, not_plain), self.k(2));
        const b = if (rd.shape == .int) yes else f.icmp(jit_c.LLVMIntEQ, f.and_(rd.tag, not_plain), self.k(2));
        return f.and_(a_, b);
    }

    /// a op b on i64s, the result's tag `tag`: an overflow an error at the
    /// node for an I64; for plain ints, Python's big int (`big`: the
    /// helper's path).
    fn checkedInt(self: *Gen, inst: *Inst, op: front.BinOp, a_: ir.Value, b: ir.Value, tag: ir.Value, big: ir.Block) Error!ir.Value {
        const f = &self.f;
        const intrinsic = switch (op) {
            .add => self.c.sadd,
            .sub => self.c.ssub,
            else => self.c.smul,
        };
        const pair = f.call(intrinsic, &.{ a_, b });
        const res = f.extract(pair, 0);
        const ovf = f.extract(pair, 1);
        const bad = try f.label("overflow");
        const good = try f.label("no_overflow");
        const checked = try f.label("i64_overflow");
        try f.condBr(ovf, bad, good);
        try f.block(bad);
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, tag, self.k(@intCast(value.PINT_TAG))), big, checked);
        try f.block(checked);
        _ = self.call("zr_overflow", &.{ self.ctx, self.k32(inst.node) });
        try f.br(self.err_label);
        try f.block(good);
        return res;
    }

    /// a op b by the runtime (both taken).
    fn binaryHelper(self: *Gen, inst: *Inst, op: front.BinOp, ld: Dyn, rd: Dyn) Error!Dyn {
        const ok = self.call("zr_binary", &.{ self.ctx, self.k32(inst.node), self.k32(@intFromEnum(op)), ld.tag, ld.bits, rd.tag, rd.bits, self.out });
        try self.drop(.{ .dyn = ld });
        try self.drop(.{ .dyn = rd });
        try self.check(ok);
        const shape: Shape = if (ld.shape == .int and rd.shape == .int and op != .div and op != .pow)
            .int
        else if ((ld.shape == .float or rd.shape == .float) and (ld.shape == .int or ld.shape == .float) and (rd.shape == .int or rd.shape == .float))
            .float
        else
            .any;
        return self.loadOut(shape);
    }

    fn isScalar(v: SVal) bool {
        return switch (v) {
            .none, .bool, .int, .pint, .float, .str => true,
            // (a big int)
            .py => |o| ph.typeOf(o) == @as(*py.c.PyTypeObject, @ptrCast(py.types.typeObject("PyLong_Type"))),
            .tuple => |t| for (t) |x| {
                if (!isScalar(x)) break false;
            } else true,
            else => false,
        };
    }

    /// A Python object for a known scalar (new reference).
    fn pyOf(self: *Gen, v: SVal) Error!*PyObject {
        return switch (v) {
            .none => blk: {
                py.Py_IncRef(py.Py_None());
                break :blk py.Py_None();
            },
            .bool => |b| blk: {
                const o = if (b) py.Py_True() else py.Py_False();
                py.Py_IncRef(o);
                break :blk o;
            },
            // (an int of the program as the reference mode has it: an I64,
            // whose arithmetic is checked)
            .int => |n| types_mod.fromInt(n) orelse error.Python,
            .pint => |n| py.c.PyLong_FromLongLong(n) orelse error.Python,
            .py => |o| blk: {
                py.Py_IncRef(o);
                break :blk o;
            },
            .float => |x| py.c.PyFloat_FromDouble(x) orelse error.Python,
            .str => |s| ph.newString(s) orelse error.Python,
            .tuple => |t| blk: {
                const out = py.c.PyTuple_New(@intCast(t.len)) orelse return error.Python;
                for (t, 0..) |x, i| _ = py.c.PyTuple_SetItem(out, @intCast(i), try self.pyOf(x));
                break :blk out;
            },
            else => self.c.unsupported("a {s} where a value known when compiling is needed (a slice's bounds...)", .{@tagName(v)}),
        };
    }

    /// Python's result for known operands, or null (then it's an error:
    /// compiled as one at run time).
    fn staticBinary(self: *Gen, inst: *Inst, op: front.BinOp, l: SVal, r: SVal) Error!?SVal {
        const x = try self.pyOf(l);
        defer py.Py_DecRef(x);
        const y = try self.pyOf(r);
        defer py.Py_DecRef(y);
        const res = switch (op) {
            .add => py.c.PyNumber_Add(x, y),
            .sub => py.c.PyNumber_Subtract(x, y),
            .mul => py.c.PyNumber_Multiply(x, y),
            .div => py.c.PyNumber_TrueDivide(x, y),
            .floordiv => py.c.PyNumber_FloorDivide(x, y),
            .mod => py.c.PyNumber_Remainder(x, y),
            .pow => py.c.PyNumber_Power(x, y, py.Py_None()),
            .lshift => py.c.PyNumber_Lshift(x, y),
            .rshift => py.c.PyNumber_Rshift(x, y),
            .bitor => py.c.PyNumber_Or(x, y),
            .bitxor => py.c.PyNumber_Xor(x, y),
            .bitand => py.c.PyNumber_And(x, y),
        } orelse {
            // (the error happens when this runs: compiled as it)
            py.c.PyErr_Clear();
            return null;
        };
        defer py.Py_DecRef(res);
        return try self.constant(res, inst.node);
    }

    fn compare(self: *Gen, inst: *Inst, op: front.CmpOp, l: SVal, r: SVal) Error!SVal {
        // Known: decided now (None checks on fields are common)
        if (l.isStatic() and r.isStatic()) {
            if (op == .is or op == .is_not) {
                const same = std.meta.activeTag(l) == std.meta.activeTag(r) and switch (l) {
                    .none => true,
                    .bool => |b| b == r.bool,
                    .node => |n| n == r.node,
                    else => false,
                };
                // (anything known that isn't None is not None)
                if (l == .none or r == .none or same) return .{ .bool = if (op == .is) same else !same };
            }
            if (isScalar(l) and isScalar(r)) {
                const x = try self.pyOf(l);
                defer py.Py_DecRef(x);
                const y = try self.pyOf(r);
                defer py.Py_DecRef(y);
                const cmp_op: c_int = switch (op) {
                    .eq => py.c.Py_EQ,
                    .ne => py.c.Py_NE,
                    .lt => py.c.Py_LT,
                    .le => py.c.Py_LE,
                    .gt => py.c.Py_GT,
                    .ge => py.c.Py_GE,
                    else => -1,
                };
                if (cmp_op >= 0) {
                    const res = py.c.PyObject_RichCompareBool(x, y, cmp_op);
                    if (res >= 0) return .{ .bool = res == 1 };
                    py.c.PyErr_Clear();
                }
            }
            if (l == .node and r == .node and (op == .eq or op == .ne)) {
                const same = l.node == r.node;
                return .{ .bool = if (op == .eq) same else !same };
            }
        }
        // A run-time value against None: its tag
        if ((op == .is or op == .is_not) and (l == .none or r == .none)) {
            const other = if (l == .none) r else l;
            if (other.isStatic()) return .{ .bool = (other == .none) == (op == .is) };
            const d = other.dyn;
            const t = self.f.icmp(if (op == .is) jit_c.LLVMIntEQ else jit_c.LLVMIntNE, d.tag, self.k(0));
            try self.drop(other);
            return self.boolDyn(t);
        }
        // ... against True or False: its tag and bits
        if ((op == .is or op == .is_not) and ((l == .bool and r == .dyn) or (r == .bool and l == .dyn))) {
            const b = if (l == .bool) l.bool else r.bool;
            const d = if (l == .dyn) l.dyn else r.dyn;
            const f = &self.f;
            const same = f.and_(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(value.Tag.bool))), f.icmp(jit_c.LLVMIntEQ, d.bits, self.k(@intFromBool(b))));
            try self.drop(.{ .dyn = d });
            return self.boolDyn(if (op == .is) same else f.xor(same, self.c.m.k1(true)));
        }
        // An int against a constant beyond 64 bits: decided by its sign
        // (for an int of either kind; anything else as Python says)
        if (bigSide(l, r)) |side| if (op == .lt or op == .le or op == .gt or op == .ge or op == .eq or op == .ne) {
            return self.intVersusBig(inst, op, l, r, side);
        };
        // A run-time str against constant strs (`op == "+"`, `op in OPS`):
        // its bytes compared inline (no container made, no hashing)
        if (op == .eq or op == .ne or op == .in or op == .not_in) {
            const swap = (op == .eq or op == .ne) and l.isStatic() and r == .dyn;
            const d_side = if (swap) r else l;
            const k_side = if (swap) l else r;
            if (d_side == .dyn) if (try self.staticStrs(op, k_side)) |keys| return self.strMatch(inst, op, l, r, d_side.dyn, keys);
        }
        const f = &self.f;
        const ld = try self.materialize(l, inst.node);
        const rd = try self.materialize(r, inst.node);
        const pred: ?jit_c.LLVMIntPredicate = switch (op) {
            .eq => jit_c.LLVMIntEQ,
            .ne => jit_c.LLVMIntNE,
            .lt => jit_c.LLVMIntSLT,
            .le => jit_c.LLVMIntSLE,
            .gt => jit_c.LLVMIntSGT,
            .ge => jit_c.LLVMIntSGE,
            else => null,
        };
        if (pred != null and canBeInt(ld) and canBeInt(rd)) {
            if (ld.shape == .int and rd.shape == .int) return self.boolDyn(f.icmp(pred.?, ld.bits, rd.bits));
            // (ints: inline, behind a check of the tags)
            const fast = try f.label("int_cmp");
            const slow = try f.label("generic_cmp");
            const join = try f.label("cmp_joined");
            try f.condBr(try self.intGuard(ld, rd), fast, slow);
            try f.block(fast);
            const res = f.zext64(f.icmp(pred.?, ld.bits, rd.bits));
            const fast_end = f.current;
            try f.br(join);
            try f.block(slow);
            const g = try self.compareHelper(inst, op, ld, rd);
            const slow_end = f.current;
            try f.br(join);
            try f.block(join);
            return dyn(self.k(1), f.phi(self.c.m.t.i64, res, fast_end, g.bits, slow_end), .bool);
        }
        return .{ .dyn = try self.compareHelper(inst, op, ld, rd) };
    }

    /// The constant strs a comparison is against: `== "s"`, `in` a constant
    /// tuple, list or dict of strs (its keys). (null: not only strs.)
    fn staticStrs(self: *Gen, op: front.CmpOp, v: SVal) Error!?[]const []const u8 {
        const items: []const SVal = switch (op) {
            .eq, .ne => if (v == .str) &.{v} else return null,
            else => switch (v) {
                .tuple => |t| t,
                .list => |l| l.items.items,
                .dict => |d| d.keys.items,
                else => return null,
            },
        };
        const keys = try self.a().alloc([]const u8, items.len);
        for (items, keys) |item, *key| {
            if (item != .str) return null;
            key.* = item.str;
        }
        return keys;
    }

    /// `d == s` / `d in (s1, s2...)` for constant strs: d a Str, its length
    /// then its bytes (8 at a time) checked inline; anything else to
    /// zr_compare (Python's rules: an unhashable key in a dict raises...).
    fn strMatch(self: *Gen, inst: *Inst, op: front.CmpOp, l: SVal, r: SVal, d: Dyn, keys: []const []const u8) Error!SVal {
        const f = &self.f;
        const m = &self.c.m;
        const t = m.t;
        const is_str = try f.label("str_cmp");
        const slow = try f.label("generic_cmp");
        const hit = try f.label("str_hit");
        const miss = try f.label("str_miss");
        const done = try f.label("str_done");
        const join = try f.label("cmp_joined");
        try f.condBr(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(value.Tag.str))), is_str, slow);
        try f.block(is_str);
        const p = f.intToPtr(d.bits);
        const len = f.load(t.i64, f.offset(p, @offsetOf(value.Str, "len")));
        for (keys) |key| {
            const bytes = try f.label("str_bytes");
            const next = try f.label("str_next");
            try f.condBr(f.icmp(jit_c.LLVMIntEQ, len, self.k(@intCast(key.len))), bytes, next);
            try f.block(bytes);
            var same = m.k1(true);
            var at: usize = 0;
            while (at < key.len) {
                // (usize: @min would make it a u4, n * 8 wrapping)
                const n: usize = @min(8, key.len - at);
                const ty = m.intType(@intCast(n * 8));
                var word: u64 = 0;
                for (key[at .. at + n], 0..) |b, i| word |= @as(u64, b) << @intCast(i * 8);
                const got = f.load(ty, f.offset(p, @intCast(@sizeOf(value.Str) + at)));
                same = f.and_(same, f.icmp(jit_c.LLVMIntEQ, got, ir.Module.kInt(ty, word)));
                at += n;
            }
            try f.condBr(same, hit, next);
            try f.block(next);
        }
        try f.br(miss);
        const yes: i64 = if (op == .eq or op == .in) 1 else 0;
        try f.block(hit);
        try f.br(done);
        try f.block(miss);
        try f.br(done);
        try f.block(done);
        const res = f.phi(t.i64, self.k(yes), hit, self.k(1 - yes), miss);
        try self.drop(.{ .dyn = d });
        const fast_end = f.current;
        try f.br(join);
        try f.block(slow);
        const ld = try self.materialize(l, inst.node);
        const rd = try self.materialize(r, inst.node);
        const g = try self.compareHelper(inst, op, ld, rd);
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        return dyn(self.k(1), f.phi(t.i64, res, fast_end, g.bits, slow_end), .bool);
    }

    /// Which side is a constant int beyond 64 bits (the other a run-time
    /// value that may be an int): 0 left, 1 right; null: not that.
    fn bigSide(l: SVal, r: SVal) ?u1 {
        const isBig = struct {
            fn f(v: SVal) bool {
                return v == .py and ph.typeOf(v.py) == @as(*py.c.PyTypeObject, @ptrCast(py.types.typeObject("PyLong_Type")));
            }
        }.f;
        if (isBig(l) and r == .dyn and canBeInt(r.dyn)) return 0;
        if (isBig(r) and l == .dyn and canBeInt(l.dyn)) return 1;
        return null;
    }

    /// `x < 2**63` and the like: for an int (of 64 bits, either kind) the
    /// answer is the constant's sign's; anything else by zr_compare.
    fn intVersusBig(self: *Gen, inst: *Inst, op: front.CmpOp, l: SVal, r: SVal, side: u1) Error!SVal {
        const f = &self.f;
        const t = self.c.m.t;
        const big = if (side == 0) l.py else r.py;
        const d = if (side == 0) r.dyn else l.dyn;
        // (beyond 64 bits: positive, every int is below it; negative, above)
        var overflow: c_int = 0;
        _ = py.c.PyLong_AsLongLongAndOverflow(big, &overflow);
        // (one that fits isn't .py: the constants beyond 64 bits are)
        if (overflow == 0) return self.c.unsupported("an int constant of 64 bits as a Python object", .{});
        const positive = overflow > 0;
        // int OP big (int on the left), as it comes out
        const int_left_lt = positive;
        const answer: bool = switch (op) {
            .lt, .le => if (side == 1) int_left_lt else !int_left_lt,
            .gt, .ge => if (side == 1) !int_left_lt else int_left_lt,
            .eq => false,
            .ne => true,
            else => unreachable,
        };
        const T = value.Tag;
        const fast = try f.label("big_int");
        const slow = try f.label("big_other");
        const join = try f.label("big_cmp");
        const is_int = f.or_(f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intFromEnum(T.int))), f.icmp(jit_c.LLVMIntEQ, d.tag, self.k(@intCast(value.PINT_TAG))));
        try f.condBr(is_int, fast, slow);
        try f.block(fast);
        try f.br(join);
        try f.block(slow);
        const ld = try self.materialize(l, inst.node);
        const rd = try self.materialize(r, inst.node);
        const g = try self.compareHelper(inst, op, ld, rd);
        const slow_end = f.current;
        try f.br(join);
        try f.block(join);
        return dyn(self.k(1), f.phi(t.i64, self.k(@intFromBool(answer)), fast, g.bits, slow_end), .bool);
    }

    fn compareHelper(self: *Gen, inst: *Inst, op: front.CmpOp, ld: Dyn, rd: Dyn) Error!Dyn {
        const ok = self.call("zr_compare", &.{ self.ctx, self.k32(inst.node), self.k32(@intFromEnum(op)), ld.tag, ld.bits, rd.tag, rd.bits, self.out });
        try self.drop(.{ .dyn = ld });
        try self.drop(.{ .dyn = rd });
        try self.check(ok);
        return self.loadOut(.bool);
    }

    fn unary(self: *Gen, inst: *Inst, op: front.UnaryOp, v: SVal) Error!SVal {
        if (op == .not_) {
            return switch (try self.truth(v, inst.node)) {
                .known => |b| .{ .bool = !b },
                .dyn => |t| self.boolDyn(self.f.xor(t, self.c.m.k1(true))),
            };
        }
        if (v.isStatic() and isScalar(v)) {
            const x = try self.pyOf(v);
            defer py.Py_DecRef(x);
            const res = switch (op) {
                .neg => py.c.PyNumber_Negative(x),
                .pos => py.c.PyNumber_Positive(x),
                else => py.c.PyNumber_Invert(x),
            };
            if (res) |r| {
                defer py.Py_DecRef(r);
                return self.constant(r, inst.node);
            }
            py.c.PyErr_Clear();
        }
        const d = try self.materialize(v, inst.node);
        const ok = self.call("zr_unary", &.{ self.ctx, self.k32(inst.node), self.k32(@intFromEnum(op)), d.tag, d.bits, self.out });
        try self.drop(.{ .dyn = d });
        try self.check(ok);
        return .{ .dyn = try self.loadOut(if (d.shape == .int or d.shape == .float) d.shape else .any) };
    }

    /// `a and b` / `a or b`: the deciding operand's value (short-circuit).
    fn boolOp(self: *Gen, inst: *Inst, is_and: bool, items: []const *const front.Expr) Error!SVal {
        var v = try self.expr(inst, items[0]);
        for (items[1..]) |next_e| {
            // Known: decided now
            if (v.isStatic()) {
                const t = (try self.truth(v, inst.node)).known;
                if (t != is_and) return v;
                v = try self.expr(inst, next_e);
                continue;
            }
            // At run time: the value, or the next one's
            const f = &self.f;
            const slot = try self.valSlot();
            try self.storeSlot(slot, v.dyn);
            try self.increfDyn(v.dyn);
            const t = (try self.truth(v, inst.node)).dyn;
            const take_next = try f.label("boolop_next");
            const join = try f.label("boolop_end");
            if (is_and) try f.condBr(t, take_next, join) else try f.condBr(t, join, take_next);
            try f.block(take_next);
            // (the first value is dropped on the path taking the next)
            const first = try self.loadSlot(slot, v.dyn.shape);
            try self.drop(.{ .dyn = first });
            const n = try self.materialize(try self.expr(inst, next_e), inst.node);
            try self.storeSlot(slot, n);
            try f.br(join);
            try f.block(join);
            v = .{ .dyn = try self.loadSlot(slot, if (n.shape == v.dyn.shape) n.shape else .any) };
        }
        return v;
    }

    fn andValues(self: *Gen, inst: *Inst, a_: SVal, b: SVal) Error!SVal {
        if (a_ == .bool and a_.bool) return b;
        if (a_ == .bool and !a_.bool) {
            try self.drop(b);
            return a_;
        }
        const x = self.truthValue(try self.truth(a_, inst.node));
        const y = self.truthValue(try self.truth(b, inst.node));
        return self.boolDyn(self.f.and_(x, y));
    }

    /// `x if cond else y` with a run-time cond.
    fn branchValue(self: *Gen, inst: *Inst, cond: ir.Value, then_e: *const front.Expr, else_e: *const front.Expr) Error!SVal {
        const f = &self.f;
        const slot = try self.valSlot();
        const yes = try f.label("ifexp_then");
        const no = try f.label("ifexp_else");
        const join = try f.label("ifexp_end");
        try f.condBr(cond, yes, no);
        inst.dyn_depth += 1;
        try f.block(yes);
        const a_ = try self.materialize(try self.expr(inst, then_e), inst.node);
        try self.storeSlot(slot, a_);
        try f.br(join);
        try f.block(no);
        const b = try self.materialize(try self.expr(inst, else_e), inst.node);
        try self.storeSlot(slot, b);
        try f.br(join);
        inst.dyn_depth -= 1;
        try f.block(join);
        return .{ .dyn = try self.loadSlot(slot, if (a_.shape == b.shape) a_.shape else .any) };
    }

    fn listComp(self: *Gen, inst: *Inst, comp: front.Comp, pos: front.Pos) Error!SVal {
        // Known iterables: the list built now (its items may be run-time);
        // a run-time one: a run-time list filled by a loop
        const out = try self.a().create(SList);
        out.* = .{};
        var sink = Sink{ .list = out, .elt = comp.elt };
        try self.compLoops(inst, comp.generators, 0, pos, &sink);
        if (sink.kind == .static_list) return .{ .list = out };
        return .{ .dyn = try self.loadSlot(sink.slot, .list) };
    }

    /// any(x for ...) / all(x for ...): the elements in turn until one
    /// decides (true for any, false for all), as Python's stop there.
    fn allAny(self: *Gen, inst: *Inst, comp: front.Comp, is_any: bool, pos: front.Pos) Error!SVal {
        const f = &self.f;
        const slot = try self.valSlot();
        try self.storeSlot(slot, self.konst(1, @intFromBool(!is_any), .bool));
        var sink = Sink{ .kind = .decide, .list = undefined, .slot = slot, .elt = comp.elt, .is_any = is_any, .exit = try f.label("decided") };
        // (each element may end it: what follows runs on some paths only)
        inst.dyn_depth += 1;
        try self.compLoops(inst, comp.generators, 0, pos, &sink);
        inst.dyn_depth -= 1;
        try f.br(sink.exit);
        try f.block(sink.exit);
        return .{ .dyn = try self.loadSlot(slot, .bool) };
    }

    /// Where a comprehension's elements go
    const Sink = struct {
        kind: enum { static_list, list_into, dict_into, decide } = .static_list,
        list: *SList,
        /// The run-time list or dict (a stack slot); any()/all()'s result
        slot: ir.Value = null,
        /// any() or all(): which, and where a deciding element jumps to
        is_any: bool = false,
        exit: ir.Block = null,
        elt: ?*const front.Expr = null,
        key: ?*const front.Expr = null,
        value: ?*const front.Expr = null,
    };

    /// From here the elements go to a run-time list (the known ones first).
    fn sinkToRuntime(self: *Gen, inst: *Inst, sink: *Sink) Error!void {
        if (sink.kind != .static_list) return;
        const items = sink.list.items.items;
        const d = try self.buildSequence("zr_list", items, inst.node, .list);
        const slot = try self.valSlot();
        try self.storeSlot(slot, d);
        sink.kind = .list_into;
        sink.slot = slot;
    }

    fn compLoops(self: *Gen, inst: *Inst, gens: []const front.Generator, level: usize, pos: front.Pos, sink: *Sink) Error!void {
        if (level == gens.len) return self.compEmit(inst, sink);
        const g = gens[level];
        var it = switch (try self.iteration(inst, g.iter)) {
            .known => |items| {
                for (items) |item| {
                    if (item == .dyn) try self.increfDyn(item.dyn);
                    try self.assign(inst, g.target, item, pos);
                    try self.compFiltered(inst, gens, level, g.ifs, pos, sink);
                }
                return;
            },
            .runtime => |x| x,
        };
        // At run time: a loop over the items
        try self.sinkToRuntime(inst, sink);
        const targets = try self.a().alloc(front.Target, gens.len - level);
        for (gens[level..], targets) |g2, *t| t.* = g2.target;
        try self.slotTargets(inst, targets);
        const f = &self.f;
        try self.iterStart(&it);
        const head = try f.label("comp");
        const body = try f.label("comp_body");
        const done = try f.label("comp_done");
        try f.br(head);
        try f.block(head);
        self.loop_level += 1;
        defer self.loop_level -= 1;
        try f.condBr(try self.iterHead(it), body, done);
        try f.block(body);
        inst.dyn_depth += 1;
        try self.iterItem(inst, it, g.target, pos);
        try self.compFiltered(inst, gens, level, g.ifs, pos, sink);
        inst.dyn_depth -= 1;
        try self.iterStep(it);
        try f.br(head);
        try f.block(done);
        try self.iterEnd(it);
    }

    /// The comprehension's filters, then its next level.
    fn compFiltered(self: *Gen, inst: *Inst, gens: []const front.Generator, level: usize, ifs: []const *const front.Expr, pos: front.Pos, sink: *Sink) Error!void {
        if (ifs.len == 0) return self.compLoops(inst, gens, level + 1, pos, sink);
        switch (try self.truth(try self.expr(inst, ifs[0]), inst.node)) {
            .known => |b| if (b) try self.compFiltered(inst, gens, level, ifs[1..], pos, sink),
            .dyn => |cond| {
                try self.sinkToRuntime(inst, sink);
                const yes = try self.f.label("comp_keep");
                const skip = try self.f.label("comp_skip");
                try self.f.condBr(cond, yes, skip);
                inst.dyn_depth += 1;
                try self.f.block(yes);
                try self.compFiltered(inst, gens, level, ifs[1..], pos, sink);
                inst.dyn_depth -= 1;
                try self.f.br(skip);
                try self.f.block(skip);
            },
        }
    }

    fn compEmit(self: *Gen, inst: *Inst, sink: *Sink) Error!void {
        switch (sink.kind) {
            .static_list => try sink.list.items.append(self.a(), try self.expr(inst, sink.elt.?)),
            .list_into => {
                const x = try self.materialize(try self.expr(inst, sink.elt.?), inst.node);
                const l = try self.loadSlot(sink.slot, .list);
                const ok = self.call("zr_append", &.{ self.ctx, self.k32(inst.node), l.tag, l.bits, x.tag, x.bits });
                try self.drop(.{ .dyn = x });
                try self.check(ok);
            },
            .dict_into => {
                const key = try self.materialize(try self.expr(inst, sink.key.?), inst.node);
                const v = try self.materialize(try self.expr(inst, sink.value.?), inst.node);
                const d = try self.loadSlot(sink.slot, .dict);
                const ok = self.call("zr_setitem", &.{ self.ctx, self.k32(inst.node), d.tag, d.bits, key.tag, key.bits, v.tag, v.bits });
                try self.drop(.{ .dyn = key });
                try self.drop(.{ .dyn = v });
                try self.check(ok);
            },
            .decide => {
                const f = &self.f;
                const t = try self.truth(try self.expr(inst, sink.elt.?), inst.node);
                // (deciding: the result, and out)
                const decides = switch (t) {
                    .known => |b| self.c.m.k1(b == sink.is_any),
                    .dyn => |b| if (sink.is_any) b else f.xor(b, self.c.m.k1(true)),
                };
                const yes = try f.label("decides");
                const go_on = try f.label("undecided");
                try f.condBr(decides, yes, go_on);
                try f.block(yes);
                try self.storeSlot(sink.slot, self.konst(1, @intFromBool(sink.is_any), .bool));
                try f.br(sink.exit);
                try f.block(go_on);
            },
        }
    }
};

/// The locals a statement list assigns (or mutates through), anywhere in it.
fn collectAssigned(body: []const front.Stmt, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    for (body) |s| switch (s.kind) {
        .assign => |x| for (x.targets) |t| try collectTarget(t, set, a),
        .aug => |x| try collectTarget(x.target, set, a),
        .for_ => |x| {
            try collectTarget(x.target, set, a);
            try collectAssigned(x.body, set, a);
            try collectAssigned(x.else_, set, a);
        },
        .if_ => |x| {
            try collectAssigned(x.body, set, a);
            try collectAssigned(x.else_, set, a);
        },
        .while_ => |x| {
            try collectAssigned(x.body, set, a);
            try collectAssigned(x.else_, set, a);
        },
        .try_ => |x| {
            try collectAssigned(x.body, set, a);
            try collectAssigned(x.else_, set, a);
            try collectAssigned(x.finally, set, a);
            for (x.handlers) |h| {
                if (h.name) |slot| try set.put(a, slot, {});
                try collectAssigned(h.body, set, a);
            }
        },
        .expr => |e| try collectMutated(e, set, a),
        else => {},
    };
}

fn collectTarget(t: front.Target, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    switch (t) {
        .local => |slot| try set.put(a, slot, {}),
        .tuple => |ts| for (ts) |x| try collectTarget(x, set, a),
        .attr => |x| if (x.obj.kind == .local) try set.put(a, x.obj.kind.local, {}),
        .index => |x| if (x.obj.kind == .local) try set.put(a, x.obj.kind.local, {}),
    }
}

/// `x.append(...)`: x is mutated (a list's or dict's mutating methods).
/// The locals statements read (or change in place), anywhere in them but
/// the statement list `skip` (a try's body).
fn collectReads(body: []const front.Stmt, skip: []const front.Stmt, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    if (body.ptr == skip.ptr and body.len == skip.len) return;
    for (body) |s| switch (s.kind) {
        .assign => |x| {
            for (x.targets) |t| try targetReads(t, set, a);
            try exprReads(x.value, set, a);
        },
        .aug => |x| {
            try targetReads(x.target, set, a);
            if (x.target == .local) try set.put(a, x.target.local, {});
            try exprReads(x.value, set, a);
        },
        .expr => |e| try exprReads(e, set, a),
        .if_ => |x| {
            try exprReads(x.test_, set, a);
            try collectReads(x.body, skip, set, a);
            try collectReads(x.else_, skip, set, a);
        },
        .while_ => |x| {
            try exprReads(x.test_, set, a);
            try collectReads(x.body, skip, set, a);
            try collectReads(x.else_, skip, set, a);
        },
        .for_ => |x| {
            try targetReads(x.target, set, a);
            try exprReads(x.iter, set, a);
            try collectReads(x.body, skip, set, a);
            try collectReads(x.else_, skip, set, a);
        },
        .return_, .raise_ => |e| if (e) |x| try exprReads(x, set, a),
        .assert_ => |x| {
            try exprReads(x.test_, set, a);
            if (x.msg) |m| try exprReads(m, set, a);
        },
        .try_ => |x| {
            try collectReads(x.body, skip, set, a);
            try collectReads(x.else_, skip, set, a);
            try collectReads(x.finally, skip, set, a);
            for (x.handlers) |h| {
                if (h.type_) |e| try exprReads(e, set, a);
                try collectReads(h.body, skip, set, a);
            }
        },
        .break_, .continue_, .pass => {},
    };
}

/// (a target's reads: an attribute's or item's object and index)
fn targetReads(t: front.Target, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    switch (t) {
        .local => {},
        .tuple => |ts| for (ts) |x| try targetReads(x, set, a),
        .attr => |x| try exprReads(x.obj, set, a),
        .index => |x| {
            try exprReads(x.obj, set, a);
            try exprReads(x.index, set, a);
        },
    }
}

fn exprReads(e: *const front.Expr, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) Allocator.Error!void {
    switch (e.kind) {
        .local => |slot| try set.put(a, slot, {}),
        .int, .big, .float, .str, .bool, .none, .global => {},
        .attr => |x| try exprReads(x.obj, set, a),
        .index => |x| {
            try exprReads(x.obj, set, a);
            try exprReads(x.index, set, a);
        },
        .slice => |x| {
            try exprReads(x.obj, set, a);
            inline for (.{ x.lo, x.hi, x.step }) |p| if (p) |y| try exprReads(y, set, a);
        },
        .call => |x| {
            try exprReads(x.func, set, a);
            for (x.args) |y| try exprReads(y, set, a);
            for (x.keywords) |k| try exprReads(k.value, set, a);
        },
        .binary => |x| {
            try exprReads(x.left, set, a);
            try exprReads(x.right, set, a);
        },
        .unary => |x| try exprReads(x.operand, set, a),
        .and_, .or_, .list, .tuple => |xs| for (xs) |y| try exprReads(y, set, a),
        .compare => |x| {
            try exprReads(x.first, set, a);
            for (x.rest) |y| try exprReads(y, set, a);
        },
        .cond => |x| {
            try exprReads(x.test_, set, a);
            try exprReads(x.then, set, a);
            try exprReads(x.else_, set, a);
        },
        .dict => |x| {
            for (x.keys) |y| try exprReads(y, set, a);
            for (x.values) |y| try exprReads(y, set, a);
        },
        .list_comp, .gen_exp => |c| {
            try exprReads(c.elt, set, a);
            for (c.generators) |g| try genReads(g, set, a);
        },
        .dict_comp => |c| {
            try exprReads(c.key, set, a);
            try exprReads(c.value, set, a);
            for (c.generators) |g| try genReads(g, set, a);
        },
        .fstring => |parts| for (parts) |p| try fpartReads(p, set, a),
    }
}

fn genReads(g: front.Generator, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) Allocator.Error!void {
    try exprReads(g.iter, set, a);
    for (g.ifs) |y| try exprReads(y, set, a);
}

fn fpartReads(p: front.FPart, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) Allocator.Error!void {
    switch (p) {
        .text => {},
        .value => |v| {
            try exprReads(v.expr, set, a);
            for (v.spec) |s| try fpartReads(s, set, a);
        },
    }
}

fn collectMutated(e: *const front.Expr, set: *std.AutoHashMapUnmanaged(u32, void), a: Allocator) !void {
    if (e.kind != .call) return;
    const func = e.kind.call.func;
    if (func.kind != .attr or func.kind.attr.obj.kind != .local) return;
    const mutating = [_][]const u8{ "append", "extend", "insert", "pop", "remove", "clear", "update", "setdefault", "sort", "reverse", "popitem" };
    for (mutating) |m| {
        if (std.mem.eql(u8, func.kind.attr.name, m)) {
            try set.put(a, func.kind.attr.obj.kind.local, {});
            return;
        }
    }
}
