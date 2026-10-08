"""Lua 5.4, run by zrun: zgram parses it, zrules checks it (lualang.py),
and the functions below say what each kind of node does.

    python lua.py script.lua

Values: nil is None, booleans and numbers are Python's (integers are
64-bit and wrap around, as Lua's do), strings are str, tables are Table,
functions are the program's (rt.function) or the library's (Builtin). A
call gives a list of values (Lua's multiple results); where one value is
wanted, the first is taken.
"""

import builtins
import math
import os
import sys
import time

import zrun
from lualang import PARSER, RULES

lang = zrun.Language(PARSER, RULES)

# Functions: `function (params) block end` (named or not, methods: below).
# Lua calls pass any number of arguments: missing ones are nil, extra ones
# are the function's `...`
lang.function("funcbody", params="names", body="block", name=None, hoist=False, missing="none", extra="keep")


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------


class Table:
    """A Lua table: an array part (keys 1..n) and a hash part."""

    __slots__ = ("arr", "hash", "meta")

    def __init__(self):
        self.arr = []
        self.hash = {}
        self.meta = None

    def get(self, key):
        if type(key) is float and key.is_integer():
            key = int(key)
        if type(key) is int or isinstance(key, int) and not isinstance(key, bool):
            if 1 <= key <= len(self.arr):
                return self.arr[key - 1]
        return self.hash.get(key)

    def set(self, key, value):
        if type(key) is float and key.is_integer():
            key = int(key)
        arr = self.arr
        if isinstance(key, int) and not isinstance(key, bool):
            n = len(arr)
            if 1 <= key <= n:
                arr[key - 1] = value
                if value is None and key == n:
                    while arr and arr[-1] is None:
                        arr.pop()
                return
            if key == n + 1:
                if value is None:
                    self.hash.pop(key, None)
                    return
                arr.append(value)
                self.hash.pop(key, None)
                # (the keys right after it move to the array part)
                nxt = key + 1
                while nxt in self.hash:
                    arr.append(self.hash.pop(nxt))
                    nxt += 1
                return
        if value is None:
            self.hash.pop(key, None)
        else:
            self.hash[key] = value

    def length(self):
        if self.arr:
            return len(self.arr)
        # a border in the hash part
        n = 0
        while self.hash.get(n + 1) is not None:
            n += 1
        return n

    def next(self, key):
        """The entry after `key` (None: the first), or None at the end."""
        arr = self.arr
        i = 0
        if key is not None:
            if type(key) is float and key.is_integer():
                key = int(key)
            if isinstance(key, int) and not isinstance(key, bool) and 1 <= key <= len(arr):
                i = key
            else:
                keys = list(self.hash)
                try:
                    j = keys.index(key) + 1
                except ValueError:
                    return "bad"
                for k in keys[j:]:
                    if self.hash[k] is not None:
                        return (k, self.hash[k])
                return None
        while i < len(arr):
            if arr[i] is not None:
                return (i + 1, arr[i])
            i += 1
        for k, v in self.hash.items():
            if v is not None:
                return (k, v)
        return None


class Method:
    """A function defined as `function t:name()`: its `self` is the first
    argument (the receiver of the call)."""

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn


class Builtin:
    """A library function: fn(rt, node, args) -> list of results."""

    __slots__ = ("name", "fn")

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_number(v):
    return is_int(v) or type(v) is float


def is_function(v):
    return isinstance(v, (Method, Builtin, zrun.Function))


def wrap(n):
    """An integer wrapped around to 64 bits, as Lua's arithmetic does."""
    n &= 0xFFFFFFFFFFFFFFFF
    return n - 0x10000000000000000 if n >= 0x8000000000000000 else n


def type_name(v):
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "boolean"
    if is_number(v):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, Table):
        return "table"
    if is_function(v):
        return "function"
    return "userdata"


def truthy(v):
    return v is not None and v is not False


def one(values):
    return values[0] if values else None


def fmt_float(x):
    if x != x:
        return "-nan" if math.copysign(1.0, x) < 0 else "nan"
    if x == math.inf:
        return "inf"
    if x == -math.inf:
        return "-inf"
    s = "%.14g" % x
    if all(c in "-0123456789" for c in s):
        s += ".0"
    return s


def address(v):
    return "0x%014x" % (id(v) & 0xFFFFFFFFFFFFFF)


def tostring(rt, node, v):
    if v is None:
        return "nil"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if is_int(v):
        return str(int(v))
    if type(v) is float:
        return fmt_float(v)
    if isinstance(v, str):
        return v
    if isinstance(v, Table):
        mm = metamethod(v, "__tostring")
        if mm is not None:
            s = call_one(rt, node, mm, [v])
            if not isinstance(s, str):
                lua_error(rt, node, "'__tostring' must return a string")
            return s
        name = metamethod(v, "__name")
        return (name if isinstance(name, str) else "table") + ": " + address(v)
    if isinstance(v, Builtin):
        return "builtin: " + address(v)
    return "function: " + address(v)


def str_to_number(s):
    """A string's number (Lua's conversion), or None."""
    t = s.strip(" \t\n\r\f\v")
    if not t:
        return None
    try:
        return parse_number(t)
    except ValueError:
        return None


def parse_number(text):
    t = text.lower()
    neg = False
    body = t
    if body[:1] in "+-" and body[1:2] != "":
        neg = body[0] == "-"
        body = body[1:]
    if body.startswith("0x"):
        if "." in body or "p" in body:
            mant, _, exp = body[2:].partition("p")
            ip, _, fp = mant.partition(".")
            if not ip and not fp:
                raise ValueError(text)
            v = int(ip or "0", 16) + (int(fp, 16) / 16 ** len(fp) if fp else 0)
            v = v * 2.0 ** int(exp) if exp else float(v)
            return -v if neg else v
        v = wrap(int(body[2:], 16))
        return wrap(-v) if neg else v
    if any(c in body for c in ".ei") or body in ("nan",):
        if "inf" in body or "nan" in body:
            raise ValueError(text)
        v = float(body)
        return -v if neg else v
    v = int(body)
    if neg:
        v = -v
    if -(2**63) <= v < 2**63:
        return v
    return float(v)


def tonumber_arith(v):
    if is_number(v):
        return v
    if isinstance(v, str):
        return str_to_number(v)
    return None


def to_integer(v):
    """A number with an exact integer value as an int (None if not)."""
    if is_int(v):
        return v
    if type(v) is float and v.is_integer() and -(2.0**63) <= v < 2.0**63:
        return int(v)
    if isinstance(v, str):
        n = str_to_number(v)
        return to_integer(n) if n is not None else None
    return None


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


def where(rt, node):
    """`chunk:line: ` of a node; nothing for None (code of the library,
    which Lua's messages give no position for)."""
    if node is None:
        return ""
    name = rt.path if rt.path is not None else "input"
    return "%s:%d: " % (name, node.line)


def lua_error(rt, node, message):
    """A Lua runtime error at a node (position first, as Lua's are)."""
    msg = where(rt, node) + message
    raise rt.Throw(msg, msg)


def describe(rt, node):
    """What a name in an error message refers to (Lua's "global 'x'")."""
    if node is None:
        return ""
    k = node.kind
    if k == "paren":
        return describe(rt, node.children[0])
    if k == "var":
        return " (%s '%s')" % ("local" if rt.scope(node) is not None else "global", node.text)
    if k == "member":
        return " (field '%s')" % node.name.text
    if k == "method_call":
        return " (method '%s')" % node.method.text
    if k == "string":
        return " (constant '%s')" % string_value(node.text)
    return ""


# ----------------------------------------------------------------------
# Metatables
# ----------------------------------------------------------------------

STRING_META = Table()


def metatable(v):
    if isinstance(v, Table):
        return v.meta
    if isinstance(v, str):
        return STRING_META
    return None


def metamethod(v, event):
    mt = metatable(v)
    if mt is None:
        return None
    return mt.get(event)


def index(rt, node, obj, key):
    """obj[key], with __index."""
    for _ in range(100):
        if isinstance(obj, Table):
            v = obj.get(key)
            if v is not None:
                return v
            h = metamethod(obj, "__index")
            if h is None:
                return None
        else:
            h = metamethod(obj, "__index")
            if h is None:
                lua_error(rt, node, "attempt to index a %s value%s" % (type_name(obj), describe(rt, target_of(node))))
        if is_function(h):
            return call_one(rt, node, h, [obj, key])
        obj = h
    lua_error(rt, node, "'__index' chain too long; possible loop")


def target_of(node):
    """What a call, an index or a member applies to (None without one)."""
    if node is None or node.kind not in ("call", "method_call", "index", "member"):
        return None
    return node.target


def setindex(rt, node, obj, key, value):
    """obj[key] = value, with __newindex."""
    for _ in range(100):
        if isinstance(obj, Table):
            if obj.get(key) is not None:
                obj.set(key, value)
                return
            h = metamethod(obj, "__newindex")
            if h is None:
                if key is None:
                    lua_error(rt, node, "table index is nil")
                if type(key) is float and key != key:
                    lua_error(rt, node, "table index is NaN")
                obj.set(key, value)
                return
        else:
            h = metamethod(obj, "__newindex")
            if h is None:
                lua_error(rt, node, "attempt to index a %s value%s" % (type_name(obj), describe(rt, target_of(node))))
        if is_function(h):
            call(rt, node, h, [obj, key, value])
            return
        obj = h
    lua_error(rt, node, "'__newindex' chain too long; possible loop")


# ----------------------------------------------------------------------
# Calls
# ----------------------------------------------------------------------


def call_raw(rt, node, f, args):
    """Call a Lua value with arguments: its results as a function returns
    them (retstat): a list, one value not nil as itself, or None for none
    (a function's end reached). `node` is the call expression, or None for
    a call made by the library (pcall's)."""
    if isinstance(f, zrun.Function):
        return rt.call(f, args)
    if isinstance(f, Builtin):
        return f.fn(rt, node, args)
    if isinstance(f, Method):
        return rt.call(f.fn, args[1:], receiver=args[0] if args else None)
    h = metamethod(f, "__call")
    if h is not None:
        return call_raw(rt, node, h, [f] + args)
    lua_error(rt, node, "attempt to call a %s value%s" % (type_name(f), describe(rt, target_of(node))))


def as_list(r):
    """A call's results (call_raw's) as a list."""
    if isinstance(r, list):
        return r
    return [] if r is None else [r]


def first(r):
    """A call's first result (call_raw's), nil if none."""
    if isinstance(r, list):
        return r[0] if r else None
    return r


def call(rt, node, f, args):
    """Call a Lua value with arguments: its results (a list)."""
    return as_list(call_raw(rt, node, f, args))


def call_one(rt, node, f, args):
    """Call a Lua value with arguments: its first result (no list made for
    a function returning one value)."""
    return first(call_raw(rt, node, f, args))


def call_args(node, rt):
    """A call's arguments: `(a, b)`, or a table or string alone."""
    args = node.args
    if not isinstance(args, list):
        args = [] if args is None else [args]
    if not args:
        skip = 2 if node.kind == "method_call" else 1
        args = [c for c in node.children[skip:] if c.kind in ("table", "string")]
    return explist(args, rt)


def call_target(node, rt):
    """What a call expression calls and with what: (function, arguments)."""
    if node.kind == "method_call":
        obj = rt.eval(node.target)
        f = index(rt, node, obj, node.method.text)
        if not is_function(f) and metamethod(f, "__call") is None:
            lua_error(rt, node, "attempt to call a %s value (method '%s')" % (type_name(f), node.method.text))
        return f, [obj] + call_args(node, rt)
    f = rt.eval(node.target)
    return f, call_args(node, rt)


def call_results(node, rt):
    """A call expression's results, as call_raw gives them."""
    f, args = call_target(node, rt)
    return call_raw(rt, node, f, args)


def call_values(node, rt):
    """A call expression's results (all of them)."""
    return as_list(call_results(node, rt))


def values(node, rt):
    """An expression's values: all a call's or `...`'s, else one."""
    k = node.kind
    if k == "call" or k == "method_call":
        return call_values(node, rt)
    if k == "vararg":
        return list(rt.varargs)
    return [rt.eval(node)]


def explist(nodes, rt):
    """A list of expressions' values: the last one's all, the others' first."""
    out = []
    n = len(nodes)
    for i in range(n):
        if i == n - 1:
            out.extend(values(nodes[i], rt))
        else:
            out.append(rt.eval(nodes[i]))
    return out


def explist_of(node, rt):
    """The values of an explist node (None: none)."""
    if node is None:
        return []
    return explist(node.children, rt)


# ----------------------------------------------------------------------
# Statements
# ----------------------------------------------------------------------


@lang.exec("chunk")
def chunk(node, rt):
    # (`return` at the top level ends the chunk: what it returns, the
    # chunk's value, a REPL's to show)
    try:
        for c in node.children:
            if c.kind == "block":
                rt.exec(c)
    except rt.Return as r:
        return r.args[0]
    return None


@lang.exec("local_stmt")
def local_stmt(node, rt):
    names = node.names
    vals = explist_of(node.values, rt)
    for i in range(len(names)):
        rt.store(names[i], vals[i] if i < len(vals) else None)


@lang.exec("local_function")
def local_function(node, rt):
    body = [c for c in node.children if c.kind == "funcbody"][0]
    rt.store(node.name, rt.function(body))


@lang.exec("function_stmt")
def function_stmt(node, rt):
    fname = node.children[0]
    body = node.children[1]
    f = rt.function(body)
    parts = fname.children
    method = fname.method
    if method is not None:
        f = Method(f)
        parts = parts[:-1]
    if len(parts) == 1 and method is None:
        assign_name(rt, parts[0], f)
        return
    obj = read_name(rt, parts[0])
    for p in parts[1:-1] if method is None else parts[1:]:
        obj = index(rt, fname, obj, p.text)
    key = method.text if method is not None else parts[-1].text
    setindex(rt, fname, obj, key, f)


@lang.exec("exprstat")
def exprstat(node, rt):
    targets = node.target
    if not isinstance(targets, list):
        targets = [targets]
    if node.values is None:
        call_values(targets[0], rt)
        return
    # The places (their tables and keys) first, then the values
    places = []
    for t in targets:
        if t.kind == "var":
            places.append((t, None, None))
        elif t.kind == "index":
            places.append((t, rt.eval(t.target), rt.eval(t.index)))
        else:
            places.append((t, rt.eval(t.target), t.name.text))
    vals = explist_of(node.values, rt)
    for i in range(len(places)):
        t, obj, key = places[i]
        v = vals[i] if i < len(vals) else None
        if t.kind == "var":
            assign_name(rt, t, v)
        else:
            setindex(rt, t, obj, key, v)


@lang.exec("do_stmt")
def do_stmt(node, rt):
    rt.exec(node.children[0])


@lang.exec("while_stmt")
def while_stmt(node, rt):
    body = node.children[1]
    while truthy(rt.eval(node.cond)):
        if not rt.loop(body):
            break


@lang.exec("repeat_stmt")
def repeat_stmt(node, rt):
    body = node.children[0]
    while rt.loop(body):
        pass


@lang.exec("repeat_body")
def repeat_body(node, rt):
    # (the condition sees the body's locals: it's evaluated in its scope,
    # and ends the loop as a break does)
    for c in node.children[:-1]:
        rt.exec(c)
    if truthy(rt.eval(node.cond)):
        raise rt.Break()


@lang.exec("if_stmt")
def if_stmt(node, rt):
    # cond, block, (cond, block)*, then the else block
    kids = node.children
    i = 0
    while i < len(kids):
        if kids[i].kind != "block":
            if truthy(rt.eval(kids[i])):
                rt.exec(kids[i + 1])
                return
            i += 2
        else:
            rt.exec(kids[i])
            return


@lang.exec("for_num")
def for_num(node, rt):
    bounds = node.bounds
    start = rt.eval(bounds[0])
    limit = rt.eval(bounds[1])
    step = rt.eval(bounds[2]) if len(bounds) > 2 else 1
    body = node.children[-1]
    for what, v in (("initial value", start), ("limit", limit), ("step", step)):
        if not is_number(v):
            lua_error(rt, node, "bad 'for' %s (number expected, got %s)" % (what, type_name(v)))
    if is_int(start) and is_int(step):
        if step == 0:
            lua_error(rt, node, "'for' step is zero")
        if is_int(limit):
            lim = limit
        elif type(limit) is float:
            if limit != limit:
                return
            lim = math.floor(limit) if step > 0 else math.ceil(limit)
            lim = max(min(lim, 2**63 - 1), -(2**63))
        else:
            lim = limit
        i = start
        while (i <= lim) if step > 0 else (i >= lim):
            # (a new variable each time round: closures keep theirs)
            rt.fresh(node)
            rt.store(node.var, i)
            if not rt.loop(body):
                break
            # (no wraparound past the limit)
            if (step > 0 and i > lim - step) or (step < 0 and i < lim - step):
                break
            i += step
        return
    fstart = float(start)
    flimit = float(limit)
    fstep = float(step)
    if fstep == 0:
        lua_error(rt, node, "'for' step is zero")
    x = fstart
    while (x <= flimit) if fstep > 0 else (x >= flimit):
        rt.fresh(node)
        rt.store(node.var, x)
        if not rt.loop(body):
            break
        x += fstep


@lang.exec("for_in")
def for_in(node, rt):
    names = node.names
    init = explist_of(node.iter, rt)
    f = init[0] if init else None
    s = init[1] if len(init) > 1 else None
    control = init[2] if len(init) > 2 else None
    body = node.children[-1]
    while True:
        vals = call(rt, node.iter, f, [s, control])
        first = vals[0] if vals else None
        if first is None:
            break
        control = first
        rt.fresh(node)
        for i in range(len(names)):
            rt.store(names[i], vals[i] if i < len(vals) else None)
        if not rt.loop(body):
            break


@lang.exec("retstat")
def retstat(node, rt):
    kids = node.children
    # `return f(x)`: a tail call, as Lua's (the function's frame given up
    # first: a chain of them takes no stack); a library function's call
    # is made as any other
    if kids and kids[0].kind == "explist":
        exps = kids[0].children
        if len(exps) == 1 and (exps[0].kind == "call" or exps[0].kind == "method_call"):
            f, args = call_target(exps[0], rt)
            if isinstance(f, zrun.Function):
                rt.tail_call(f, args)
            if isinstance(f, Method):
                rt.tail_call(f.fn, args[1:], receiver=args[0] if args else None)
            raise rt.Return(call_raw(rt, exps[0], f, args))
    vals = explist_of(kids[0], rt) if kids and kids[0].kind == "explist" else []
    # (one value, not nil: itself, no list made (call_raw's callers take
    # it so); nil stays a list: None is a function's end, no value)
    if len(vals) == 1:
        v = vals[0]
        if v is not None:
            raise rt.Return(v)
    raise rt.Return(vals)


@lang.exec("break_stmt")
def break_stmt(node, rt):
    raise rt.Break()


@lang.exec("goto_stmt")
def goto_stmt(node, rt):
    lua_error(rt, node, "goto isn't supported")


@lang.exec("label")
def label(node, rt):
    pass


# ----------------------------------------------------------------------
# Names: locals (zrules' scopes) and globals (_G)
# ----------------------------------------------------------------------

G = Table()


def read_name(rt, node):
    if rt.scope(node) is not None:
        return rt.load(node)
    name = node.text
    if name == "self":
        r = rt.receiver
        if r is not None:
            return r
    return G.get(name)


def assign_name(rt, node, value):
    if rt.scope(node) is not None:
        rt.store(node, value)
    else:
        G.set(node.text, value)


@lang.eval("var")
def var(node, rt):
    return read_name(rt, node)


# ----------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------


@lang.eval("nil")
def nil(node, rt):
    return None


@lang.eval("true")
def true(node, rt):
    return True


@lang.eval("false")
def false(node, rt):
    return False


@lang.eval("number")
def number(node, rt):
    return parse_number(node.text)


ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "\\": "\\", '"': '"', "'": "'", "\n": "\n"}


def string_value(text):
    """A string literal's value."""
    if text.startswith("["):
        level = text.index("[", 1) - 1
        body = text[level + 2 : len(text) - level - 2]
        if body.startswith("\r\n"):
            body = body[2:]
        elif body.startswith("\n") or body.startswith("\r"):
            body = body[1:]
        return body
    s = text[1:-1]
    out = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        e = s[i]
        if e in ESCAPES:
            out.append(ESCAPES[e])
            i += 1
        elif e == "x":
            out.append(chr(int(s[i + 1 : i + 3], 16)))
            i += 3
        elif e == "z":
            i += 1
            while i < n and s[i] in " \t\r\n\f\v":
                i += 1
        elif e.isdigit():
            j = i
            while j < n and j < i + 3 and s[j].isdigit():
                j += 1
            out.append(chr(int(s[i:j])))
            i = j
        elif e == "u":
            j = s.index("}", i)
            out.append(chr(int(s[i + 2 : j], 16)))
            i = j + 1
        else:
            out.append(e)
            i += 1
    return "".join(out)


@lang.eval("string")
def string(node, rt):
    return string_value(node.text)


@lang.eval("vararg")
def vararg(node, rt):
    return one(list(rt.varargs))


@lang.eval("function_exp")
def function_exp(node, rt):
    return rt.function(node.children[0])


@lang.eval("paren")
def paren(node, rt):
    return rt.eval(node.children[0])


@lang.eval("table")
def table(node, rt):
    t = Table()
    fields = node.children
    n = 1
    for i in range(len(fields)):
        f = fields[i]
        k = f.kind
        if k == "keyed_field":
            key = rt.eval(f.key)
            if key is None:
                lua_error(rt, f, "index is nil")
            t.set(key, rt.eval(f.value))
        elif k == "named_field":
            t.set(f.key.text, rt.eval(f.value))
        elif i == len(fields) - 1:
            for v in values(f, rt):
                t.set(n, v)
                n += 1
        else:
            t.set(n, rt.eval(f))
            n += 1
    return t


@lang.eval(["call", "method_call"])
def call_exp(node, rt):
    return first(call_results(node, rt))


@lang.eval("index")
def index_exp(node, rt):
    return index(rt, node, rt.eval(node.target), rt.eval(node.index))


@lang.eval("member")
def member_exp(node, rt):
    return index(rt, node, rt.eval(node.target), node.name.text)


ARITH_EVENTS = {"+": "__add", "-": "__sub", "*": "__mul", "/": "__div", "%": "__mod", "^": "__pow", "//": "__idiv"}
BIT_EVENTS = {"&": "__band", "|": "__bor", "~": "__bxor", "<<": "__shl", ">>": "__shr"}


@lang.eval(["exp", "and_exp", "cmp_exp", "bor_exp", "bxor_exp", "band_exp", "shift_exp", "concat_exp", "add_exp", "mul_exp", "pow_exp"])
def binary(node, rt):
    op = node.op.text
    a = rt.eval(node.left)
    if op == "or":
        return a if truthy(a) else rt.eval(node.right)
    if op == "and":
        return rt.eval(node.right) if truthy(a) else a
    b = rt.eval(node.right)
    return binop(rt, node, op, a, b)


def binop(rt, node, op, a, b):
    if op in ARITH_EVENTS:
        return arith(rt, node, op, a, b)
    if op == "..":
        return concat(rt, node, a, b)
    if op == "==":
        return equal(rt, node, a, b)
    if op == "~=":
        return not equal(rt, node, a, b)
    if op == "<":
        return less(rt, node, a, b)
    if op == ">":
        return less(rt, node, b, a)
    if op == "<=":
        return less_equal(rt, node, a, b)
    if op == ">=":
        return less_equal(rt, node, b, a)
    return bitwise(rt, node, op, a, b)


def arith(rt, node, op, a, b):
    if is_number(a) and is_number(b):
        return arith_numbers(rt, node, op, a, b)
    # Anything else: a metamethod (strings have them: their numbers, as
    # Lua's string library converts them)
    event = ARITH_EVENTS[op]
    h = metamethod(a, event)
    if h is None:
        h = metamethod(b, event)
    if h is not None:
        return call_one(rt, node, h, [a, b])
    # (Lua blames the first operand that isn't a number)
    bad, badv = (node.left, a) if not is_number(a) else (node.right, b)
    lua_error(rt, node, "attempt to perform arithmetic on a %s value%s" % (type_name(badv), describe(rt, bad)))


def string_arith(event):
    """A string's arithmetic metamethod (`"10" + 1`): both operands as
    numbers, else the other operand's metamethod, else an error."""
    op = [k for k, v in ARITH_EVENTS.items() if v == event][0]

    def fn(rt, node, args):
        a = args[0] if args else None
        b = args[1] if len(args) > 1 else None
        x = tonumber_arith(a)
        y = tonumber_arith(b)
        if x is not None and y is not None:
            return [arith_numbers(rt, node, op, x, y)]
        if not isinstance(b, str):
            h = metamethod(b, event)
            if h is not None:
                return call(rt, node, h, [a, b])
        lua_error(rt, node, "attempt to %s a '%s' with a '%s'" % (event[2:], type_name(a), type_name(b)))

    return Builtin(event, fn)


def arith_numbers(rt, node, op, x, y):
    if is_int(x) and is_int(y):
        # (64 bits, wrapping around, as Lua's)
        if op == "+":
            return rt.wrapping_add(x, y)
        if op == "-":
            return rt.wrapping_sub(x, y)
        if op == "*":
            return rt.wrapping_mul(x, y)
        xi = int(x)
        yi = int(y)
        if op == "//":
            if yi == 0:
                lua_error(rt, node, "attempt to divide by zero")
            return wrap(xi // yi)
        if op == "%":
            if yi == 0:
                lua_error(rt, node, "attempt to perform 'n%0'")
            return wrap(xi % yi)
    fx = float(x)
    fy = float(y)
    if op == "+":
        return fx + fy
    if op == "-":
        return fx - fy
    if op == "*":
        return fx * fy
    if op == "/":
        if fy == 0:
            if fx == 0 or fx != fx:
                return math.nan
            return math.copysign(math.inf, fx) * math.copysign(1.0, fy)
        return fx / fy
    if op == "^":
        try:
            return math.pow(fx, fy)
        except (OverflowError, ValueError):
            return math.inf if fx > 0 else math.nan
    if op == "//":
        if fy == 0:
            if fx == 0 or fx != fx:
                return math.nan
            return math.copysign(math.inf, fx) * math.copysign(1.0, fy)
        q = fx / fy
        return float(math.floor(q)) if math.isfinite(q) else q
    # %
    if fy == 0:
        return math.nan
    if fx != fx or fy != fy or fx in (math.inf, -math.inf):
        return math.nan
    if fy in (math.inf, -math.inf):
        return fx if (fx >= 0) == (fy > 0) else fy
    r = math.fmod(fx, fy)
    if r != 0 and (r < 0) != (fy < 0):
        r += fy
    return r


def bitwise(rt, node, op, a, b):
    x = to_integer(a) if is_number(a) or isinstance(a, str) else None
    y = to_integer(b) if is_number(b) or isinstance(b, str) else None
    if x is None or y is None:
        h = metamethod(a, BIT_EVENTS[op])
        if h is None:
            h = metamethod(b, BIT_EVENTS[op])
        if h is not None:
            return call_one(rt, node, h, [a, b])
        badv = a if x is None else b
        if is_number(badv):
            lua_error(rt, node, "number has no integer representation")
        lua_error(rt, node, "attempt to perform bitwise operation on a %s value%s" % (type_name(badv), describe(rt, node.left if x is None else node.right)))
    x = int(x)
    y = int(y)
    # (of two ints of 64 bits, & | ~ give one)
    if op == "&":
        return x & y
    if op == "|":
        return x | y
    if op == "~":
        return x ^ y
    if op == "<<":
        return shift_left(rt, x, y)
    return shift_left(rt, x, -y)


def shift_left(rt, x, n):
    """Lua's shift: logical, to the right for a negative n, 0 past 63."""
    if n <= -64 or n >= 64:
        return 0
    if n >= 0:
        return rt.wrapping_shl(x, n)
    return rt.wrapping_ushr(x, -n)


def concat(rt, node, a, b):
    if (isinstance(a, str) or is_number(a)) and (isinstance(b, str) or is_number(b)):
        return tostring(rt, node, a) + tostring(rt, node, b)
    h = metamethod(a, "__concat")
    if h is None:
        h = metamethod(b, "__concat")
    if h is not None:
        return call_one(rt, node, h, [a, b])
    bad = a if not (isinstance(a, str) or is_number(a)) else b
    badn = node.left if bad is a else node.right
    lua_error(rt, node, "attempt to concatenate a %s value%s" % (type_name(bad), describe(rt, badn)))


def raw_equal(a, b):
    if is_number(a) and is_number(b):
        return a == b
    if type(a) is not type(b) and not (is_int(a) and is_int(b)):
        return False
    if isinstance(a, (Table, Method, Builtin, zrun.Function)):
        return a is b
    return a == b


def equal(rt, node, a, b):
    if raw_equal(a, b):
        return True
    if isinstance(a, Table) and isinstance(b, Table):
        h = metamethod(a, "__eq")
        if h is None:
            h = metamethod(b, "__eq")
        if h is not None:
            return truthy(call_one(rt, node, h, [a, b]))
    return False


def compare_error(rt, node, a, b):
    ta = type_name(a)
    tb = type_name(b)
    if ta == tb:
        lua_error(rt, node, "attempt to compare two %s values" % ta)
    lua_error(rt, node, "attempt to compare %s with %s" % (ta, tb))


def less(rt, node, a, b):
    if is_number(a) and is_number(b):
        return a < b
    if isinstance(a, str) and isinstance(b, str):
        return a < b
    h = metamethod(a, "__lt")
    if h is None:
        h = metamethod(b, "__lt")
    if h is not None:
        return truthy(call_one(rt, node, h, [a, b]))
    compare_error(rt, node, a, b)


def less_equal(rt, node, a, b):
    if is_number(a) and is_number(b):
        return a <= b
    if isinstance(a, str) and isinstance(b, str):
        return a <= b
    h = metamethod(a, "__le")
    if h is None:
        h = metamethod(b, "__le")
    if h is not None:
        return truthy(call_one(rt, node, h, [a, b]))
    compare_error(rt, node, a, b)


@lang.eval("unop_exp")
def unop_exp(node, rt):
    op = node.op.text
    v = rt.eval(node.operand)
    if op == "not":
        return not truthy(v)
    if op == "-":
        x = tonumber_arith(v)
        if x is None:
            h = metamethod(v, "__unm")
            if h is not None:
                return call_one(rt, node, h, [v, v])
            lua_error(rt, node, "attempt to perform arithmetic on a %s value%s" % (type_name(v), describe(rt, node.operand)))
        if is_int(x):
            return wrap(-int(x))
        return -x
    if op == "#":
        if isinstance(v, str):
            return len(v.encode("utf-8"))
        h = metamethod(v, "__len")
        if h is not None:
            return call_one(rt, node, h, [v])
        if isinstance(v, Table):
            return v.length()
        lua_error(rt, node, "attempt to get length of a %s value%s" % (type_name(v), describe(rt, node.operand)))
    # ~
    x = to_integer(v) if is_number(v) or isinstance(v, str) else None
    if x is None:
        h = metamethod(v, "__bnot")
        if h is not None:
            return call_one(rt, node, h, [v, v])
        lua_error(rt, node, "attempt to perform bitwise operation on a %s value%s" % (type_name(v), describe(rt, node.operand)))
    return wrap(~int(x))


# ----------------------------------------------------------------------
# The library
# ----------------------------------------------------------------------


def arg(rt, node, args, i, fname, expected=None):
    """Argument i (0-based) of a library function, checked."""
    v = args[i] if i < len(args) else None
    if expected == "table" and not isinstance(v, Table):
        arg_error(rt, node, i, fname, "table expected, got " + got(v, i, args))
    if expected == "number":
        n = tonumber_arith(v)
        if n is None:
            arg_error(rt, node, i, fname, "number expected, got " + got(v, i, args))
        return n
    if expected == "integer":
        n = tonumber_arith(v)
        if n is None:
            arg_error(rt, node, i, fname, "number expected, got " + got(v, i, args))
        k = to_integer(n)
        if k is None:
            arg_error(rt, node, i, fname, "number has no integer representation")
        return k
    if expected == "string":
        if is_number(v):
            return tostring(rt, node, v)
        if not isinstance(v, str):
            arg_error(rt, node, i, fname, "string expected, got " + got(v, i, args))
    if expected == "function" and not is_function(v):
        arg_error(rt, node, i, fname, "function expected, got " + got(v, i, args))
    if expected == "any" and i >= len(args):
        arg_error(rt, node, i, fname, "value expected")
    return v


def got(v, i, args):
    return "no value" if i >= len(args) else type_name(v)


def arg_error(rt, node, i, fname, message):
    """A bad argument to a library function. Called from Lua code: at the
    call, named as it was called (a method's self not counted); called by
    the library (node None): no position, its full name (`string.rep`)."""
    if node is None:
        name = QUALIFIED.get(fname, fname)
    else:
        name = call_name(node) or fname
        if node.kind == "method_call":
            i -= 1
    lua_error(rt, node, "bad argument #%d to '%s' (%s)" % (i + 1, name, message))


def call_name(node):
    """The name a call expression calls its function by, if it has one."""
    if node.kind == "method_call":
        return node.method.text
    if node.kind != "call":
        return None
    t = node.target
    if t.kind == "var":
        return t.text
    if t.kind == "member":
        return t.name.text
    return None


def opt(args, i, default):
    v = args[i] if i < len(args) else None
    return default if v is None else v


# Library functions' full names (string.rep), by their names
QUALIFIED = {}


def lib(table, name, qualified=None):
    def register(fn):
        table.set(name, Builtin(qualified or name, fn))
        prefix = LIB_NAMES.get(id(table), "")
        QUALIFIED.setdefault(name, prefix + name)
        return fn

    return register


LIB_NAMES = {}


OUT = []


def write(text):
    sys.stdout.write(text)


@lib(G, "print")
def lua_print(rt, node, args):
    write("\t".join(tostring(rt, node, a) for a in args) + "\n")
    return []


@lib(G, "type")
def lua_type(rt, node, args):
    arg(rt, node, args, 0, "type", "any")
    return [type_name(args[0])]


@lib(G, "tostring")
def lua_tostring(rt, node, args):
    arg(rt, node, args, 0, "tostring", "any")
    return [tostring(rt, node, args[0])]


@lib(G, "tonumber")
def lua_tonumber(rt, node, args):
    v = arg(rt, node, args, 0, "tonumber", "any")
    base = opt(args, 1, None)
    if base is None:
        if is_number(v):
            return [v]
        if isinstance(v, str):
            return [str_to_number(v)]
        return [None]
    b = to_integer(base)
    s = arg(rt, node, args, 0, "tonumber", "string").strip().lower()
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    try:
        n = int(s, b)
    except ValueError:
        return [None]
    return [wrap(-n if neg else n)]


@lib(G, "rawequal")
def lua_rawequal(rt, node, args):
    return [raw_equal(opt(args, 0, None), opt(args, 1, None))]


@lib(G, "rawlen")
def lua_rawlen(rt, node, args):
    v = opt(args, 0, None)
    if isinstance(v, Table):
        return [v.length()]
    if isinstance(v, str):
        return [len(v.encode("utf-8"))]
    lua_error(rt, node, "table or string expected")


@lib(G, "rawget")
def lua_rawget(rt, node, args):
    t = arg(rt, node, args, 0, "rawget", "table")
    return [t.get(opt(args, 1, None))]


@lib(G, "rawset")
def lua_rawset(rt, node, args):
    t = arg(rt, node, args, 0, "rawset", "table")
    t.set(opt(args, 1, None), opt(args, 2, None))
    return [t]


@lib(G, "setmetatable")
def lua_setmetatable(rt, node, args):
    t = arg(rt, node, args, 0, "setmetatable", "table")
    mt = opt(args, 1, None)
    if mt is not None and not isinstance(mt, Table):
        arg_error(rt, node, 1, "setmetatable", "nil or table expected")
    if t.meta is not None and t.meta.get("__metatable") is not None:
        lua_error(rt, node, "cannot change a protected metatable")
    t.meta = mt
    return [t]


@lib(G, "getmetatable")
def lua_getmetatable(rt, node, args):
    mt = metatable(opt(args, 0, None))
    if mt is None:
        return [None]
    protected = mt.get("__metatable")
    return [protected if protected is not None else mt]


@lib(G, "assert")
def lua_assert(rt, node, args):
    if not args or not truthy(args[0]):
        # (as error() at level 1: a string message gets the position)
        if len(args) > 1:
            m = args[1]
            if isinstance(m, str):
                m = where(rt, node) + m
            raise rt.Throw(m, tostring(rt, node, m))
        lua_error(rt, node, "assertion failed!")
    return list(args)


@lib(G, "error")
def lua_error_fn(rt, node, args):
    v = opt(args, 0, None)
    level = to_integer(opt(args, 1, 1))
    # (level 1: where error was called; 2 blames the caller, which this
    # Lua doesn't track: no position, as for a call made by the library)
    if isinstance(v, str) and level == 1:
        v = where(rt, node) + v
    raise rt.Throw(v, tostring(rt, node, v) if v is not None else "nil")


@lib(G, "pcall")
def lua_pcall(rt, node, args):
    f = arg(rt, node, args, 0, "pcall", "any")
    try:
        # (called by the library: errors it raises itself have no position)
        return [True] + call(rt, None, f, list(args[1:]))
    except rt.Throw as e:
        return [False, e.value]
    except zrun.Error as e:
        return [False, e.diagnostic.message]


@lib(G, "xpcall")
def lua_xpcall(rt, node, args):
    f = opt(args, 0, None)
    handler = opt(args, 1, None)
    try:
        return [True] + call(rt, None, f, list(args[2:]))
    except rt.Throw as e:
        return [False] + call(rt, None, handler, [e.value])
    except zrun.Error as e:
        return [False] + call(rt, None, handler, [e.diagnostic.message])


@lib(G, "select")
def lua_select(rt, node, args):
    n = opt(args, 0, None)
    if n == "#":
        return [len(args) - 1]
    i = arg(rt, node, args, 0, "select", "integer")
    if i < 0:
        i = len(args) + i
        if i < 1:
            arg_error(rt, node, 0, "select", "index out of range")
        return list(args[i:])
    if i == 0:
        arg_error(rt, node, 0, "select", "index out of range")
    return list(args[i:])


@lib(G, "next")
def lua_next(rt, node, args):
    t = arg(rt, node, args, 0, "next", "table")
    r = t.next(opt(args, 1, None))
    if r == "bad":
        lua_error(rt, node, "invalid key to 'next'")
    return [None] if r is None else [r[0], r[1]]


NEXT = G.get("next")


@lib(G, "pairs")
def lua_pairs(rt, node, args):
    v = arg(rt, node, args, 0, "pairs", "any")
    h = metamethod(v, "__pairs")
    if h is not None:
        return call(rt, node, h, [v])[:3]
    if not isinstance(v, Table):
        arg_error(rt, node, 0, "pairs", "table expected, got " + type_name(v))
    return [NEXT, v, None]


def ipairs_step(rt, node, args):
    t = args[0]
    i = int(args[1]) + 1
    v = index(rt, node, t, i) if not isinstance(t, Table) or t.meta is not None else t.get(i)
    return [None] if v is None else [i, v]


IPAIRS_STEP = Builtin("ipairs_iterator", ipairs_step)


@lib(G, "ipairs")
def lua_ipairs(rt, node, args):
    v = arg(rt, node, args, 0, "ipairs", "any")
    return [IPAIRS_STEP, v, 0]


@lib(G, "unpack")
def lua_unpack(rt, node, args):
    t = arg(rt, node, args, 0, "unpack", "table")
    i = to_integer(opt(args, 1, 1))
    j = to_integer(opt(args, 2, None)) if opt(args, 2, None) is not None else t.length()
    return [t.get(k) for k in range(i, j + 1)]


G.set("_G", G)
G.set("_VERSION", "Lua 5.4")

# -- table --

TABLE = Table()
G.set("table", TABLE)
LIB_NAMES[id(TABLE)] = "table."
TABLE.set("unpack", Builtin("unpack", lua_unpack))


@lib(TABLE, "insert", "insert")
def table_insert(rt, node, args):
    t = arg(rt, node, args, 0, "insert", "table")
    n = t.length()
    if len(args) == 2:
        t.set(n + 1, args[1])
        return []
    if len(args) != 3:
        lua_error(rt, node, "wrong number of arguments to 'insert'")
    pos = arg(rt, node, args, 1, "insert", "integer")
    if pos < 1 or pos > n + 1:
        arg_error(rt, node, 1, "insert", "position out of bounds")
    for k in range(n, pos - 1, -1):
        t.set(k + 1, t.get(k))
    t.set(pos, args[2])
    return []


@lib(TABLE, "remove", "remove")
def table_remove(rt, node, args):
    t = arg(rt, node, args, 0, "remove", "table")
    n = t.length()
    pos = to_integer(opt(args, 1, n))
    if len(args) > 1 and n + 1 != pos and (pos < 1 or pos > n + 1) and not (n == 0 and pos == 0):
        arg_error(rt, node, 1, "remove", "position out of bounds")
    v = t.get(pos)
    for k in range(pos, n):
        t.set(k, t.get(k + 1))
    if pos <= n:
        t.set(n, None)
    return [v]


@lib(TABLE, "concat", "concat")
def table_concat(rt, node, args):
    t = arg(rt, node, args, 0, "concat", "table")
    sep = arg(rt, node, args, 1, "concat", "string") if opt(args, 1, None) is not None else ""
    i = to_integer(opt(args, 2, 1))
    j = to_integer(opt(args, 3, None)) if opt(args, 3, None) is not None else t.length()
    parts = []
    for k in range(i, j + 1):
        v = t.get(k)
        if not (isinstance(v, str) or is_number(v)):
            lua_error(rt, node, "invalid value (at index %d) in table for 'concat'" % k)
        parts.append(tostring(rt, node, v))
    return [sep.join(parts)]


@lib(TABLE, "sort", "sort")
def table_sort(rt, node, args):
    t = arg(rt, node, args, 0, "sort", "table")
    comp = opt(args, 1, None)
    n = t.length()
    items = [t.get(k) for k in range(1, n + 1)]

    def lt(a, b):
        if comp is not None:
            return truthy(call_one(rt, node, comp, [a, b]))
        return less(rt, node, a, b)

    # (merge sort: a comparison function that isn't a strict order still
    # gives some order, without Python's sort raising)
    def msort(xs):
        if len(xs) <= 1:
            return xs
        mid = len(xs) // 2
        left = msort(xs[:mid])
        right = msort(xs[mid:])
        out = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if lt(right[j], left[i]):
                out.append(right[j])
                j += 1
            else:
                out.append(left[i])
                i += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out

    for k, v in enumerate(msort(items), 1):
        t.set(k, v)
    return []


@lib(TABLE, "pack", "pack")
def table_pack(rt, node, args):
    t = Table()
    for k, v in enumerate(args, 1):
        t.set(k, v)
    t.set("n", len(args))
    return [t]


# -- math --

MATH = Table()
G.set("math", MATH)
LIB_NAMES[id(MATH)] = "math."
MATH.set("pi", math.pi)
MATH.set("huge", math.inf)
MATH.set("maxinteger", 2**63 - 1)
MATH.set("mininteger", -(2**63))


def float_to_int_or_float(x):
    if x.is_integer() and -(2.0**63) <= x < 2.0**63:
        return int(x)
    return x


@lib(MATH, "floor", "floor")
def math_floor(rt, node, args):
    x = arg(rt, node, args, 0, "floor", "number")
    if is_int(x):
        return [x]
    return [float_to_int_or_float(float(math.floor(x)))] if x == x and abs(x) != math.inf else [x]


@lib(MATH, "ceil", "ceil")
def math_ceil(rt, node, args):
    x = arg(rt, node, args, 0, "ceil", "number")
    if is_int(x):
        return [x]
    return [float_to_int_or_float(float(math.ceil(x)))] if x == x and abs(x) != math.inf else [x]


@lib(MATH, "abs", "abs")
def math_abs(rt, node, args):
    x = arg(rt, node, args, 0, "abs", "number")
    return [wrap(abs(int(x))) if is_int(x) else abs(x)]


@lib(MATH, "max", "max")
def math_max(rt, node, args):
    best = arg(rt, node, args, 0, "max", "number")
    for i in range(1, len(args)):
        x = arg(rt, node, args, i, "max", "number")
        if best < x:
            best = x
    return [best]


@lib(MATH, "min", "min")
def math_min(rt, node, args):
    best = arg(rt, node, args, 0, "min", "number")
    for i in range(1, len(args)):
        x = arg(rt, node, args, i, "min", "number")
        if x < best:
            best = x
    return [best]


@lib(MATH, "sqrt", "sqrt")
def math_sqrt(rt, node, args):
    x = float(arg(rt, node, args, 0, "sqrt", "number"))
    return [math.sqrt(x) if x >= 0 else math.nan]


@lib(MATH, "fmod", "fmod")
def math_fmod(rt, node, args):
    a = arg(rt, node, args, 0, "fmod", "number")
    b = arg(rt, node, args, 1, "fmod", "number")
    if is_int(a) and is_int(b):
        if b == 0:
            arg_error(rt, node, 1, "fmod", "zero")
        return [int(math.fmod(a, b))]
    return [math.fmod(float(a), float(b)) if b != 0 else math.nan]


@lib(MATH, "modf", "modf")
def math_modf(rt, node, args):
    x = float(arg(rt, node, args, 0, "modf", "number"))
    if x in (math.inf, -math.inf):
        return [x, 0.0]
    f, i = math.modf(x)
    return [float_to_int_or_float(i) if True else i, f]


@lib(MATH, "tointeger", "tointeger")
def math_tointeger(rt, node, args):
    v = opt(args, 0, None)
    return [to_integer(v) if is_number(v) or isinstance(v, str) else None]


@lib(MATH, "type", "type")
def math_type(rt, node, args):
    v = arg(rt, node, args, 0, "type", "any")
    return ["integer" if is_int(v) else "float" if type(v) is float else None]


@lib(MATH, "exp", "exp")
def math_exp(rt, node, args):
    return [math.exp(arg(rt, node, args, 0, "exp", "number"))]


@lib(MATH, "log", "log")
def math_log(rt, node, args):
    x = float(arg(rt, node, args, 0, "log", "number"))
    base = opt(args, 1, None)
    if x == 0:
        return [-math.inf]
    if x < 0:
        return [math.nan]
    if base is None:
        return [math.log(x)]
    b = float(base)
    if b == 2:
        return [math.log2(x)]
    if b == 10:
        return [math.log10(x)]
    return [math.log(x) / math.log(b)]


for _name in ("sin", "cos", "tan", "asin", "acos"):
    MATH.set(_name, Builtin(_name, (lambda f, n: lambda rt, node, args: [f(arg(rt, node, args, 0, n, "number"))])(getattr(math, _name), _name)))


@lib(MATH, "atan", "atan")
def math_atan(rt, node, args):
    y = float(arg(rt, node, args, 0, "atan", "number"))
    x = float(opt(args, 1, 1.0))
    return [math.atan2(y, x)]


@lib(MATH, "ult", "ult")
def math_ult(rt, node, args):
    a = arg(rt, node, args, 0, "ult", "integer")
    b = arg(rt, node, args, 1, "ult", "integer")
    return [(int(a) & 0xFFFFFFFFFFFFFFFF) < (int(b) & 0xFFFFFFFFFFFFFFFF)]


# (a fixed sequence: runs give the same numbers in every mode)
RANDOM_STATE = [0x2545F4914F6CDD1D]


def next_random():
    x = RANDOM_STATE[0]
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 7
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    RANDOM_STATE[0] = x
    return x


@lib(MATH, "random", "random")
def math_random(rt, node, args):
    r = next_random()
    if not args:
        return [(r >> 11) / float(1 << 53)]
    lo = 1
    hi = arg(rt, node, args, 0, "random", "integer")
    if len(args) > 1:
        lo = hi
        hi = arg(rt, node, args, 1, "random", "integer")
    if lo > hi:
        arg_error(rt, node, len(args) - 1, "random", "interval is empty")
    return [lo + r % (hi - lo + 1)]


@lib(MATH, "randomseed", "randomseed")
def math_randomseed(rt, node, args):
    RANDOM_STATE[0] = int(to_integer(opt(args, 0, 0)) or 0) & 0xFFFFFFFFFFFFFFFF or 0x2545F4914F6CDD1D
    return []


# -- os, io --

OS = Table()
G.set("os", OS)
LIB_NAMES[id(OS)] = "os."


@lib(OS, "time", "time")
def os_time(rt, node, args):
    return [int(time.time())]


@lib(OS, "clock", "clock")
def os_clock(rt, node, args):
    return [time.process_time()]


@lib(OS, "getenv", "getenv")
def os_getenv(rt, node, args):
    return [os.environ.get(arg(rt, node, args, 0, "getenv", "string"))]


IO = Table()
G.set("io", IO)
LIB_NAMES[id(IO)] = "io."


@lib(IO, "write", "write")
def io_write(rt, node, args):
    for i in range(len(args)):
        v = args[i]
        # (numbers as C's printf writes them: floats without Lua's ".0")
        if type(v) is float:
            write("%.14g" % v)
        else:
            write(arg(rt, node, args, i, "write", "string"))
    return []


# -- string --

STRING = Table()
G.set("string", STRING)
LIB_NAMES[id(STRING)] = "string."
STRING_META.set("__index", STRING)
for _event in ARITH_EVENTS.values():
    STRING_META.set(_event, string_arith(_event))
STRING_META.set("__unm", Builtin("__unm", lambda rt, node, args: string_arith("__sub").fn(rt, node, [0, args[0]]) if tonumber_arith(args[0]) is not None else lua_error(rt, node, "attempt to perform arithmetic on a string value")))


def str_index(i, n):
    """A Lua string index (1-based, negative from the end) as 0-based."""
    if i > 0:
        return i - 1
    if i == 0:
        return 0
    return max(n + i, 0)


@lib(STRING, "len", "len")
def string_len(rt, node, args):
    return [len(arg(rt, node, args, 0, "len", "string"))]


@lib(STRING, "sub", "sub")
def string_sub(rt, node, args):
    s = arg(rt, node, args, 0, "sub", "string")
    n = len(s)
    i = arg(rt, node, args, 1, "sub", "integer") if len(args) > 1 and args[1] is not None else 1
    j = arg(rt, node, args, 2, "sub", "integer") if len(args) > 2 and args[2] is not None else -1
    if i < 0:
        i = max(n + i + 1, 1)
    elif i == 0:
        i = 1
    if j < 0:
        j = n + j + 1
    elif j > n:
        j = n
    return [s[i - 1 : j] if i <= j else ""]


@lib(STRING, "upper", "upper")
def string_upper(rt, node, args):
    return [arg(rt, node, args, 0, "upper", "string").upper()]


@lib(STRING, "lower", "lower")
def string_lower(rt, node, args):
    return [arg(rt, node, args, 0, "lower", "string").lower()]


@lib(STRING, "rep", "rep")
def string_rep(rt, node, args):
    s = arg(rt, node, args, 0, "rep", "string")
    n = arg(rt, node, args, 1, "rep", "integer")
    sep = arg(rt, node, args, 2, "rep", "string") if opt(args, 2, None) is not None else ""
    if n <= 0:
        return [""]
    return [sep.join([s] * n)]


@lib(STRING, "reverse", "reverse")
def string_reverse(rt, node, args):
    return [arg(rt, node, args, 0, "reverse", "string")[::-1]]


@lib(STRING, "byte", "byte")
def string_byte(rt, node, args):
    s = arg(rt, node, args, 0, "byte", "string")
    i = to_integer(opt(args, 1, 1))
    j = to_integer(opt(args, 2, i))
    n = len(s)
    if i < 0:
        i = n + i + 1
    if j < 0:
        j = n + j + 1
    i = max(i, 1)
    j = min(j, n)
    return [ord(s[k - 1]) for k in range(i, j + 1)]


@lib(STRING, "char", "char")
def string_char(rt, node, args):
    out = []
    for i in range(len(args)):
        c = arg(rt, node, args, i, "char", "integer")
        if c < 0 or c > 255:
            arg_error(rt, node, i, "char", "value out of range")
        out.append(chr(c))
    return ["".join(out)]


def format_q(rt, node, v):
    if isinstance(v, str):
        out = ['"']
        for ch in v:
            if ch in '"\\':
                out.append("\\" + ch)
            elif ch == "\n":
                out.append("\\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\0":
                out.append("\\0")
            elif ord(ch) < 32 or ord(ch) == 127:
                out.append("\\%d" % ord(ch))
            else:
                out.append(ch)
        out.append('"')
        return "".join(out)
    if is_int(v):
        return "0x%x" % (int(v) & 0xFFFFFFFFFFFFFFFF) if v == -(2**63) else str(int(v))
    if type(v) is float:
        if v == math.inf:
            return "1e9999"
        if v == -math.inf:
            return "-1e9999"
        if v != v:
            return "(0/0)"
        return float.hex(v) if not v.is_integer() else "%d.0" % v if abs(v) < 1e16 else float.hex(v)
    return tostring(rt, node, v)


@lib(STRING, "format", "format")
def string_format(rt, node, args):
    fmt = arg(rt, node, args, 0, "format", "string")
    out = []
    i = 0
    n = len(fmt)
    a = 1
    while i < n:
        ch = fmt[i]
        if ch != "%":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i < n and fmt[i] == "%":
            out.append("%")
            i += 1
            continue
        j = i
        while j < n and fmt[j] in "-+ #0":
            j += 1
        while j < n and fmt[j].isdigit():
            j += 1
        if j < n and fmt[j] == ".":
            j += 1
            while j < n and fmt[j].isdigit():
                j += 1
        if j >= n:
            lua_error(rt, node, "invalid conversion '%%%s' to 'format'" % fmt[i:j])
        spec = fmt[i:j]
        conv = fmt[j]
        i = j + 1
        if a >= len(args) and conv != "%":
            arg_error(rt, node, a, "format", "no value")
        v = args[a] if a < len(args) else None
        a += 1
        if conv in "di":
            k = arg(rt, node, args, a - 1, "format", "integer")
            out.append(("%" + spec + "d") % k)
        elif conv == "u":
            k = arg(rt, node, args, a - 1, "format", "integer")
            out.append(("%" + spec + "d") % (int(k) & 0xFFFFFFFFFFFFFFFF))
        elif conv == "c":
            out.append(chr(arg(rt, node, args, a - 1, "format", "integer")))
        elif conv in "xXo":
            k = arg(rt, node, args, a - 1, "format", "integer")
            out.append(("%" + spec + conv) % (int(k) & 0xFFFFFFFFFFFFFFFF))
        elif conv in "eEfFgG":
            x = float(arg(rt, node, args, a - 1, "format", "number"))
            if x != x or x in (math.inf, -math.inf):
                s = ("-" if x < 0 else "") + ("inf" if x == x else "nan")
                if conv in "EFG":
                    s = s.upper()
                out.append(("%" + spec.replace("0", "") + "s") % s)
            else:
                out.append(("%" + spec + conv) % x)
        elif conv == "a" or conv == "A":
            x = float(arg(rt, node, args, a - 1, "format", "number"))
            s = float.hex(x)
            out.append(s.upper() if conv == "A" else s)
        elif conv == "s":
            s = tostring(rt, node, v)
            out.append(("%" + spec + "s") % s)
        elif conv == "q":
            out.append(format_q(rt, node, v))
        else:
            lua_error(rt, node, "invalid conversion '%%%s' to 'format'" % (spec + conv))
    return ["".join(out)]


# Lua patterns (lstrlib's matcher)

SPECIALS = "^$*+?.([%-"
MAXCCALLS = 200


class MatchState:
    __slots__ = ("src", "pat", "level", "capture", "depth", "rt", "node")

    def __init__(self, rt, node, src, pat):
        self.src = src
        self.pat = pat
        self.level = 0
        self.capture = []
        self.depth = 0
        self.rt = rt
        self.node = node


CAP_UNFINISHED = -1
CAP_POSITION = -2


def class_end(ms, p):
    pat = ms.pat
    if p >= len(pat):
        lua_error(ms.rt, ms.node, "malformed pattern (ends with '%')")
    c = pat[p]
    p += 1
    if c == "%":
        if p >= len(pat):
            lua_error(ms.rt, ms.node, "malformed pattern (ends with '%')")
        return p + 1
    if c == "[":
        if p < len(pat) and pat[p] == "^":
            p += 1
        while True:
            if p >= len(pat):
                lua_error(ms.rt, ms.node, "malformed pattern (missing ']')")
            c = pat[p]
            p += 1
            if c == "%":
                p += 1
            if p < len(pat) and pat[p] == "]":
                return p + 1
            if p >= len(pat):
                lua_error(ms.rt, ms.node, "malformed pattern (missing ']')")
    return p


def single_class(c, cl):
    o = ord(c)
    k = cl.lower()
    if k == "a":
        r = c.isalpha() and o < 128
    elif k == "c":
        r = o < 32 or o == 127
    elif k == "d":
        r = "0" <= c <= "9"
    elif k == "g":
        r = 32 < o < 127
    elif k == "l":
        r = "a" <= c <= "z"
    elif k == "p":
        r = 32 < o < 127 and not c.isalnum()
    elif k == "s":
        r = c in " \t\n\r\f\v"
    elif k == "u":
        r = "A" <= c <= "Z"
    elif k == "w":
        r = c.isalnum() and o < 128
    elif k == "x":
        r = c in "0123456789abcdefABCDEF"
    else:
        return cl == c
    return not r if cl.isupper() else r


def match_bracket_class(ms, c, p, ec):
    pat = ms.pat
    sig = True
    p += 1
    if pat[p] == "^":
        sig = False
        p += 1
    while p < ec:
        if pat[p] == "%":
            p += 1
            if single_class(c, pat[p]):
                return sig
            p += 1
        elif p + 2 < ec and pat[p + 1] == "-":
            if pat[p] <= c <= pat[p + 2]:
                return sig
            p += 3
        else:
            if pat[p] == c:
                return sig
            p += 1
    return not sig


def single_match(ms, s, p, ep):
    if s >= len(ms.src):
        return False
    c = ms.src[s]
    pc = ms.pat[p]
    if pc == ".":
        return True
    if pc == "%":
        return single_class(c, ms.pat[p + 1])
    if pc == "[":
        return match_bracket_class(ms, c, p, ep - 1)
    return pc == c


def do_match(ms, s, p):
    ms.depth += 1
    if ms.depth > MAXCCALLS:
        lua_error(ms.rt, ms.node, "pattern too complex")
    try:
        pat = ms.pat
        while True:
            if p >= len(pat):
                return s
            pc = pat[p]
            if pc == "(":
                if p + 1 < len(pat) and pat[p + 1] == ")":
                    return start_capture(ms, s, p + 2, CAP_POSITION)
                return start_capture(ms, s, p + 1, CAP_UNFINISHED)
            if pc == ")":
                return end_capture(ms, s, p + 1)
            if pc == "$" and p + 1 == len(pat):
                return s if s == len(ms.src) else None
            if pc == "%" and p + 1 < len(pat):
                nx = pat[p + 1]
                if nx == "b":
                    s = match_balance(ms, s, p + 2)
                    if s is None:
                        return None
                    p += 4
                    continue
                if nx == "f":
                    p += 2
                    if p >= len(pat) or pat[p] != "[":
                        lua_error(ms.rt, ms.node, "missing '[' after '%f' in pattern")
                    ep = class_end(ms, p)
                    prev = ms.src[s - 1] if s > 0 else "\0"
                    cur = ms.src[s] if s < len(ms.src) else "\0"
                    if not match_bracket_class(ms, prev, p, ep - 1) and match_bracket_class(ms, cur, p, ep - 1):
                        p = ep
                        continue
                    return None
                if nx.isdigit():
                    s = match_capture(ms, s, int(nx))
                    if s is None:
                        return None
                    p += 2
                    continue
            ep = class_end(ms, p)
            epc = pat[ep] if ep < len(pat) else ""
            if epc == "?":
                if single_match(ms, s, p, ep):
                    r = do_match(ms, s + 1, ep + 1)
                    if r is not None:
                        return r
                p = ep + 1
                continue
            if epc == "+":
                return max_expand(ms, s + 1, p, ep) if single_match(ms, s, p, ep) else None
            if epc == "*":
                return max_expand(ms, s, p, ep)
            if epc == "-":
                return min_expand(ms, s, p, ep)
            if not single_match(ms, s, p, ep):
                return None
            s += 1
            p = ep
    finally:
        ms.depth -= 1


def max_expand(ms, s, p, ep):
    i = 0
    while single_match(ms, s + i, p, ep):
        i += 1
    while i >= 0:
        r = do_match(ms, s + i, ep + 1)
        if r is not None:
            return r
        i -= 1
    return None


def min_expand(ms, s, p, ep):
    while True:
        r = do_match(ms, s, ep + 1)
        if r is not None:
            return r
        if single_match(ms, s, p, ep):
            s += 1
        else:
            return None


def start_capture(ms, s, p, what):
    ms.capture.append([s, what])
    ms.level += 1
    r = do_match(ms, s, p)
    if r is None:
        ms.level -= 1
        ms.capture.pop()
    return r


def end_capture(ms, s, p):
    lvl = -1
    for i in range(ms.level - 1, -1, -1):
        if ms.capture[i][1] == CAP_UNFINISHED:
            lvl = i
            break
    if lvl < 0:
        lua_error(ms.rt, ms.node, "invalid pattern capture")
    ms.capture[lvl][1] = s - ms.capture[lvl][0]
    r = do_match(ms, s, p)
    if r is None:
        ms.capture[lvl][1] = CAP_UNFINISHED
    return r


def match_balance(ms, s, p):
    if p + 1 >= len(ms.pat):
        lua_error(ms.rt, ms.node, "malformed pattern (missing arguments to '%b')")
    if s >= len(ms.src) or ms.src[s] != ms.pat[p]:
        return None
    b = ms.pat[p]
    e = ms.pat[p + 1]
    cont = 1
    i = s + 1
    while i < len(ms.src):
        c = ms.src[i]
        if c == e:
            cont -= 1
            if cont == 0:
                return i + 1
        elif c == b:
            cont += 1
        i += 1
    return None


def match_capture(ms, s, n):
    n -= 1
    if n < 0 or n >= ms.level or ms.capture[n][1] == CAP_UNFINISHED:
        lua_error(ms.rt, ms.node, "invalid capture index %%%d" % (n + 1))
    start, length = ms.capture[n]
    cap = ms.src[start : start + length]
    if ms.src[s : s + len(cap)] == cap:
        return s + len(cap)
    return None


def get_capture(ms, i, s, e):
    if i >= ms.level:
        if i == 0:
            return ms.src[s:e]
        lua_error(ms.rt, ms.node, "invalid capture index %%%d" % (i + 1))
    start, length = ms.capture[i]
    if length == CAP_POSITION:
        return start + 1
    return ms.src[start : start + length]


def captures(ms, s, e, whole):
    n = ms.level if (ms.level or not whole) else 1
    if ms.level == 0 and whole:
        return [ms.src[s:e]]
    return [get_capture(ms, i, s, e) for i in range(n)]


def str_find_aux(rt, node, args, find):
    fname = "find" if find else "match"
    s = arg(rt, node, args, 0, fname, "string")
    p = arg(rt, node, args, 1, fname, "string")
    init = to_integer(opt(args, 2, 1))
    ls = len(s)
    if init < 0:
        init = max(ls + init + 1, 1)
    elif init == 0:
        init = 1
    if init > ls + 1:
        return [None]
    plain = truthy(opt(args, 3, None))
    if find and (plain or not any(c in SPECIALS for c in p)):
        k = s.find(p, init - 1)
        return [k + 1, k + len(p)] if k >= 0 else [None]
    anchor = p.startswith("^")
    pi = 1 if anchor else 0
    si = init - 1
    while True:
        ms = MatchState(rt, node, s, p)
        e = do_match(ms, si, pi)
        if e is not None:
            if find:
                return [si + 1, e] + (captures(ms, si, e, False) if ms.level else [])
            return captures(ms, si, e, True)
        si += 1
        if anchor or si > ls:
            return [None]


@lib(STRING, "find", "find")
def string_find(rt, node, args):
    return str_find_aux(rt, node, args, True)


@lib(STRING, "match", "match")
def string_match(rt, node, args):
    return str_find_aux(rt, node, args, False)


@lib(STRING, "gmatch", "gmatch")
def string_gmatch(rt, node, args):
    s = arg(rt, node, args, 0, "gmatch", "string")
    p = arg(rt, node, args, 1, "gmatch", "string")
    state = {"pos": 0, "last": None}

    def step(rt, node, _args):
        while state["pos"] <= len(s):
            ms = MatchState(rt, node, s, p)
            src = state["pos"]
            e = do_match(ms, src, 0)
            if e is not None and e != state["last"]:
                state["pos"] = state["last"] = e
                if e == src:
                    state["pos"] = e
                return captures(ms, src, e, True)
            state["pos"] += 1
        return [None]

    return [Builtin("gmatch_iterator", step)]


@lib(STRING, "gsub", "gsub")
def string_gsub(rt, node, args):
    src = arg(rt, node, args, 0, "gsub", "string")
    p = arg(rt, node, args, 1, "gsub", "string")
    repl = opt(args, 2, None)
    if not (isinstance(repl, (str, Table)) or is_number(repl) or is_function(repl)):
        arg_error(rt, node, 2, "gsub", "string/function/table expected, got " + got(repl, 2, args))
    max_n = to_integer(opt(args, 3, len(src) + 1))
    anchor = p.startswith("^")
    pi = 1 if anchor else 0
    s = 0
    n = 0
    out = []
    while n < max_n:
        ms = MatchState(rt, node, src, p)
        e = do_match(ms, s, pi)
        if e is not None:
            n += 1
            whole = src[s:e]
            caps = captures(ms, s, e, True)
            if isinstance(repl, str) or is_number(repl):
                r = tostring(rt, node, repl)
                buf = []
                i = 0
                while i < len(r):
                    if r[i] == "%":
                        i += 1
                        d = r[i] if i < len(r) else ""
                        if d == "%":
                            buf.append("%")
                        elif d.isdigit():
                            v = whole if d == "0" else get_capture(ms, int(d) - 1, s, e)
                            buf.append(tostring(rt, node, v))
                        else:
                            lua_error(rt, node, "invalid use of '%' in replacement string")
                    else:
                        buf.append(r[i])
                    i += 1
                out.append("".join(buf))
            else:
                if isinstance(repl, Table):
                    v = index(rt, node, repl, caps[0])
                else:
                    v = call_one(rt, node, repl, caps)
                if v is None or v is False:
                    out.append(whole)
                elif isinstance(v, str) or is_number(v):
                    out.append(tostring(rt, node, v))
                else:
                    lua_error(rt, node, "invalid replacement value (a %s)" % type_name(v))
        if e is not None and e > s:
            s = e
        elif s < len(src):
            out.append(src[s])
            s += 1
        else:
            break
        if anchor:
            break
    out.append(src[s:])
    return ["".join(out), n]


# ----------------------------------------------------------------------
# Running a script
# ----------------------------------------------------------------------


def run(source, path="input", mode="python", args=()):
    """Run a script; `arg` is its path (arg[0]) and arguments."""
    a = Table()
    a.set(0, path)
    for i, v in enumerate(args, 1):
        a.set(i, v)
    G.set("arg", a)
    lang.load(source, path).run(mode=mode)


def show(values):
    """What a REPL entry returned (`return ...`), as print shows it (a
    table's __tostring aside: no program runs to call it)."""
    values = as_list(values)
    if values:
        write("\t".join("table: 0x%014x" % (id(v) & 0xFFFFFFFFFFFFFF) if isinstance(v, Table) else tostring(None, None, v) for v in values) + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # (no script: interactive, `return` showing values)
        lang.repl(show=show)
        sys.exit(0)
    path = sys.argv[1]
    with open(path) as f:
        src = f.read()
    try:
        run(src, path, args=sys.argv[2:])
    except (zrun.LoadError, zrun.Error) as e:
        builtins.print(e, file=sys.stderr)
        sys.exit(1)
