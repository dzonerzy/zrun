"""Values crossing between compiled code and Python (host functions,
semantics run as Python) are the same objects, as in the reference mode:
whatever Python does to a list, a dict or a record the semantics made is
seen by them, and the other way round. Every program runs in every mode."""

import builtins
from collections import namedtuple
from dataclasses import dataclass
from enum import IntEnum

import pytest
import zrun
from conftest import tiny
from test_modes import same_in_every_mode
from zrules import Rules, scopes

BUILTINS = (
    "print",
    "show",
    "mutate_list",
    "mutate_dict",
    "mutate_point",
    "keep",
    "kept",
    "ops",
    "py_side",
    "reset",
    "grow_shared",
    "shared_dict",
    "fresh",
    "point",
    "keys",
    "unlive",
    "live",
    "same",
    "back",
    "bigmath",
    "slices",
    "genexp",
    "trying",
    "strs",
    "wide",
    "tables",
    "untables",
    "defaults",
    "fields",
    "matches",
    "isinst",
    "folds",
)


@dataclass
class Point:
    x: int
    y: int


class Box:
    """A plain class (compared and hashed by identity)."""

    def __init__(self, v):
        self.v = v


KEPT = []
SHARED = []
SHARED_DICT = {}
Pt = namedtuple("Pt", "x y")


class Color(IntEnum):
    RED = 1


class Holder:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def get(self, k):
        return self.v + k


class Slot2:
    __slots__ = ("x", "y")

    def __init__(self, x):
        self.x = x


class Slot3(Slot2):
    __slots__ = ("z",)

    def __init__(self, x, y):
        self.x = x
        self.y = y


@dataclass(frozen=True)
class Frozen:
    v: int


def scaled(x, factor=2, offset=0):
    return x * factor + offset


FOLD_LOG = []


def parsed(text):
    """Pure, and big (out of line): given a constant, run while compiling."""
    t = text.strip().lower()
    neg = t.startswith("-")
    if neg:
        t = t[1:]
    base = 16 if t.startswith("0x") else 10
    digits = t[2:] if base == 16 else t
    n = 0
    for ch in digits:
        if ch == "_":
            continue
        d = "0123456789abcdef".find(ch)
        if d < 0 or d >= base:
            raise ValueError("bad digit " + repr(ch) + " in " + repr(text))
        n = n * base + d
    return -n if neg else n


def logged(text):
    """Big too, but it changes a list of the module's: run each time."""
    FOLD_LOG.append(text)
    t = text.strip().lower()
    neg = t.startswith("-")
    if neg:
        t = t[1:]
    n = 0
    for ch in t:
        n = n * 10 + "0123456789".find(ch)
    return -n if neg else n


def fresh_list(text):
    """Big, pure, its result a new list each call: run each time."""
    t = text.strip().lower()
    out = []
    for ch in t:
        if ch == " ":
            continue
        out.append(ch)
    if t.startswith("x"):
        out.append("x")
    if t.endswith("y"):
        out.append("y")
    return out


def returns_in_try(log):
    try:
        return "ret"
    finally:
        log.append("rf")


FROZEN_TABLE = {"a": 1, "b": 2}
OPS = {"+": 1, "..": 2, "//": 3, "": 4, "abcdefgh": 5}


class MyStr(str):
    pass


LIVE_TABLE = {"a": 1}

COUNTER = 0
HOLDER = Holder(1)
STACK = []


def make_lang():
    rules = Rules(
        tiny.PARSER,
        [
            scopes(
                scope=("Program", "FuncDef"),
                define=("Let > .name", "FuncDef > .params"),
                define_outer="FuncDef > .name",
                use="Name",
                hoist="FuncDef > .name",
                after="Let > .name",
                builtins=BUILTINS,
            )
        ],
    )
    lang = zrun.Language(tiny.PARSER, rules)
    lang.exec("While")(tiny.while_)
    lang.exec("If")(tiny.if_)
    lang.exec("Return")(tiny.return_)
    lang.exec("Break")(tiny.break_)
    lang.eval("BinOp")(tiny.binop)
    lang.eval("Neg")(tiny.neg)
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.function("FuncDef")

    @lang.eval("Call")
    def call(node, rt):
        name = node.name.text
        args = rt.eval(node.args)
        # Values the semantics make, given to Python and looked at again
        if name == "mutate_list":
            xs = [1, 2, 3, 4, 5]
            rt.call(rt.load(node.name), [xs])
            return len(xs) * 1000 + xs[0] * 100 + xs[-1]
        if name == "mutate_dict":
            d = {"a": 1, "b": 2}
            rt.call(rt.load(node.name), [d])
            return len(d) * 100 + d.get("c", 0) * 10 + d.get("a", 7)
        if name == "mutate_point":
            p = Point(1, 2)
            rt.call(rt.load(node.name), [p])
            return p.x * 10 + p.y
        # Values Python made, changed by the semantics: Python sees it
        if name == "grow_shared":
            xs = rt.call(rt.load(node.name), [])
            xs.append(7)
            xs[0] = xs[0] + 1
            again = rt.call(rt.load(node.name), [])
            return len(again) * 100 + again[0] + isinstance(xs, list)
        if name == "shared_dict":
            m = rt.call(rt.load(node.name), [])
            m["k"] = 5
            return rt.call(rt.load(node.name), [])["k"]
        if name == "fresh":
            xs = rt.call(rt.load(node.name), [])
            xs.append(4)
            return len(xs)
        if name == "point":
            p = rt.call(rt.load(node.name), [])
            return p.x * 10 + p[1]
        if name == "keys":
            # Python objects as keys: equal and hashed as Python does
            k = rt.call(rt.load(node.name), [])
            d = {}
            d[k[0]] = "red"
            d[k[1]] = "fs"
            d[k[3]] = "box"
            return d.get(1, "no") + d.get(k[2], "no") + d.get(k[3], "no") + d.get(Box(1), "no") + str(len(d))
        if name == "slices":
            xs = rt.call(rt.load(node.name), [])
            n = len(xs)
            return (xs[1 : n - 1], xs[:: n - 7], xs[n:], tuple(xs)[n - 7 :], "hello"[n - 4 :], "héllo"[n - 4 :], xs[n - 105 : 2], xs[n - 5 : n - 6])
        if name == "genexp":
            xs = rt.call(rt.load(node.name), [])
            seen = []
            # (any() stops at the first true one: 2)
            a = any(seen.append(x) or x > 1 for x in xs)
            b = all(x < 3 for x in xs)
            return (a, b, sum(x * 2 for x in xs), len(seen), all(x for x in []), any(x for x in ()), "".join(str(x) for x in xs))
        if name == "trying":
            out = []
            try:
                int("x")
            except ValueError as e:
                out.append("value:" + str(e))
            try:
                rt.call(rt.load(node.name), [])
            except (TypeError, zrun.Error):
                # (a host function's exception: a zrun.Error, as rt.call
                # raises it)
                out.append("key")
            else:
                out.append("no")
            finally:
                out.append("fin")
            try:
                out.append("body")
            except Exception:
                out.append("bad")
            else:
                out.append("else")
            for i in range(3):
                try:
                    if i == 1:
                        continue
                    if i == 2:
                        break
                    out.append("loop" + str(i))
                finally:
                    out.append("f" + str(i))
            for x in rt.call(rt.load(node.name), [1]):
                try:
                    if x == 2:
                        break
                finally:
                    out.append("g" + str(x))
            out.append(returns_in_try(out))
            try:
                try:
                    raise rt.Throw("v", "m")
                except rt.Throw as t:
                    out.append(t.value)
                    raise
            except rt.Throw:
                out.append("again")
            try:
                try:
                    out.append(1 // 0)
                finally:
                    out.append("inner")
            except ZeroDivisionError:
                out.append("zero")
            return out
        if name == "strs":
            # str operations on run-time strs, as Python does them
            words = rt.call(rt.load(node.name), [])
            out = []
            for w in words:
                out.append((w.lower(), w.upper(), w.strip(), w.isupper(), w.islower(), w.isdigit(), w.isspace(), w.startswith("A"), w.endswith("é"), w.find("b"), w.count(""), w[1:3], w[::-1], [c for c in w]))
            out.append("-".join(words))
            nums = []
            for t in (" -1_000 ", "+42", "007", "\x0b9\x0c"):
                nums.append(int(t))
            for bad in ("1__0", "  ", "x'y", "_1", "1_", "\x1c9"):
                try:
                    int(bad)
                except ValueError as e:
                    nums.append(str(e))
            out.append(nums)
            return out
        if name == "wide":
            # ints past 64 bits a semantic computes (native up to 128)
            vals = rt.call(rt.load(node.name), [])
            a = vals[0] * 4
            b = a - 1
            m = vals[1] & 0xFFFFFFFFFFFFFFFF
            d = -vals[1]
            big = a * a * a
            return (a, b, m, d, m - 0x10000000000000000, a // 3, a % 7, -a // 3, -a % 7, a >> 3, b & 0xFF, 1 << 100, big, big // a,
                    a > 2**63, a == 2**64, a == float(2**64), 2**53 + 1 == float(2**53), 2**53 + 1 > float(2**53), {a: "k"}.get(2**64),
                    str(a), int(a), abs(-a), float(a), -a, ~a, a == d * 2, isinstance(a, int), type(a) is int)
        if name == "tables":
            # a table only read (known when compiling) and one a host
            # function changes (read when the code runs)
            before = LIVE_TABLE["a"]
            rt.call(rt.load(node.name), [])
            return ("a" in FROZEN_TABLE, FROZEN_TABLE.get("b"), [k for k in FROZEN_TABLE], before, LIVE_TABLE["a"], "z" in LIVE_TABLE)
        if name == "matches":
            # run-time values against constant strs (`==`, `in` a constant
            # table or tuple): strs of every length, anything else
            out = []
            for w in rt.call(rt.load(node.name), []):
                r = (w == "+", w != "..", "//" == w, w == "", w == "abcdefgh", w in ("abcdefghi", "é", "abcdefg"), w not in ["", "x"])
                try:
                    r = r + (w in OPS, w not in OPS)
                except TypeError as e:
                    r = r + (str(e),)
                out.append(r)
            return out
        if name == "folds":
            # helpers given constants: a pure one's result known when
            # compiling; one changing the module's state, one making a new
            # list each call, run each time; one raising, raising when run
            out = [parsed(" 0x1F "), parsed("-1_000"), parsed("42")]
            FOLD_LOG.clear()
            out.append(logged("7") + logged("-8"))
            out.append(list(FOLD_LOG))
            a = fresh_list("x b y")
            a.append("!")
            out.append(a)
            out.append(fresh_list("x b y"))
            try:
                out.append(parsed("12z"))
            except ValueError as e:
                out.append(str(e))
            # (a str method's result known when compiling, not a constant:
            # bytes, kept for the code)
            out.append(len("héllo".encode("utf-8")))
            out.append("ab".encode())
            return out
        if name == "isinst":
            # isinstance() of values of every kind: Python's, natively made
            # ones (records of a class, of a subclass)
            vals = rt.call(rt.load(node.name), [])
            vals = vals + [Slot3(1, 2), Holder(3), 2**70 + len(vals), True, 7, 1.5, "s", None]
            out = []
            for v in vals:
                out.append("".join("1" if x else "0" for x in (
                    isinstance(v, int), isinstance(v, bool), isinstance(v, float), isinstance(v, str), isinstance(v, list),
                    isinstance(v, tuple), isinstance(v, dict), isinstance(v, Holder), isinstance(v, Slot2), isinstance(v, Slot3),
                    isinstance(v, (str, list)), isinstance(v, Point))))
            return out
        if name == "defaults":
            # helpers' defaults and keyword arguments
            n = rt.call(rt.load(node.name), [])
            return (scaled(n), scaled(n, 3), scaled(n, offset=1), scaled(x=n, factor=4, offset=1), scaled(n, offset=2, factor=0))
        if name == "fields":
            # record fields: a class's, a subclass's, unset, frozen
            n = rt.call(rt.load(node.name), [])
            b = Slot2(n)
            s = Slot3(n, n + 1)
            s.x = s.x + 10
            out = [b.x, s.x, s.y]
            try:
                out.append(b.y)
            except AttributeError as e:
                out.append(str(e))
            fz = Frozen(n)
            try:
                fz.v = 2
            except Exception as e:
                out.append(type(e).__name__)
            out.append(fz.v)
            return out
        if name == "bigmath":
            # A semantic's own ints are Python's (beyond 64 bits on the way)
            m = -1 & 0xFFFFFFFFFFFFFFFF
            back = m - 0x10000000000000000 if m >= 0x8000000000000000 else m
            lim = max(min(2**63 + 5, 2**63 - 1), -(2**63))
            return back * 100 + (lim - (2**63 - 1)) + len(str(2**70))
        if name == "same":
            # One object, the same each time Python sees it; Python keeping
            # it after compiled code let go, changing it, giving it back
            keep = rt.load(node.name)
            xs = [1]
            rt.call(keep, [xs])
            rt.call(keep, [xs])
            return rt.call(keep, [None])
        if name == "back":
            ys = rt.call(rt.load(node.name), [])
            ys.append(3)
            return len(ys) * 10 + ys[1]
        if name == "live":
            # module-level objects Python changes: read when the code runs
            before = COUNTER * 1000 + HOLDER.get(5) * 10 + (1 if STACK else 0)
            rt.call(rt.load(node.name), [])
            after = COUNTER * 1000 + HOLDER.get(5) * 10 + (1 if STACK else 0)
            return before * 10000 + after
        if name == "py_side":
            # a list a semantic run as Python changes
            xs = [10, 20]
            rt.exec(node)
            return len(xs)
        return rt.call(rt.load(node.name), args)

    @lang.host
    def print(*args):
        builtins.print(*args)

    @lang.host
    def mutate_list(xs):
        assert isinstance(xs, list)
        xs.append(6)
        xs.extend([7, 8])
        xs.insert(0, 0)
        xs.insert(-1, 99)
        xs.remove(99)
        assert xs.pop() == 8 and xs.pop(0) == 0
        xs[0] = 100
        xs[-1] = 70
        del xs[1]
        xs[1:3] = [30, 31, 32]
        del xs[::3]
        assert xs.index(31) >= 0 and xs.count(31) == 1 and 31 in xs and 1000 not in xs
        xs.reverse()
        xs.sort()
        xs.sort(key=lambda v: -v, reverse=True)
        assert xs == sorted(xs) and xs != [] and xs < xs + [1] and xs + [0] != xs
        assert [0] + xs == [0] + list(xs) and xs * 2 == list(xs) * 2 and 2 * xs == list(xs) * 2
        assert xs[1:] == list(xs)[1:] and xs[::-1] == list(xs)[::-1]
        assert repr(xs) == repr(list(xs)) and str(xs) == str(list(xs))
        assert list(iter(xs)) == list(xs) and list(reversed(xs)) == list(xs)[::-1] and bool(xs)
        c = xs.copy()
        c.append(1)
        assert len(c) == len(xs) + 1
        try:
            xs[100]
        except IndexError as e:
            assert str(e) == "list index out of range"
        try:
            xs["a"]
        except TypeError as e:
            assert str(e) == "list indices must be integers or slices, not str"
        try:
            hash(xs)
        except TypeError:
            pass
        else:
            raise AssertionError("a list is unhashable")

    @lang.host
    def mutate_dict(d):
        assert isinstance(d, dict)
        d["c"] = 3
        del d["b"]
        assert d.pop("zz", 5) == 5 and d.setdefault("a", 9) == 1
        d.update({"e": 4}, f=6)
        assert d.pop("e") == 4 and d.pop("f") == 6
        assert sorted(d.keys()) == ["a", "c"] and sorted(d.values()) == [1, 3]
        assert sorted(d.items()) == [("a", 1), ("c", 3)] and "a" in d and "b" not in d
        assert dict(d) == {"a": 1, "c": 3} and d == {"a": 1, "c": 3} and repr(d) == repr(dict(d))
        assert list(d) == list(dict(d)) and len(d) == 2 and d.get("q") is None
        try:
            d["missing"]
        except KeyError as e:
            assert e.args == ("missing",)
        try:
            d[[1]] = 2
        except TypeError as e:
            assert str(e) == "unhashable type: 'list'"
        c = d.copy()
        c.clear()
        assert len(c) == 0 and len(d) == 2

    @lang.host
    def mutate_point(p):
        assert isinstance(p, Point) and p.__class__ is Point
        assert repr(p) == "Point(x=1, y=2)" and p == Point(1, 2) or p == p
        p.x = 5
        p.y = p.y + 1
        try:
            p.z = 1
        except AttributeError:
            pass

    @lang.host
    def reset():
        SHARED[:] = [1]
        SHARED_DICT.clear()
        return 0

    @lang.host
    def grow_shared():
        return SHARED

    @lang.host
    def shared_dict():
        return SHARED_DICT

    @lang.host
    def same(xs):
        if xs is not None:
            KEPT.append((xs, id(xs)))
            return 0
        (a, ida), (b, idb) = KEPT[-2:]
        assert a is b and ida == idb
        return 1

    @lang.host
    def back():
        xs = KEPT[-1][0]
        del KEPT[:]
        xs.append(2)
        return xs

    @lang.host
    def unlive():
        global COUNTER
        COUNTER = 5
        HOLDER.v = 3
        STACK.clear()
        return 0

    @lang.host
    def live():
        global COUNTER
        COUNTER = 1
        HOLDER.v = 11
        STACK.append(1)
        return 0

    @lang.host
    def keys():
        return (Color.RED, frozenset({1}), frozenset({1}), Box(2))

    @lang.host
    def trying(*args):
        if args:
            return [1, 2, 3]
        raise KeyError("k")

    @lang.host
    def defaults():
        return 3

    @lang.host
    def folds():
        return 0

    @lang.host
    def isinst():
        return [1, True, 2**70, 2**200, 2.0, "s", MyStr("x"), [1], (1,), {}, None, Color.RED, Point(1, 2), Holder(1), Slot3(1, 2), Box(1)]

    @lang.host
    def matches():
        return ["+", "..", "//", "", "abcdefgh", "abcdefghi", "abcdefg", "abcdefgx", "é", "/", "++", 1, None, True, 2.0, [1], MyStr("+")]

    @lang.host
    def fields():
        return 5

    @lang.host
    def untables():
        LIVE_TABLE.clear()
        LIVE_TABLE["a"] = 1
        return 0

    @lang.host
    def tables():
        LIVE_TABLE["a"] = LIVE_TABLE["a"] + 1
        LIVE_TABLE["z"] = 0
        return 0

    @lang.host
    def wide():
        return [2**62, -(2**63)]

    @lang.host
    def strs():
        return ["Abc", "ÉcolÉ é", " \t\x1c ", "ABC1", "12", "", "a'b\"c"]

    @lang.host
    def genexp():
        return [1, 2, 3]

    @lang.host
    def slices():
        return [1, 2, 3, 4, 5]

    @lang.host
    def fresh():
        return [1, 2, 3]

    @lang.host
    def point():
        return Pt(3, 4)

    @lang.host
    def keep(v):
        KEPT.append(v)
        return 0

    @lang.host
    def kept():
        return len(KEPT)

    @lang.host
    def show(v):
        builtins.print(repr(v))
        return 0

    @lang.exec("Program")
    def program(node, rt):
        for c in node.children:
            rt.exec(c)

    return lang


lang = make_lang()

PROGRAMS = {
    "list": "print(mutate_list());\n",
    "dict": "print(mutate_dict());\n",
    "record": "print(mutate_point());\n",
    "from_python": "print(reset(), grow_shared(), shared_dict(), fresh(), point());\n",
    "keys": "print(keys());\n",
    "live": "print(unlive(), live());\n",
    "identity": "print(same(), back());\n",
    "bigmath": "print(bigmath());\n",
    "slices": "print(slices());\n",
    "genexp": "print(genexp());\n",
    "trying": "print(trying());\n",
    "strs": "print(strs());\n",
    "wide": "print(wide());\n",
    "tables": "print(untables(), tables());\n",
    "defaults": "print(defaults());\n",
    "fields": "print(fields());\n",
    "matches": "print(matches());\n",
    "isinst": "print(isinst());\n",
    "folds": "print(folds());\n",
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_shared_with_python(name, capsys):
    out, err = same_in_every_mode(lang, PROGRAMS[name], capsys)
    assert err is None, err
    # (len * 1000 + first * 100 + last, after the host's changes...; none:
    # the reference mode's output is what's expected)
    expected = {
        "list": "4531",
        "dict": "231",
        "record": "53",
        "from_python": "0 203 5 4 34",
        "keys": "redfsboxno3",
        "live": "0 50801161",
        "identity": "1 32",
        "bigmath": "-78",
        "slices": "([2, 3, 4], [5, 3, 1], [], (4, 5), 'ello', 'éllo', [1, 2], [1, 2, 3, 4])",
        "genexp": "(True, False, 12, 2, True, False, '123')",
        "defaults": "(6, 9, 7, 13, 2)",
        "folds": "[31, -1000, 42, -1, ['7', '-8'], ['x', 'b', 'y', 'x', 'y', '!'], ['x', 'b', 'y', 'x', 'y'], \"bad digit 'z' in '12z'\", 6, b'ab']",
        "trying": "[\"value:invalid literal for int() with base 10: 'x'\", 'key', 'fin', 'body', 'else', 'loop0', 'f0', 'f1', 'f2', 'g1', 'g2', 'rf', 'ret', 'v', 'again', 'inner', 'zero']",
    }.get(name)
    if expected is not None:
        assert out.strip() == expected
    if name == "from_python":
        assert SHARED == [2, 7] and SHARED_DICT == {"k": 5}
    # (compiled: none of them ran as Python)
    assert lang.python_semantics() == {}
