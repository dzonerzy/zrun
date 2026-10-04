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
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_shared_with_python(name, capsys):
    out, err = same_in_every_mode(lang, PROGRAMS[name], capsys)
    assert err is None, err
    # (len * 1000 + first * 100 + last, after the host's changes...)
    assert out.strip() == {"list": "4531", "dict": "231", "record": "53", "from_python": "0 203 5 4 34", "keys": "redfsboxno3", "live": "0 50801161"}[name]
    if name == "from_python":
        assert SHARED == [2, 7] and SHARED_DICT == {"k": 5}
    # (compiled: none of them ran as Python)
    assert lang.python_semantics() == {}
