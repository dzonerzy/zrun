"""Builtins and math functions compiled code runs natively (zr_builtin,
zr_int_base, zr_min_max, zr_math, zr_utf8_len): the same results and the
same errors as the reference mode, for values of every kind (what they
don't do natively, Python does: the same again)."""

import math

import pytest
import zrun
from test_strict import tiny


def _program(compute):
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.function("FuncDef")

    @lang.exec("Return")
    def return_(node, rt):
        raise rt.Return(rt.eval(node.value))

    @lang.eval("BinOp")
    def binop(node, rt):
        return compute(rt.eval(node.left), rt.eval(node.right))

    return lang.load("fn f(a, b) { return a + b; }", "f.tiny")


def outcome(p, mode, a, b):
    try:
        r = p.call("f", a, b, mode=mode)
        # (NaN: equal to itself here)
        return ("ok", repr(r))
    except zrun.Error as e:
        return ("error", str(e).split("\n")[0])


def same_in_both(p, cases):
    for a, b in cases:
        assert outcome(p, "compiled", a, b) == outcome(p, "python", a, b), (a, b)


def int_base(a, b):
    return int(a, b)


def test_int_with_a_base():
    p = _program(int_base)
    same_in_both(p, [
        ("ff", 16), ("FF", 16), ("0xff", 16), ("0x_ff", 16), (" -1_0 ", 16), ("z", 36), ("10", 2), ("12", 2),
        ("0o17", 0), ("0b101", 0), ("0x1F", 0), ("017", 0), ("00", 0), ("", 10), ("_1", 10), ("1__0", 10), ("1_", 10),
        ("7fffffffffffffff", 16), ("8000000000000000", 16), ("-8000000000000000", 16), ("ffffffffffffffffff", 16),
        (12, 16), (1.5, 16), ("ff", 1), ("ff", 37),
    ])


def one_argument(a, b):
    if b == 0:
        return chr(a)
    if b == 1:
        return ord(a)
    if b == 2:
        return math.floor(a)
    if b == 3:
        return math.ceil(a)
    if b == 4:
        return list(a)
    return len(a.encode("utf-8"))


def test_one_argument_builtins():
    p = _program(one_argument)
    cases = [(x, 0) for x in (0, 65, 0x7FF, 0xD800, 0x10FFFF, 0x110000, -1, 2**40, True)]
    cases += [(x, 1) for x in ("a", "é", "€", "😀", "", "ab")]
    cases += [(x, b) for b in (2, 3) for x in (2.5, -2.5, 3, True, 1e300, float("inf"), float("nan"), -0.0)]
    cases += [(x, 4) for x in ([1, 2], (1, "a"), [])]
    cases += [(x, 5) for x in ("abc", "é€😀", "")]
    same_in_both(p, cases)


def min_max(a, b):
    return (min(a, b), max(a, b), min(a, b, 0), max(b, a, 0))


def test_min_max():
    p = _program(min_max)
    nan = float("nan")
    same_in_both(p, [(1, 2), (2, 1), (1, 1.0), (1.0, 1), (-0.0, 0.0), (nan, 1), (1, nan), (2**62, 2.0**62), (True, 0), ("a", "b"), ("b", "a")])
    with pytest.raises(zrun.Error):
        p.call("f", "a", 1, mode="compiled")


def identity(a, b):
    t = [a]
    u = [a]
    # (what Python promises of id(): the same object, the same id; two
    # alive at once, two ids)
    return (id(t) == id(t), id(t) != id(u), id(a) == id(a), id(None) == id(None), id(True) != id(False), id(t) != id(a), id(a) > 0)


def test_id():
    p = _program(identity)
    for a in (1, -1, 2.5, "s", None, True, 2**62, [1], (2, 3)):
        for mode in ("compiled", "python"):
            assert outcome(p, mode, a, 0) == ("ok", repr((True,) * 7)), (a, mode)


def math_one(a, b):
    # (each called by name: compiled to zr_math)
    if b == 0:
        return math.sqrt(a)
    if b == 1:
        return math.fabs(a)
    if b == 2:
        return math.exp(a)
    if b == 3:
        return math.log(a)
    if b == 4:
        return math.sin(a)
    if b == 5:
        return math.atan(a)
    if b == 6:
        return math.log1p(a)
    if b == 7:
        return math.degrees(a)
    return math.radians(a)


def math_two(a, b):
    return (math.copysign(a, b), math.fmod(a, b), math.atan2(a, b), math.pow(a, b))


def test_math_functions():
    p = _program(math_one)
    values = (0.5, 2, -1.0, 0.0, -0.0, 1e308, 710, float("inf"), float("-inf"), float("nan"), True, 2**60)
    same_in_both(p, [(x, i) for i in range(9) for x in values])
    p = _program(math_two)
    pairs = [(1.5, -2.0), (-3, 2), (5.0, 0.0), (0.0, -1.0), (2.0, 0.5), (-8.0, 1 / 3), (float("inf"), 2.0), (2.0, float("inf")), (float("nan"), 0.0), (10.0, 1000.0)]
    same_in_both(p, pairs)
