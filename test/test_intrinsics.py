"""Builtins and math functions compiled code runs natively (zr_builtin,
zr_int_base, zr_min_max, zr_math, zr_utf8_len): the same results and the
same errors as the reference mode, for values of every kind (what they
don't do natively, Python does: the same again)."""

import math
import re
import sys

import pytest
import zrun
from test_strict import tiny


def _program(compute, strict=False):
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=strict)
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


def formatted(a, b):
    return f"{a:{b}}"


def test_format_specs():
    # (the mini-language natively: ints, floats, strs; Python's errors)
    import random

    rng = random.Random(11)
    values = [0, 7, -42, 123456789, 2**62, 0.0, -0.0, 1.5, -2.25, 1e-7, 123456.789, 1e300, float("inf"), float("-inf"), float("nan"), "abc", "héllo", ""]
    specs = []
    for _ in range(400):
        s = ""
        if rng.random() < 0.3:
            s += rng.choice(["", "*", "é"]) + rng.choice("<>=^")
        s += rng.choice(["", "", "+", "-", " "])
        s += rng.choice(["", "", "#"])
        s += rng.choice(["", "", "0"])
        s += rng.choice(["", "", "8", "12", "1"])
        s += rng.choice(["", "", ",", "_"])
        s += rng.choice(["", "", ".0", ".3", ".14"])
        s += rng.choice(["", "d", "x", "X", "o", "b", "e", "E", "f", "F", "g", "G", "%", "s"])
        specs.append(s)
    same_in_both(_program(formatted), [(v, s) for v in values for s in specs[:60]] + [(rng.choice(values), s) for s in specs])


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


def kind_of(x):
    return type(x).__name__


def calling_a_value(a, b):
    # (a callable known only at run time, called by the semantic itself:
    # as Python calls it, its arguments as they are, its error its own)
    fs = (math.sqrt, kind_of, int)
    return fs[b](a)


def test_calling_a_value():
    p = _program(calling_a_value)
    same_in_both(p, [(4.0, 0), (-1.0, 0), (3, 1), (2.5, 1), ("x", 2), ("12", 2)])


def str_method(a, b):
    if b == 0:
        return (a.find("é"), a.find("b"), a.find(""), a.find("", 3), a.find("", 99), a.find("a", -3), a.find("a", 1, -1))
    if b == 1:
        return (a.rfind("a"), a.rfind("é", 0, 4), a.count("a"), a.count(""), a.count("a", 2), a.count("", 1, 3))
    if b == 2:
        return a.index("é")
    if b == 3:
        return a.rindex("a", 1)
    if b == 4:
        return (a.startswith("ab"), a.endswith(("x", "é")), a.startswith(("é", "a")))
    if b == 5:
        return (a.replace("a", "XY"), a.replace("", "-"), a.replace("é", ""))
    if b == 6:
        return (a.strip(), a.lstrip(), a.rstrip(), a.strip("aé "), a.rstrip(None))
    if b == 7:
        return (a.split(), a.split("a"), a.split("a", 1), a.split(None, 1), a.split(maxsplit=0) if False else a.split(None, 0))
    if b == 8:
        return a.split("")
    if b == 9:
        return (a.partition("a"), a.rpartition("a"), a.partition("é"), a.rpartition("zz"), a.partition(" "))
    if b == 10:
        return a.partition("")
    return "|".join([a, a.upper() if a.isascii() else a, "é"])


def test_str_methods():
    p = _program(str_method)
    texts = ("abcab", "aébécé", "  a b\tc\n", "　é a\xa0b ", "", "aaaa", "é", "  ")
    same_in_both(p, [(t, i) for i in range(12) for t in texts])
    # (partition's, natively: strict)
    strict = _program(str_method, strict=True)
    assert [strict.call("f", t, 9) for t in texts] == [str_method(t, 9) for t in texts]


def float_of(a, b):
    if b == 0:
        return float(a)
    return float(a).is_integer()


def test_float_of_a_str():
    p = _program(float_of)
    texts = ("1.5", " -2.25e3 ", "1_000.5", "1__0", "1_", "_1", ".5", "1.", ".", "inf", "-Infinity", "NaN", "nan ", "1e", "1e+5", "0x10", "　1.0\xa0", "", "3.141592653589793238462643383279", "1e400", "2.5e-324", "12abc")
    same_in_both(p, [(t, b) for t in texts for b in (0, 1)])


def float_text(a, b):
    if b == 0:
        return str(a)
    return f"{a}|{a!r}|{True}|{None}"


def test_str_of_floats():
    # (their shortest digits and Python's layout: random bit patterns, and
    # the edges of fixed and exponential notation)
    import random
    import struct

    rng = random.Random(1234)
    xs = [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(50000)]
    xs += [0.0, -0.0, 1.0, 0.1, 1e15, 1e16, 9999999999999998.0, 1e-4, 1e-5, 123456789.123, 5e-324, 1.7976931348623157e308, float("inf"), float("-inf"), float("nan"), 2.5, 100.0]
    xs = [x for x in xs if x == x] + [float("nan")]
    # (strict: done natively, no Python)
    p = _program(float_text, strict=True)
    p.run(mode="compiled")
    assert p.map("f", [(x, 0) for x in xs]) == [str(x) for x in xs]
    assert p.map("f", [(x, 1) for x in xs[:2000]]) == [f"{x}|{x!r}|True|None" for x in xs[:2000]]


FORMATS = (
    "%d", "%5d", "%-5d|", "%05d", "%+d", "% d", "%.3d", "%05.3d", "%x", "%#x", "%X", "%#o", "%o", "%i", "%u",
    "%f", "%.2f", "%10.3f", "%-10.1f|", "%+.1f", "%010.2f", "%e", "%.3E", "%g", "%.3g", "%#g", "%G", "%F",
    "%s", "%10s", "%-6s|", "%.2s", "%r", "%c", "%%d", "[%s]", "%*d", "%.*f", "%-*s|",
)


def percent(a, b):
    fmt = FORMATS[b]
    if "*" in fmt:
        return fmt % (7, a)
    return fmt % a


def test_percent_format():
    p = _program(percent)
    values = (0, 42, -42, 255, 2**40, -7, True, 3.7, -2.675, 2.675, 0.0, -0.0, 1e300, 1.5e-7, 123456.789, float("inf"), float("nan"), "ab", "é", "", "a'b", 65)
    cases = [(v, i) for i in range(len(FORMATS)) for v in values]
    same_in_both(p, cases)
    # (those Python formats, natively: strict, no Python; but a repr of a
    # str not ASCII, its escapes Unicode's)
    strict = _program(percent, strict=True)
    for v, i in cases:
        try:
            want = percent(v, i)
        except (TypeError, ValueError, OverflowError):
            continue
        if FORMATS[i] == "%r" and isinstance(v, str) and not v.isascii():
            continue
        # (a float's digits: the C library's, natively where it's the one
        # Python formats with (Linux); elsewhere Python's own)
        if sys.platform != "linux" and (isinstance(v, float) or re.search(r"%[^a-zA-Z%]*[eEfFgG]", FORMATS[i])):
            continue
        try:
            got = strict.call("f", v, i)
        except zrun.StrictError as e:
            raise AssertionError(f"{FORMATS[i]!r} % {v!r}: {e}") from None
        assert got == want, (v, FORMATS[i])


def power(a, b):
    return a**b


def test_powers():
    # (natively: ints exactly, floats as CPython's float_pow (the C
    # library's pow, its special cases and errors); a complex result or an
    # int beyond 128 bits, Python's)
    p = _program(power)
    vals = [0, 1, -1, 2, -2, 3, 10, 63, 64, 127, 200, -3, 0.0, -0.0, 0.5, -0.5, 1.5, -2.5, 2.0, 1e308, -1e308, 1e-308,
            float("inf"), float("-inf"), float("nan"), 2**62, 2**70, True]
    # (an int to a huge int power: Python's own result too big to make)
    same_in_both(p, [(a, b) for a in vals for b in vals if not (isinstance(a, int) and isinstance(b, int) and abs(b) > 300 and abs(a) > 1)])
    # (strict: no Python, the reference mode's results and errors; ints
    # handed over by call() are I64s, 2 ** 64 an overflow)
    strict = _program(power, strict=True)
    for a, b in [(2, 10), (2, 64), (-3, 3), (2.0, 0.5), (10, -2), (0.0, -1), (1e308, 2.0)]:
        try:
            got = outcome(strict, "compiled", a, b)
        except zrun.StrictError as e:
            raise AssertionError(f"{a!r} ** {b!r}: {e}") from None
        assert got == outcome(p, "python", a, b), (a, b)


def hex_of(a, b):
    return (float.hex(a), a.hex()) if b == 0 else int(a, b)


def test_float_hex_and_int_bases():
    import random
    import struct

    rng = random.Random(99)
    xs = [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(20000)]
    xs = [x for x in xs if x == x] + [0.0, -0.0, 1.0, 0.5, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, float("inf"), float("-inf")]
    p = _program(hex_of, strict=True)
    assert p.map("f", [(x, 0) for x in xs]) == [(x.hex(), x.hex()) for x in xs]
    # (the base known only at run time)
    same_in_both(_program(hex_of), [("ff", 16), ("z", 36), ("0b11", 0), ("12", 1), ("12", 37), ("12", -1), ("777", 8)])


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


def math_modf(a, b):
    return math.modf(a)


def test_math_modf():
    same_in_both(_program(math_modf), [(x, 0) for x in (2.5, -2.5, -3.0, 3.0, 0.0, -0.0, 1e300, float("inf"), float("-inf"), float("nan"), 7, True)])


def math_two(a, b):
    return (math.copysign(a, b), math.fmod(a, b), math.atan2(a, b), math.pow(a, b))


def test_math_functions():
    p = _program(math_one)
    values = (0.5, 2, -1.0, 0.0, -0.0, 1e308, 710, float("inf"), float("-inf"), float("nan"), True, 2**60)
    same_in_both(p, [(x, i) for i in range(9) for x in values])
    p = _program(math_two)
    pairs = [(1.5, -2.0), (-3, 2), (5.0, 0.0), (0.0, -1.0), (2.0, 0.5), (-8.0, 1 / 3), (float("inf"), 2.0), (2.0, float("inf")), (float("nan"), 0.0), (10.0, 1000.0)]
    same_in_both(p, pairs)
