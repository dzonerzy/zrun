"""print(), sys.stdout.write() and sys.stderr.write() natively (zr_print,
zr_write): what Python writes, str() and repr() of every kind of value
included, to sys.stdout and sys.stderr as they are when it's written."""

import sys
from dataclasses import dataclass

import pytest
import zrun
from test_strict import tiny


@dataclass
class Point:
    x: int
    y: object


def _program(compute, strict=True):
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=strict)
    lang.function("FuncDef")

    @lang.exec("Return")
    def return_(node, rt):
        raise rt.Return(rt.eval(node.value))

    @lang.eval("BinOp")
    def binop(node, rt):
        return compute(rt.eval(node.left), rt.eval(node.right))

    return lang.load("fn f(a, b) { return a + b; }", "f.tiny")


def both(capsys, compute, *args):
    """What the compiled code and the reference mode write and return"""
    p = _program(compute)
    got = []
    for mode in ("compiled", "python"):
        try:
            r = ("ok", p.call("f", *args, mode=mode))
        except zrun.Error as e:
            r = ("error", str(e).split("\n")[0])
        out = capsys.readouterr()
        got.append((r, out.out, out.err))
    assert got[0] == got[1]
    return got[0]


def values(a, b):
    cyclic = [a]
    cyclic.append(cyclic)
    d = {a: b, "k": [b, (a,)]}
    d["self"] = d
    print(a, b, [a, b], (a,), (), (a, b), {a: b}, {a, b}, set(), d, cyclic)
    print("it's", 'say "x"', "both ' \"", "tab\tnew\nline\\", "\x01\x7f", sep="|")
    print(["it's", 'say "x"', "both ' \"", "tab\tnew\nline\\", "\x01\x7f", ""])
    print(None, True, False, 1.5, -0.0, 1e16, 1e-5, 0.1 + 0.2, 2**70, -(2**64), float("inf"), float("nan"))
    print([None, True, 1.5, 1e16, 2**70, [], {}, ()], end="")
    print(b"a'b\x00\xff\n", [b"", b'"', b"'\""])
    print(Point(a, [b, "s"]), [Point(1, None)])
    print(ValueError("bad"), [ValueError("bad", 2), KeyError("k")], KeyError("k"), ValueError())
    return 0


def test_str_and_repr_of_values(capsys):
    r, out, err = both(capsys, values, 3, "x")
    assert r == ("ok", 0) and err == ""
    assert "[3, [...]]" in out and "'self': {...}" in out and "Point(x=3, y=['x', 's'])" in out


def separators(a, b):
    print(a, b, sep=b, end=b)
    print(a, sep=None, end=None)
    print()
    print(*[a, b, a], sep="")
    items = [a] * a
    print(b, *items, b)
    print(a, file=sys.stderr)
    print(b, file=sys.stdout, flush=True)
    return sys.stdout.write(b * 2) + sys.stderr.write("é!")


def test_separators_files_and_write(capsys):
    r, out, err = both(capsys, separators, 2, "--")
    assert r == ("ok", 6)
    assert out == "2------2\n\n2--2\n-- 2 2 --\n--\n----"
    assert err == "2\né!"


def write_text(text):
    sys.stdout.write(text)


def branches(a, b):
    for v in [a, b, 1.5, a]:
        if type(v) is float:
            write_text(f"{v:.3g}")
        elif v == a:
            sys.stdout.write(str(v))
        else:
            print(v, end=";")
    return 0


def test_writes_in_branches(capsys):
    r, out, err = both(capsys, branches, 2, "--")
    assert r == ("ok", 0) and out == "2--;1.52"


def bad_sep(a, b):
    print(a, sep=a)
    return 0


def bad_write(a, b):
    return sys.stdout.write(a)


@pytest.mark.parametrize("compute", [bad_sep, bad_write])
def test_errors(capsys, compute):
    r, out, err = both(capsys, compute, 1, 2)
    assert r[0] == "error"


def test_strict_tiny_prints(capsys):
    # (tiny's print(*args) a host function: compiled, *args and all;
    # test_strict's tiny is strict)
    src = open(tiny.__file__.replace("tiny.py", "fib.tiny")).read()
    p = tiny.lang.load(src, "fib.tiny")
    p.run(mode="compiled")
    assert capsys.readouterr().out == "".join(f"{n}\n" for n in (0, 1, 1, 2, 3, 5, 8, 13, 21, 34))
