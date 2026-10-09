"""@zrun.comptime: a function whose result depends only on its arguments,
its author says. Compiled code calling it with values known when compiling
calls it then, once for those values in the process: the result is a
constant of the code (any Python in the function: the subset or not), its
lists, dicts and tuples native and read-only (changing one isn't compiled:
the semantic runs as Python). With values known only at run time it's
called as any function is. The reference mode calls it as ever."""

import os

import pytest
import zrun
from conftest import HERE  # noqa: F401  (examples/ on the path)
from test_strict import tiny


@zrun.comptime
def crc_table(poly):
    # (outside the compilable subset: a nested def, a generator)
    def entry(i):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        return c

    return list(entry(i) for i in range(256))


CALLS = []


@zrun.comptime
def described(x):
    CALLS.append(x)
    return {"double": x * 2, "nested": [[x, x + 1], (x,)]}


@zrun.comptime
def fails(x):
    raise ValueError(f"no table for {x}")


def _language(strict, binop):
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=strict)
    lang.function("FuncDef")

    @lang.exec(["Let", "Assign"])
    def assign(node, rt):
        rt.store(node.name, rt.eval(node.value))

    @lang.eval("Call")
    def call(node, rt):
        return rt.call(rt.eval(node.name), rt.eval(node.args))

    @lang.exec("Return")
    def return_(node, rt):
        raise rt.Return(rt.eval(node.value))

    lang.eval("BinOp")(binop)
    return lang


PROGRAM = "fn f(a, b) { return a + b; }\nlet x = f(1, 2);\n"


def lookup(node, rt):
    table = crc_table(0xEDB88320)
    info = described(21)
    a = rt.eval(node.left)
    b = rt.eval(node.right)
    return table[(a + b) % 256] + info["double"] + info["nested"][0][1]


@pytest.mark.parametrize("strict", [True, False])
def test_a_table_computed_when_compiling(strict):
    p = _language(strict, lookup).load(PROGRAM, "t.tiny")
    p.run(mode="compiled")
    expected = crc_table(0xEDB88320)
    assert [p.call("f", a, 5) for a in (0, 1, 250)] == [expected[(a + 5) % 256] + 42 + 22 for a in (0, 1, 250)]
    p.run(mode="python")


def test_computed_once_for_its_arguments():
    _language(False, lookup).load(PROGRAM, "t.tiny").run(mode="compiled")
    n = len(CALLS)
    # (other languages compiling it, the code running: not called again)
    for strict in (True, False, False):
        p = _language(strict, lookup).load(PROGRAM, "t.tiny")
        p.run(mode="compiled")
        p.call("f", 1, 2)
    assert len(CALLS) == n
    # (the reference mode: called as ever)
    p.run(mode="python")
    assert len(CALLS) == n + 1


def runtime_argument(node, rt):
    # (its argument known only at run time: called as any function)
    return len(crc_table(rt.eval(node.left)))


def test_values_known_only_at_run_time():
    p = _language(False, runtime_argument).load(PROGRAM, "t.tiny")
    p.run(mode="compiled")
    assert p.call("f", 1, 2) == 256
    with pytest.raises(zrun.CompileError, match="strict"):
        _language(True, runtime_argument).load(PROGRAM, "t.tiny").run(mode="compiled")


def raising(node, rt):
    if rt.eval(node.left) > 100:
        return fails(1)
    return rt.eval(node.left) + rt.eval(node.right)


def test_raising_while_compiling():
    # (not strict: raised when the code runs it, as the reference mode does)
    p = _language(False, raising).load(PROGRAM, "t.tiny")
    p.run(mode="compiled")
    assert p.call("f", 1, 2) == 3
    with pytest.raises(zrun.Error, match="no table for 1"):
        p.call("f", 200, 1)
    with pytest.raises(zrun.CompileError, match="raised while compiling: ValueError: no table for 1"):
        _language(True, raising).load(PROGRAM, "t.tiny").run(mode="compiled")


def changing(node, rt):
    t = crc_table(0x82F63B78)
    t[0] = rt.eval(node.left)
    return t[0] + rt.eval(node.right)


def test_changing_a_result():
    # (not compiled: the semantic as Python, each call its own list)
    p = _language(False, changing).load(PROGRAM, "t.tiny")
    p.run(mode="compiled")
    assert p.call("f", 7, 1) == 8
    assert crc_table(0x82F63B78)[0] == 0
    with pytest.raises(zrun.CompileError, match="a zrun.comptime function's result"):
        _language(True, changing).load(PROGRAM, "t.tiny").run(mode="compiled")


def test_only_functions():
    with pytest.raises(TypeError):
        zrun.comptime(os.getpid)
    with pytest.raises(TypeError):
        zrun.comptime(3)
