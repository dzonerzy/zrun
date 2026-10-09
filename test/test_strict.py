"""Language(strict=True): every semantic compiled and the code calling
nothing in Python. A semantic outside the compilable subset (or marked
native=False) is a CompileError when it's registered; code that would call
into Python (a Python function, a Python object as a value, isinstance() of
any class, a semantic run as Python...) a CompileError when the program
compiles, saying where in the semantic and why; what only shows when the
code runs (a host function in Python) a StrictError, saying where in the
program. Errors are still Python's exceptions; code compiled while the
program runs is allowed."""

import functools
import importlib.util
import os

import pytest
import zrun
from conftest import HERE

TINY = os.path.join(HERE, "..", "examples", "tiny", "tiny.py")


def _strict_tiny():
    # (tiny.py's language, made strict: its module loaded again with
    # zrun.Language(strict=True))
    spec = importlib.util.spec_from_file_location("tiny_strict", TINY)
    module = importlib.util.module_from_spec(spec)
    plain = zrun.Language
    zrun.Language = functools.partial(plain, strict=True)
    try:
        spec.loader.exec_module(module)
    finally:
        zrun.Language = plain
    return module


tiny = _strict_tiny()

FIB = """
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
let x = fib(20);
"""


def test_native_code_runs():
    p = tiny.lang.load(FIB, "fib.tiny")
    p.run(mode="compiled")
    assert p.call("fib", 20) == 6765
    assert p.map("fib", [10, 15, 20]) == [55, 610, 6765]
    # (the reference mode is Python's, strict or not)
    p.run(mode="python")


def test_a_host_function_in_python_is_an_error_where_it_runs():
    src = FIB + "print(x);\n"
    p = tiny.lang.load(src, "fib.tiny")
    with pytest.raises(zrun.StrictError) as e:
        p.run(mode="compiled")
    assert "fib.tiny:7:1:" in str(e.value) and "calling print()" in str(e.value)
    assert issubclass(zrun.StrictError, zrun.CompileError)


def _language(strict):
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=strict)
    lang.function("FuncDef")

    @lang.exec(["Let", "Assign"])
    def assign(node, rt):
        rt.store(node.name, rt.eval(node.value))

    @lang.exec("Return")
    def return_(node, rt):
        raise rt.Return(rt.eval(node.value))

    return lang


def test_a_python_call_refused_when_compiling():
    for strict in (True, False):
        lang = _language(strict)

        @lang.eval("BinOp")
        def binop(node, rt):
            # (a builtin compiled code calls in Python)
            return rt.eval(node.left) + rt.eval(node.right) + os.getpid() * 0

        p = lang.load("let x = 1 + 2;", "add.tiny")
        if strict:
            with pytest.raises(zrun.CompileError) as e:
                p.run(mode="compiled")
            assert "in binop(): strict: calls the Python function getpid()" in str(e.value)
            assert "test_strict.py:" in str(e.value)
        else:
            # (not strict: the call made in Python, as before)
            p.run(mode="compiled")


def test_a_semantic_outside_the_subset_refused_when_registered():
    lang = _language(True)
    with pytest.raises(zrun.CompileError):

        @lang.eval("BinOp")
        def binop(node, rt):
            with open(os.devnull):
                return 0

    with pytest.raises(zrun.CompileError) as e:

        @lang.eval("Neg", native=False)
        def neg(node, rt):
            return -rt.eval(node.operand)

    assert "native=False" in str(e.value)


def test_strict_is_part_of_the_definition(tmp_path):
    # (a module compiled by the language not strict isn't the strict one's)
    lax = importlib.util.module_from_spec(importlib.util.spec_from_file_location("tiny_lax", TINY))
    lax.__spec__.loader.exec_module(lax)
    path = str(tmp_path / "fib.zrc")
    lax.lang.compile(FIB, path)
    with pytest.raises(Exception, match="definition"):
        tiny.lang.load_compiled(path)


def produce(rt, node):
    # (a helper raising the jump: caught by the semantic calling it)
    raise rt.Return(rt.eval(node.left) * 10)


def test_a_jump_caught_by_its_value():
    # `except rt.Return as r` using r.args only: the jump's value, natively
    for strict in (True, False):
        lang = _language(strict)

        @lang.eval("BinOp")
        def binop(node, rt):
            try:
                produce(rt, node)
            except rt.Return as r:
                got = r.args[0]
            return got + rt.eval(node.right)

        p = lang.load("let x = 4 + 2;\nfn f(a, b) { return a + b; }", "jump.tiny")
        p.run(mode="compiled")
        assert p.call("f", 4, 2) == 42


def test_a_jump_caught_as_an_exception():
    # (its name used otherwise: Python's exception, as before; not strict)
    lang = _language(False)

    @lang.eval("BinOp")
    def binop(node, rt):
        try:
            produce(rt, node)
        except rt.Return as r:
            e = r
        return e.args[0] + rt.eval(node.right)

    p = lang.load("fn f(a, b) { return a + b; }", "jump.tiny")
    p.run(mode="compiled")
    assert p.call("f", 4, 2) == 42


def test_type_of_a_known_value():
    # `type(v) is float` with v known: decided while compiling
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        v = 2.5
        if type(v) is float and type(v) is not int:
            return rt.eval(node.left) + rt.eval(node.right)
        return 0

    p = lang.load("let x = 40 + 2;\nfn f(a, b) { return a + b; }", "type.tiny")
    p.run(mode="compiled")
    assert p.call("f", 40, 2) == 42
    assert p.call("f", 40, 2, mode="python") == 42
