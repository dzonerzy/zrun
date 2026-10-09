"""Errors natively: exceptions of Python's builtin classes and zrun's
(rt.Throw, zrun.Error) raised, caught, matched (CPython's hierarchy), their
args, value, message, str; the same as the reference mode, natively in
strict mode."""

import zrun
from test_intrinsics import _program, same_in_both
from test_strict import tiny


def catching(a, b):
    out = []
    for i in range(6):
        try:
            if i == 0:
                raise ValueError("bad", a)
            if i == 1:
                out.append(a // b)
            if i == 2:
                out.append([1][a])
            if i == 3:
                raise KeyError(a)
            if i == 4:
                raise LookupError
            out.append(int("x"))
        except (KeyError, IndexError) as e:
            out.append(("lookup", type(e).__name__, str(e), e.args))
        except ArithmeticError as e:
            out.append(("arith", str(e)))
        except Exception as e:
            out.append(("other", type(e).__name__, str(e), len(e.args)))
    return out


def test_catching():
    same_in_both(_program(catching), [(5, 0), (0, 1), ("k", 2)])
    assert _program(catching, strict=True).call("f", 5, 0) == catching(5, 0)


def uncaught(a, b):
    if a == 0:
        raise ValueError("no good")
    if a == 1:
        raise KeyError(b)
    if a == 2:
        raise ZeroDivisionError
    try:
        raise TypeError("inner")
    except TypeError as e:
        raise RuntimeError("outer") from e


def test_uncaught_errors_worded_as_the_reference_mode():
    same_in_both(_program(uncaught), [(0, 0), (1, "k"), (2, 0), (3, 0)])


def reraising(a, b):
    try:
        try:
            raise IndexError("deep")
        except IndexError:
            raise
    except LookupError as e:
        return str(e)


def test_bare_raise_again():
    same_in_both(_program(reraising), [(0, 0)])
    assert _program(reraising, strict=True).call("f", 0, 0) == "deep"


def throwing(node_value, rt_message):
    return 0


def _throw_language(strict):
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=strict)
    lang.function("FuncDef")

    @lang.exec("Return")
    def return_(node, rt):
        raise rt.Return(rt.eval(node.value))

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        try:
            if b == 0:
                raise rt.Throw(a, "thrown " + str(a))
            return a // b
        except rt.Throw as e:
            if a < 0:
                raise
            return [e.value, e.message, len(e.args)]

    return lang.load("fn f(a, b) { return a + b; }", "t.tiny")


def test_throw_caught_and_uncaught():
    for strict in (False, True):
        p = _throw_language(strict)
        for mode in ("compiled", "python"):
            assert p.call("f", 7, 0, mode=mode) == [7, "thrown 7", 2]
            assert p.call("f", 7, 2, mode=mode) == 3
            try:
                p.call("f", -7, 0, mode=mode)
                raise AssertionError("no error")
            except zrun.Error as e:
                assert "thrown -7" in str(e)


def native_errors_in_python(a, b):
    return ValueError("x", a)


def test_an_exception_value_given_to_python():
    e = _program(native_errors_in_python).call("f", 1, 0, mode="compiled")
    assert type(e) is ValueError and e.args == ("x", 1)
