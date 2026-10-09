"""Functions a semantic makes as values (lambdas, nested defs), `nonlocal`
and `global`: compiled natively, the same as the reference mode."""

from dataclasses import dataclass

import pytest
import zrun
from test_intrinsics import _program, same_in_both


def lambdas(a, b):
    add = lambda x: x + a  # noqa: E731
    twice = lambda f, x: f(f(x))  # noqa: E731
    return (add(b), twice(add, b), (lambda: a * b)())


def test_lambda():
    same_in_both(_program(lambdas), [(1, 2), (5, -3), ("a", "b")])
    assert _program(lambdas, strict=True).call("f", 1, 2) == lambdas(1, 2)


def counter(a, b):
    n = a

    def bump(k):
        nonlocal n
        n += k
        return n

    first = bump(b)
    inc = bump
    inc(1)
    return (first, n, inc(b))


def test_nonlocal():
    same_in_both(_program(counter), [(0, 1), (10, -4)])
    assert _program(counter, strict=True).call("f", 0, 1) == counter(0, 1)


def make_adder(n):
    def add(x):
        return x + n

    return add


def late(a, b):
    # (each lambda reads i as it is when it's called: the last)
    fs = [lambda: i * a for i in range(b)]
    adders = [make_adder(k) for k in range(3)]
    return ([f() for f in fs], [g(a) for g in adders])


def test_late_binding_and_helpers_returning_closures():
    same_in_both(_program(late), [(2, 3), (1, 1), (3, 0)])
    assert _program(late, strict=True).call("f", 2, 3) == late(2, 3)


def nested(a, b):
    def outer(x):
        def inner(y):
            return x * 10 + y + a

        return inner

    g = outer(b)
    h = outer(b + 1)
    return (g(1), h(2), outer(a)(b))


def test_nested_closures():
    same_in_both(_program(nested), [(1, 2), (0, 0), (-5, 7)])
    assert _program(nested, strict=True).call("f", 1, 2) == nested(1, 2)


def recursive(a, b):
    def fact(n):
        return 1 if n <= 1 else n * fact(n - 1)

    f = fact
    return (f(a), [fact(k) for k in range(b)])


def test_recursive_closure():
    same_in_both(_program(recursive), [(5, 4), (1, 0)])


def wrong_args(a, b):
    f = lambda x, y: x + y  # noqa: E731
    if a == 0:
        return f(b)
    return f(a, b, 3)


def test_wrong_argument_counts_are_pythons_errors():
    same_in_both(_program(wrong_args), [(0, 1), (1, 2)])


@dataclass
class Holder:
    name: str
    fn: object


def iterator(s):
    state = {"pos": 0}

    def step():
        if state["pos"] >= len(s):
            return None
        c = s[state["pos"]]
        state["pos"] += 1
        return c

    return Holder("it", step)


def kept(a, b):
    # (closures kept in a record and a dict, called later: an iterator's
    # state in a dict it reads, as the Lua example's string.gmatch)
    h = iterator(a)
    table = {"next": h.fn}
    out = []
    while True:
        c = table["next"]()
        if c is None:
            break
        out.append(c * b)
    return out


def test_closures_kept_and_called_later():
    same_in_both(_program(kept), [("abc", 2), ("", 1)])
    assert _program(kept, strict=True).call("f", "abc", 2) == kept("abc", 2)


TOTAL = 0


def accumulate(a, b):
    global TOTAL
    TOTAL = a
    TOTAL += b
    return TOTAL


def test_global():
    global TOTAL
    p = _program(accumulate)
    for mode in ("compiled", "python"):
        TOTAL = 0
        assert p.call("f", 3, 4, mode=mode) == 7
        assert TOTAL == 7
    # (a module variable assigned: Python's, not native)
    with pytest.raises(zrun.CompileError):
        _program(accumulate, strict=True).call("f", 3, 4)


def closure_to_python(a, b):
    return lambda: a


def test_a_closure_given_to_python_is_an_error():
    p = _program(closure_to_python)
    with pytest.raises(TypeError, match="can't be given to Python"):
        p.call("f", 1, 2, mode="compiled")
