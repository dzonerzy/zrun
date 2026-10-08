"""rt.tail_call(f, args, receiver=None): a return of what calling f returns,
the function's frame given up first, so chains of tail calls take no depth;
the same in every mode."""

import builtins

import pytest
import zrun
from conftest import tiny
from test_modes import same_in_every_mode


def language(return_semantic):
    """tiny, max_depth 100, with its `return` given."""
    lang = zrun.Language(tiny.PARSER, tiny.RULES, max_depth=100)
    lang.function("FuncDef")
    lang.exec("While")(tiny.while_)
    lang.exec("If")(tiny.if_)
    lang.exec("Break")(tiny.break_)
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.eval("BinOp")(tiny.binop)
    lang.eval("Neg")(tiny.neg)
    lang.eval("Call")(tiny.call)
    lang.exec("Return")(return_semantic)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


def is_call(v):
    # (tiny's numbers and strings are Python's, its expressions nodes)
    return v is not None and not isinstance(v, (int, str)) and v.kind == "Call"


def tail_return(node, rt):
    # `return f(x);`: a tail call
    v = node.value
    if is_call(v):
        rt.tail_call(rt.eval(v.name), rt.eval(v.args))
    raise rt.Return(rt.eval(v) if v is not None else None)


TAIL = language(tail_return)

COUNT = "fn count(n, acc) { if n == 0 { return acc; } return count(n - 1, acc + 1); }\nprint(count(5000, 0));\n"


def test_deep_tail_recursion(capsys):
    # (50 times max_depth: no depth taken)
    out, err = same_in_every_mode(TAIL, COUNT, capsys)
    assert (out, err) == ("5000\n", None)
    # (compiled: not a semantic run as Python, whose tail calls are calls)
    assert TAIL.python_semantics() == {}


def test_mutual_recursion(capsys):
    src = "fn even(n) { if n == 0 { return 1; } return odd(n - 1); }\nfn odd(n) { if n == 0 { return 0; } return even(n - 1); }\nprint(even(3001), odd(3001));\n"
    assert same_in_every_mode(TAIL, src, capsys) == ("0 1\n", None)


def test_not_a_tail_call_still_counts(capsys):
    # (`1 + f(x)` isn't returned as is: each call stays on the stack)
    src = "fn down(n) { if n == 0 { return 0; } return 1 + down(n - 1); }\nprint(down(500));\n"
    out, err = same_in_every_mode(TAIL, src, capsys)
    assert "call stack too deep" in err[0]


def test_host_function_and_top_level(capsys):
    # (a host function in tail position, and a tail call at the top level,
    # where there's no frame to leave: made as calls)
    src = "fn show(x) { return print(x); }\nprint(show(7));\n"
    assert same_in_every_mode(TAIL, src, capsys) == ("7\nNone\n", None)


def test_errors_keep_the_stack_of_the_calls_left(capsys):
    src = "fn boom(n) { if n == 0 { return 1 / 0; } return boom(n - 1); }\nfn outer() { let r = boom(3); return r; }\nouter();\n"
    out, err = same_in_every_mode(TAIL, src, capsys)
    assert err[0] == "division by zero"
    # (boom's tail calls one frame: called where outer called it)
    assert [name for name, _ in err[2]] == ["boom", "outer"]


def finally_return(node, rt):
    v = node.value
    try:
        if is_call(v):
            rt.tail_call(rt.eval(v.name), rt.eval(v.args))
        raise rt.Return(rt.eval(v) if v is not None else None)
    finally:
        builtins.print("left")


def caught_return(node, rt):
    v = node.value
    try:
        if is_call(v):
            rt.tail_call(rt.eval(v.name), rt.eval(v.args))
        raise rt.Return(rt.eval(v) if v is not None else None)
    except zrun.TailCall:
        raise rt.Return(-1)


def test_a_finally_runs_before_the_call(capsys):
    # (the function is left first: its finally, then the call)
    lang = language(finally_return)
    src = "fn g(x) { print(x); return x; }\nfn f() { return g(5); }\nprint(f());\n"
    assert same_in_every_mode(lang, src, capsys) == ("left\n5\nleft\n5\n", None)
    assert lang.python_semantics() == {}


def test_a_handler_of_it_takes_it(capsys):
    lang = language(caught_return)
    src = "fn g(x) { print(x); return x; }\nfn f() { return g(5); }\nprint(f());\n"
    # (the call isn't made: the handler returned instead)
    assert same_in_every_mode(lang, src, capsys) == ("-1\n", None)


@pytest.mark.parametrize("mode", ["python", "compiled"])
def test_tail_calls_from_python(mode):
    # (program.call of a function ending with tail calls: the result)
    p = TAIL.load("fn count(n, acc) { if n == 0 { return acc; } return count(n - 1, acc + 1); }\n", "c.tiny")
    p.run(mode=mode)
    assert p.call("count", 3000, 0) == 3000
