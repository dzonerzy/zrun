"""Engines: a program loaded once and called many times (program.call), with a
context per call (rt.context), in every mode."""

import builtins

import pytest
import zrun
from conftest import tiny

MODES = ["python", "compiled"]


def _engine_lang():
    """tiny whose `context()` is the call's context (rt.context)."""
    from zrules import Rules, scopes

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
                builtins=("print", "context"),
            ),
        ],
    )
    lang = zrun.Language(tiny.PARSER, rules)
    lang.function("FuncDef")
    for kind, fn in (("Let", tiny.assign), ("Assign", tiny.assign), ("Return", tiny.return_), ("If", tiny.if_), ("While", tiny.while_)):
        lang.exec(kind)(fn)
    lang.eval("BinOp")(tiny.binop)

    @lang.eval("Call")
    def call(node, rt):
        if node.name.text == "context":
            return rt.context
        return rt.call(rt.eval(node.name), rt.eval(node.args))

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


lang = _engine_lang()

SOURCE = """
let count = 0;
fn bump(n) { count = count + n; return count; }
fn tag(x) { return context() + x; }
fn div(a, b) { return a / b; }
"""


@pytest.mark.parametrize("mode", MODES)
def test_calls_keep_the_programs_variables(mode):
    program = lang.load(SOURCE, "engine")
    assert [program.call("bump", n, mode=mode) for n in (1, 2, 3)] == [1, 3, 6]


@pytest.mark.parametrize("mode", MODES)
def test_context(mode):
    program = lang.load(SOURCE, "engine")
    assert program.call("tag", "!", mode=mode, context="file-a") == "file-a!"
    assert program.call("tag", 1, mode=mode, context=41) == 42


@pytest.mark.parametrize("mode", MODES)
def test_errors(mode):
    program = lang.load(SOURCE, "engine")
    with pytest.raises(zrun.Error) as e:
        program.call("div", 1, 0, mode=mode)
    assert e.value.diagnostic.message == "division by zero"
    with pytest.raises(KeyError):
        program.call("nothing", mode=mode)
    # (a call after an error: the program's variables as they were)
    assert program.call("bump", 5, mode=mode) == 5


DEEP = "fn down(n) { if n == 0 { return 0; } return 1 + down(n - 1); }\n"


@pytest.mark.parametrize("mode", MODES)
def test_deep_calls(mode):
    program = lang.load(DEEP, "deep")
    assert program.call("down", 900, mode=mode) == 900
    with pytest.raises(zrun.Error) as e:
        program.call("down", 5000, mode=mode)
    assert e.value.diagnostic.message == "call stack too deep (more than 1000 calls)"


def test_deep_calls_from_a_small_thread():
    # (a thread whose stack hasn't room for the deepest calls: the call made
    # on a stack of its own, as deep as on the main thread)
    import threading

    program = lang.load(DEEP, "deep")
    out = {}

    def work():
        out["ok"] = program.call("down", 900, mode="compiled")
        try:
            program.call("down", 5000, mode="compiled")
        except zrun.Error as e:
            out["err"] = e.diagnostic.message

    old = threading.stack_size(512 * 1024)
    try:
        t = threading.Thread(target=work)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    assert out == {"ok": 900, "err": "call stack too deep (more than 1000 calls)"}


def test_the_mode_last_run():
    program = lang.load(SOURCE, "engine")
    # (never run: compiled)
    assert program.call("bump", 2) == 2
    program.run(mode="python")
    assert program.call("bump", 2) == 2
