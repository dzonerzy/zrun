"""Module state the semantics change is made native when a program compiles
(adopt.zig): the module's names then refer to it (proxies: isinstance() is
true), Python and compiled code see the same objects, and what can't be
shared that way (held elsewhere too, keys native dicts can't hash,
attributes beyond a record's fields) stays Python's."""

import gc

import pytest
import state_lang as S
import zrun
from test_modes import MODES, same_in_every_mode

PROGRAM = "reset(); put(1, 10); put(2, 20); print(get(1), get(2), get(3)); print(held(7), bykey(1), bykey(2), box(2));\n"


def test_shared_with_python(capsys):
    for _ in range(2):
        # (the second time, the reference mode too sees it native)
        out, err = same_in_every_mode(S.lang, PROGRAM, capsys)
        assert err is None and out == "10 20 -1\n1 3 0 3\n"
    # The module's names: the native objects
    assert type(S.ENV) is not S.Env and isinstance(S.ENV, S.Env)
    assert S.ENV is S.ALIAS and S.ENV.parent is S.ENV
    assert S.ENV.vars == {"k1": 10, "k2": 20} and S.ENV.log == [1, 2]
    assert type(S.COUNTS) is not dict and S.COUNTS == {"calls": 11}
    # What stays Python's
    assert type(S.HELD) is list and S.HELD == [7] and S.HOLDERS[0] is S.HELD
    assert type(S.BY_KEY) is dict
    assert type(S.BOX) is S.Box and S.BOX.v == 3 and S.BOX.extra == 5


def test_python_changes_are_seen(capsys):
    S.lang.load("reset();").run(mode="compiled")
    S.ENV.vars["k5"] = 55
    S.ENV.log.append("py")
    S.lang.load("print(get(5)); put(6, 1);").run(mode="compiled")
    assert capsys.readouterr().out == "55\n"
    assert S.ENV.log == ["py", 6]


def test_values_outlive_their_program():
    p = S.lang.load("lit(); kinds();")
    p.run(mode="compiled")
    del p
    gc.collect()
    # (other programs made and freed after: the memory reused)
    for i in range(3):
        S.lang.load(f"put({i}, {i});").run(mode="compiled")
    gc.collect()
    assert S.ENV.vars["lit"] == "a literal str"
    assert S.ENV.vars["big"] == 2**70 + 1
    assert S.ENV.vars["kind"] == "Call"


def test_report(capsys):
    # what's native and why not, where compiled code went through Python
    p = S.lang.load(PROGRAM)
    p.run(mode="compiled", report=True)
    capsys.readouterr()
    r = p.report()
    state = r["module_state"]
    # (what the program's code uses: lit() isn't called, ALIAS isn't read)
    assert state["ENV"] == "native" and state["COUNTS"] == "native" and "ALIAS" not in state
    assert "referred to from somewhere besides" in state["HELD"]
    assert state["BY_KEY"] == "constant (only read)" and "attributes beyond its fields" in state["BOX"]
    # (HELD, BOX: Python's; reading them goes through it)
    assert r["python_crossings"] and all(isinstance(n, int) and n > 0 for n in r["python_crossings"].values())
    assert set(r["cache"]) == {"loaded", "compiled"}
    # (a run without report=True: counts nothing, the last report stays)
    p.run(mode="compiled")
    capsys.readouterr()
    assert p.report()["python_crossings"] == r["python_crossings"]


def test_settings():
    with pytest.raises(ValueError):
        S.zrun.Language(S.tiny.PARSER, hot_calls=0)
    with pytest.raises(TypeError):
        zrun.configure(cache=3)
    # (unchanged: the usual place, no perf map)
    zrun.configure(cache=True, perf_map=False)


@pytest.mark.parametrize("mode", MODES)
def test_another_programs_function(mode):
    first = S.lang.load("fn f(n) { return n + 1; } put(1, f); print(callk(1, 2));")
    first.run(mode=mode)
    second = S.lang.load("print(callk(1, 2));")
    with pytest.raises(zrun.Error, match="a function of another program can't be called here"):
        second.run(mode=mode)
