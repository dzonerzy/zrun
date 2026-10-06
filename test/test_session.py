"""Sessions (lang.session()) and the REPL (lang.repl()): programs run one
after another, each seeing the variables and functions the ones before it
defined at their top level: the same variables, functions run in their own
program; an expression entry's value; errors keep what was defined."""

import builtins
import os
import sys

import pytest
import zrun
from conftest import HERE, tiny

sys.path.insert(0, os.path.join(HERE, "..", "examples", "lua"))
import lua  # noqa: E402


def test_variables_and_functions_across_entries(capsys):
    s = tiny.lang.session()
    assert s.run("let x = 1;") is None
    assert s.run("fn inc(n) { x = x + n; return x; }") is None
    assert s.run("inc(5);") == 6
    assert s.run("x;") == 6
    # (the same variable: assigned here, the earlier function sees it)
    assert s.run("x = 100;") is None
    assert s.run("inc(1);") == 101
    # (a function calling one of an earlier entry, and the other way round)
    s.run("fn twice(n) { return inc(n) + inc(n); }")
    assert s.run("twice(2);") == 103 + 105
    s.run("fn apply(f, v) { return f(v); }")
    s.run("fn half(v) { return v / 2; }")
    assert s.run("apply(half, 10);") == 5
    assert s.names() == ["apply", "half", "inc", "twice", "x"]
    s.run("print(x, inc(0));")
    assert capsys.readouterr().out == "105 105\n"


def test_expression_values():
    s = tiny.lang.session()
    s.run("let x = 6;")
    assert [s.run(src) for src in ("x;", "42;", '"hi";', "x * 7;", "let y = 1;")] == [6, 42, "hi", 42, None]


def test_defined_again():
    # (a name defined again is the new entry's from then on: the entries
    # naming it see the new variable; the entry that defined it first keeps
    # its own)
    s = tiny.lang.session()
    s.run("let x = 1; fn own() { return x; }")
    s.run("fn get() { return x; }")
    s.run('let x = "new";')
    assert s.run("x;") == "new"
    assert s.run("get();") == "new"
    assert s.run("own();") == 1


def test_errors_keep_what_was_defined():
    s = tiny.lang.session()
    with pytest.raises(zrun.LoadError, match="undefined name 'y'"):
        s.run("y + 1;")
    with pytest.raises(zrun.LoadError, match="syntax"):
        s.run("let q = 1 +;")
    with pytest.raises(zrun.Error):
        s.run("let a = 1; let b = 1 / 0; let c = 3;")
    assert s.run("a;") == 1
    assert "b" not in s.names() and "c" not in s.names()
    assert s.names() == ["a"]


def test_lua_returns_and_globals():
    s = lua.lang.session()
    s.run("x = 10")
    s.run("function sq(n) return n * n end")
    assert s.run("return sq(x), 2") == [100, 2]
    assert s.run("return") == []
    assert s.run("print(x)") is None


def _repl(lang, lines, monkeypatch, capsys, **kwargs):
    """Run lang.repl() on lines typed (KeyboardInterrupt: Ctrl-C); what it
    printed (out, err)."""
    feed = iter(lines)

    def fake_input(prompt=""):
        builtins.print(prompt, end="")
        line = next(feed, None)
        if line is None:
            raise EOFError
        if line is KeyboardInterrupt:
            raise KeyboardInterrupt
        return line

    monkeypatch.setattr(builtins, "input", fake_input)
    assert lang.repl(**kwargs) is None
    return capsys.readouterr()


def test_repl(monkeypatch, capsys):
    out, err = _repl(
        tiny.lang,
        [
            "let x = 2;",
            "",
            "fn sq(n) {",  # (unfinished: more lines)
            "  return n * n;",
            "}",
            "sq(x) + 1;",
            "print(x)",  # (unfinished: ; missing...)
            "",  # (an empty line ends it anyway: the error)
            "nope;",
            "fn broken() {",
            KeyboardInterrupt,  # (the entry dropped)
            "x;",
        ],
        monkeypatch,
        capsys,
    )
    assert out == "> > > ... ... > 5\n> ... > > ... > 2\n> \n"
    lines = err.splitlines()
    assert "expected" in lines[0] and "undefined name 'nope'" in err and "KeyboardInterrupt" in err


def test_repl_show_and_prompts(monkeypatch, capsys):
    shown = []
    out, _ = _repl(tiny.lang, ["1 + 2;", '"a";'], monkeypatch, capsys, prompt="tiny> ", more="  ", show=shown.append)
    assert shown == [3, "a"]
    assert out == "tiny> tiny> tiny> \n"


def test_repl_lua(monkeypatch, capsys):
    def show(values):
        if values:
            builtins.print(*values, sep="\t")

    out, _ = _repl(lua.lang, ["function f(n)", "  return n + 1", "end", "return f(1), f(2)"], monkeypatch, capsys, show=show)
    assert out == "> ... ... > 2\t3\n> \n"
