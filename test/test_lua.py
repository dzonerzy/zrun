"""Lua 5.4 (examples/lua): each program in examples/lua/tests prints what
real Lua 5.4.7 printed for it (its .expected file, made with `lua`)."""

import contextlib
import glob
import io
import os
import sys

import pytest
import zrun
from conftest import HERE

LUA_DIR = os.path.join(HERE, "..", "examples", "lua")
sys.path.insert(0, LUA_DIR)
import lua  # noqa: E402

PROGRAMS = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(LUA_DIR, "tests", "*.lua")))


@pytest.mark.parametrize("mode", ["python", "compiled"])
@pytest.mark.parametrize("name", PROGRAMS)
def test_like_real_lua(name, mode, capsys):
    path =os.path.join(LUA_DIR, "tests", name + ".lua")
    with open(path) as f:
        source = f.read()
    with open(path[:-4] + ".expected") as f:
        expected = f.read()
    lua.run(source, name + ".lua", mode=mode)
    assert capsys.readouterr().out == expected


def test_all_native():
    # every Lua semantic is compiled: none runs as Python (the ones that
    # would are learned while compiling: lang.python_semantics())
    for name in PROGRAMS:
        with open(os.path.join(LUA_DIR, "tests", name + ".lua")) as f, contextlib.redirect_stdout(io.StringIO()):
            lua.run(f.read(), name + ".lua", mode="compiled")
    assert lua.lang.python_semantics() == {}


SPECIALIZED = """
import sys
sys.path.insert(0, sys.argv[1])
import contextlib, functools, io, zrun
# (lua.py's language made with every call site hot at once)
zrun.Language = functools.partial(zrun.Language, hot_calls=1)
import lua
bad = []
for name in sys.argv[2:]:
    path = sys.argv[1] + "/tests/" + name + ".lua"
    expected = open(path[:-4] + ".expected").read()
    for _ in range(2):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lua.run(open(path).read(), name + ".lua", mode="compiled")
        if out.getvalue() != expected:
            bad.append(name)
print(bad)
"""


@pytest.mark.slow
def test_specialized():
    # every call site of a helper compiled out of line hot at its first
    # call: each compiled for what it knows (Language(hot_calls=1)), the
    # programs printing what real Lua does, the second run too (the
    # specialized code then); in a process of its own (lua.py's language
    # made so). (The cache keeps the code compiled for the IR it is: the
    # same code; compiling it all is slow, after a change of the compiler)
    import subprocess

    some = [p for p in ("funcs", "programs", "strings") if p in PROGRAMS]
    r = subprocess.run([sys.executable, "-c", SPECIALIZED, LUA_DIR, *some], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    assert r.stdout.strip() == "[]"


def test_uncaught_error():
    with pytest.raises(zrun.Error) as e:
        lua.run("local t = nil\nprint(t.x)\n", "boom.lua")
    assert e.value.diagnostic.message == "boom.lua:2: attempt to index a nil value (local 't')"
    with pytest.raises(zrun.Error) as e:
        lua.run('local function f() error({code = 1}) end\nf()\n', "obj.lua")
    # (a table error value: Lua's message for it)
    assert e.value.diagnostic.message.startswith("table: ")


def test_closures_keep_their_iteration():
    # each closure sees the loop variable of the iteration it was made in
    src = "local fs = {}\nfor i = 1, 3 do fs[i] = function() return i end end\nprint(fs[1](), fs[2](), fs[3]())\n"
    lua.run(src)


def test_function_variables_rebound(capsys):
    # calls of a variable that held one function (called directly, behind a
    # check) and then another: each call the one it holds then
    src = (
        "local function f(x) return x + 1 end\n"
        "local g = f\n"
        "local out = {}\n"
        "for i = 1, 4 do\n"
        "  out[#out + 1] = f(i) + g(i)\n"
        "  if i == 2 then f = function(x) return x * 10 end end\n"
        "  if i == 3 then f = 'not a function' end\n"
        "  if i == 3 then f = g end\n"
        "end\n"
        "print(table.concat(out, ' '))\n"
    )
    for mode in ("python", "compiled"):
        lua.lang.load(src, "rebind.lua").run(mode=mode)
    # 2+2, 3+3, 30+4, 5+5
    assert capsys.readouterr().out == "4 6 34 10\n" * 2


def test_speculation_guards_fail(capsys):
    # functions hot with ints (a typed entry made for them, taken when the
    # arguments are ints), then called with anything else: the generic code,
    # as the reference does it
    src = (
        "local function f(a, b) return a * 2 + b end\n"
        "local s = 0\n"
        "for i = 1, 1500 do s = s + f(i, 1) end\n"
        "print(s)\n"
        "print(f(1.5, 2), f(2, 0.5), f('3', 4), f(7, 1, 99), pcall(f, true, 1))\n"
        "local function g(n) if n < 1 then return 0 end return n + g(n - 1) end\n"
        "for i = 1, 1500 do g(3) end\n"
        "print(g(10), g(2.5), g(4), pcall(g, 'x'))\n"
        "local function h(x) return x end\n"
        "for i = 1, 1500 do h(i) end\n"
        "print(h(1), h(nil), h('s'), h(1.25), h({}) ~= nil)\n"
    )
    outs = []
    for mode in ("python", "compiled"):
        program = lua.lang.load(src, "guards.lua")
        program.run(mode=mode)
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]
    assert outs[1].startswith("2253000\n5.0\t4.5\t10\t15\tfalse\t")
    # (f, g and h: typed entries for ints)
    assert program.report()["speculated"] == {"line 1": "int, int", "line 6": "int", "line 9": "int"}


def test_arguments(capsys):
    lua.run("print(#arg, arg[0], arg[1], ...)\n", "script.lua", args=["a", "b"])
    assert capsys.readouterr().out == "2\tscript.lua\ta\n"
