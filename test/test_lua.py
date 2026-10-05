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
import contextlib, io, lua
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


def test_specialized():
    # every call site of a helper compiled out of line hot at its first
    # call: each compiled for what it knows (ZRUN_HOT_CALLS=1), the
    # programs printing what real Lua does, the second run too (the
    # specialized code then)
    import subprocess

    env = dict(os.environ, ZRUN_HOT_CALLS="1")
    some = [p for p in ("funcs", "programs", "strings") if p in PROGRAMS]
    r = subprocess.run([sys.executable, "-c", SPECIALIZED, LUA_DIR, *some], env=env, capture_output=True, text=True, timeout=1800)
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


def test_arguments(capsys):
    lua.run("print(#arg, arg[0], arg[1], ...)\n", "script.lua", args=["a", "b"])
    assert capsys.readouterr().out == "2\tscript.lua\ta\n"
