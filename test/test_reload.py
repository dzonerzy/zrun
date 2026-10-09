"""A program loaded again in the process (driver.zig's shared code): it runs
the code its first load compiled, not compiled again (its names, so its IR,
would be another prefix's: nothing in the cache for them), while the first
load lives and for a while after; with tiers, the optimized code the first
load was having made is the next load's. Each load is its own program (its
path in its messages, its own runs); a module attribute that may be rebound
(sys.stdout) is read when the code runs, not when it compiled."""

import contextlib
import io
import json
import os
import subprocess
import sys

from conftest import HERE

LUA_DIR = os.path.join(HERE, "..", "examples", "lua")
sys.path.insert(0, LUA_DIR)
import lua  # noqa: E402

RUN = """
import contextlib, gc, io, json, sys
sys.path.insert(0, sys.argv[1])
import zrun
settings = json.loads(sys.argv[3])
zrun.configure(cache=sys.argv[2], **settings["configure"])
import lua
outs = []
keep = []
for i in range(settings["loads"]):
    p = lua.lang.load(settings["source"], settings["path"])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        p.run(mode="compiled")
    outs.append(out.getvalue())
    if settings["keep"]:
        keep.append(p)
    else:
        del p
        gc.collect()
print(json.dumps(outs))
"""

PROGRAM = """
local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end
local t = {}
for i = 1, 3 do t[i] = fib(i + 10) end
print(table.concat(t, ","))
"""

EXPECTED = "89,144,233\n"


def run(cache, loads, keep=True, source=PROGRAM, path="prog.lua", **configure):
    """What each load printed (`loads` loads of `source` in one process,
    the earlier ones kept or not)."""
    settings = {"configure": configure, "loads": loads, "keep": keep, "source": source, "path": path}
    r = subprocess.run([sys.executable, "-c", RUN, LUA_DIR, str(cache), json.dumps(settings)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout)


def objects(cache):
    return sorted(f for f in os.listdir(cache) if f.endswith(".o"))


def test_loaded_again_compiled_once(tmp_path):
    a, b = tmp_path / "once", tmp_path / "many"
    assert run(a, 1, tiers=False) == [EXPECTED]
    # (as many objects as one load makes: the later loads compiled nothing)
    assert run(b, 6, tiers=False) == [EXPECTED] * 6
    assert len(objects(b)) == len(objects(a))


def test_loaded_again_after_the_first_load_went(tmp_path):
    a, b = tmp_path / "once", tmp_path / "many"
    run(a, 1, tiers=False)
    assert run(b, 6, keep=False, tiers=False) == [EXPECTED] * 6
    assert len(objects(b)) == len(objects(a))


def test_loaded_again_with_tiers(tmp_path):
    # the code compiled fast, and the optimized code: each made once, the
    # optimized code the first load was having made the later loads'
    a, b = tmp_path / "once", tmp_path / "many"
    run(a, 1)
    assert run(b, 6, keep=False) == [EXPECTED] * 6
    assert len(objects(b)) == len(objects(a))


def test_each_load_its_own_path():
    # (the same source, another path: its own code, its messages its own)
    src = "local t = nil\nprint(t.x)\n"
    for path in ("first.lua", "second.lua", "first.lua"):
        p = lua.lang.load(src, path)
        try:
            p.run(mode="compiled")
        except Exception as e:
            assert str(e).startswith(path), str(e)
        else:
            raise AssertionError("no error")


def test_loads_run_apart():
    # two loads of a program alive at once: each run's variables its own;
    # Lua's globals the module's (lua.py's G), as in every mode
    src = "local n = 0\nn = n + 1\nshared_count = (shared_count or 0) + 1\nprint(n, shared_count)\n"
    progs = [lua.lang.load(src, "count.lua") for _ in range(2)]
    outs = []
    for p in progs + progs:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            p.run(mode="compiled")
        outs.append(out.getvalue())
    assert outs == ["1\t1\n", "1\t2\n", "1\t3\n", "1\t4\n"]


def test_a_module_attribute_read_when_the_code_runs():
    # lua.py's print writes to sys.stdout: rebound (redirect_stdout) between
    # runs of one program, and of its loads, each run writes where it's
    # bound then
    p =lua.lang.load('print("hi")', "hi.lua")
    for prog in (p, p, lua.lang.load('print("hi")', "hi.lua")):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            prog.run(mode="compiled")
        assert out.getvalue() == "hi\n"
