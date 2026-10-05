"""Compiled code kept between processes (cache.zig): a program compiled
once is loaded from the cache the next time, doing what it did; the code
refers to the process's objects by name, so a program differing only in its
constants gets its own values; a cache file that won't load is compiled
again; zrun.configure(cache=False) keeps nothing."""

import json
import os
import subprocess
import sys

from conftest import HERE

LUA_DIR = os.path.join(HERE, "..", "examples", "lua")

RUN = """
import io, json, sys, contextlib
sys.path.insert(0, sys.argv[1])
import zrun
zrun.configure(cache=json.loads(sys.argv[3]))
import lua
p = lua.lang.load(sys.argv[2], "prog.lua")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    p.run(mode="compiled")
print(json.dumps([out.getvalue(), p.report()["cache"]]))
"""

PROGRAM = """
local t = {}
for i = 1, 3 do t[i] = "hello" .. i end
local function twice(x) return x * 2 end
print(table.concat(t, ","), twice(21), #t, 2^62 + 1 > 0)
"""

EXPECTED = "hello1,hello2,hello3\t42\t3\ttrue\n"


def run(cache, source):
    """What the program printed, and the cache's report of it."""
    r = subprocess.run([sys.executable, "-c", RUN, LUA_DIR, source, json.dumps(cache)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout)


def test_kept_and_loaded(tmp_path):
    out, cache = run(str(tmp_path), PROGRAM)
    assert out == EXPECTED
    assert cache["compiled"] > 0 and cache["loaded"] == 0
    files = sorted(os.listdir(tmp_path))
    assert files and all(f.endswith(".o") for f in files)
    # (another process: the same code, from the cache)
    out, cache = run(str(tmp_path), PROGRAM)
    assert out == EXPECTED
    assert cache["loaded"] > 0 and cache["compiled"] == 0
    assert sorted(os.listdir(tmp_path)) == files


def test_other_constants_their_own_values(tmp_path):
    run(str(tmp_path), PROGRAM)
    # (the same code but its strs: their own, the code names them)
    out, _ = run(str(tmp_path), PROGRAM.replace('"hello"', '"world"'))
    assert out == EXPECTED.replace("hello", "world")


def test_a_file_that_wont_load(tmp_path):
    run(str(tmp_path), PROGRAM)
    for f in os.listdir(tmp_path):
        with open(tmp_path / f, "wb") as fh:
            fh.write(b"not an object file")
    out, cache = run(str(tmp_path), PROGRAM)
    assert out == EXPECTED and cache["compiled"] > 0
    # (and kept right again)
    out, cache = run(str(tmp_path), PROGRAM)
    assert cache["loaded"] > 0 and cache["compiled"] == 0


def test_off(tmp_path):
    out, cache = run(False, PROGRAM)
    assert out == EXPECTED
    assert cache == {"loaded": 0, "compiled": 0}
