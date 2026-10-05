"""Compiled code kept between processes (cache.zig): a program compiled
once is loaded from the cache the next time, doing what it did; the code
refers to the process's objects by name, so a program differing only in its
constants gets its own values; a cache file that won't load is compiled
again; ZRUN_CACHE=0 keeps nothing."""

import os
import subprocess
import sys

from conftest import HERE

LUA_DIR = os.path.join(HERE, "..", "examples", "lua")

RUN = """
import sys
sys.path.insert(0, sys.argv[1])
import lua
lua.run(sys.argv[2], "prog.lua", mode="compiled")
"""

PROGRAM = """
local t = {}
for i = 1, 3 do t[i] = "hello" .. i end
local function twice(x) return x * 2 end
print(table.concat(t, ","), twice(21), #t, 2^62 + 1 > 0)
"""


def run(cache, source, **env):
    e = dict(os.environ, ZRUN_CACHE_DIR=str(cache), ZRUN_STATS="1", **env)
    r = subprocess.run([sys.executable, "-c", RUN, LUA_DIR, source], env=e, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout, r.stderr


def test_kept_and_loaded(tmp_path):
    out, err = run(tmp_path, PROGRAM)
    assert out == "hello1,hello2,hello3\t42\t3\ttrue\n"
    assert "compiled, kept" in err and "from the cache" not in err
    files = sorted(os.listdir(tmp_path))
    assert files and all(f.endswith(".o") for f in files)
    # (another process: the same code, from the cache)
    out2, err2 = run(tmp_path, PROGRAM)
    assert out2 == out
    assert "from the cache" in err2 and "compiled, kept" not in err2
    assert sorted(os.listdir(tmp_path)) == files


def test_other_constants_their_own_values(tmp_path):
    run(tmp_path, PROGRAM)
    # (the same code but its strs: their own, the code names them)
    out, _ = run(tmp_path, PROGRAM.replace('"hello"', '"world"'))
    assert out == "world1,world2,world3\t42\t3\ttrue\n"


def test_a_file_that_wont_load(tmp_path):
    run(tmp_path, PROGRAM)
    for f in os.listdir(tmp_path):
        with open(tmp_path / f, "wb") as fh:
            fh.write(b"not an object file")
    out, err = run(tmp_path, PROGRAM)
    assert out == "hello1,hello2,hello3\t42\t3\ttrue\n"
    assert "compiled, kept" in err
    # (and kept right again)
    out, err = run(tmp_path, PROGRAM)
    assert "from the cache" in err and "compiled, kept" not in err


def test_off(tmp_path):
    out, err = run(tmp_path, PROGRAM, ZRUN_CACHE="0")
    assert out == "hello1,hello2,hello3\t42\t3\ttrue\n"
    assert os.listdir(tmp_path) == [] and "kept" not in err
