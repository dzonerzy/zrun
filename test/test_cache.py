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
zrun.configure(cache=json.loads(sys.argv[3]), tiers=False, **json.loads(sys.argv[4]))
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


def run(cache, source, **settings):
    """What the program printed, and the cache's report of it (more
    zrun.configure() settings given)."""
    r = subprocess.run([sys.executable, "-c", RUN, LUA_DIR, source, json.dumps(cache), json.dumps(settings)], capture_output=True, text=True, timeout=600)
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


def _old(path, age):
    """`path`'s times made `age` seconds ago."""
    t = os.path.getmtime(path) - age
    os.utime(path, (t, t))


def _size(folder):
    return sum(os.path.getsize(folder / f) for f in os.listdir(folder) if f.endswith(".o"))


def test_size_limited(tmp_path):
    # (what the program's objects take, in a cache of their own)
    (tmp_path / "alone").mkdir()
    run(str(tmp_path / "alone"), PROGRAM)
    ours = _size(tmp_path / "alone")
    folder = tmp_path / "cache"
    folder.mkdir()
    # (objects of other programs, each as big as the program's, used long
    # ago and less long ago; a write a stopped process left, old, and one
    # being written now)
    for i in range(10):
        f = folder / f"{i:064x}.o"
        f.write_bytes(b"x" * ours)
        _old(f, 10_000 - i * 100)
    (folder / "stale.o.1.tmp").write_bytes(b"x" * 10)
    _old(folder / "stale.o.1.tmp", 7200)
    (folder / "fresh.o.2.tmp").write_bytes(b"x" * 10)
    # (room for 4 after trimming: the program's and the 3 newest)
    limit = ours * 5
    tmp_path = folder
    out, cache = run(str(tmp_path), PROGRAM, cache_size=limit)
    assert out == EXPECTED and cache["compiled"] > 0
    files = set(os.listdir(tmp_path))
    # (down to 80% of the limit: the oldest deleted first, the newest and
    # what this run made kept)
    assert _size(tmp_path) <= limit * 8 // 10
    kept = [i for i in range(10) if f"{i:064x}.o" in files]
    assert kept == [7, 8, 9]
    assert "stale.o.1.tmp" not in files and "fresh.o.2.tmp" in files
    # (this run's: still there, loaded the next time)
    out, cache = run(str(tmp_path), PROGRAM, cache_size=limit)
    assert cache["loaded"] > 0 and cache["compiled"] == 0


def test_a_hit_is_a_use(tmp_path):
    # (an object loaded from the cache counts as recently used: it outlives
    # newer ones not used since)
    run(str(tmp_path), PROGRAM)
    ours = set(os.listdir(tmp_path))
    for f in ours:
        _old(tmp_path / f, 100_000)
    small = _size(tmp_path)
    for i in range(5):
        f = tmp_path / f"{i:064x}.o"
        f.write_bytes(b"x" * (small * 2))
        _old(f, 1000)
    _, cache = run(str(tmp_path), PROGRAM)
    assert cache["loaded"] > 0
    assert all(os.path.getmtime(tmp_path / f) > os.path.getmtime(tmp_path / f"{0:064x}.o") for f in ours)
    # (a run that compiles something, the limit under what's there: the
    # others go, ours stay (with room for what that run compiles))
    run(str(tmp_path), PROGRAM.replace("twice(21)", "twice(22)"), cache_size=small * 4)
    files = set(os.listdir(tmp_path))
    assert ours <= files and not any(f"{i:064x}.o" in files for i in range(5))


def test_clear_cache(tmp_path):
    run(str(tmp_path), PROGRAM)
    assert os.listdir(tmp_path)
    r = subprocess.run([sys.executable, "-c", "import sys, zrun; zrun.configure(cache=sys.argv[1]); zrun.clear_cache()", str(tmp_path)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".o")]


def test_cache_size_checked():
    import pytest
    import zrun

    with pytest.raises(ValueError):
        zrun.configure(cache_size=-1)
    with pytest.raises(TypeError):
        zrun.configure(cache_size="big")
