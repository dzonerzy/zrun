"""The cycle collector (gc.zig): values of compiled code that only reference
one another are freed (tables holding themselves, closures and the frames
they were made in), during a run, at its end, and on zrun.collect()."""

import os
import sys

import pytest
import zrun
from conftest import HERE

sys.path.insert(0, os.path.join(HERE, "..", "examples", "lua"))
import lua  # noqa: E402


@pytest.fixture
def run(capsys):
    """Run a Lua program compiled; what it printed."""

    def run_(program):
        program.run(mode="compiled")
        return capsys.readouterr().out

    return run_


def _rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4


CYCLES = """
local made = 0
for i = 1, %d do
  local t = {n = i}
  t.me = t                                  -- a table holding itself
  local a, b = {}, {}
  a.other, b.other = b, a                   -- two holding each other
  local function again() return again end   -- a closure in its own frame
  local x = {f = function() return t end}   -- a closure holding the table
  made = made + 1
end
print(made)
"""


def test_cycles_freed_at_the_end_of_a_run(run):
    program = lua.lang.load(CYCLES % 2000, "cycles.lua")
    run(program)
    before = zrun._blocks()
    for _ in range(3):
        assert run(program) == "2000\n"
    assert zrun._blocks() == before


def test_cycles_freed_during_a_run(run):
    # (a million iterations of garbage cycles: freed as the run goes, not
    # kept until its end)
    run(lua.lang.load(CYCLES % 10, "warm.lua"))
    program = lua.lang.load(CYCLES % 1000000, "many.lua")
    rss = _rss_kb()
    assert run(program) == "1000000\n"
    assert _rss_kb() - rss < 100_000


CALLED = """
local function make(n)
  local t = {n = n}
  t.me = t
  local function f() return t end
  t.f = f
  return n
end
"""


def test_cycles_of_calls():
    program = lua.lang.load(CALLED, "called.lua")
    # (hot by then: speculation's compiling done)
    for i in range(2000):
        program.call("make", i)
    zrun.collect()
    before = zrun._blocks()
    for i in range(20000):
        program.call("make", i)
    # (collected as they piled up: far fewer left than made)
    assert zrun._blocks() - before < 20000
    assert zrun.collect() > 0
    assert zrun._blocks() == before


def test_cycles_in_map_workers():
    program = lua.lang.load(CALLED, "mapped.lua")
    program.call("make", 0)
    program.map("make", range(20000), threads=4)
    rss = _rss_kb()
    for _ in range(20):
        # (a Lua function's results: a list)
        assert program.map("make", range(20000), threads=4) == [[i] for i in range(20000)]
    assert _rss_kb() - rss < 30_000


def test_a_long_cycle(run):
    # (a ring of 300000 tables: gone through without recursion)
    program = lua.lang.load(
        """
        local first = {}
        local t = first
        for i = 1, 300000 do
          local n = {}
          t.next = n
          t = n
        end
        t.next = first
        print("made")
        """,
        "ring.lua",
    )
    run(program)
    before = zrun._blocks()
    assert run(program) == "made\n"
    assert zrun._blocks() == before


def test_lua_programs_leave_nothing():
    # (every Lua test program, run again and again: no block left over;
    # the leaks of compiled code it found: errors caught (pcall), known
    # tuples and lists, loops over temporaries)
    import gc
    import glob
    import io
    import contextlib

    for path in sorted(glob.glob(os.path.join(HERE, "..", "examples", "lua", "tests", "*.lua"))):
        with open(path) as f:
            program = lua.lang.load(f.read(), os.path.basename(path))
        counts = []
        for _ in range(3):
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    program.run(mode="compiled")
                except zrun.Error:
                    pass
            gc.collect()
            counts.append(zrun._blocks())
        assert counts[2] == counts[1], path


def test_what_python_holds_is_kept(run):
    # (cyclic tables Python got hold of, through a library function: kept
    # while Python has them, whatever the collector finds)
    kept = []

    def keep(rt, node, args):
        kept.append(args[0])
        return []

    lua.lib(lua.G, "keep")(keep)
    try:
        program = lua.lang.load(
            """
            for i = 1, 3 do
              local t = {n = i}
              t.me = t
              keep(t)
            end
            local junk = {}
            junk.me = junk
            """,
            "kept.lua",
        )
        run(program)
        zrun.collect()
        assert [t.get("n") for t in kept] == [1, 2, 3]
        assert all(t.get("me").get("n") == t.get("n") for t in kept)
    finally:
        lua.G.set("keep", None)
