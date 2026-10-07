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
    if sys.platform == "win32":
        # (the working set: GetProcessMemoryInfo's)
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                          "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
            ]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        get = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        current = ctypes.WinDLL("kernel32").GetCurrentProcess
        current.restype = wintypes.HANDLE
        if not get(current(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return c.WorkingSetSize // 1024
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


def test_a_language_in_a_cycle_is_collected():
    # (a language whose semantics name it (a closure): Python's collector
    # frees it, Language traversing what it holds)
    import gc
    import weakref

    from conftest import tiny

    def make():
        lang = zrun.Language(tiny.PARSER, tiny.RULES)

        @lang.eval("BinOp")
        def binop(node, rt):
            return lang and rt.eval(node.left)

        return weakref.ref(binop)

    gone = make()
    gc.collect()
    assert gone() is None


# (functions of the top level called without a reference of their own:
# their variables stored to while they run, the old ones kept till the code's
# done (Ctx.bury), then let go of)
REBOUND = {
    # (rebound by its own call, called again after)
    "own_call": "fn g(n) { return n * 2; }\nfn f(n) { if n == 3 { f = g; } if n > 0 { return f(n - 1) + 1; } return 0; }\nprint(f(5), f(5));\n",
    # (kept elsewhere before it's rebound: the copy its own)
    "escaped": "fn g(n) { return n + 100; }\nfn f(n) { return n; }\nfn swap() { let h = f; f = g; return h(1) + f(1); }\nprint(swap(), f(2));\n",
    # (rebound again and again while called)
    "many": "fn a(n) { return n; }\nfn b(n) { return n + 1; }\nfn f(n) { let i = 0; let t = 0; while i < n { if i % 2 == 0 { a = b; } else { a = f; } t = t + i; i = i + 1; } return t; }\nfn g(n) { let t = 0; let i = 0; while i < n { t = t + a(1); i = i + 1; } return t + f(n); }\nprint(g(50));\n",
}


@pytest.mark.parametrize("name", sorted(REBOUND))
def test_top_functions_rebound_while_called(name, capsys):
    from conftest import tiny
    from test_modes import same_in_every_mode
    import gc

    out, err = same_in_every_mode(tiny.lang, REBOUND[name], capsys)
    assert err is None and out
    # (nothing kept: the buried let go of as each run ends; its calls too)
    p = tiny.lang.load(REBOUND[name], "prog")
    counts = []
    for _ in range(4):
        p.run(mode="compiled")
        gc.collect()
        counts.append(zrun._blocks())
    capsys.readouterr()
    assert counts[3] == counts[2] == counts[1]
    q = tiny.lang.load(REBOUND["own_call"], "prog")
    q.run(mode="compiled")
    before = None
    for _ in range(4):
        # (its run left f rebound to g: g's result)
        assert q.call("f", 5) == 10
        gc.collect()
        now = zrun._blocks()
        assert before is None or now == before
        before = now
    capsys.readouterr()
