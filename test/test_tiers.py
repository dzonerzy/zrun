"""Tiers (driver.zig's Pending): a program run compiled whose optimized code
isn't cached runs compiled fast first while the optimized code is made in
the background, which runs from the run after it's done (and is in the
cache for the next process, even one exiting before it's done);
mode="auto" runs it as Python meanwhile; calls wait for the optimized code;
zrun.configure(tiers=False) compiles it optimized at once."""

import json
import subprocess
import sys

from conftest import HERE

# A fresh process (the cache: a directory of the test's), steps:
# "run:<mode>" (a run: what it printed, the code that ran), "wait" (runs
# until the optimized code took over), "call" (fib(15)), "drop" (the
# program gone, mid-compile), "exit" (at once: the work in the background
# waited for)
RUN = r"""
import sys, io, contextlib, json, time
sys.path.insert(0, {test!r})
import zrun
zrun.configure(cache={cache!r}, tiers={tiers!r})
from conftest import tiny
p = tiny.lang.load({source!r}, "t.tiny")
out = []
for step in {steps!r}:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if step.startswith("run:"):
            p.run(mode=step[4:])
        elif step == "wait":
            t = time.time()
            while p.report()["code"] != "optimized":
                assert time.time() - t < 300, "never optimized"
                time.sleep(0.05)
                p.run(mode="compiled")
        elif step == "call":
            buf.write(str(p.call("fib", 15)))
        elif step == "drop":
            p = None
            import gc
            gc.collect()
            p = tiny.lang.load({source!r}, "t2.tiny")
    r = p.report()
    out.append([step, buf.getvalue(), r["code"], r["optimizing"], r["cache"]])
print(json.dumps(out))
"""

PROGRAM = """
fn fib(n) { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }
print(fib(12));
"""


def _run(tmp_path, steps, tiers=True, source=PROGRAM):
    script = RUN.format(test=HERE, cache=str(tmp_path / "cache"), tiers=tiers, source=source, steps=steps)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_fast_first_then_optimized(tmp_path):
    out = _run(tmp_path, ["run:compiled", "wait", "run:compiled"])
    (_, printed, code, optimizing, _), _, (_, after, code2, optimizing2, _) = out
    assert printed == after == "144\n"
    assert (code, optimizing) == ("fast", True)
    assert (code2, optimizing2) == ("optimized", False)


def test_the_next_process_has_it_even_if_this_one_exited(tmp_path):
    # (the first process exits as soon as its run is done: the optimized
    # code, still being made, is waited for and kept)
    first = _run(tmp_path, ["run:compiled"])
    assert first[0][2] == "fast"
    (_, printed, code, optimizing, cache), = _run(tmp_path, ["run:compiled"])
    assert printed == "144\n"
    assert (code, optimizing) == ("optimized", False)
    assert cache["loaded"] > 0


def test_auto_runs_python_meanwhile(tmp_path):
    out = _run(tmp_path, ["run:auto", "wait", "run:auto"])
    (_, printed, code, optimizing, _) = out[0]
    assert printed == "144\n"
    # (as Python: nothing compiled fast)
    assert (code, optimizing) == (None, True)
    assert out[2][1:4] == ["144\n", "optimized", False]
    # (the next process: compiled from its first run)
    (_, printed, code, _, cache), = _run(tmp_path, ["run:auto"])
    assert (printed, code) == ("144\n", "optimized") and cache["loaded"] > 0


def test_calls_wait_for_the_optimized_code(tmp_path):
    out = _run(tmp_path, ["run:compiled", "call", "run:compiled"])
    assert out[0][2] == "fast"
    # (its top level run first, by the optimized code: the calls' state)
    assert out[1][1:4] == ["144\n610", "optimized", False]
    # (the runs after: the optimized code too)
    assert out[2][1:3] == ["144\n", "optimized"]


def test_dropped_while_compiling(tmp_path):
    out = _run(tmp_path, ["run:compiled", "drop", "run:compiled", "wait"])
    assert out[2][1] == "144\n"
    assert out[3][2] == "optimized"


def test_off(tmp_path):
    (_, printed, code, optimizing, _), = _run(tmp_path, ["run:compiled"], tiers=False)
    assert (printed, code, optimizing) == ("144\n", "optimized", False)
