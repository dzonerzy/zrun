"""Compiled modules (aot.zig): a program compiled ahead of time, saved, and
loaded in a fresh process without compiling; refused when the language's
definition changed."""

import os
import subprocess
import sys

import pytest
from conftest import HERE, tiny

PROGRAM = """
fn fib(n) { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }
let i = 0;
while i < 3 { print(fib(10 + i)); i = i + 1; }
"""

# A fresh process: tiny's language (maybe changed), the cache off (what it
# runs compiled comes from the module, or is compiled there)
FRESH = r"""
import sys, io, contextlib, json
sys.path.insert(0, {test!r})
import zrun
zrun.configure(cache=False)
from conftest import tiny
change = {change!r}
if change == "host":
    @tiny.lang.host("extra")
    def extra():
        return 1
elif change == "semantic":
    def neg(node, rt):
        return 0 - rt.eval(node.operand)
    tiny.lang.eval("Neg")(neg)
out = {{}}
try:
    p = tiny.lang.load_compiled({module!r})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p.run(mode="compiled")
        if {call!r}:
            out["call"] = p.call("fib", 20)
    out["printed"] = buf.getvalue()
    out["cache"] = p.report()["cache"]
except ValueError as e:
    out["refused"] = str(e)
print(json.dumps(out))
"""


def _fresh(module, change=None, call=False):
    # (a file: a semantic defined there has its source)
    script = module.parent / ("fresh_%s.py" % (change or "none"))
    script.write_text(FRESH.format(test=HERE, change=change, module=str(module), call=call))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=os.environ, timeout=600)
    assert r.returncode == 0, r.stderr
    import json

    return json.loads(r.stdout.strip().splitlines()[-1])


def test_compile_and_load_in_a_fresh_process(tmp_path, capsys):
    module = tmp_path / "fib.zrc"
    p = tiny.lang.compile(PROGRAM, str(module), path="fib.tiny")
    p.run(mode="compiled")
    expected = capsys.readouterr().out
    out = _fresh(module)
    assert out["printed"] == expected
    # (nothing compiled there: every object the module's)
    assert out["cache"]["compiled"] == 0 and out["cache"]["loaded"] > 0


def test_saved_after_running_has_what_was_compiled_as_it_ran(tmp_path, capsys):
    # (a call made before saving: the code compiled for it (a typed entry
    # for fib, made as calls got hot) goes in the module too)
    module = tmp_path / "hot.zrc"
    p = tiny.lang.load(PROGRAM, "hot.tiny")
    p.run(mode="compiled")
    for _ in range(3):
        assert p.call("fib", 20) == 6765
    p.save(str(module))
    capsys.readouterr()
    out = _fresh(module, call=True)
    assert out["call"] == 6765
    assert out["cache"]["compiled"] == 0


@pytest.mark.parametrize("change", ["host", "semantic"])
def test_refused_for_another_definition(tmp_path, change):
    module = tmp_path / "fib.zrc"
    tiny.lang.compile(PROGRAM, str(module))
    out = _fresh(module, change=change)
    assert "another definition of the language" in out["refused"]


MOVED = r"""
import importlib.util, json, sys
import zrun
zrun.configure(cache=False)
spec = importlib.util.spec_from_file_location("tiny_moved", {path!r})
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)
try:
    tiny.lang.load_compiled({module!r})
    print(json.dumps("loaded"))
except ValueError as e:
    print(json.dumps(str(e)))
"""


def test_loaded_where_the_language_moved(tmp_path):
    # (the language's files elsewhere, the same: an executable's, an
    # install's)
    module = tmp_path / "fib.zrc"
    tiny.lang.compile(PROGRAM, str(module))
    moved = tmp_path / "elsewhere" / "tiny.py"
    moved.parent.mkdir()
    moved.write_text(open(tiny.__file__, encoding="utf-8").read(), encoding="utf-8")
    script = tmp_path / "moved.py"
    script.write_text(MOVED.format(path=str(moved), module=str(module)))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=os.environ, timeout=600)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == '"loaded"'


def test_not_a_module(tmp_path):
    bad = tmp_path / "bad.zrc"
    bad.write_bytes(b"not a module at all")
    with pytest.raises(ValueError, match="isn't a compiled module"):
        tiny.lang.load_compiled(str(bad))
    with pytest.raises(OSError):
        tiny.lang.load_compiled(str(tmp_path / "missing.zrc"))
    # (cut short)
    good = tmp_path / "good.zrc"
    tiny.lang.compile(PROGRAM, str(good))
    cut = tmp_path / "cut.zrc"
    cut.write_bytes(good.read_bytes()[:200])
    with pytest.raises(ValueError):
        tiny.lang.load_compiled(str(cut))
