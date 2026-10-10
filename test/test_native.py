"""Standalone programs (zrun.build_native): a strict language's program
compiled ahead of time and linked with zrun's runtime into an executable
with no Python. It runs as the compiled mode runs it: the same output, the
same runtime errors, worded and placed the same. Building needs Zig (the
ziglang package, or `zig` on the PATH here)."""

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest
import zrun
from conftest import HERE
from test_strict import tiny

LUA_TESTS = os.path.join(HERE, "..", "examples", "lua", "tests")

HAVE_ZIGLANG = importlib.util.find_spec("ziglang") is not None

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or (not HAVE_ZIGLANG and shutil.which("zig") is None),
    reason="standalone programs: Linux, with Zig",
)


@pytest.fixture
def zig(monkeypatch, tmp_path_factory):
    """`python -m ziglang` as build_native runs it: the package, or a stand
    in running the `zig` on the PATH."""
    if HAVE_ZIGLANG:
        return
    shim = tmp_path_factory.mktemp("shim")
    (shim / "ziglang").mkdir()
    (shim / "ziglang" / "__init__.py").write_text("")
    (shim / "ziglang" / "__main__.py").write_text("import os, sys\nos.execvp('zig', ['zig'] + sys.argv[1:])\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(shim), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.syspath_prepend(str(shim))


def native(lang, source, tmp_path, name="prog.tiny"):
    """The program built standalone and run: (exit code, out, err)."""
    exe = zrun.build_native(lang, source, str(tmp_path / "prog"), path=name)
    r = subprocess.run([exe], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout, r.stderr


def compiled(lang, source, capsys, name="prog.tiny"):
    """The program run compiled here: (exit code, out, err) as a standalone
    one gives them (an error written, exit 1)."""
    try:
        lang.load(source, name).run(mode="compiled")
        code, err = 0, ""
    except zrun.Error as e:
        code, err = 1, str(e) + "\n"
    out = capsys.readouterr().out
    return code, out, err


def test_fib(zig, tmp_path, capsys):
    src = open(os.path.join(HERE, "..", "examples", "tiny", "fib.tiny")).read()
    got = native(tiny.lang, src, tmp_path)
    assert got == (0, "".join(f"{n}\n" for n in (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)), "")
    assert got == compiled(tiny.lang, src, capsys)


def test_runtime_error_as_the_compiled_mode_words_it(zig, tmp_path, capsys):
    src = "fn f(n) {\n    return 10 / n;\n}\nfn g(n) { return f(n) + 1; }\nprint(g(5));\nlet x = g(0);\n"
    got = native(tiny.lang, src, tmp_path)
    assert got[0] == 1 and got[1] == "3\n"
    assert got == compiled(tiny.lang, src, capsys)
    assert "in f(), called at prog.tiny:4:18" in got[2]


def test_overflow(zig, tmp_path, capsys):
    src = "let a = 9223372036854775807;\nlet b = a +\n   1;\n"
    got = native(tiny.lang, src, tmp_path)
    assert got == compiled(tiny.lang, src, capsys)
    assert "integer overflow" in got[2]


def test_output_of_values(zig, tmp_path, capsys):
    # (str() of native values, as Python writes them, without Python)
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=True)
    lang.function("FuncDef")

    @lang.exec(["Let", "Assign"])
    def assign(node, rt):
        rt.store(node.name, rt.eval(node.value))

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        print([a, b, (a,)], {a: [b]}, {a}, a / 4, -0.0, 2**70, "x'y", (1.5, None, True), sep=" | ")
        return a + b

    src = "let x = 3 + 4;\n"
    got = native(lang, src, tmp_path)
    assert got == compiled(lang, src, capsys)
    assert got[1] == "[3, 4, (3,)] | {3: [4]} | {3} | 0.75 | -0.0 | 1180591620717411303424 | x'y | (1.5, None, True)\n"


def _strict_lua(monkeypatch):
    # (the Lua example made strict: its module loaded again with
    # zrun.Language(strict=True))
    monkeypatch.syspath_prepend(os.path.join(HERE, "..", "examples", "lua"))
    spec = importlib.util.spec_from_file_location("lua_strict", os.path.join(HERE, "..", "examples", "lua", "lua.py"))
    module = importlib.util.module_from_spec(spec)
    # (a module of its own, as imported: its tables the semantics' state)
    monkeypatch.setitem(sys.modules, "lua_strict", module)
    plain = zrun.Language
    monkeypatch.setattr(zrun, "Language", lambda *a, **k: plain(*a, **k, strict=True))
    spec.loader.exec_module(module)
    monkeypatch.setattr(zrun, "Language", plain)
    return module


@pytest.mark.slow
def test_pruned(zig, tmp_path, monkeypatch):
    # (prune=True: only the library functions the program names compiled:
    # the same output, smaller; one named only at run time stops it, saying
    # why)
    lua = _strict_lua(monkeypatch)
    src = 'print("hi", string.rep("ab", 2))\nlocal f = _G["string"][("up" .. "per")]\nprint(f("x"))\n'
    lua.set_args("p.lua", ())
    left = []
    pruned = zrun.build_native(lua.lang, src, str(tmp_path / "pruned"), path="p.lua", prune=True, left_out=left)
    whole = zrun.build_native(lua.lang, src, str(tmp_path / "whole"), path="p.lua")
    assert "string_upper" in left and "string_rep" not in left and "lua_print" not in left
    assert os.path.getsize(pruned) < os.path.getsize(whole) / 2
    r = subprocess.run([pruned], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and r.stdout == "hi\tabab\n"
    assert "string_upper() wasn't compiled ahead of time: the build was pruned" in r.stderr
    r = subprocess.run([whole], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout == "hi\tabab\nX\n"


def test_not_strict(tmp_path):
    from conftest import tiny as plain

    with pytest.raises(zrun.CompileError, match="strict"):
        zrun.build_native(plain.lang, "print(1);", str(tmp_path / "x"))


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(n for n in os.listdir(LUA_TESTS) if n.endswith(".lua")))
def test_lua(zig, tmp_path, name, monkeypatch):
    # (the Lua example made strict: every test program, standalone, as
    # real Lua runs it)
    module = _strict_lua(monkeypatch)
    module.set_args(name, ())
    src = open(os.path.join(LUA_TESTS, name)).read()
    code, out, err = native(module.lang, src, tmp_path, name)
    expected = open(os.path.join(LUA_TESTS, name[:-4] + ".expected")).read()
    assert out + (err.split("\n")[0] + "\n" if err else "") == expected
