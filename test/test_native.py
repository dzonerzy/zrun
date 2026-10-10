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
    sys.platform not in ("linux", "win32") or (not HAVE_ZIGLANG and shutil.which("zig") is None),
    reason="standalone programs: Linux or Windows, with Zig",
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


@pytest.mark.slow
def test_setup(zig, tmp_path, monkeypatch):
    # (setup=: a function compiled with the program, called with its path
    # and arguments as it starts: Lua's `arg`)
    lua = _strict_lua(monkeypatch)
    src = 'print(#arg, arg[0])\nfor i = 1, #arg do io.write(arg[i], ";") end\nprint(tonumber(arg[1]) + 1)\n'
    exe = zrun.build_native(lua.lang, src, str(tmp_path / "args"), path="args.lua", prune=True, setup=lua.set_args)
    r = subprocess.run([exe, "41", "héllo", ""], capture_output=True, encoding="utf-8", timeout=60)
    assert (r.returncode, r.stdout) == (0, "3\targs.lua\n41;héllo;;42\n")
    r = subprocess.run([exe], capture_output=True, encoding="utf-8", timeout=60)
    assert r.returncode == 1 and r.stdout == "0\targs.lua\n" and "arithmetic on a nil value" in r.stderr


def test_setup_not_compiled(tmp_path, monkeypatch):
    lua = _strict_lua(monkeypatch)

    def setup(path, args, **kw):
        pass

    with pytest.raises(zrun.CompileError, match="the setup can't be compiled.*kwargs"):
        lua.lang.load("print(1)", "x.lua").native_objects(setup=setup)
    with pytest.raises(TypeError, match="setup must be a Python function"):
        lua.lang.load("print(1)", "x.lua").native_objects(setup=print)


@pytest.mark.parametrize("target", ["x86_64-linux", "x86_64-windows"])
def test_targets(zig, tmp_path, target):
    # (a program for another machine: Linux's or Windows' executable, from
    # either; run where it can be)
    src = open(os.path.join(HERE, "..", "examples", "tiny", "fib.tiny")).read()
    exe = zrun.build_native(tiny.lang, src, str(tmp_path / "fib"), path="fib.tiny", target=target)
    windows = target == "x86_64-windows"
    assert exe.endswith(".exe") == windows
    with open(exe, "rb") as f:
        assert f.read(4)[: 2 if windows else 4] == (b"MZ" if windows else b"\x7fELF")
    if (sys.platform == "win32") == windows:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        assert (r.returncode, r.stdout.replace("\r\n", "\n")) == (0, "".join(f"{n}\n" for n in (0, 1, 1, 2, 3, 5, 8, 13, 21, 34)))


def test_shared(zig, tmp_path):
    # (a library with a C API: zrun_init, zrun_call, zrun_error (zrun.h);
    # called here through ctypes as C would)
    import ctypes

    src = (
        "let base = 100;\n"
        "fn add(a, b) { return a + b + base; }\n"
        "fn half(x) { return x / 2; }\n"
        "fn twice(s) { return s + s; }\n"
        "fn boom(n) { return 10 / n; }\n"
    )
    lib_path = zrun.build_native(tiny.lang, src, str(tmp_path / "prog"), path="prog.tiny", shared=True)
    assert lib_path.endswith(".dll" if sys.platform == "win32" else ".so")
    assert "int zrun_call(" in open(str(tmp_path / "prog.h")).read()

    class S(ctypes.Structure):
        _fields_ = [("ptr", ctypes.c_char_p), ("len", ctypes.c_size_t)]

    class U(ctypes.Union):
        _fields_ = [("i", ctypes.c_int64), ("f", ctypes.c_double), ("s", S)]

    class V(ctypes.Structure):
        _fields_ = [("kind", ctypes.c_int), ("as_", U)]

    lib = ctypes.CDLL(lib_path)
    lib.zrun_error.restype = ctypes.c_char_p
    assert lib.zrun_init(0, None) == 0

    def call(name, *args):
        vals = (V * max(len(args), 1))()
        for v, a in zip(vals, args):
            if isinstance(a, int):
                v.kind, v.as_.i = 2, a
            elif isinstance(a, float):
                v.kind, v.as_.f = 3, a
            else:
                b = a.encode()
                v.kind, v.as_.s = 4, S(b, len(b))
        out = V()
        if lib.zrun_call(name.encode(), vals, len(args), ctypes.byref(out)) != 0:
            return ("error", lib.zrun_error().decode())
        return {2: lambda: out.as_.i, 3: lambda: out.as_.f, 4: lambda: out.as_.s.ptr[: out.as_.s.len].decode()}[out.kind]()

    assert call("add", 1, 2) == 103
    # (tiny's /: ints' floored, as the compiled mode's)
    assert call("half", 5) == 2
    assert call("half", 5.0) == 2.0
    assert call("twice", "hé") == "héhé"
    kind, msg = call("boom", 0)
    assert kind == "error" and msg.startswith("prog.tiny:5:21: error: division by zero [runtime]") and "in boom()" in msg
    assert call("add", 1, 1) == 102
    assert call("nope") == ("error", "prog.tiny: error: nope isn't a name the program's top level defines [runtime]")


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
