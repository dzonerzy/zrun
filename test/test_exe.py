"""Executables (exe/): zrun.build_executable() makes one file running a
program, with a Python runtime and zrun in it. Building downloads the
runtime once and needs the ziglang package: the tests doing it are slow."""

import importlib.util
import os
import subprocess
import sys

import pytest
import zrun
from conftest import HERE

TINY = os.path.join(HERE, "..", "examples", "tiny")

needs_zig = pytest.mark.skipif(importlib.util.find_spec("ziglang") is None, reason="needs the ziglang package")


@pytest.fixture
def tiny_path(monkeypatch):
    """examples/tiny importable as `tiny` ('tiny:lang')."""
    monkeypatch.syspath_prepend(TINY)
    return "tiny:lang"


def run(exe, cache, *args):
    """The executable run with its cache under `cache`."""
    env = dict(os.environ, XDG_CACHE_HOME=str(cache), LOCALAPPDATA=str(cache))
    return subprocess.run([exe, *args], capture_output=True, text=True, env=env, timeout=600)


def test_unknown_target(tiny_path, tmp_path):
    with pytest.raises(ValueError, match="target"):
        zrun.build_executable(tiny_path, "print(1);", str(tmp_path / "x"), target="riscv64-linux")


def test_unknown_python(tiny_path, tmp_path):
    with pytest.raises(ValueError, match="3.10 to 3.14"):
        zrun.build_executable(tiny_path, "print(1);", str(tmp_path / "x"), python="3.9")


def test_language_not_found(tmp_path):
    with pytest.raises(TypeError):
        zrun.build_executable(42, "print(1);", str(tmp_path / "x"))


@pytest.mark.slow
@needs_zig
def test_runs_the_program(tiny_path, tmp_path):
    src = tmp_path / "fib.tiny"
    src.write_text("fn fib(n) { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }\nprint(fib(20));\n")
    exe = zrun.build_executable(tiny_path, str(src), str(tmp_path / "fib"))
    assert os.path.exists(exe)
    cache = tmp_path / "cache"
    first = run(exe, cache)
    assert (first.returncode, first.stdout, first.stderr) == (0, "6765\n", "")
    # (unpacked once: the second run uses it)
    unpacked = os.listdir(cache / "zrun" / "exe")
    again = run(exe, cache)
    assert (again.returncode, again.stdout) == (0, "6765\n")
    assert os.listdir(cache / "zrun" / "exe") == unpacked


@pytest.mark.slow
@needs_zig
def test_errors_exit_1(tiny_path, tmp_path):
    exe = zrun.build_executable(tiny_path, "print(1);\nprint(1 / 0);\n", str(tmp_path / "err"), path="err.tiny")
    r = run(exe, tmp_path / "cache")
    assert r.returncode == 1
    assert r.stdout == "1\n"
    assert "err.tiny:2" in r.stderr


@pytest.mark.slow
@needs_zig
def test_errors_found_when_building(tiny_path, tmp_path):
    with pytest.raises(zrun.LoadError):
        zrun.build_executable(tiny_path, "print(;", str(tmp_path / "bad"))
    with pytest.raises(zrun.LoadError):
        other = "x86_64-linux" if sys.platform == "win32" else "x86_64-windows"
        zrun.build_executable(tiny_path, "print(;", str(tmp_path / "bad"), target=other)
    assert not os.path.exists(tmp_path / "bad") and not os.path.exists(tmp_path / "bad.exe")
