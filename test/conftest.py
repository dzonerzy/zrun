"""The tiny language (examples/tiny) for the tests, and helpers."""

import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)


def _load_tiny():
    path = os.path.join(HERE, "..", "examples", "tiny", "tiny.py")
    spec = importlib.util.spec_from_file_location("tiny_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tiny = _load_tiny()

# The typed language (examples/typed): typed.py imports typedlang.py from its folder
sys.path.insert(0, os.path.join(HERE, "..", "examples", "typed"))
import typed  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run the slow tests too (compiling a lot: before committing compiler changes)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: compiles a lot; run with --slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow: run with --slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def run(capsys):
    """Run a tiny program; what it printed."""

    def run_(source):
        tiny.lang.load(source).run()
        return capsys.readouterr().out

    return run_
