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

# Python 3.12 and 3.13 on Windows allow 3000 calls nested through C (10000
# elsewhere), and a call of a language run as Python is several: the
# reference mode stops at a few hundred calls deep there, before zrun's
# limit (compiled code doesn't)
shallow_python = pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info[:2] in ((3, 12), (3, 13)),
    reason="Python 3.12/3.13 on Windows: nested calls through C are limited to 3000",
)

# The typed language (examples/typed): typed.py imports typedlang.py from its folder
sys.path.insert(0, os.path.join(HERE, "..", "examples", "typed"))
import typed  # noqa: E402, F401  (the tests import it from here)


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
