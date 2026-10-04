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


@pytest.fixture
def run(capsys):
    """Run a tiny program; what it printed."""

    def run_(source):
        tiny.lang.load(source).run()
        return capsys.readouterr().out

    return run_
