"""The typed language: structs, methods (a receiver), fields seen from
methods (rt.scope), lists, optionals, every loop."""

import os

import pytest
import zrun
from conftest import HERE, typed


def run(source, capsys):
    typed.lang.load(source).run()
    return capsys.readouterr().out


def test_shapes(capsys):
    with open(os.path.join(HERE, "..", "examples", "typed", "shapes.ty")) as f:
        assert run(f.read(), capsys) == "somewhere else 4.0\n3.0\n"


def test_methods_see_the_receiver(capsys):
    src = """
struct Counter {
    n: int;
    fn bump(by: int) -> int {
        n = n + by;
        return n;
    }
}
let c = Counter(1);
c.bump(2);
print(c.bump(3), c.n);
"""
    assert run(src, capsys) == "6 6\n"


def test_loops(capsys):
    src = """
let i = 0;
let out = 0;
loop {
    i = i + 1;
    if i % 2 == 0 { continue; }
    if i > 7 { break; }
    out = out + i;
}
do { out = out * 10; } while false;
print(out);
"""
    # 1 + 3 + 5 + 7, then once * 10
    assert run(src, capsys) == "160\n"


def test_int_where_float_is_declared(capsys):
    assert run("let x: float = 2;\nprint(x);\n", capsys) == "2.0\n"


def test_runtime_errors():
    with pytest.raises(zrun.Error) as e:
        typed.lang.load("let xs = [1, 2];\nprint(xs[5]);\n").run()
    assert e.value.diagnostic.message == "list index out of range"
    with pytest.raises(zrun.Error, match="integer overflow"):
        typed.lang.load("let x = 9223372036854775807;\nlet y = x + 1;\n").run()


def test_types_and_symbols():
    seen = {}
    lang = zrun.Language(typed.PARSER, typed.RULES)

    @lang.exec("let_stmt")
    def let(node, rt):
        seen[node.name.text] = (rt.type_of(node.value), rt.symbol(node.name).name, rt.scope(node.name).kind)

    lang.load('let a = 1.5;\nlet b = "s";\n').run()
    # (the rules make `program` a scope: the top level's names are in it)
    assert seen == {"a": ("float", "a", "program"), "b": ("str", "b", "program")}


def test_receiver_outside_a_method():
    lang = zrun.Language(typed.PARSER, typed.RULES)
    got = []

    @lang.exec("let_stmt")
    def let(node, rt):
        got.append(rt.receiver)

    lang.load("let a = 1;\n").run()
    assert got == [None]
