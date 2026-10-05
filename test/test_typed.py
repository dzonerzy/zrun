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


class TestTypes:
    """Language.types(): what the values of the language's types are. Compiled
    code knows a node's value's kind from it; a value not of the kind said
    is an error at the node, in every mode."""

    PROGRAM = "fn half(x: float) -> float { return x; }\nprint(half(4), 1 + 2);\n"

    def outcomes(self, source):
        from test_modes import outcome

        out = {}
        for mode in ("python", "compiled"):
            program = typed.lang.load(source, "prog")
            try:
                program.run(mode=mode)
                out[mode] = None
            except zrun.Error as e:
                out[mode] = e.diagnostic.message
        return out

    def test_declared_right(self, capsys):
        # (the example's own: int, bool, str, nil; not float, an int's one too)
        assert self.outcomes(self.PROGRAM) == {"python": None, "compiled": None}
        assert capsys.readouterr().out == "4 3\n4 3\n"

    def test_declared_wrong(self, capsys):
        # float said to be float, but an int where a float's expected stays one
        try:
            typed.lang.types(dict(typed.SCALARS, float=float))
            outs = self.outcomes(self.PROGRAM)
        finally:
            typed.lang.types(typed.value_type)
        assert outs["python"] == outs["compiled"] == "the value isn't what types() says the type float is"

    def test_by_name_and_generic_name(self, capsys):
        # a dict: 'int' by name; 'list' for 'list[int]' (lists aren't checked:
        # their kind isn't one tag)
        try:
            typed.lang.types({"int": int, "list": list, "fn": zrun.Function})
            outs = self.outcomes("let xs = [1, 2];\nfn f(n: int) -> int { return n * 2; }\nprint(f(xs[1]));\n")
        finally:
            typed.lang.types(typed.value_type)
        # (fn: a builtin's value (print) isn't a zrun.Function: an error, in
        # both modes)
        assert outs["python"] == outs["compiled"] == "the value isn't what types() says the type fn(...) -> void is"

    def test_typed_entries(self, capsys):
        # functions whose parameters are declared ints and bools: called with
        # them plain where the arguments' kinds are known (n - 1, a literal), by
        # the argument list where not (a list's item), from methods too
        src = """
struct Acc {
    total: int;
    fn add(by: int, twice: bool) -> int {
        if twice { total = total + by; }
        total = total + by;
        return total;
    }
}
fn fib(n: int) -> int { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }
fn pick(a: int, b: int, first: bool) -> int { if first { return a; } return b; }
let xs = [7, 8];
let acc = Acc(0);
acc.add(2, true);
print(fib(15), pick(3, 4, false), pick(xs[0], 2, false), pick(1, xs[1], true), acc.add(xs[1], false));
"""
        assert self.outcomes(src) == {"python": None, "compiled": None}
        assert capsys.readouterr().out == "610 4 2 1 12\n610 4 2 1 12\n"
        ir = typed.lang.load(src, "prog").compiled_ir()
        # (fib's and pick's typed entries)
        assert ir.count("define i1 @zr_ir_t") == 2

    def test_not_a_mapping(self):
        with pytest.raises(TypeError):
            typed.lang.types(3)
