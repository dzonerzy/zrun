"""The typed language: structs, methods (a receiver), fields seen from
methods (rt.scope), lists, optionals, every loop."""

import os

import pytest
import zrun
from conftest import HERE, shallow_python, typed


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
        # (fib's, pick's and Acc.add's typed entries: called directly, or
        # through the generic entries for arguments of their kinds)
        assert ir.count("define { i64, i64 } @zr_ir_t") == 3

    def test_list_parameters_borrowed(self, capsys):
        # a typed entry takes a list declared as a parameter borrowed from
        # its caller: read, returned, kept in another list, changed, the
        # parameter stored to (a reference taken first), recursion with it;
        # the same in both modes, and nothing left allocated
        src = """
fn total(xs: list[int]) -> int { let t = 0; let i = 0; while i < len(xs) { t = t + xs[i]; i = i + 1; } return t; }
fn same(xs: list[int]) -> list[int] { return xs; }
fn pair(xs: list[int]) -> list[list[int]] { return [xs, xs]; }
fn bump(xs: list[int], n: int) -> int { xs[0] = xs[0] + n; return xs[0]; }
fn swap(xs: list[int], ys: list[int]) -> int { let a = total(xs); xs = ys; return a * 100 + total(xs); }
fn down(xs: list[int], n: int) -> int { if n == 0 { return total(xs); } return down(xs, n - 1) + 1; }
let a = [1, 2, 3];
let b = [10, 20];
let kept = pair(a);
let k = 0;
let sum = 0;
while k < 300 {
    let p = pair(b);
    sum = sum + total(a) + total(same(b)) + total(p[1]) + bump(b, 1) + swap(a, [4, 5]) + down(a, 3);
    k = k + 1;
}
print(sum, total(kept[0]), total(b), len(kept));
"""
        assert self.outcomes(src) == {"python": None, "compiled": None}
        out = capsys.readouterr().out
        assert out.splitlines()[0] == out.splitlines()[1]
        # (once more: what the first run kept for good, kept; the rest freed)
        zrun.collect()
        before = zrun._blocks()
        assert self.outcomes(src) == {"python": None, "compiled": None}
        assert capsys.readouterr().out == out
        zrun.collect()
        assert zrun._blocks() == before

    @pytest.mark.parametrize(
        "name, src",
        [
            # (an error deep in typed recursion: its stack, each call's)
            ("deep_error", "fn f(n: int) -> int { if n == 0 { return 9223372036854775807 + n + 1; } return f(n - 1) + 1; }\nprint(f(6));\n"),
            # (the calls' limit reached in typed code: the same call failing)
            pytest.param("too_deep", "fn f(n: int) -> int { return f(n + 1) + 1; }\nprint(f(0));\n", marks=shallow_python),
            # (typed code calling a function taking a list (the list
            # borrowed) that fails, and back)
            ("through_generic", 'fn g(xs: list[int], i: int) -> int { return xs[i]; }\nfn f(n: int) -> int { if n == 0 { return g([1, 2], 5); } return f(n - 1); }\nprint(f(4));\n'),
            # (a typed function failing after typed calls returned)
            ("after_calls", "fn h(n: int) -> int { return n; }\nfn f(n: int) -> int { let a = h(n) + h(n); return a + 9223372036854775807; }\nprint(f(3));\n"),
        ],
    )
    def test_typed_errors_have_their_stacks(self, name, src, capsys):
        # (typed entries keep the calls' depth in a register: the errors'
        # stacks and the depth limit as the reference mode has them)
        from test_modes import same_in_every_mode

        # (hot enough for typed entries: run a few times first)
        p = typed.lang.load(src, "prog")
        for _ in range(3):
            try:
                p.run(mode="compiled")
            except zrun.Error:
                pass
        capsys.readouterr()
        out, err = same_in_every_mode(typed.lang, src, capsys)
        assert err is not None
        message, _, stack, _ = err
        if name == "too_deep":
            assert "call stack too deep" in message and len(stack) > 100
        else:
            assert len(stack) >= 1

    def test_not_a_mapping(self):
        with pytest.raises(TypeError):
            typed.lang.types(3)
