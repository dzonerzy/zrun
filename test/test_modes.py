"""Differential tests: every program runs in every mode, and the output and
the errors (message, place, call stack) must be the same."""

import builtins
import os

import pytest
import zrun
from conftest import HERE, tiny, typed

MODES = ["python", "compiled"]


def outcome(lang, source, mode, capsys):
    """What running a program does: (output, error or None)."""
    program = lang.load(source, "prog")
    try:
        program.run(mode=mode)
        err = None
    except zrun.Error as e:
        d = e.diagnostic
        err = (d.message, d.span, [(name, n.span) for name, n in e.stack], str(e))
    return capsys.readouterr().out, err


def same_in_every_mode(lang, source, capsys):
    results = {m: outcome(lang, source, m, capsys) for m in MODES}
    assert results["compiled"] == results["python"], results
    return results["python"]


TINY = {
    "fib": """
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
let i = 0;
while i < 15 { print(fib(i)); i = i + 1; }
""",
    "break": "let i = 0;\nwhile 1 { if i == 3 { break; } print(i); i = i + 1; }\nprint(99);\n",
    "hoisted": "print(twice(4));\nfn twice(x) { return x * 2; }\n",
    "globals": "let count = 0;\nfn bump() { count = count + 1; }\nbump(); bump();\nprint(count);\n",
    "frames": "fn f(n) { let x = n; if n > 0 { f(n - 1); } print(x); }\nf(3);\n",
    "strings": 'let s = "a" + "b";\nprint(s, s + "c");\n',
    "no_return_value": "fn f() { return; }\nprint(f());\n",
    "arith": "print(7 / 2, 7 % 3, -7 / 2, -7 % 3, 2 * 3 - 4, 10 - 2 - 3);\n",
    "compare": "print(1 < 2, 2 <= 1, 3 == 3, 3 != 3, 1 > 0, 0 >= 1);\n",
    "neg": "let x = 5;\nprint(-x, - -x);\n",
    "nested_calls": "fn add(a, b) { return a + b; }\nfn mul(a, b) { return a * b; }\nprint(add(mul(2, 3), add(1, mul(4, 5))));\n",
    "loop_in_function": "fn sum(n) { let s = 0; let i = 0; while i < n { s = s + i; i = i + 1; } return s; }\nprint(sum(100));\n",
    "shadow_params": "let n = 10;\nfn f(n) { return n + 1; }\nprint(f(1), n);\n",
    "early_return_in_loop": "fn first(n) { let i = 0; while 1 { if i * i > n { return i; } i = i + 1; } }\nprint(first(50));\n",
}

TINY_ERRORS = {
    "div_zero": "fn div(a, b) { return a / b; }\nfn go(x) { return div(x, 0); }\nprint(go(1));\n",
    "overflow": "let x = 4611686018427387904;\nlet y = x * 2;\n",
    "overflow_add": "let x = 9223372036854775807;\nprint(x + 1);\n",
    "type_error": 'let x = "a" - 1;\n',
    "concat_error": 'let x = "a" + 1;\n',
    "compare_error": 'print("a" < 1);\n',
    "arity": "fn f(a) { return a; }\nf(1, 2);\n",
    "not_callable": "let x = 1;\nx();\n",
    "too_deep": "fn f(n) { return f(n + 1); }\nf(0);\n",
    "deep_in_loop": "fn f(n) { let i = 0; while i < 3 { if n > 2 { return 1 / 0; } f(n + 1); i = i + 1; } }\nf(0);\n",
    "big_literal": "let x = 99999999999999999999;\n",
}


TYPED = {
    "methods": """
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
""",
    "loops": """
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
""",
    "coerce": "let x: float = 2;\nprint(x);\n",
    "lists": "let xs = [1, 2, 3];\nxs[1] = 20;\nlet i = 0;\nlet s = 0;\nwhile i < len(xs) { s = s + xs[i]; i = i + 1; }\nprint(xs, s);\n",
    "optional": "fn first(v: list[float]) -> float? { if len(v) == 0 { return nil; } return v[0]; }\nlet a: float? = first([]);\nlet b: float? = first([2.5]);\nprint(a, b);\n",
    "struct_fields": "struct P { x: float; y: float; }\nlet p = P(1, 2);\np.x = 5.0;\nprint(p.x + p.y);\n",
    "nested_structs": """
struct V { x: int; y: int;
    fn add(o: V) -> V { return V(x + o.x, y + o.y); }
}
fn total(vs: list[V]) -> V {
    let t = V(0, 0);
    let i = 0;
    while i < len(vs) { t = t.add(vs[i]); i = i + 1; }
    return t;
}
let r = total([V(1, 2), V(3, 4), V(5, 6)]);
print(r.x, r.y);
""",
    "recursion": "fn fact(n: int) -> int { if n <= 1 { return 1; } return n * fact(n - 1); }\nprint(fact(20));\n",
    "strings": 'fn greet(name: str) -> str { return "hi " + name; }\nprint(greet("bob"));\n',
}

TYPED_ERRORS = {
    "index": "let xs = [1, 2];\nprint(xs[5]);\n",
    "overflow": "let x = 9223372036854775807;\nlet y = x + 1;\n",
    "div_zero": "fn f(a: int) -> int { return a % 0; }\nprint(f(1));\n",
    "deep": "fn f(n: int) -> int { return f(n + 1); }\nprint(f(0));\n",
}


def test_typed_shapes(capsys):
    with open(os.path.join(HERE, "..", "examples", "typed", "shapes.ty")) as f:
        out, err = same_in_every_mode(typed.lang, f.read(), capsys)
    assert err is None and out == "somewhere else 4.0\n3.0\n"


@pytest.mark.parametrize("name", sorted(TYPED))
def test_typed(name, capsys):
    out, err = same_in_every_mode(typed.lang, TYPED[name], capsys)
    assert err is None and out


@pytest.mark.parametrize("name", sorted(TYPED_ERRORS))
def test_typed_errors(name, capsys):
    out, err = same_in_every_mode(typed.lang, TYPED_ERRORS[name], capsys)
    assert err is not None


@pytest.mark.parametrize("name", sorted(TINY))
def test_tiny(name, capsys):
    out, err = same_in_every_mode(tiny.lang, TINY[name], capsys)
    assert err is None and out


@pytest.mark.parametrize("name", sorted(TINY_ERRORS))
def test_tiny_errors(name, capsys):
    out, err = same_in_every_mode(tiny.lang, TINY_ERRORS[name], capsys)
    assert err is not None


def test_a_host_function_failing(capsys):
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.eval("Call")(tiny.call)
    lang.function("FuncDef")

    @lang.host
    def print(*args):
        raise ValueError("no output")

    _, err = same_in_every_mode(lang, "print(1);\n", capsys)
    assert err[0] == "print: ValueError: no output"


def _python_lang():
    """tiny with calls whose semantics use Python's loops, comprehensions,
    unpacking and item assignment on values known only at run time."""
    from zrules import Rules, scopes

    rules = Rules(
        tiny.PARSER,
        [
            scopes(
                scope=("Program", "FuncDef"),
                define=("Let > .name", "FuncDef > .params"),
                define_outer="FuncDef > .name",
                use="Name",
                hoist="FuncDef > .name",
                after="Let > .name",
                builtins=("print", "nums", "total", "evens", "grid", "table", "swap", "setat", "bump", "grow", "pairs", "indexed", "pairs_whole", "triples", "has", "text"),
            ),
        ],
    )
    lang = zrun.Language(tiny.PARSER, rules)
    for kind in ("While", "If", "Return", "Break", "BinOp", "Neg"):
        fn = {"While": tiny.while_, "If": tiny.if_, "Return": tiny.return_, "Break": tiny.break_, "BinOp": tiny.binop, "Neg": tiny.neg}[kind]
        (lang.exec if kind in ("While", "If", "Return", "Break") else lang.eval)(kind)(fn)
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.function("FuncDef")

    @lang.eval("Call")
    def call(node, rt):
        name = node.name.text
        args = rt.eval(node.args)
        if name == "total":
            s = 0
            for x in args[0]:
                if x < 0:
                    break
                s += x
            else:
                s += 1000
            return s
        if name == "evens":
            return [x * 10 for x in args[0] if x % 2 == 0]
        if name == "grid":
            return [a * b for a in args[0] for b in args[1] if a != b]
        if name == "table":
            return {k: v for k, v in zip(args[0], args[1])}
        if name == "swap":
            a, b = args[0]
            return [b, a]
        if name == "setat":
            xs = args[0]
            xs[args[1]] = args[2]
            return xs
        if name == "grow":
            xs = args[0]
            for x in xs:
                if x < 5:
                    xs.append(x + 3)
            return xs
        if name == "pairs":
            return [a * b for a, b in zip(args[0], args[1])]
        if name == "indexed":
            s = 0
            for i, x in enumerate(args[0]):
                s += i * x
            return s
        if name == "pairs_whole":
            return [p for p in zip(args[0], args[1])]
        if name == "triples":
            return [a for a, b, c in zip(args[0], args[1])]
        if name == "has":
            return [args[0] in args[1], args[0] not in args[1]]
        if name == "bump":
            xs = args[0]
            xs[0] += args[1]
            return xs
        return rt.call(rt.eval(node.name), args)

    @lang.host
    def nums(n):
        return list(range(n))

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


python_lang = _python_lang()

PYTHON_FEATURES = {
    "for": "print(total(nums(5)));\n",
    "for_break": "let xs = nums(4);\nprint(total(xs), total(evens(xs)));\n",
    "comp": "print(evens(nums(7)));\n",
    "comp_nested": "print(grid(nums(3), nums(4)));\n",
    "dict_comp": "print(table(nums(3), evens(nums(6))));\n",
    "unpack": "print(swap(nums(2)));\n",
    "setitem": "print(setat(nums(3), 1, 9), setat(nums(3), -1, 7));\n",
    "aug_item": "print(bump(nums(2), 5));\n",
    "grow": "print(grow(nums(3)));\n",
    "zip": "print(pairs(nums(4), nums(3)), pairs_whole(nums(2), evens(nums(4))));\n",
    "enumerate": "print(indexed(nums(5)));\n",
    "in": 'print(has(2, nums(3)), has(5, nums(3)), has("b", "abc"), has(1, table(nums(2), nums(2))));\n',
    "in_function": "fn f(n) { let s = 0; let i = 0; while i < n { s = s + total(nums(i)); i = i + 1; } return s; }\nprint(f(6));\n",
}

PYTHON_FEATURE_ERRORS = {
    "unpack_many": "print(swap(nums(3)));\n",
    "unpack_few": "print(swap(nums(1)));\n",
    "unpack_int": "print(swap(5));\n",
    "setitem_range": "print(setat(nums(2), 5, 1));\n",
    "zip_width": "print(triples(nums(2), nums(2)));\n",
    "in_int": "print(has(1, 2));\n",
    "for_over_int": "print(total(3));\n",
}


@pytest.mark.parametrize("name", sorted(PYTHON_FEATURES))
def test_python_features(name, capsys):
    out, err = same_in_every_mode(python_lang, PYTHON_FEATURES[name], capsys)
    assert err is None and out


@pytest.mark.parametrize("name", sorted(PYTHON_FEATURE_ERRORS))
def test_python_feature_errors(name, capsys):
    out, err = same_in_every_mode(python_lang, PYTHON_FEATURE_ERRORS[name], capsys)
    assert err is not None
