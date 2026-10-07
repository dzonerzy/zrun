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
    # (a function's name bound to another function: called as that one)
    "rebound_function": "fn a(x) { return x + 1; }\nfn b(x) { return x * 2; }\nprint(a(1));\na = b;\nprint(a(3));\n",
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
    "rebound_to_a_value": "fn a(x) { return x; }\nprint(a(1));\na = 5;\na(2);\n",
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


def _keeping_lang():
    """tiny whose assignments read the variable's old value, store the new
    one, then use the old one: a value read while its variable changes."""
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    for kind, fn in (("While", tiny.while_), ("If", tiny.if_), ("Return", tiny.return_)):
        lang.exec(kind)(fn)
    for kind, fn in (("BinOp", tiny.binop), ("Call", tiny.call)):
        lang.eval(kind)(fn)
    lang.function("FuncDef")

    @lang.exec(["Let", "Assign"])
    def assign(node, rt):
        if node.kind == "Let":
            rt.store(node.name, rt.eval(node.value))
            return
        old = rt.load(node.name)
        rt.store(node.name, rt.eval(node.value))
        rt.store(node.name, rt.load(node.name) + old)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


KEEPING = {
    # (each a function's: its variables on the stack, read borrowed; the
    # strings made at run time, the variable's the only reference)
    "straight": 'fn f(a) { let s = a + "b"; s = a + "d"; s = s + "!"; print(s); }\nf("x"); f("y");\n',
    "in_a_loop": 'fn g(n) { let s = "a" + "b"; let i = 0; while i < n { if i % 2 == 0 { s = s + "x"; } i = i + 1; } print(s); }\ng(5);\n',
    "returned": 'fn h(a) { let s = a + "1"; s = a + "2"; return s; }\nprint(h("p"), h("q"));\n',
}


@pytest.mark.parametrize("name", sorted(KEEPING))
def test_value_kept_while_its_variable_changes(name, capsys):
    out, err = same_in_every_mode(_keeping_lang(), KEEPING[name], capsys)
    assert err is None and out


# Loops reading values borrowed (a list's items, a record's fields, values
# kept while their variable changes), run n times
BORROWING_LOOPS = {
    "typed": (
        typed.lang,
        "struct P {{ x: int; }}\n"
        "fn f(n: int) -> int {{ let xs = [1, 2, 3]; let p = P(4); let t = 0; let i = 0;\n"
        "  while i < n {{ t = t + xs[i % 3] + p.x; if i % 2 == 0 {{ xs = [i, i, i]; p = P(i); }} i = i + 1; }} return t; }}\n"
        "print(f({}));\n",
    ),
    "keeping": (_keeping_lang(), 'fn g(n) {{ let s = "a" + "b"; let i = 0; while i < n {{ s = s + "x"; i = i + 1; }} print(s); }}\ng({});\n'),
}


@pytest.mark.parametrize("name", sorted(BORROWING_LOOPS))
def test_borrowing_gives_back_what_it_takes(name, capsys):
    # what a run keeps doesn't grow with the iterations (a run keeps its
    # program's functions now: their cycle with its frame)
    lang, src = BORROWING_LOOPS[name]
    kept = []
    for n in (10, 40):
        program = lang.load(src.format(n), "prog")
        program.run(mode="compiled")
        before = zrun._blocks()
        program.run(mode="compiled")
        kept.append(zrun._blocks() - before)
    capsys.readouterr()
    assert kept[0] == kept[1] >= 0


def _host_lang():
    """tiny with host functions called from compiled code by their own
    compiled code (kept at the call site): one adding (its ints I64s, as
    rt.call hands them over: overflow checked), one raising, one compiled
    code can't run (Python runs it)."""
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
                builtins=("print", "add", "picky", "opened", "count", "item"),
            )
        ],
    )
    lang = zrun.Language(tiny.PARSER, rules)
    lang.function("FuncDef")
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.exec("Return")(tiny.return_)
    lang.exec("While")(tiny.while_)
    lang.eval("Call")(tiny.call)
    lang.eval("BinOp")(tiny.binop)

    @lang.host
    def print(*args):
        builtins.print(*args)

    @lang.host
    def add(a, b):
        return a + b

    @lang.host
    def picky(x):
        if x < 0:
            raise ValueError("negative: %d" % x)
        return x * 2

    @lang.host
    def count(xs):
        return len(xs)

    @lang.host
    def item(xs, i):
        return xs[i]

    @lang.host
    def opened(n):
        with open(os.devnull) as f:
            f.read()
        return n + 1

    return lang


HOSTS = {
    "loop": "let i = 0; let t = 0;\nwhile i < 300 { t = add(t, picky(i)) + opened(i); i = i + 1; }\nprint(t);\n",
    "overflow": "fn f(n) { return add(n, n); }\nprint(f(4611686018427387904));\n",
    "raises": "fn f(n) { return picky(n); }\nprint(f(3));\nprint(f(0 - 1));\n",
    "python_runs_it": "let i = 0;\nwhile i < 3 { print(opened(i)); i = i + 1; }\n",
}


@pytest.mark.parametrize("name", sorted(HOSTS))
def test_host_functions(name, capsys):
    out, err = same_in_every_mode(_host_lang(), HOSTS[name], capsys)
    if name == "overflow":
        assert err is not None and "overflow" in err[0]
    elif name == "raises":
        assert out == "6\n" and "negative: -1" in err[0]
    else:
        assert err is None and out


def test_host_calls_dont_take_the_gil():
    # (a call site knowing its host function's code calls it directly:
    # nothing of Python's, no GIL taken, once the site has it)
    p = _host_lang().load("fn f(n) { let i = 0; let t = 0; while i < n { t = add(t, picky(i)); i = i + 1; } return t; }\n", "prog")
    assert p.call("f", 10) == 90
    taken = p.report()["gil_taken"]
    assert p.call("f", 1000) == 999000
    assert p.report()["gil_taken"] == taken


PY_DATA = [
    [1, 2, 3],
    (4, 5, -6),
    [2**63 - 1, -(2**63)],
    [2**64, "s", None, [7, 8], 1.5],
    {"a": 1, "b": 2},
    [],
]


@pytest.mark.parametrize("data", PY_DATA, ids=lambda d: type(d).__name__ + str(len(d)))
def test_python_data_items(data):
    # (Python's lists and tuples read by compiled code item by item, their
    # len() natively: as the reference mode reads them, errors too)
    p = _host_lang().load("fn at(xs, i) { return item(xs, i); }\nfn size(xs) { return count(xs); }\n", "prog")
    keys = ["a", "z"] if isinstance(data, dict) else [0, 1, -1, len(data) - 1, len(data), -len(data) - 1]
    for name, args in [("at", (data, k)) for k in keys] + [("size", (data,))]:
        got = {}
        for mode in MODES:
            try:
                got[mode] = ("ok", p.call(name, *args, mode=mode))
            except zrun.Error as e:
                got[mode] = ("error", e.diagnostic.message)
        assert got["compiled"] == got["python"], (name, args, got)
        # (a host function's failure as Python words its exception)
        if args[1:] == ("z",):
            assert got["python"] == ("error", "item: KeyError: 'z'")


def _wrapping_lang():
    """tiny whose + - * wrap around at 64 bits (rt.wrapping_add...), and
    whose / % < are the 64-bit shifts << >> >>> (rt.wrapping_shl...)."""
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.function("FuncDef")
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.exec("Return")(tiny.return_)
    lang.eval("Call")(tiny.call)

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        if node.op == "+":
            return rt.wrapping_add(a, b)
        if node.op == "-":
            return rt.wrapping_sub(a, b)
        if node.op == "/":
            return rt.wrapping_shl(a, b)
        if node.op == "%":
            return rt.wrapping_shr(a, b)
        if node.op == "<":
            return rt.wrapping_ushr(a, b)
        return rt.wrapping_mul(a, b)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


WRAPPING = {
    "known": "print(9223372036854775807 + 1, 3 * 9223372036854775807, 0 - 9223372036854775807 - 2);\n",
    "run_time": "fn f(a, b) { return a * b + a - b; }\nprint(f(9223372036854775807, 3), f(5, 7), f(0 - 4611686018427387904, 2));\n",
    "not_ints": 'fn f(a, b) { return a + b; }\nprint(f("a", 1));\n',
    # (shifts: known and at run time; negative values; by 63, 64, 65, -1)
    "shifts_known": "print(1 / 63, 3 / 62, 0 - 8 % 1, 0 - 8 < 60, 1 / 64, 1 / 65, 5 % (0 - 1), 0 - 1 < 0 - 1);\n",
    "shifts_run_time": "fn f(a, n) { return (a / n) + (a % n) + (a < n); }\nprint(f(0 - 12345, 3), f(9223372036854775807, 1), f(1, 64), f(0 - 1, 63), f(7, 0 - 2));\n",
    "shift_not_ints": 'fn f(a, n) { return a < n; }\nprint(f(1, "x"));\n',
}

# (1 << 63; 3 << 62; -(8 >> 1); -8 >>> 60; by 64 and 65: by 0 and 1; by -1:
# by 63; -1 >>> 63)
SHIFTS_KNOWN = [-(2**63), -(2**62), -4, 15, 1, 2, 0, 1]


@pytest.mark.parametrize("name", sorted(WRAPPING))
def test_wrapping_arithmetic(name, capsys):
    out, err = same_in_every_mode(_wrapping_lang(), WRAPPING[name], capsys)
    if name == "not_ints":
        assert err[0] == "rt.wrapping_add() takes ints of 64 bits"
    elif name == "shift_not_ints":
        assert err[0] == "rt.wrapping_ushr() takes ints of 64 bits"
    else:
        assert err is None and out
    if name == "shifts_known":
        assert out == " ".join(str(x) for x in SHIFTS_KNOWN) + "\n"


def combine(a, b):
    """A helper too big to run inline whose first `if` is its common case
    (ints): that runs inline, the rest out of line."""
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(a, str) and isinstance(b, str):
        joined = a + b
        if len(joined) > 10:
            return joined[:5] + "..." + joined[-5:]
        return joined
    if isinstance(a, str):
        return a + str(b) + "/" + str(len(a)) + "/" + str(b) + "/" + str(len(str(b)))
    if isinstance(b, str):
        return str(a) + b + "/" + str(len(b)) + "/" + str(a) + "/" + str(len(str(a)))
    return [a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b, a, b]


def _common_case_lang():
    """tiny whose + goes through combine(), with a semantic run as Python
    (Neg: the variables in frames, helpers too big run out of line)."""
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.function("FuncDef")
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.exec("Return")(tiny.return_)
    lang.eval("Call")(tiny.call)
    lang.eval("Neg", native=False)(tiny.neg)

    @lang.eval("BinOp")
    def binop(node, rt):
        return combine(rt.eval(node.left), rt.eval(node.right))

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


COMMON_CASE = {
    "ints": "fn f(a, b) { return a + b + 1; }\nprint(f(2, 3), f(-4, 4));\n",
    "the_rest": 'fn f(a, b) { return a + b; }\nprint(f("ab", "cd"), f("abcdefgh", "ijklmn"), f("x", 7), f(7, "x"), f(1, 2));\n',
}


@pytest.mark.parametrize("name", sorted(COMMON_CASE))
def test_common_case_inline(name, capsys):
    out, err = same_in_every_mode(_common_case_lang(), COMMON_CASE[name], capsys)
    assert err is None and out


def store_twice(rt, node, value):
    """Big: its common case (a negative int) inline, the rest out of line,
    which reads the variable, stores the value, then the old one joined to
    it."""
    if isinstance(value, int) and value < 0:
        return rt.store(node.name, value)
    old = rt.load(node.name)
    rt.store(node.name, value)
    if isinstance(old, str) and isinstance(value, str):
        joined = value + "<" + old
        if len(joined) > 40:
            joined = joined[:20] + "~" + joined[-19:]
        rt.store(node.name, joined)
    elif isinstance(old, int) and isinstance(value, int):
        rt.store(node.name, value * 100 + old % 100 + len(str(value)) * 0 + len(str(old)) * 0)
    else:
        rt.store(node.name, value)


def _spill_lang():
    """tiny whose assignments run a helper out of line that reads and
    stores the function's variables (on the stack: moved into a frame for
    the call, back after)."""
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.function("FuncDef")
    lang.exec("Return")(tiny.return_)
    lang.exec("While")(tiny.while_)
    lang.eval("Call")(tiny.call)
    lang.eval("BinOp")(tiny.binop)

    @lang.exec(["Let", "Assign"])
    def assign(node, rt):
        if node.kind == "Let":
            rt.store(node.name, rt.eval(node.value))
        else:
            store_twice(rt, node, rt.eval(node.value))

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


SPILL = {
    "strs": 'fn f(a) { let s = a + "1"; let t = s; s = a + "2"; s = s + "3"; print(s, t); return s; }\nprint(f("x"));\n',
    "ints_in_a_loop": "fn g(n) { let i = 0; let acc = 7; while i < n { acc = i + 1; i = i + 1; } return acc; }\nprint(g(5), g(0));\n",
    "both_paths": "fn h(n) { let a = 5; a = 0 - n; let b = a; a = n; return a + b; }\nprint(h(3), h(0 - 2));\n",
}


@pytest.mark.parametrize("name", sorted(SPILL))
def test_variables_moved_into_a_frame(name, capsys):
    out, err = same_in_every_mode(_spill_lang(), SPILL[name], capsys)
    assert err is None and out


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
