"""Differential tests: every program runs in every mode, and the output and
the errors (message, place, call stack) must be the same."""

import pytest
import zrun
from conftest import tiny, typed

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
