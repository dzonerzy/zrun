"""f-strings in semantics: formatted natively where they can be (a str, an
int, no format spec or `d`), by Python otherwise; the same text in every
mode."""

import builtins

import zrun
from conftest import tiny
from test_modes import same_in_every_mode


def make():
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.function("FuncDef")
    lang.exec("While")(tiny.while_)
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.eval("BinOp")(tiny.binop)
    lang.eval("Neg")(tiny.neg)

    @lang.eval("Call")
    def call(node, rt):
        args = rt.eval(node.args)
        if node.name.text == "show":
            x = args[0]
            if isinstance(x, int) and not isinstance(x, bool):
                return f"<{x}|{x!s}|{x!r}|{x:d}|{x:>4}|{x:x}>"
            return f"<{x}|{x!s}|{x!r}>"
        return rt.call(rt.eval(node.name), args)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


LANG = make()


def test_formatted_the_same(capsys):
    src = """
fn show(x) {}
print(show(42));
print(show(-7));
print(show(9223372036854775807));
print(show("text"));
print(show(""));
print(show(1 < 2));
let i = 0;
while i < 3 { print(show(i * 1000)); i = i + 1; }
"""
    out, err = same_in_every_mode(LANG, src, capsys)
    assert err is None
    assert out.splitlines()[:6] == [
        "<42|42|42|42|  42|2a>",
        "<-7|-7|-7|-7|  -7|-7>",
        "<9223372036854775807|9223372036854775807|9223372036854775807|9223372036854775807|9223372036854775807|7fffffffffffffff>",
        "<text|text|'text'>",
        "<||''>",
        "<True|True|True>",
    ]
