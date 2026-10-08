"""The program's variables no function refers to, kept on the stack while
its code runs (compile.zig: Gen's mirrors): the frame has them wherever
code that sees the frames runs, and when the program ends."""

import builtins

import pytest
import zrun
from conftest import tiny
from test_modes import same_in_every_mode


def _lang(python=()):
    """tiny, the semantics in `python` run as Python (they see the
    variables through the frames)."""
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.function("FuncDef")
    for kind, fn in (("While", tiny.while_), ("If", tiny.if_), ("Return", tiny.return_), ("Break", tiny.break_)):
        lang.exec(kind)(fn)
    lang.exec(["Let", "Assign"], native="Assign" not in python)(tiny.assign)
    for kind, fn in (("BinOp", tiny.binop), ("Call", tiny.call)):
        lang.eval(kind)(fn)
    lang.eval("Neg", native="Neg" not in python)(tiny.neg)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


LOOP = "let s = 0;\nlet i = 0;\nwhile i < 6 { s = s + i; i = i + 1; print(-s); }\nprint(s, i);\n"


@pytest.mark.parametrize("python", [(), ("Neg",), ("Assign",), ("Neg", "Assign")])
def test_python_semantics_see_them(python, capsys):
    # (a semantic run as Python reads them in a loop; one assigning them,
    # its stores are what the compiled code reads after)
    out, err = same_in_every_mode(_lang(python), LOOP, capsys)
    assert err is None
    assert out.splitlines()[-1] == "15 6"


def test_a_function_that_refers_to_one_still_sees_it(capsys):
    src = "let s = 0;\nlet seen = 0;\nfn look() { return seen; }\nlet i = 0;\nwhile i < 4 { s = s + i; seen = s; i = i + 1; }\nprint(s, look());\n"
    out, err = same_in_every_mode(_lang(), src, capsys)
    assert err is None and out == "6 6\n"


def test_code_out_of_line_sees_them(capsys):
    # (combine(), too big to inline, runs out of line in the frames (every
    # binary operator goes through it there); its Neg runs as Python: both
    # read the loop's variables)
    from test_modes import _common_case_lang

    src = 'let s = "";\nlet n = 0;\nwhile n < 3 { s = s + "ab"; n = n + 1; print(-n); }\nprint(s, n);\n'
    out, err = same_in_every_mode(_common_case_lang(), src, capsys)
    assert err is None


def test_an_error_in_the_loop(capsys):
    src = "let n = 0;\nwhile n < 10 { n = n + 1; if n == 4 { print(n / 0); } }\n"
    out, err = same_in_every_mode(_lang(), src, capsys)
    assert err is not None


def test_the_program_frame_after_a_run():
    p = _lang().load("let k = 0;\nwhile k < 3 { k = k + 1; }\nfn get() { return 7; }\n")
    p.run(mode="compiled")
    assert p.call("get") == 7
