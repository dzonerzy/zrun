"""str.lower() and str.upper() natively (src/casemap.zig, made from
Python's Unicode data by tools/gen_casemap.py): the same as Python's for
every character, the final sigma's context, and strings mixing them; in a
strict language (no Python)."""

import random
import unicodedata

import pytest
import zrun
from conftest import tiny


def cases(s):
    return [s.lower(), s.upper()]


def make():
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=True)
    lang.function("FuncDef")
    lang.exec("Return")(tiny.return_)

    @lang.eval("Call")
    def call(node, rt):
        return cases(rt.eval(node.args[0]))

    return lang.load("fn f(s) { return print(s); }", "case.tiny")


@pytest.fixture(scope="module")
def program():
    return make()


# (the tables are this Python's: another's Unicode may map a few more)
GENERATED = open(__file__.replace("test/test_case.py", "src/casemap.zig"), encoding="utf-8").read()
SAME_UNICODE = f"(Unicode\n//! {unicodedata.unidata_version})" in GENERATED


@pytest.mark.skipif(not SAME_UNICODE, reason="casemap.zig made from another Unicode version")
def test_every_character(program):
    chars = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    for k in range(0, len(chars), 4096):
        s = "".join(chars[k : k + 4096])
        assert program.call("f", s) == cases(s), hex(k)


SIGMAS = ["Σ", "AΣ", "AΣB", "AΣ.", "A'Σ", "A'Σ'", "ΑΣ", "ΑΣ ", "ΟΔΥΣΣΕΥΣ", "áΣ", "aΣ́", "aΣ́b",
          "1Σ", "Σa", " Σ ", "ἈΣ", "ÉΣ!", "ǅΣ", "ʰΣ", "AΣ­", "AΣ­b", "ꭜΣ"]


def test_final_sigma(program):
    for s in SIGMAS:
        assert program.call("f", s) == cases(s), s


@pytest.mark.skipif(not SAME_UNICODE, reason="casemap.zig made from another Unicode version")
def test_mixed(program):
    rng = random.Random(7)
    pool = "aZ ._'-́­ΣσςİıßŉǰΐﬀﬃǅǈÉéДдჀⴀꭜ𐐀𐐨\U0001e900" + "".join(chr(rng.randrange(0x80, 0x3000)) for _ in range(200))
    for _ in range(2000):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))
        assert program.call("f", s) == cases(s), s
