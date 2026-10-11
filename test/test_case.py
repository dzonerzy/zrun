"""str.lower(), str.upper() and repr() natively (src/unicode.zig, made from
Python's Unicode data by tools/gen_unicode.py): the same as Python's for
every character, the final sigma's context, quotes and escapes, and strings
mixing them; in a strict language (no Python)."""

import random
import unicodedata

import pytest
import zrun
from conftest import tiny


def cases(s):
    return [s.lower(), s.upper(), repr(s), repr([s, 1, 2.5, None, True])]


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
GENERATED = open(__file__.replace("test/test_case.py", "src/unicode.zig"), encoding="utf-8").read()
SAME_UNICODE = f"(Unicode\n//! {unicodedata.unidata_version})" in GENERATED
other_unicode = pytest.mark.skipif(not SAME_UNICODE, reason="unicode.zig made from another Unicode version")


@other_unicode
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


QUOTES = ["", "'", '"', "'\"", "a'b", 'a"b', "\\", "\t\n\r\x00\x1f\x7f", "\x85\xa0\xad", "​ ﻿",
          "\U0001f600\U000e0001", "é'"]


def test_quotes_and_escapes(program):
    for s in QUOTES:
        assert program.call("f", s) == cases(s), repr(s)


@other_unicode
def test_mixed(program):
    rng = random.Random(7)
    pool = "aZ ._'\"-\\\t́­ΣσςİıßŉǰΐﬀﬃǅǈÉéДдჀⴀꭜ𐐀𐐨\U0001e900\x85​" + \
        "".join(chr(rng.randrange(0x80, 0x3000)) for _ in range(200))
    for _ in range(2000):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))
        assert program.call("f", s) == cases(s), s
