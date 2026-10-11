"""References compiled code holds and gives up: known lists holding known
containers or run-time values, made at run time or let go of, a value in
a known tuple unrolled over, a parsed node kept. Each shape called many
times, inline (a helper the semantic names) and as a function a table
holds (compiled when it's first called): the same results as the
reference mode, nothing left allocated (zrun._blocks()), nothing freed
twice."""

import pytest
import zgram
import zrun
from conftest import tiny


def nested_appended(s):
    # (a known list holding a known one, changed in run-time control flow:
    # made at run time, the inner one with it)
    fields = [[]]
    for c in s:
        fields.append(c)
    return len(fields)


def nested_dict(s):
    fields = [{}]
    for c in s:
        fields.append(c)
    return len(fields)


def inner_appended(s):
    # (the last read of a variable holding a known list: the list a
    # temporary, its inner list's items given up with it)
    fields = [[]]
    fields[0].append(s)
    return len(fields[0])


def inner_split(s):
    fields = [[]]
    for c in s:
        if c == " ":
            fields.append([])
            continue
        fields[len(fields) - 1].append(c)
    return [len(f) for f in fields]


def unpacked(s):
    # (a known tuple's item unrolled over: the target's own reference)
    roots = [(s, 0), (s + "!", 1)]
    n = 0
    for a, b in roots:
        n += len(a) + b
    return n


JSON = zgram.compile(r"""
value   = ws (array | number) ws
array   = '[' ws (items:value (',' items:value)*)? ws ']'
number  = [0-9]+
@silent ws = [ \t\n]*
""")


def nodes_kept(rt, s):
    # (a parsed tree's nodes kept past the list they came in)
    root = rt.parse(JSON, s)
    kept = [(c, c.text) for c in root.children[0].children]
    out = []
    for node, text in kept:
        out.append(node.rule + ":" + text.strip())
    return out


SHAPES = [nested_appended, nested_dict, inner_appended, inner_split, unpacked]

# (a table the module changes: its functions called as values, each
# compiled when it's first called)
TABLE = {}
TABLE.update({f.__name__: f for f in SHAPES})
PARSED = {}
PARSED.update({"nodes_kept": nodes_kept})


def make(which, by_table):
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=True)
    lang.function("FuncDef")
    lang.exec("Return")(tiny.return_)
    f = SHAPES[which] if which < len(SHAPES) else nodes_kept
    name = f.__name__

    if f is nodes_kept:
        if by_table:
            @lang.eval("Call")
            def call(node, rt):
                return PARSED[name](rt, rt.eval(node.args[0]))
        else:
            @lang.eval("Call")
            def call(node, rt):
                return nodes_kept(rt, rt.eval(node.args[0]))
    elif by_table:
        @lang.eval("Call")
        def call(node, rt):
            return TABLE[name](rt.eval(node.args[0]))
    else:
        @lang.eval("Call")
        def call(node, rt):
            return f(rt.eval(node.args[0]))

    return lang.load("fn f(s) { return print(s); }", "refs.tiny")


@pytest.mark.parametrize("by_table", [False, True])
@pytest.mark.parametrize("which", range(len(SHAPES) + 1))
def test_nothing_kept_nothing_freed_twice(which, by_table):
    program = make(which, by_table)
    arg = "[1, [2], 3]" if which == len(SHAPES) else "ab cd e"
    want = program.call("f", arg, mode="python")
    for _ in range(50):
        assert program.call("f", arg) == want
    zrun.collect()
    before = zrun._blocks()
    for _ in range(500):
        assert program.call("f", arg) == want
    zrun.collect()
    assert zrun._blocks() == before
