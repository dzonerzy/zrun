"""rt.parse: text parsed with a zgram grammar while the program runs, its
tree read as the program's: the same in every mode."""

import pytest
import zgram
import zrun
from zrules import Rules

# The data: JSON, its rules labelled, an action or two
JSON = zgram.compile(r"""
value   = ws (object | array | string | number | lit) ws
object  = '{' ws (members:pair (ws ',' ws members:pair)*)? ws '}'        -> Object
pair    = key:string ws ':' value:value                                   -> Pair
array   = '[' ws (items:value (',' items:value)*)? ws ']'                 -> Array
string  = '"' ('\\' . | [^"\\])* '"'
number  = '-'? [0-9]+ ('.' [0-9]+)?
lit     = 'true' | 'false' | 'null'
@silent ws = [ \t\n]*
""")

# The program: statements naming what to do with a string of JSON
GRAMMAR = r"""
program = ws (body:stmt ws)*                         -> Program
@silent stmt = sum | keys | bad | rec | depth
sum     = 'sum' ws text:string ws ';'                -> Sum
keys    = 'keys' ws text:string ws ';'               -> Keys
bad     = 'bad' ws text:string ws ';'                -> Bad
rec     = 'rec' ws text:string ws ';'                -> Rec
depth   = 'depth' ws text:string ws ';'              -> Depth
string  = '"' ('\\' . | [^"\\])* '"'                 -> unquote
@silent ws = [ \t\n]*
"""
PARSER = zgram.compile(GRAMMAR)
lang = zrun.Language(PARSER, Rules(PARSER, []))


def total(n):
    """The sum of the numbers in a JSON tree."""
    if n.kind == "number":
        return float(n.text)
    s = 0.0
    for c in n.children:
        s += total(c)
    return s


def deepest(n):
    d = 0
    for c in n.children:
        k = deepest(c)
        if k > d:
            d = k
    return d + 1


@lang.exec("Program")
def program(node, rt):
    for s in node.body:
        rt.exec(s)


@lang.exec("Sum")
def sum_(node, rt):
    print(total(rt.parse(JSON, node.text)))


@lang.exec("Keys")
def keys(node, rt):
    root = rt.parse(JSON, node.text)
    obj = root.children[0]
    print(obj.kind, [p.key.text for p in obj.members], [p.value.text.strip() for p in obj.members])
    print(root.rule, root.start, root.end, len(root.children), obj.parent.kind)


@lang.exec("Bad")
def bad(node, rt):
    try:
        rt.parse(JSON, node.text)
        print("parsed")
    except zgram.ParseError as e:
        print("error", e.line, e.column, e.offset, e.message)


@lang.exec("Rec")
def rec(node, rt):
    root, errors = rt.parse(JSON, node.text, recover=True)
    print([e[3] for e in errors], [c.kind for c in root.children[0].children])


@lang.exec("Depth")
def depth(node, rt):
    print(deepest(rt.parse(JSON, node.text, start="value")))


SOURCE = r"""
sum "[1, 2, {\"a\": [3.5, -4]}, 10]";
keys "{\"x\": 1, \"y\": [true, null]}";
bad "[1, 2,, 3]";
bad "{\"a\" 1}";
rec "[1,, 2, {,}]";
depth "[[[[1]]], [2]]";
"""

EXPECTED = """12.5
Object ['"x"', '"y"'] ['1', '[true, null]']
value 0 27 1 value
error 1 7 6 expected value
error 1 6 5 expected ':'
['expected value', "expected pair or '}'"] ['value', '<error>', 'value', 'value']
10
"""


@pytest.mark.parametrize("mode", ["python", "compiled"])
def test_parse_in_every_mode(mode, capsys):
    lang.load(SOURCE, "data.prog").run(mode=mode)
    assert capsys.readouterr().out == EXPECTED


def test_compiled_without_python(capsys):
    program = lang.load(SOURCE, "data.prog")
    program.run(mode="compiled", report=True)
    assert program.report()["python_crossings"] == {}
    assert lang.python_semantics() == {}


# Fields whose values the grammar's actions make: the same values in every
# mode, natively in compiled code
ACTIONS = zgram.compile(r"""
record  = ws 'rec' ws name:name ws n:int ws f:float ws s:qstr ws flags:flags ws pairs:pairs ws first:firstof ws
name    = [a-z]+                                       -> str
int     = '-'? [0-9_]+                                 -> int
float   = '-'? [0-9]+ '.' [0-9]+ ('e' '-'? [0-9]+)?    -> float
qstr    = '"' ('\\' . | [^"\\])* '"'                   -> unquote
flags   = '[' flag* ']'                                -> tuple
@silent flag = ws (yes | no | nil)
yes     = 'yes'                                        -> True
no      = 'no'                                         -> False
nil     = 'nil'                                        -> None
pairs   = '{' (ws pair)* ws '}'                        -> dict
pair    = name ws '=' ws int                           -> tuple
firstof = '<' ws int (ws int)* ws '>'                  -> first
@silent ws = [ \t\n]*
""")

AGRAMMAR = r"""
program = ws (body:stmt ws)*                         -> Program
stmt    = 'show' ws text:string ws ';'               -> Show
string  = '"' ('\\' . | [^"\\])* '"'                 -> unquote
@silent ws = [ \t\n]*
"""
APARSER = zgram.compile(AGRAMMAR)
alang = zrun.Language(APARSER, Rules(APARSER, []))


@alang.exec("Program")
def aprogram(node, rt):
    for s in node.body:
        rt.exec(s)


@alang.exec("Show")
def show(node, rt):
    r = rt.parse(ACTIONS, node.text)
    print(r.name, r.n, r.n + 1, r.f, r.s, len(r.s), r.flags, r.pairs, r.first)


ASOURCE = r"""
show "rec abc 1_000 2.5e-3 \"a\\tb\\u00e9\\ud83d\\ude00\" [yes no nil] {x = 1 y = -2} <7 8 9>";
show "rec z -42 0.1 \"\" [] {} <0>";
"""


@pytest.mark.parametrize("mode", ["python", "compiled"])
def test_actions(mode, capsys):
    alang.load(ASOURCE, "a.prog").run(mode=mode)
    assert capsys.readouterr().out == (
        "abc 1000 1001 0.0025 a\tbé😀 5 (True, False, None) {'x': 1, 'y': -2} 7\n"
        "z -42 -41 0.1  0 () {} 0\n")


def test_actions_without_python(capsys):
    program = alang.load(ASOURCE, "a.prog")
    program.run(mode="compiled", report=True)
    assert program.report()["python_crossings"] == {}
