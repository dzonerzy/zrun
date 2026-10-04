"""Block scopes: a closure keeps the variables of the block (or the loop
iteration) it was made in, in every mode."""

import builtins

import pytest
import zgram
import zrun
from test_modes import same_in_every_mode
from zrules import Rules, scopes

GRAMMAR = r"""
program     = ws (body:stmt ws)*                                     -> Program
@silent stmt = for_stmt | let_stmt | assign | return_stmt | expr_stmt
for_stmt    = 'for' kw ws var:ident ws 'in' kw ws count:expr ws body:block  -> For
block       = '{' ws (stmt ws)* '}'                                 -> Block
let_stmt    = 'let' kw ws name:ident ws '=' ws value:expr ws ';'     -> Let
assign      = name:ident ws '=' !'=' ws value:expr ws ';'            -> Assign
return_stmt = 'return' kw ws value:expr ws ';'                       -> Return
@silent expr_stmt = expr ws ';'

@left expr "expression" = left:term (ws op:addop ws right:term)*     -> BinOp
@silent term = fn | call | number | ident | '(' ws expr ws ')'
fn          = 'fn' kw ws '(' ws (params:ident (ws ',' ws params:ident)*)? ws ')' ws body:block  -> Fn
call        = callee:callee ws '(' ws (args:expr (ws ',' ws args:expr)*)? ws ')'  -> Call
@silent callee = ident | '(' ws expr ws ')'
number      = [0-9]+                                                  -> int
ident "name" = !keyword [a-z_] [a-z0-9_]*                              -> Name
addop       = [+\-]                                                   -> str
@silent keyword = ('for' | 'in' | 'let' | 'return' | 'fn') kw
@silent kw  = ![a-z0-9_]
@silent ws  = [ \t\n]*
"""

PARSER = zgram.compile(GRAMMAR)
RULES = Rules(
    PARSER,
    [
        scopes(
            scope=("Program", "Fn", "Block", "For"),
            define=("Let > .name", "Fn > .params", "For > .var"),
            use="Name",
            after="Let > .name",
            builtins=("print", "push", "list", "get", "size"),
        )
    ],
)


def make_lang():
    lang = zrun.Language(PARSER, RULES)
    lang.function("Fn", params="params", body="body", name=None, hoist=False)

    @lang.exec("For")
    def for_(node, rt):
        n = rt.eval(node.count)
        i = 0
        while i < n:
            # (a new loop variable each time round)
            rt.fresh(node)
            rt.store(node.var, i)
            if not rt.loop(node.body):
                break
            i = i + 1

    @lang.exec(["Let", "Assign"])
    def assign(node, rt):
        rt.store(node.name, rt.eval(node.value))

    @lang.exec("Return")
    def return_(node, rt):
        raise rt.Return(rt.eval(node.value))

    @lang.eval("Fn")
    def fn(node, rt):
        return rt.function(node)

    @lang.eval("Call")
    def call(node, rt):
        return rt.call(rt.eval(node.callee), rt.eval(node.args))

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        if node.op == "+":
            return a + b
        return a - b

    @lang.host
    def print(*args):
        builtins.print(*args)

    @lang.host
    def list():
        return []

    @lang.host
    def push(xs, x):
        xs.append(x)
        return xs

    @lang.host
    def get(xs, i):
        return xs[i]

    @lang.host
    def size(xs):
        return len(xs)

    return lang


lang = make_lang()

PROGRAMS = {
    # each closure keeps the iteration's variable
    "loop_variable": """
let fs = list();
for i in 3 { fs = push(fs, fn() { return i; }); }
print((get(fs, 0))(), (get(fs, 1))(), (get(fs, 2))());
""",
    # and a variable declared in the loop's body
    "body_variable": """
let fs = list();
for i in 3 { let x = i + 10; fs = push(fs, fn() { return x; }); }
print((get(fs, 0))(), (get(fs, 1))(), (get(fs, 2))());
""",
    # a counter shared by the closures made in one block
    "shared_in_block": """
let make = fn() {
    let n = 0;
    let inc = fn() { n = n + 1; return n; };
    let read = fn() { return n; };
    inc(); inc();
    return read;
};
let r = make();
print(r());
""",
    # nested blocks: each level's variables, from the innermost closure
    "nested": """
let fs = list();
for i in 2 {
    let a = i + 100;
    for j in 2 {
        let b = j + 10;
        fs = push(fs, fn() { return a + b + i + j; });
    }
}
print((get(fs, 0))(), (get(fs, 1))(), (get(fs, 2))(), (get(fs, 3))());
""",
    # a return from inside the loop (its scopes' frames left by the jump)
    "return_in_loop": """
let first = fn() {
    for i in 5 { let y = i + 7; return fn() { return y + i; }; }
    return fn() { return 0; };
};
let g = first();
print(g(), g());
""",
}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_block_scopes(name, capsys):
    out, err = same_in_every_mode(lang, PROGRAMS[name], capsys)
    assert err is None, err
    assert out


def test_expected_values(capsys):
    lang.load(PROGRAMS["loop_variable"]).run(mode="compiled")
    lang.load(PROGRAMS["nested"]).run(mode="compiled")
    # (a + b + i + j = 2i + 2j + 110)
    assert capsys.readouterr().out == "0 1 2\n110 112 112 114\n"
