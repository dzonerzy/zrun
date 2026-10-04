"""tiny, run: zgram parses it, zrules checks it, zrun runs it.

    python tiny.py fib.tiny

The language's meaning is the functions below, one per kind of node: an
interpreter, written the natural way. zrun runs it.
"""

import builtins
import sys

import zgram
import zrun
from zrules import Rules, forbid, inside, scopes

GRAMMAR = r"""
program     = ws (body:stmt ws)*                                      -> Program
@silent stmt = funcdef | while_stmt | if_stmt | return_stmt | break_stmt
             | let_stmt | assign | expr_stmt
funcdef     = 'fn' kw ws name:ident ws '(' ws (params:ident (ws ',' ws params:ident)*)? ws ')' ws body:block  -> FuncDef
block       = '{' ws (stmt ws)* '}'                                   -> list
while_stmt  = 'while' kw ws cond:expr ws body:block                   -> While
if_stmt     = 'if' kw ws cond:expr ws then:block (ws 'else' kw ws else_:block)?  -> If
return_stmt = 'return' kw (ws value:expr)? ws ';'                     -> Return
break_stmt  = 'break' kw ws ';'                                       -> Break()
let_stmt    = 'let' kw ws name:ident ws '=' ws value:expr ws ';'      -> Let
assign      = name:ident ws '=' !'=' ws value:expr ws ';'             -> Assign
@silent expr_stmt = expr ws ';'

@left expr "expression" = left:sum (ws op:cmpop ws right:sum)?        -> BinOp
@left sum  "expression" = left:term (ws op:addop ws right:term)*      -> BinOp
@left term "expression" = left:operand (ws op:mulop ws right:operand)*  -> BinOp
@silent operand = neg | primary
neg         = '-' ws operand:operand                                  -> Neg
@silent primary = number | string | call | ident | '(' ws expr ws ')'
call        = name:ident ws '(' ws (args:expr (ws ',' ws args:expr)*)? ws ')'  -> Call

number      = [0-9]+                                                  -> int
string      = '"' ('\\' . | [^"\\])* '"'                              -> unquote
ident "name"       = !keyword [a-zA-Z_] [a-zA-Z0-9_]*                 -> Name
cmpop "operator"   = '==' | '!=' | '<=' | '>=' | '<' | '>'            -> str
addop "operator"   = [+\-]                                            -> str
mulop "operator"   = [*/%]                                            -> str

@silent keyword = ('fn' | 'while' | 'if' | 'else' | 'return' | 'break' | 'let') kw
@silent kw      = ![a-zA-Z0-9_]
@silent ws      = ([ \t\n\r] | '#' [^\n]*)*
"""

PARSER = zgram.compile(GRAMMAR)

RULES = Rules(
    PARSER,
    [
        inside("Break", within="While", stop_at="FuncDef", code="break-outside-loop", message="'break' outside loop"),
        inside("Return", within="FuncDef", code="return-outside-function", message="'return' outside function"),
        forbid("FuncDef FuncDef", code="nested-function", message="functions cannot be defined inside functions"),
        scopes(
            scope=("Program", "FuncDef"),
            define=("Let > .name", "FuncDef > .params"),
            define_outer="FuncDef > .name",
            use="Name",
            hoist="FuncDef > .name",
            after="Let > .name",
            builtins=("print",),
        ),
    ],
)

lang = zrun.Language(PARSER, RULES)

# Functions: `fn name(params) { body }`, callable before their definition
lang.function("FuncDef")


@lang.exec("While")
def while_(node, rt):
    while rt.eval(node.cond):
        if not rt.loop(node.body):
            break


@lang.exec("If")
def if_(node, rt):
    if rt.eval(node.cond):
        rt.exec(node.then)
    elif node.else_ is not None:
        rt.exec(node.else_)


@lang.exec("Return")
def return_(node, rt):
    raise rt.Return(rt.eval(node.value) if node.value is not None else None)


@lang.exec("Break")
def break_(node, rt):
    raise rt.Break()


@lang.exec(["Let", "Assign"])
def assign(node, rt):
    rt.store(node.name, rt.eval(node.value))


@lang.eval("BinOp")
def binop(node, rt):
    a = rt.eval(node.left)
    b = rt.eval(node.right)
    op = node.op
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a // b
    if op == "%":
        return a % b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


@lang.eval("Neg")
def neg(node, rt):
    return -rt.eval(node.operand)


@lang.eval("Call")
def call(node, rt):
    return rt.call(rt.eval(node.name), rt.eval(node.args))


@lang.host
def print(*args):
    builtins.print(*args)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "fib.tiny"
    with open(path) as f:
        source = f.read()
    try:
        lang.load(source, path).run()
    except (zrun.LoadError, zrun.Error) as e:
        builtins.print(e, file=sys.stderr)
        sys.exit(1)
