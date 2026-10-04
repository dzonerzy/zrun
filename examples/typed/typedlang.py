"""The typed language of zrules' examples/typed (vendored): its grammar and
rules. typed.py gives it semantics."""

import zgram
from zrules import Rules, flow, forbid, inside, scopes, types

GRAMMAR = r"""
program    = ws (stmt ws)*
@silent stmt = import_stmt | from_stmt | struct_def | funcdef | if_stmt | while_stmt | loop_stmt | do_stmt | return_stmt | break_stmt | continue_stmt | let_stmt | assign | expr_stmt
import_stmt = 'import' kw ws module:ident ws ';'
from_stmt   = 'from' kw ws module:ident ws 'import' kw ws names:ident (ws ',' ws names:ident)* ws ';'
struct_def = 'struct' kw ws name:ident ws '{' ws (field ws)* (funcdef ws)* '}'
field      = name:ident ws ':' ws type:type_expr ws ';'
funcdef    = 'fn' kw ws name:ident ws '(' ws (params:param (ws ',' ws params:param)*)? ws ')' (ws '->' ws returns:type_expr)? ws block
param      = name:ident (ws ':' ws type:type_expr)?
block      = '{' ws (stmt ws)* '}'
if_stmt    = 'if' kw ws cond:expr ws block (ws 'else' kw ws (if_stmt | block))?
while_stmt = 'while' kw ws cond:expr ws block
loop_stmt  = 'loop' kw ws block
do_stmt    = 'do' kw ws block ws 'while' kw ws cond:expr ws ';'
break_stmt = 'break' kw ws ';'
continue_stmt = 'continue' kw ws ';'
return_stmt = 'return' kw (ws value:expr)? ws ';'
let_stmt   = 'let' kw ws name:ident (ws ':' ws type:type_expr)? (ws '=' ws value:expr)? ws ';'
assign     = target:postfix ws '=' !'=' ws value:expr ws ';'
@silent expr_stmt = expr ws ';'

@postfix type_expr = target:simple_type optional_type*
optional_type = '?'
@silent simple_type = generic_type | type_name
generic_type = base:type_ident '[' ws args:type_expr (ws ',' ws args:type_expr)* ws ']'
type_ident = [a-zA-Z_]+
type_name  = [a-zA-Z_] [a-zA-Z0-9_]*

@left expr  = left:sum (ws op:cmpop ws right:sum)?
@left sum   = left:term (ws op:addop ws right:term)*
@left term  = left:unary (ws op:mulop ws right:unary)*
@silent unary = neg | not_expr | postfix
neg        = '-' ws operand:unary
not_expr   = 'not' kw ws operand:unary
@postfix postfix = target:primary (call_args | index_op | member)*
call_args  = '(' ws (args:expr (ws ',' ws args:expr)*)? ws ')'
index_op   = '[' ws index:expr ws ']'
member     = '.' name:ident
@silent primary = float_lit | int_lit | string | bool_lit | nil_lit | list_lit | ident | '(' ws expr ws ')'
list_lit   = '[' ws (items:expr (ws ',' ws items:expr)*)? ws ']'
float_lit  = [0-9]+ '.' [0-9]+
int_lit    = [0-9]+
string     = '"' [^"]* '"'
bool_lit   = ('true' | 'false') kw
nil_lit    = 'nil' kw
ident      = !keyword [a-zA-Z_] [a-zA-Z0-9_]*
cmpop      = '==' | '!=' | '<=' | '>=' | '<' | '>'
addop      = [+\-]
mulop      = [*/%]
@silent keyword = ('import' | 'from' | 'struct' | 'fn' | 'if' | 'else' | 'while' | 'loop' | 'do' | 'break' | 'continue' | 'return' | 'let' | 'not' | 'true' | 'false' | 'nil') kw
@silent kw = ![a-zA-Z0-9_]
@silent ws = ([ \t\n] | '#' [^\n]*)*
"""

PARSER = zgram.compile(GRAMMAR)

NUMERIC = [("int", "int", "int"), ("float", "float", "float")]
COMPARE = [("int", "int", "bool"), ("float", "float", "bool"), ("str", "str", "bool")]

SCOPES = dict(
    scope=("program", "funcdef", "struct_def"),
    define=("let_stmt > .name", "param > .name", "field > .name"),
    define_outer=("funcdef > .name", "struct_def > .name"),
    use="ident, type_name",
    hoist=("funcdef > .name", "struct_def > .name"),
    after="let_stmt > .name",
    members="member",
    imports="import_stmt, from_stmt",
    builtins=("print", "len"),
)

TYPES = dict(
    basic=("int", "float", "str", "bool", "void", "nil"),
    coerce={"int": "float"},
    literals={"int_lit": "int", "float_lit": "float", "string": "str", "bool_lit": "bool", "nil_lit": "nil"},
    containers={"list_lit": "list"},
    type_names="type_name",
    type_args="generic_type",
    optional="optional_type",
    variables="let_stmt, param, field",
    functions="funcdef",
    structs="struct_def",
    binary="expr, sum, term",
    unary="neg, not_expr",
    calls="call_args",
    index="index_op",
    assigns="assign",
    returns="return_stmt",
    conditions="if_stmt > .cond, while_stmt > .cond, do_stmt > .cond",
    operators={
        "+": NUMERIC + [("str", "str", "str")],
        "-": NUMERIC + [("int", "int"), ("float", "float")],
        "*": NUMERIC,
        "/": [("float", "float", "float")],
        "%": [("int", "int", "int")],
        "==": [("T", "T", "bool")],
        "!=": [("T", "T", "bool")],
        "<": COMPARE,
        "<=": COMPARE,
        ">": COMPARE,
        ">=": COMPARE,
        "not": [("bool", "bool")],
    },
    builtins={"print": "fn(...) -> void", "len": "fn(any) -> int"},
)


FLOW = dict(
    sequences="program, block",
    functions="funcdef",
    branches="if_stmt",
    otherwise="if_stmt > block + block",
    loops="while_stmt",
    forever="loop_stmt",
    at_least_once="do_stmt",
    exits="return_stmt",
    breaks="break_stmt",
    continues="continue_stmt",
    must_return="funcdef:has(> .returns):not([returns=void])",
    variables="let_stmt",
    assigns="assign",
)


LOOPS = "while_stmt, loop_stmt, do_stmt"

STRUCTURE = [
    inside("break_stmt", within=LOOPS, stop_at="funcdef", code="break-outside-loop", message="'break' outside a loop"),
    inside("continue_stmt", within=LOOPS, stop_at="funcdef", code="continue-outside-loop", message="'continue' outside a loop"),
    inside("return_stmt", within="funcdef", code="return-outside-function", message="'return' outside a function"),
    forbid("funcdef struct_def", code="nested-struct", message="a struct can't be declared inside a function"),
]

RULES = Rules(PARSER, STRUCTURE + [scopes(**SCOPES), types(**TYPES), flow(**FLOW)])

