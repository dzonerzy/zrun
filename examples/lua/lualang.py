"""A Lua 5.4 checker: the grammar in zgram, everything else in zrules.

(zrules' examples/lua/lua.py, vendored for zrun's Lua: the parameter list
is folded into the function body, so `funcbody > .names` are a function's
parameters for Language.function.)

What `luac` refuses beyond syntax, and what a linter warns about:

    errors    'break' outside a loop, '...' outside a vararg function,
              assignment to a <const> variable, 'goto' without a label,
              a label defined twice, a parameter named twice, an expression
              used as a statement, an assignment to something that can't be
              assigned to
    warnings  a local that is never used, a local that hides another one,
              a local read before it is given a value, unreachable code

    python lua.py file.lua [more.lua ...]
"""

import sys

import zgram
from zrules import Rules, flow, forbid, inside, scopes, unique

GRAMMAR = r"""
chunk       = shebang? ws block
@silent shebang = '#' [^\n]*
block       = (stmt ws)* (retstat ws)?
@silent stmt = ';' | label | break_stmt | goto_stmt | do_stmt | while_stmt | repeat_stmt | if_stmt
             | for_num | for_in | function_stmt | local_function | local_stmt | exprstat

label       = '::' ws name:label_name ws '::'
break_stmt  = 'break' kw
goto_stmt   = 'goto' kw ws name:label_name
do_stmt     = 'do' kw ws block 'end' kw
while_stmt  = 'while' kw ws cond:exp ws 'do' kw ws block 'end' kw
repeat_stmt = 'repeat' kw ws repeat_body
repeat_body = (stmt ws)* (retstat ws)? 'until' kw ws cond:exp
if_stmt     = 'if' kw ws cond:exp ws 'then' kw ws block
              ('elseif' kw ws cond:exp ws 'then' kw ws block)*
              ('else' kw ws else:block)? 'end' kw
for_num     = 'for' kw ws var:name ws '=' ws bounds:exp ws ',' ws bounds:exp (ws ',' ws bounds:exp)? ws 'do' kw ws block 'end' kw
for_in      = 'for' kw ws names:name (ws ',' ws names:name)* ws 'in' kw ws iter:explist ws 'do' kw ws block 'end' kw
function_stmt  = 'function' kw ws funcname ws funcbody
funcname    = var (ws '.' ws field_name)* (ws ':' ws method:field_name)?
local_function = 'local' kw ws 'function' kw ws name:name ws funcbody
local_stmt  = 'local' kw ws names:name (ws attrib)? (ws ',' ws names:name (ws attrib)?)* (ws '=' ws values:explist)?
attrib      = '<' ws field_name ws '>'
exprstat    = target:suffixed (ws ',' ws target:suffixed)* (ws '=' !'=' ws values:explist)?
retstat     = 'return' kw (ws explist)? (ws ';')?
explist     = exp (ws ',' ws exp)*

funcbody    = '(' ws parlist? ws ')' ws block 'end' kw
@silent parlist = names:name (ws ',' ws names:name)* (ws ',' ws dots)? | dots
dots        = '...'

@left exp        "expression" = left:and_exp (ws op:or_op ws right:and_exp)*
@left and_exp    "expression" = left:cmp_exp (ws op:and_op ws right:cmp_exp)*
@left cmp_exp    "expression" = left:bor_exp (ws op:cmp_op ws right:bor_exp)*
@left bor_exp    "expression" = left:bxor_exp (ws op:bor_op ws right:bxor_exp)*
@left bxor_exp   "expression" = left:band_exp (ws op:bxor_op ws right:band_exp)*
@left band_exp   "expression" = left:shift_exp (ws op:band_op ws right:shift_exp)*
@left shift_exp  "expression" = left:concat_exp (ws op:shift_op ws right:concat_exp)*
@right concat_exp "expression" = left:add_exp (ws op:concat_op ws right:add_exp)*
@left add_exp    "expression" = left:mul_exp (ws op:add_op ws right:mul_exp)*
@left mul_exp    "expression" = left:unary (ws op:mul_op ws right:unary)*
@silent unary = unop_exp | pow_exp
unop_exp    = op:unop ws operand:unary
@right pow_exp "expression" = left:simple (ws op:pow_op ws right:unary)*
@silent simple = nil | false | true | number | string | vararg | function_exp | table | suffixed

or_op       = 'or' kw
and_op      = 'and' kw
cmp_op      = '<=' | '>=' | '==' | '~=' | '<' !'<' | '>' !'>'
bor_op      = '|'
bxor_op     = '~' !'='
band_op     = '&'
shift_op    = '<<' | '>>'
concat_op   = '..' !'.'
add_op      = '+' | '-'
mul_op      = '//' | '*' | '/' | '%'
pow_op      = '^'
unop        = 'not' kw | '#' | '-' | '~' !'='

@postfix suffixed "expression" = target:atom (ws (call | method_call | index | member))*
@silent atom = var | paren
paren       = '(' ws exp ws ')'
call        = '(' ws (args:exp (ws ',' ws args:exp)*)? ws ')' | table | string
method_call = ':' ws method:field_name ws ('(' ws (args:exp (ws ',' ws args:exp)*)? ws ')' | table | string)
index       = '[' ![=\[] ws index:exp ws ']'
member      = '.' !'.' ws name:field_name

function_exp = 'function' kw ws funcbody
table       = '{' ws (field (ws [,;] ws field)* (ws [,;])?)? ws '}'
@silent field = keyed_field | named_field | exp
keyed_field = '[' ![=\[] ws key:exp ws ']' ws '=' ws value:exp
named_field = key:field_name ws '=' !'=' ws value:exp

nil         = 'nil' kw
false       = 'false' kw
true        = 'true' kw
vararg      = '...'
number      = '0' [xX] ([0-9a-fA-F]+ ('.' [0-9a-fA-F]*)? | '.' [0-9a-fA-F]+) ([pP] [+\-]? [0-9]+)?
            | ([0-9]+ ('.' [0-9]*)? | '.' [0-9]+) ([eE] [+\-]? [0-9]+)?
string      = '"' (escape | [^"\\\n])* '"' | '\'' (escape | [^'\\\n])* '\'' | long_bracket
@silent escape = '\\z' [ \t\r\n]* | '\\' .
# A PEG can't count the '=' of a long bracket: one with any is closed by the first `]=...=]`
@silent long_bracket = '[[' (!']]' .)* ']]' | '[' '='+ '[' (!long_close .)* long_close
@silent long_close = ']' '='+ ']'

var         "name" = ident
name        "name" = ident
field_name  "name" = [a-zA-Z_] [a-zA-Z0-9_]*
label_name  "name" = ident
@silent ident = !keyword [a-zA-Z_] [a-zA-Z0-9_]*
@silent keyword = ('and' | 'break' | 'do' | 'elseif' | 'else' | 'end' | 'false' | 'for' | 'function' | 'goto' | 'if' | 'in'
                | 'local' | 'nil' | 'not' | 'or' | 'repeat' | 'return' | 'then' | 'true' | 'until' | 'while') kw
@silent kw  = ![a-zA-Z0-9_]
@silent ws  = ([ \t\r\n] | '--' long_bracket | '--' [^\n]*)*
"""

PARSER = zgram.compile(GRAMMAR)

LOOPS = "while_stmt, for_num, for_in, repeat_stmt"

RULES = Rules(
    PARSER,
    [
        # ── What luac refuses ──
        inside("break_stmt", within=LOOPS, stop_at="funcbody", code="break-outside-loop", message="break outside a loop"),
        inside(
            "vararg",
            within="chunk, funcbody:has(> dots)",
            stop_at="funcbody",
            code="vararg-outside-vararg-function",
            message="cannot use '...' outside a vararg function",
        ),
        unique("funcbody > .names", code="duplicate-parameter", message="parameter '{text}' is named twice"),
        unique("label > .name", within="funcbody, chunk", code="duplicate-label", message="label '{text}' already defined"),
        forbid(
            "exprstat:not(:has(> .values)) > :not(call):not(method_call)",
            code="not-a-statement",
            message="syntax error: this expression is not a statement",
        ),
        forbid(
            "exprstat:has(> .values) > .target:not(var):not(index):not(member)",
            code="cannot-assign",
            message="cannot assign to this expression",
        ),
        # Labels are names of their own: a goto needs one in a block around it
        scopes(
            namespace="label",
            scope="chunk, block, repeat_body, funcbody",
            define="label > .name",
            use="goto_stmt > .name",
            hoist="label > .name",
            on_redefine="ignore",
            messages={"undefined": "no visible label '{text}' for goto"},
            codes={"undefined": "no-label"},
        ),
        # ── Names ──
        scopes(
            scope="chunk, block, repeat_body, funcbody, for_num, for_in",
            define="local_stmt > .names, funcbody > .names, for_num > .var, for_in > .names, local_function > .name",
            use="var",
            # `local x = x` reads the outer x; `local function f` can call itself
            after="local_stmt > .names",
            # the bounds of a for are worked out before its variable exists
            outside="for_num > .bounds, for_in > .iter",
            # a name that isn't a local is a global: nothing to report
            on_undefined="ignore",
            on_redefine="ignore",
            on_unused="warning",
            on_shadow="warning",
            messages={"unused": "unused variable '{text}'", "shadowed": "'{text}' shadows an outer local"},
            codes={"unused": "unused", "shadowed": "shadowing"},
        ),
        # ── Paths ──
        flow(
            namespace="name",
            sequences="block, repeat_body",
            functions="funcbody",
            branches="if_stmt",
            otherwise="if_stmt > .else",
            loops="while_stmt, for_num, for_in",
            forever="while_stmt:has(> true.cond)",
            at_least_once="repeat_stmt",
            exits="retstat",
            breaks="break_stmt",
            variables="local_stmt",
            assigns="exprstat:has(> .values)",
            labels={"name": "names", "value": "values"},
            on_unassigned="warning",
            messages={
                "unassigned": "'{text}' is read before it is given a value",
                "maybe_unassigned": "'{text}' may be read before it is given a value",
            },
            codes={"unassigned": "uninitialized", "maybe_unassigned": "uninitialized"},
        ),
    ],
)


@RULES.rule("exprstat > .target", code="assign-to-const")
def no_assignment_to_const(node, ctx):
    """`local x <const> = 1` ... `x = 2`: luac's "attempt to assign to const variable"."""
    symbol = ctx.resolve(node)
    if symbol is None or symbol.namespace != "name" or symbol.node is None:
        return
    # the attribute, if there is one, is the node right after the name
    if symbol.node + 1 >= len(ctx.tree):
        return
    attribute = ctx.tree.node(symbol.node + 1)
    if attribute.rule() == "attrib" and attribute.text().strip("<> \t") in ("const", "close"):
        ctx.error(node, f"attempt to assign to const variable '{node.text()}'")


def check(source):
    """The diagnostics of a Lua source text, in source order.

    Lua's own conventions are applied here, in plain Python: `_` and names
    starting with `_` are meant to be unused, and so is an unused `self`.
    """
    found = []
    for d in RULES.check(source):
        if d.code in ("unused", "shadowing"):
            name = d.message.split("'")[1]
            if name.startswith("_") or name == "self":
                continue
        found.append(d)
    return found


def main(paths):
    failed = False
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            source = f.read()
        try:
            diagnostics = check(source)
        except zgram.ParseError as e:
            print(e.diagnostic.render(source, path) if hasattr(e, "diagnostic") else f"{path}: {e}", file=sys.stderr)
            failed = True
            continue
        for d in diagnostics:
            print(d.render(source, path), file=sys.stderr)
            failed = failed or d.severity == "error"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
