"""The typed language of zrules' examples, run by zrun: structs with fields
and methods, functions, optionals, lists.

    python typed.py shapes.ty

typedlang.py is the grammar and the rules (zrules checks names, types and
flow before anything runs); below is what each kind of node does.
"""

import builtins
import sys
from dataclasses import dataclass

import zrun
from typedlang import PARSER, RULES

lang = zrun.Language(PARSER, RULES)

# Functions and methods: `fn name(params) -> type { block }`
lang.function("funcdef", params="params", body="block", name="name")


@dataclass
class Struct:
    """A struct type: its name, fields (name, type) and methods."""

    name: str
    fields: list
    methods: dict


@dataclass
class Instance:
    struct: Struct
    values: dict


# What the values of the language's types are (zrules works out each node's
# type; compiled code knows the kind of its value then). Not said: float
# (an int is a float where one's expected, unconverted: either), functions'
# types (a builtin's value is a Python function, a fn's a zrun.Function).
SCALARS = {"int": int, "bool": bool, "str": str, "nil": type(None)}


def value_type(t):
    if t in SCALARS:
        return SCALARS[t]
    if t.startswith("list["):
        return list
    if t.startswith("type["):
        return Struct
    if t[:1].isupper() and t.isidentifier():
        return Instance
    return None


lang.types(value_type)


# ----------------------------------------------------------------------
# Statements
# ----------------------------------------------------------------------


@lang.exec("struct_def")
def struct_def(node, rt):
    fields = []
    methods = {}
    for child in node.children:
        if child.kind == "field":
            fields.append((child.name.text, child.type.text))
        elif child.kind == "funcdef":
            methods[child.name.text] = rt.function(child)
    rt.store(node.name, Struct(node.name.text, fields, methods))


@lang.exec(["import_stmt", "from_stmt"])
def import_(node, rt):
    rt.error(node, "imports need a project: run single files")


@lang.exec("let_stmt")
def let(node, rt):
    if node.value is not None:
        rt.store(node.name, coerce(rt.eval(node.value), node.type))


@lang.exec("assign")
def assign(node, rt):
    value = rt.eval(node.value)
    target = node.target
    if target.kind == "member":
        rt.eval(target.target).values[target.name.text] = value
    elif target.kind == "index_op":
        rt.eval(target.target)[rt.eval(target.index)] = value
    elif is_field(target, rt):
        rt.receiver.values[target.text] = value
    else:
        rt.store(target, value)


@lang.exec("if_stmt")
def if_(node, rt):
    branches = node.children  # cond, block, then a block or an if_stmt
    if rt.eval(node.cond):
        rt.exec(branches[1])
    elif len(branches) > 2:
        rt.exec(branches[2])


@lang.exec("while_stmt")
def while_(node, rt):
    body = node.children[1]
    while rt.eval(node.cond):
        if not rt.loop(body):
            break


@lang.exec("loop_stmt")
def loop(node, rt):
    while rt.loop(node.children[0]):
        pass


@lang.exec("do_stmt")
def do(node, rt):
    body = node.children[0]
    while rt.loop(body) and rt.eval(node.cond):
        pass


@lang.exec("return_stmt")
def return_(node, rt):
    raise rt.Return(rt.eval(node.value) if node.value is not None else None)


@lang.exec("break_stmt")
def break_(node, rt):
    raise rt.Break()


@lang.exec("continue_stmt")
def continue_(node, rt):
    raise rt.Continue()


# ----------------------------------------------------------------------
# Expressions
# ----------------------------------------------------------------------


@lang.eval(["expr", "sum", "term"])
def binary(node, rt):
    a = rt.eval(node.left)
    b = rt.eval(node.right)
    op = node.op.text
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
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


@lang.eval("neg")
def neg(node, rt):
    return -rt.eval(node.operand)


@lang.eval("not_expr")
def not_(node, rt):
    return not rt.eval(node.operand)


@lang.eval("call_args")
def call(node, rt):
    args = rt.eval(node.args)
    callee = node.target
    if callee.kind == "member":
        # obj.method(args): called on obj
        obj = rt.eval(callee.target)
        return rt.call(obj.struct.methods[callee.name.text], args, receiver=obj)
    f = rt.eval(callee)
    if isinstance(f, Struct):
        return Instance(f, {name: coerce_text(value, t) for (name, t), value in zip(f.fields, args)})
    return rt.call(f, args)


@lang.eval("member")
def member(node, rt):
    obj = rt.eval(node.target)
    name = node.name.text
    if name in obj.values:
        return obj.values[name]
    return obj.struct.methods[name]


@lang.eval("index_op")
def index(node, rt):
    return rt.eval(node.target)[rt.eval(node.index)]


@lang.eval("list_lit")
def list_(node, rt):
    return rt.eval(node.items)


@lang.eval("ident")
def ident(node, rt):
    # a struct's field, inside one of its methods
    if is_field(node, rt):
        return rt.receiver.values[node.text]
    return rt.load(node)


@lang.eval("int_lit")
def int_lit(node, rt):
    return int(node.text)


@lang.eval("float_lit")
def float_lit(node, rt):
    return float(node.text)


@lang.eval("string")
def string(node, rt):
    return node.text[1:-1]


@lang.eval("bool_lit")
def bool_lit(node, rt):
    return node.text == "true"


@lang.eval("nil_lit")
def nil(node, rt):
    return None


def is_field(name, rt):
    scope = rt.scope(name)
    return scope is not None and scope.kind == "struct_def" and rt.receiver is not None


def coerce(value, type_node):
    """An int where a float is declared becomes a float."""
    return coerce_text(value, type_node.text if type_node is not None else None)


def coerce_text(value, type_text):
    if type_text in ("float", "float?") and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


# ----------------------------------------------------------------------
# Builtins
# ----------------------------------------------------------------------


@lang.host
def print(*args):
    builtins.print(*("nil" if a is None else a for a in args))


@lang.host
def len(value):
    return builtins.len(value)


if __name__ == "__main__":
    path = sys.argv[1] if builtins.len(sys.argv) > 1 else "shapes.ty"
    with open(path) as f:
        source = f.read()
    try:
        lang.load(source, path).run()
    except (zrun.LoadError, zrun.Error) as e:
        builtins.print(e, file=sys.stderr)
        sys.exit(1)
