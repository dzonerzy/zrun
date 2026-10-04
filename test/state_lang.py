"""A language keeping state at module level that its semantics change
(test_state.py): compiling a program makes it native (zrun's adopt.zig),
shared with Python through the module's names. Its own module: what's
adopted stays adopted for the process."""

import builtins
from dataclasses import dataclass

import zrun
from conftest import tiny
from zrules import Rules, scopes

BUILTINS = ("print", "reset", "put", "get", "held", "bykey", "box", "lit", "kinds", "callk")


class Env:
    __slots__ = ("vars", "log", "parent")

    def __init__(self):
        self.vars = {}
        self.log = []
        self.parent = None


# a cycle, a second name: one object, adopted with what it reaches
ENV = Env()
ENV.parent = ENV
ALIAS = ENV
COUNTS = {"calls": 0}

# held by something else too (a tuple): stays Python's
HELD = []
HOLDERS = (HELD,)


@dataclass(frozen=True)
class Key:
    k: int


# a key native dicts can't hash as Python does: stays Python's
BY_KEY = {Key(1): "one"}


@dataclass
class Box:
    v: int


# an attribute beyond its fields: stays Python's
BOX = Box(1)
BOX.extra = 5


def make_lang():
    rules = Rules(
        tiny.PARSER,
        [
            scopes(
                scope=("Program", "FuncDef"),
                define=("Let > .name", "FuncDef > .params"),
                define_outer="FuncDef > .name",
                use="Name",
                hoist="FuncDef > .name",
                after="Let > .name",
                builtins=BUILTINS,
            )
        ],
    )
    lang = zrun.Language(tiny.PARSER, rules)
    lang.exec("While")(tiny.while_)
    lang.exec("If")(tiny.if_)
    lang.eval("BinOp")(tiny.binop)
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.function("FuncDef")

    @lang.eval("Call")
    def call(node, rt):
        name = node.name.text
        args = rt.eval(node.args)
        COUNTS["calls"] += 1
        if name == "reset":
            ENV.vars.clear()
            ENV.log.clear()
            COUNTS["calls"] = 0
            HELD.clear()
            BOX.v = 1
            return 0
        if name == "put":
            ENV.vars["k" + str(args[0])] = args[1]
            ENV.log.append(args[0])
            return len(ENV.vars)
        if name == "get":
            return ENV.parent.vars.get("k" + str(args[0]), -1)
        if name == "held":
            HELD.append(args[0])
            return len(HELD)
        if name == "bykey":
            return len(BY_KEY.get(Key(args[0]), ""))
        if name == "box":
            BOX.v = BOX.v + args[0]
            return BOX.v
        if name == "lit":
            # (strs and ints of the code's own, kept after the program)
            ALIAS.vars["lit"] = "a literal str"
            ALIAS.vars["big"] = 2**70 + 1
            return 0
        if name == "kinds":
            # (a node's kind, kept after the program)
            ENV.vars["kind"] = node.kind
            return 0
        if name == "callk":
            # (a function kept there: another program's, maybe)
            return rt.call(ENV.vars["k" + str(args[0])], args[1:])
        return rt.call(rt.load(node.name), args)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


lang = make_lang()
