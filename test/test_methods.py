"""Methods of records called where the record's class isn't known while
compiling (a variable's value): the class's function's compiled code, the
record self; staticmethods, classmethods and properties as Python finds
them. The same in every mode."""

import builtins

import zrun
from conftest import tiny
from test_modes import same_in_every_mode


class Box:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def get(self, k):
        return self.v + k

    @staticmethod
    def twice(k):
        return 2 * k

    @classmethod
    def name(cls, k):
        return k + 100

    @property
    def size(self):
        return self.v * 10


class Wide(Box):
    __slots__ = ("w",)

    def __init__(self, v):
        self.v = v
        self.w = v + 1

    def get(self, k):
        return self.w * 1000 + k


def make():
    lang = zrun.Language(tiny.PARSER, tiny.RULES)
    lang.function("FuncDef")
    lang.exec("While")(tiny.while_)
    lang.exec("If")(tiny.if_)
    lang.exec("Return")(tiny.return_)
    lang.exec(["Let", "Assign"])(tiny.assign)
    lang.eval("BinOp")(tiny.binop)

    @lang.eval("Call")
    def call(node, rt):
        name = node.name.text
        args = rt.eval(node.args)
        # (the record's class, a variable's value: known only at run time)
        if name == "box":
            return Box(args[0])
        if name == "wide":
            return Wide(args[0])
        if name == "get":
            return args[0].get(args[1])
        if name == "twice":
            return args[0].twice(args[1])
        if name == "named":
            return args[0].name(args[1])
        if name == "size":
            return args[0].size
        return rt.call(rt.eval(node.name), args)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


LANG = make()


def test_methods_of_records(capsys):
    # (the names declared for tiny's rules; the semantic takes the calls)
    src = """
fn box(v) {} fn wide(v) {} fn get(o, k) {} fn twice(o, k) {} fn named(o, k) {} fn size(o) {}
let b = box(3);
let w = wide(4);
let i = 0;
while i < 3 {
    print(get(b, i), get(w, i), twice(b, i), named(w, i), size(b));
    i = i + 1;
}
"""
    out, err = same_in_every_mode(LANG, src, capsys)
    assert err is None
    assert out.splitlines() == ["3 5000 0 100 30", "4 5001 2 101 30", "5 5002 4 102 30"]
    assert LANG.python_semantics() == {}
