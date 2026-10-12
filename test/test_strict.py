"""Language(strict=True): every semantic compiled and the code calling
nothing in Python. A semantic outside the compilable subset (or marked
native=False) is a CompileError when it's registered; code that would call
into Python (a Python function, a Python object as a value, isinstance() of
any class, a semantic run as Python...) a CompileError when the program
compiles, saying where in the semantic and why; what only shows when the
code runs (a host function in Python) a StrictError, saying where in the
program. Errors are still Python's exceptions; code compiled while the
program runs is allowed."""

import functools
import importlib.util
import os
from dataclasses import dataclass

import pytest
import zrun
from conftest import HERE

TINY = os.path.join(HERE, "..", "examples", "tiny", "tiny.py")


def _strict_tiny():
    # (tiny.py's language, made strict: its module loaded again with
    # zrun.Language(strict=True))
    spec = importlib.util.spec_from_file_location("tiny_strict", TINY)
    module = importlib.util.module_from_spec(spec)
    plain = zrun.Language
    zrun.Language = functools.partial(plain, strict=True)
    try:
        spec.loader.exec_module(module)
    finally:
        zrun.Language = plain
    return module


tiny = _strict_tiny()

FIB = """
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
let x = fib(20);
"""


def test_native_code_runs():
    p = tiny.lang.load(FIB, "fib.tiny")
    p.run(mode="compiled")
    assert p.call("fib", 20) == 6765
    assert p.map("fib", [10, 15, 20]) == [55, 610, 6765]
    # (the reference mode is Python's, strict or not)
    p.run(mode="python")


def kw_print(*args, **kw):
    print(*args, **kw)


def test_a_host_function_in_python_is_an_error_where_it_runs():
    # (a print taking **kwargs: not compiled, run in Python)
    own = _strict_tiny()
    own.lang.host("print", kw_print)
    src = FIB + "print(x);\n"
    p = own.lang.load(src, "fib.tiny")
    with pytest.raises(zrun.StrictError) as e:
        p.run(mode="compiled")
    assert "fib.tiny:7:1:" in str(e.value) and "calling kw_print()" in str(e.value)
    assert issubclass(zrun.StrictError, zrun.CompileError)


def _language(strict):
    lang = zrun.Language(tiny.PARSER, tiny.RULES, strict=strict)
    lang.function("FuncDef")

    @lang.exec(["Let", "Assign"])
    def assign(node, rt):
        rt.store(node.name, rt.eval(node.value))

    @lang.exec("Return")
    def return_(node, rt):
        raise rt.Return(rt.eval(node.value))

    return lang


def test_a_python_call_refused_when_compiling():
    for strict in (True, False):
        lang = _language(strict)

        @lang.eval("BinOp")
        def binop(node, rt):
            # (a builtin compiled code calls in Python)
            return rt.eval(node.left) + rt.eval(node.right) + os.getpid() * 0

        p = lang.load("let x = 1 + 2;", "add.tiny")
        if strict:
            with pytest.raises(zrun.CompileError) as e:
                p.run(mode="compiled")
            assert "in binop(): strict: calls the Python function getpid()" in str(e.value)
            assert "test_strict.py:" in str(e.value)
        else:
            # (not strict: the call made in Python, as before)
            p.run(mode="compiled")


def test_a_semantic_outside_the_subset_refused_when_registered():
    lang = _language(True)
    with pytest.raises(zrun.CompileError):

        @lang.eval("BinOp")
        def binop(node, rt):
            class Inner:
                pass

            return 0

    with pytest.raises(zrun.CompileError) as e:

        @lang.eval("Neg", native=False)
        def neg(node, rt):
            return -rt.eval(node.operand)

    assert "native=False" in str(e.value)


def test_strict_is_part_of_the_definition(tmp_path):
    # (a module compiled by the language not strict isn't the strict one's)
    lax = importlib.util.module_from_spec(importlib.util.spec_from_file_location("tiny_lax", TINY))
    lax.__spec__.loader.exec_module(lax)
    path = str(tmp_path / "fib.zrc")
    lax.lang.compile(FIB, path)
    with pytest.raises(Exception, match="definition"):
        tiny.lang.load_compiled(path)


def produce(rt, node):
    # (a helper raising the jump: caught by the semantic calling it)
    raise rt.Return(rt.eval(node.left) * 10)


def test_a_jump_caught_by_its_value():
    # `except rt.Return as r` using r.args only: the jump's value, natively
    for strict in (True, False):
        lang = _language(strict)

        @lang.eval("BinOp")
        def binop(node, rt):
            try:
                produce(rt, node)
            except rt.Return as r:
                got = r.args[0]
            return got + rt.eval(node.right)

        p = lang.load("let x = 4 + 2;\nfn f(a, b) { return a + b; }", "jump.tiny")
        p.run(mode="compiled")
        assert p.call("f", 4, 2) == 42


def test_a_jump_caught_as_an_exception():
    # (its name used otherwise: Python's exception, as before; not strict)
    lang = _language(False)

    @lang.eval("BinOp")
    def binop(node, rt):
        try:
            produce(rt, node)
        except rt.Return as r:
            e = r
        return e.args[0] + rt.eval(node.right)

    p = lang.load("fn f(a, b) { return a + b; }", "jump.tiny")
    p.run(mode="compiled")
    assert p.call("f", 4, 2) == 42


def test_type_of_a_known_value():
    # `type(v) is float` with v known: decided while compiling
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        v = 2.5
        if type(v) is float and type(v) is not int:
            return rt.eval(node.left) + rt.eval(node.right)
        return 0

    p = lang.load("let x = 40 + 2;\nfn f(a, b) { return a + b; }", "type.tiny")
    p.run(mode="compiled")
    assert p.call("f", 40, 2) == 42
    assert p.call("f", 40, 2, mode="python") == 42


def same_type(a, b):
    return type(a) is type(b)


def test_type_of_a_known_value_and_one_of_run_time():
    # `type(None) is type(b)`, b known only at run time: both types' keys
    # compared natively, no class object made
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        return [same_type(None, b), same_type(b, 1.5), type(a) is not type(None), same_type(a, b)]

    p = lang.load("fn f(a, b) { return a + b; }", "type.tiny")
    for a, b in [(1, 2), (1, 2.5), (1, "x"), (1.5, 2.5)]:
        assert p.call("f", a, b) == p.call("f", a, b, mode="python")


@dataclass
class Command:
    name: str
    options: set
    roles: dict
    sub: dict


# (a table only read holding records, sets, tables: native all through in
# a strict language, a constant)
COMMANDS = {
    "git": Command("git", {"-c", "-C"}, {"-C": "cwd"}, {"log": Command("log", {"-n"}, {}, {})}),
    "cp": Command("cp", {"-t"}, {"-t": "write"}, {}),
}


def test_a_module_table_of_records_only_read():
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        name = rt.eval(node.left)
        opt = rt.eval(node.right)
        if name not in COMMANDS:
            return None
        c = COMMANDS[name]
        subs = sorted([k for k, v in c.sub.items()])
        return [c.name, opt in c.options, c.roles.get(opt, "plain"), subs, len(COMMANDS)]

    p = lang.load("fn f(a, b) { return a + b; }", "records.tiny")
    for a, b in [("git", "-C"), ("git", "-x"), ("cp", "-t"), ("rm", "-f")]:
        assert p.call("f", a, b) == p.call("f", a, b, mode="python")


def pairs(d):
    out = []
    for k, v in d.items():
        out.append(k + "=" + v)
    return out


def test_items_of_a_known_dict():
    # (a dict known while compiling, given to a helper: its items unrolled,
    # its values' references its own)
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        return pairs({"a": a, "b": b}) + [k for k in {"x": a}.keys()] + [v for k, v in {"y": b}.items()]

    p = lang.load("fn f(a, b) { return a + b; }", "items.tiny")
    assert p.call("f", "1", "2") == p.call("f", "1", "2", mode="python") == ["a=1", "b=2", "x", "2"]


KEYWORDS = {"if", "else", "while", 3, (1, "x")}


def test_a_module_set_only_read():
    # a module's set the semantics only read (`in`, iteration): native, a
    # constant of the code, as its lists and dicts are
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        n = 0
        for k in KEYWORDS:
            if k == b:
                n += 1
        return [a in KEYWORDS, b not in KEYWORDS, n, len(KEYWORDS)]

    p = lang.load("fn f(a, b) { return a + b; }", "set.tiny")
    for a, b in [("if", "x"), (3, "while"), ((1, "x"), 4), ("", 3)]:
        assert p.call("f", a, b) == p.call("f", a, b, mode="python")


@dataclass
class Finding:
    kind: str
    text: str


def add_once(out, kind, text):
    for f in out:
        if f.kind == kind and f.text == text:
            return
    out.append(Finding(kind, text))


def test_a_known_list_changed_after_a_return_from_a_loop():
    # a helper returning from inside a loop on some paths, the list it was
    # given changed after (on the others): the caller's list is the run's,
    # whichever path ran (it was the change's on all: a value LLVM rejected)
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        out = [Finding("x", a)]
        add_once(out, "x", b)
        add_once(out, "y", b)
        add_once(out, "y", b)
        return [f.kind + "=" + f.text for f in out]

    p = lang.load("fn f(a, b) { return a + b; }", "once.tiny")
    for a, b in [("1", "1"), ("1", "2")]:
        assert p.call("f", a, b) == p.call("f", a, b, mode="python")
    assert p.call("f", "1", "2") == ["x=1", "x=2", "y=2"]


def joined(d):
    out = ""
    for k in d:
        out += k
    return out


def logged(text, entry):
    try:
        n = int(text)
        s = joined(entry)
        if n > 1:
            s = s + "!"
    except ValueError:
        pass


def filled(out, text):
    try:
        n = int(text)
        out.append(n)
    except ValueError:
        pass


def test_known_containers_in_a_helpers_try():
    # a helper's try body, after a call that may fail: a known dict it's
    # given made a run-time one there (it escapes), and a known list of its
    # caller's changed there: on the paths that get that far only (the dict
    # made there was released after the try, on every path: a value LLVM
    # rejected; the list's change was made while compiling)
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        logged(a, {"t": 1, "a": b})
        out = []
        filled(out, a)
        return [len(out), out]

    p = lang.load("fn f(a, b) { return a + b; }", "try.tiny")
    for a, b in [("5", "x"), ("x", "y"), ("1", "")]:
        assert p.call("f", a, b) == p.call("f", a, b, mode="python")
    assert p.call("f", "x", "y") == [0, []]
    assert p.call("f", "5", "y") == [1, [5]]


def push(out, s):
    out.append(s)


def test_a_known_list_a_helper_changes_in_its_callers_flow():
    # a helper changing its caller's list, called inside the caller's `if`
    # or loop over run-time items: the change the run's, on the paths it's
    # on (it was made while compiling: once, whichever ran)
    lang = _language(True)

    @lang.eval("BinOp")
    def binop(node, rt):
        a = rt.eval(node.left)
        b = rt.eval(node.right)
        out = []
        if a == "yes":
            push(out, "if")
        for c in b:
            if c == "x":
                push(out, c)
        seen = []
        for c in "axbx":
            if c == b:
                add_once(seen, "c", c)
        return [len(out), out, [f.text for f in seen]]

    p = lang.load("fn f(a, b) { return a + b; }", "flow.tiny")
    for a, b in [("yes", "axbx"), ("no", ""), ("no", "x"), ("yes", "b")]:
        assert p.call("f", a, b) == p.call("f", a, b, mode="python")
    assert p.call("f", "no", "axbx") == [2, ["x", "x"], []]
    assert p.call("f", "yes", "x") == [2, ["if", "x"], ["x"]]
