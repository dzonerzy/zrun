"""The semantics compiler's front: semantics read from their Python source,
checked against the compilable subset (errors at the line; a registered
semantic outside it runs as Python, lang.python_semantics() says why), and
what it makes of them (lang.ir())."""

import inspect

import pytest
import zrun
from conftest import tiny, typed


def lang():
    return zrun.Language(tiny.PARSER, tiny.RULES)


def line_of(fn, text):
    """The file line of the first source line of `fn` containing `text`."""
    lines, first = inspect.getsourcelines(fn)
    for i, line in enumerate(lines):
        if text in line:
            return first + i
    raise AssertionError(text)


class TestSubset:
    def test_examples_compile(self):
        # every semantic of the two example languages is in the subset
        for module in (tiny, typed):
            for fn in vars(module).values():
                if inspect.isfunction(fn) and fn.__module__ == module.__name__ and fn.__name__ not in ("print", "len"):
                    module.lang.ir(fn)

    @pytest.mark.parametrize(
        "source, reason",
        [
            ("    f = lambda x=1: x\n", "a lambda's parameters must be plain ones (no defaults, keyword-only or positional-only ones)"),
            ("    with open('x') as f:\n        pass\n", "`with` can't be compiled"),
            ("    f = inner\n    def inner():\n        pass\n", "the nested function inner used before its def can't be compiled"),
            ("    @staticmethod\n    def inner():\n        pass\n", "a nested def with decorators can't be compiled"),
            ("    class Inner:\n        pass\n", "a nested `class` can't be compiled"),
            ("    return {1, 2}\n", "a set can't be compiled"),
            ("    return [*node.children]\n", "*unpacking can't be compiled"),
            ("    return node @ rt\n", "the operator MatMult can't be compiled"),
            ("    yield 1\n", "`yield` can't be compiled"),
        ],
    )
    def test_outside(self, source, reason):
        ns = {}
        exec(compile("def semantic(node, rt):\n" + source, "semantics.py", "exec"), ns)
        # (inspect needs the source: make it findable)
        import linecache

        text = "def semantic(node, rt):\n" + source
        linecache.cache["semantics.py"] = (len(text), None, text.splitlines(True), "semantics.py")
        with pytest.raises(zrun.CompileError) as e:
            lang().ir(ns["semantic"])
        assert e.value.reason == reason
        assert e.value.file == "semantics.py" and e.value.line == 2
        assert str(e.value).startswith(f"semantics.py:2:{e.value.column}: in semantic(): {reason}")
        # registered, it runs as Python in compiled programs, saying why
        l = lang()
        l.eval("Call")(ns["semantic"])
        assert l.python_semantics() == {"semantic": str(e.value)}

    def test_the_line_in_a_real_file(self):
        def semantic(node, rt):
            x = 1
            with open("f") as f:
                x = 2
            return x

        with pytest.raises(zrun.CompileError) as e:
            lang().ir(semantic)
        assert e.value.line == line_of(semantic, "with ") and e.value.column == 13

    def test_parameters(self):
        def defaults(node, rt=None):
            return 1

        def keyword_only(node, *, rt):
            return 1

        def star(node, *rest):
            return 1

        # (defaults: the function's own, when called without)
        assert lang().ir(defaults).splitlines()[0] == "def defaults(node, rt):"
        with pytest.raises(zrun.CompileError, match="only plain parameters"):
            lang().ir(keyword_only)
        with pytest.raises(zrun.CompileError, match=r"\*args and \*\*kwargs"):
            lang().ir(star)

    def test_native_false_isnt_read(self, capsys):
        l = lang()
        l.function("FuncDef")
        l.host("print", lambda *a: print(*a))

        @l.eval("Call", native=False)
        def call(node, rt):
            try:  # outside the subset: fine, it runs as Python
                return rt.call(rt.eval(node.name), rt.eval(node.args))
            finally:
                pass

        l.load("print(7);\n").run()
        assert capsys.readouterr().out == "7\n"


class TestReading:
    def ir(self, fn):
        return lang().ir(fn)

    def test_locals_and_globals(self):
        def f(node, rt):
            x = len(node.children)
            for child in node.children:
                x += 1
            return [c for c in node.children if c], x

        assert self.ir(f) == (
            "def f(node, rt):\n"
            "    x#2 = global len(node#0.children)\n"
            "    for child#3 in node#0.children:\n"
            "        x#2 add= 1\n"
            "    return (tuple [c#4 for c#4 in node#0.children if c#4], x#2)\n"
        )

    def test_try(self):
        def f(node, rt):
            try:
                x = 1
            except (ValueError, KeyError) as e:
                raise
            except Exception:
                pass
            else:
                x = 2
            finally:
                x = 3

        assert self.ir(f).splitlines()[1:] == [
            "    try:",
            "        x#2 = 1",
            "    except (tuple global ValueError, global KeyError) as e#3:",
            "        raise",
            "    except global Exception:",
            "        pass",
            "    else:",
            "        x#2 = 2",
            "    finally:",
            "        x#2 = 3",
        ]

    def test_big_ints(self):
        def f(node, rt):
            return 2**70 + 99999999999999999999

        # (Python's ints: beyond 64 bits too)
        assert self.ir(f).splitlines()[1] == "    return (add (pow 2 70) 99999999999999999999)"

    def test_a_comprehension_variable_doesnt_leak(self):
        def f(node, rt):
            c = 1
            xs = [c for c in node.children]
            return c

        # (the comprehension's c is a slot of its own)
        assert self.ir(f).splitlines()[2:] == ["    xs#3 = [c#4 for c#4 in node#0.children]", "    return c#2"]

    def test_expressions(self):
        def f(node, rt):
            a = -node.x if not node.y else node.z[1:2]
            b = 1 < node.a <= 3 and node.b or None
            c = {"k": node.v, **{}} if False else {"k": 1}
            return f"{a!r:>{b}} and {c}"

        with pytest.raises(zrun.CompileError, match=r"\*\*unpacking"):
            self.ir(f)

        def g(node, rt):
            a = -node.x if not node.y else node.z[1:2]
            b = 1 < node.a <= 3 and node.b or None
            return f"{a!r:>{b}} and {node.text}"

        assert self.ir(g).splitlines()[1:] == [
            "    a#2 = ((neg node#0.x) if (not_ node#0.y) else node#0.z[1:2])",
            "    b#3 = (or_ (and_ (compare 1 lt node#0.a le 3), node#0.b), None)",
            '    return f"{a#2!r:>{b#3}} and {node#0.text}"',
        ]

    def test_targets(self):
        def f(node, rt):
            a, (b, c) = node.x
            node.y = a
            node.z[b] = c
            n: int = 3
            m: int
            while a:
                break
            else:
                pass
            assert a, "message"
            raise rt.Return(n)

        assert self.ir(f).splitlines()[1:] == [
            "    (a#2, (b#3, c#4)) = node#0.x",
            "    node#0.y = a#2",
            "    node#0.z[b#3] = c#4",
            "    n#5 = 3",
            "    pass",
            "    while a#2:",
            "        break",
            "    else:",
            "        pass",
            '    assert a#2, "message"',
            "    raise rt#1.Return(n#5)",
        ]
