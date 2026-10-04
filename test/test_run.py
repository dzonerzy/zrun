"""Running programs: semantics as Python, everything around them native."""

import gc

import pytest
import zgram
import zrun
from conftest import tiny

FIB = """
fn fib(n) {
    if n < 2 { return n; }
    return fib(n - 1) + fib(n - 2);
}
let i = 0;
while i < 10 {
    print(fib(i));
    i = i + 1;
}
"""


class TestPrograms:
    def test_fib(self, run):
        assert run(FIB).split() == ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34"]

    def test_break(self, run):
        src = "let i = 0;\nwhile 1 { if i == 3 { break; } print(i); i = i + 1; }\nprint(99);\n"
        assert run(src).split() == ["0", "1", "2", "99"]

    def test_functions_are_hoisted(self, run):
        # called before its definition
        assert run('print(twice(4));\nfn twice(x) { return x * 2; }\n').split() == ["8"]

    def test_globals_from_a_function(self, run):
        src = "let count = 0;\nfn bump() { count = count + 1; }\nbump(); bump();\nprint(count);\n"
        assert run(src).split() == ["2"]

    def test_each_call_its_own_variables(self, run):
        src = "fn f(n) { let x = n; if n > 0 { f(n - 1); } print(x); }\nf(2);\n"
        assert run(src).split() == ["0", "1", "2"]

    def test_strings(self, run):
        assert run('print("a\\tb" + "c");\n') == "a\tbc\n"

    def test_return_without_value(self, run):
        assert run("fn f() { return; }\nprint(f());\n") == "None\n"

    def test_call_from_the_host(self):
        program = tiny.lang.load("fn add(a, b) { return a + b; }\nfn fib(n) { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }\n")
        assert program.call("add", 2, 3) == 5
        assert program.call("fib", 20) == 6765
        assert type(program.call("add", 1, 1)) is zrun.I64
        with pytest.raises(KeyError, match="no function 'nope'"):
            program.call("nope")


class TestIntegers:
    def test_i64(self):
        x = zrun.I64(5)
        assert isinstance(x, int) and x == 5 and repr(x) == "5"
        assert type(x + 1) is zrun.I64 and type(1 + x) is zrun.I64 and type(-x) is zrun.I64
        assert x / 2 == 2.5 and type(x * 1.5) is float
        assert x // 2 == 2 and x % 3 == 2 and divmod(x, 2) == (2, 1) and x ** 2 == 25

    def test_overflow(self):
        big = zrun.I64(2**62)
        with pytest.raises(zrun.IntegerOverflow):
            big * 2
        with pytest.raises(zrun.IntegerOverflow):
            -zrun.I64(-(2**63))
        with pytest.raises(zrun.IntegerOverflow):
            zrun.I64(2**63)
        assert zrun.I64(2**63 - 1) == 2**63 - 1

    def test_overflow_in_a_program(self):
        src = "let x = 4611686018427387904;\nlet y = x * 2;\n"
        with pytest.raises(zrun.Error) as e:
            tiny.lang.load(src).run()
        assert e.value.diagnostic.message == "integer overflow"
        assert e.value.diagnostic.span == (src.index("x * 2"), src.index("x * 2") + 5)

    def test_literal_too_big(self):
        with pytest.raises(zrun.Error, match="integer overflow"):
            tiny.lang.load("let x = 99999999999999999999;\n").run()


class TestErrors:
    def test_a_runtime_error_with_its_stack(self):
        src = "fn div(a, b) { return a / b; }\nfn go(x) { return div(x, 0); }\nprint(go(1));\n"
        with pytest.raises(zrun.Error) as e:
            tiny.lang.load(src, "prog.tiny").run()
        err = e.value
        d = err.diagnostic
        assert (d.severity, d.code, d.message) == ("error", "runtime", "division by zero")
        assert src[d.span[0] : d.span[1]] == "a / b" and (d.line, d.column) == (1, 23)
        # innermost call first
        assert [(name, src[n.span[0] : n.span[1]]) for name, n in err.stack] == [("div", "div(x, 0)"), ("go", "go(1)")]
        text = str(err)
        assert text.startswith("prog.tiny:1:23: error: division by zero [runtime]")
        assert "  in div(), called at prog.tiny:2:19" in text and "  in go(), called at prog.tiny:3:7" in text

    def test_a_python_error_in_semantics(self):
        with pytest.raises(zrun.Error) as e:
            tiny.lang.load('let x = "a" - 1;\n').run()
        assert e.value.diagnostic.message == "unsupported operand type(s) for -: 'str' and 'int'"

    def test_arity(self):
        with pytest.raises(zrun.Error, match=r"f\(\) takes 1 argument, 2 given"):
            tiny.lang.load("fn f(a) { return a; }\nf(1, 2);\n").run()

    def test_not_callable(self):
        with pytest.raises(zrun.Error, match="'int' value is not callable"):
            tiny.lang.load("let x = 1;\nx();\n").run()

    def test_call_stack_too_deep(self):
        with pytest.raises(zrun.Error, match="call stack too deep"):
            tiny.lang.load("fn f(n) { return f(n + 1); }\nf(0);\n").run()

    def test_deep_but_allowed(self, run):
        src = "fn down(n) { if n == 0 { return 0; } return down(n - 1) + 1; }\nprint(down(900));\n"
        assert run(src) == "900\n"

    def calls(self):
        lang = zrun.Language(tiny.PARSER, tiny.RULES)
        lang.eval("Call")(tiny.call)
        return lang

    def test_a_host_function_raising(self):
        lang = self.calls()

        @lang.host
        def print(*args):
            raise ValueError("no output")

        with pytest.raises(zrun.Error, match="print: ValueError: no output"):
            lang.load("print(1);\n").run()

    def test_no_host_for_a_builtin(self):
        with pytest.raises(zrun.Error, match="no host function for the builtin 'print'"):
            self.calls().load("print(1);\n").run()

    def test_rt_error(self):
        lang = zrun.Language(tiny.PARSER, tiny.RULES)

        @lang.eval("Call")
        def call(node, rt):
            rt.error(node, "calls are off", code="no-calls")

        with pytest.raises(zrun.Error) as e:
            lang.load("let x = f(1);\nfn f(a) { return a; }\n").run()
        assert (e.value.diagnostic.code, e.value.diagnostic.message) == ("no-calls", "calls are off")


class TestLoading:
    def test_syntax_errors(self):
        with pytest.raises(zrun.LoadError) as e:
            tiny.lang.load("let x = ;\nlet y = 1\n", "bad.tiny")
        codes = [d.code for d in e.value.diagnostics]
        assert codes == ["syntax", "syntax"]
        assert str(e.value).startswith("bad.tiny:1:9: error:")

    def test_rules_errors(self):
        with pytest.raises(zrun.LoadError) as e:
            tiny.lang.load("break;\nprint(nope);\n")
        assert sorted(d.code for d in e.value.diagnostics) == ["break-outside-loop", "undefined-name"]

    def test_warnings_kept(self):
        program = tiny.lang.load("fn f(unused) { return 1; }\nprint(f(2));\n")
        assert all(d.severity != "error" for d in program.diagnostics)
        assert program.source.startswith("fn f") and program.analysis is not None
        assert program.root.kind == "Program"


class TestNodes:
    def program(self, src):
        return tiny.lang.load(src)

    def test_fields(self):
        root = self.program("let x = 1 + 2;\n").root
        let = root.body[0]
        assert let.kind == "Let" and let.rule == "let_stmt"
        assert let.name.kind == "Name" and let.name.text == "x"
        assert let.value.kind == "BinOp" and let.value.op == "+"
        assert let.value.left == 1 and type(let.value.left) is zrun.I64
        assert let.fields == ["name", "value"]
        assert let.span == (0, 14) and let.text == "let x = 1 + 2;"
        assert let.parent == root and let.value.parent == let

    def test_lists_and_absent(self):
        fn = self.program("fn f(a, b) { return; }\nfn g() { return; }\n").root.body
        assert [p.text for p in fn[0].params] == ["a", "b"]
        assert fn[1].params == []
        ret = fn[0].body[0]
        assert ret.kind == "Return" and ret.value is None

    def test_unknown_field(self):
        let = self.program("let x = 1;\n").root.body[0]
        with pytest.raises(AttributeError, match=r"Let has no field 'nope' \(its fields: name, value\)"):
            let.nope

    def test_equality(self):
        program = self.program("let x = 1;\n")
        a, b = program.root.body[0], program.root.body[0]
        assert a == b and hash(a) == hash(b) and a is not b
        assert a != program.root and len({a, b}) == 1
        assert repr(a) == "<Let 'let x = 1;'>"

    def test_labels_come_first(self):
        # a label named like a node attribute is the field
        p = zgram.compile(
            r"""
            root = 'x[' index:num ']' kind:num
            num = [0-9]+ -> int
            """
        )
        lang = zrun.Language(p)
        seen = {}

        @lang.exec("root")
        def root(node, rt):
            seen.update(index=node.index, kind=node.kind, rt_kind=rt.kind(node), text=rt.text(node))

        lang.load("x[3]4").run()
        assert seen == {"index": 3, "kind": 4, "rt_kind": "root", "text": "x[3]4"}


class TestLanguage:
    def test_unknown_kind(self):
        lang = zrun.Language(tiny.PARSER, tiny.RULES)
        with pytest.raises(ValueError, match="no rule or -> class named 'Wihle'"):
            lang.exec("Wihle")

    def test_semantics_by_rule_name(self, capsys):
        lang = zrun.Language(tiny.PARSER, tiny.RULES)
        lang.host("print", lambda *a: None)

        @lang.exec("let_stmt")
        def let(node, rt):
            print("let", node.name.text)

        lang.load("let a = 1;\n").run()
        assert capsys.readouterr().out == "let a\n"

    def test_function_labels_checked(self):
        lang = zrun.Language(tiny.PARSER, tiny.RULES)
        with pytest.raises(ValueError, match="no label 'arguments'"):
            lang.function("FuncDef", params="arguments")

    def test_frames_and_functions_are_collected(self):
        # a function stored in the frame it was made in: a cycle
        def ours():
            gc.collect()
            return sum(1 for o in gc.get_objects() if type(o) in (zrun.Frame, zrun.Function, zrun.Node, zrun.State))

        before = ours()
        program = tiny.lang.load("fn f() { return 1; }\nfn g() { return f(); }\n")
        program.run()
        assert program.call("g") == 1
        assert ours() > before
        del program
        assert ours() == before

    def test_a_node_in_a_variable_is_collected(self):
        # the program's frame refers to the node, the node to the program
        lang = zrun.Language(tiny.PARSER, tiny.RULES)

        @lang.exec("Let")
        def let(node, rt):
            rt.store(node.name, node)

        def ours():
            gc.collect()
            return sum(1 for o in gc.get_objects() if type(o) in (zrun.Node, zrun.State, zrun.Frame))

        before = ours()
        program = lang.load("let x = 1;\n")
        program.run()
        del program
        assert ours() == before
