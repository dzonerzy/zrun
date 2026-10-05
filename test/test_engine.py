"""Engines: a program loaded once and called many times (program.call), with a
context per call (rt.context), in every mode."""

import builtins

import pytest
import zrun
from conftest import tiny

MODES = ["python", "compiled"]


def _engine_lang(more_builtins=()):
    """tiny whose `context()` is the call's context (rt.context)."""
    from zrules import Rules, scopes

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
                builtins=("print", "context", "u8", "u32le", "u32be", "i16le", "u64le", "size", "at", "part") + tuple(more_builtins),
            ),
        ],
    )
    lang = zrun.Language(tiny.PARSER, rules)
    lang.function("FuncDef")
    for kind, fn in (("Let", tiny.assign), ("Assign", tiny.assign), ("Return", tiny.return_), ("If", tiny.if_), ("While", tiny.while_)):
        lang.exec(kind)(fn)
    lang.eval("BinOp")(tiny.binop)

    @lang.eval("Call")
    def call(node, rt):
        name = node.name.text
        if name == "context":
            return rt.context
        args = rt.eval(node.args)
        if name == "u8":
            return rt.u8(args[0], args[1])
        if name == "u32le":
            return rt.u32le(args[0], args[1])
        if name == "u32be":
            return rt.u32be(args[0], args[1])
        if name == "i16le":
            return rt.i16le(args[0], args[1])
        if name == "u64le":
            return rt.u64le(args[0], args[1])
        if name == "size":
            return len(args[0])
        if name == "at":
            return args[0][args[1]]
        if name == "part":
            return args[0][args[1] : args[2]]
        return rt.call(rt.eval(node.name), args)

    @lang.host
    def print(*args):
        builtins.print(*args)

    return lang


lang = _engine_lang()

SOURCE = """
let count = 0;
fn bump(n) { count = count + n; return count; }
fn tag(x) { return context() + x; }
fn div(a, b) { return a / b; }
"""


@pytest.mark.parametrize("mode", MODES)
def test_calls_keep_the_programs_variables(mode):
    program = lang.load(SOURCE, "engine")
    assert [program.call("bump", n, mode=mode) for n in (1, 2, 3)] == [1, 3, 6]


@pytest.mark.parametrize("mode", MODES)
def test_context(mode):
    program = lang.load(SOURCE, "engine")
    assert program.call("tag", "!", mode=mode, context="file-a") == "file-a!"
    assert program.call("tag", 1, mode=mode, context=41) == 42


@pytest.mark.parametrize("mode", MODES)
def test_errors(mode):
    program = lang.load(SOURCE, "engine")
    with pytest.raises(zrun.Error) as e:
        program.call("div", 1, 0, mode=mode)
    assert e.value.diagnostic.message == "division by zero"
    with pytest.raises(KeyError):
        program.call("nothing", mode=mode)
    # (a call after an error: the program's variables as they were)
    assert program.call("bump", 5, mode=mode) == 5


DEEP = "fn down(n) { if n == 0 { return 0; } return 1 + down(n - 1); }\n"


@pytest.mark.parametrize("mode", MODES)
def test_deep_calls(mode):
    program = lang.load(DEEP, "deep")
    assert program.call("down", 900, mode=mode) == 900
    with pytest.raises(zrun.Error) as e:
        program.call("down", 5000, mode=mode)
    assert e.value.diagnostic.message == "call stack too deep (more than 1000 calls)"


def test_deep_calls_from_a_small_thread():
    # (a thread whose stack hasn't room for the deepest calls: the call made
    # on a stack of its own, as deep as on the main thread)
    import threading

    program = lang.load(DEEP, "deep")
    out = {}

    def work():
        out["ok"] = program.call("down", 900, mode="compiled")
        try:
            program.call("down", 5000, mode="compiled")
        except zrun.Error as e:
            out["err"] = e.diagnostic.message

    old = threading.stack_size(512 * 1024)
    try:
        t = threading.Thread(target=work)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    assert out == {"ok": 900, "err": "call stack too deep (more than 1000 calls)"}


RULES = """
fn header(d) { return u32le(d, 0); }
fn fields(d) { return u8(d, 1) + u32be(d, 2) + i16le(d, 6) + size(d) + at(d, 0 - 1); }
fn big(d) { return u64le(d, 0) > 0; }
fn magic(d) { let m = part(d, 0, 2); return m == part(d, 0, 2); }
fn slice_len(d) { return size(part(d, 2, 100)); }
fn read_past(d) { return u32le(d, size(d) - 2); }
"""

DATA = bytes([0x7F, 0x45, 0x4C, 0x46, 2, 1, 0xFE, 0xFF, 0x80, 0x90])


def _data_kinds(tmp_path):
    import mmap

    path = tmp_path / "data.bin"
    path.write_bytes(DATA)
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return f, {"bytes": DATA, "bytearray": bytearray(DATA), "memoryview": memoryview(DATA), "mmap": mm, "Bytes": zrun.Bytes(DATA)}


def test_reading_data(tmp_path):
    f, kinds = _data_kinds(tmp_path)
    try:
        results = {}
        for mode in MODES:
            program = lang.load(RULES, "rules")
            for kind, data in kinds.items():
                got = [program.call(fn, data, mode=mode) for fn in ("header", "fields", "magic", "slice_len")]
                for fn in ("big", "read_past"):
                    try:
                        program.call(fn, data, mode=mode)
                    except zrun.Error as e:
                        got.append(e.diagnostic.message)
                results[(mode, kind)] = got
        # (big: a u64 beyond 63 bits, which a node's value can't be: an int
        # of the program is 64 bits)
        fields = DATA[1] + int.from_bytes(DATA[2:6], "big") + int.from_bytes(DATA[6:8], "little", signed=True) + len(DATA) + DATA[-1]
        expected = [int.from_bytes(DATA[:4], "little"), fields, True, 8, "integer overflow", "rt.u32le(): offset 8 out of range"]
        for key, got in results.items():
            assert got == expected, key
    finally:
        kinds["mmap"].close()
        f.close()


def test_bytes_in_python():
    b = zrun.Bytes(b"hello")
    assert (len(b), b[1], b[-1], bytes(b[1:3]), b[1:3] == b"el", b == bytearray(b"hello"), hash(b) == hash(b"hello")) == (5, 101, 111, b"el", True, True, True)
    with pytest.raises(IndexError):
        b[5]


def _native(signature, fn, error=None):
    """A zrun.native.v1 capsule over a Python function (through ctypes: a
    native library's would be C)."""
    import ctypes

    class Data(ctypes.Structure):
        _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_uint64)]

    class Arg(ctypes.Union):
        _fields_ = [("i", ctypes.c_int64), ("f", ctypes.c_double), ("b", Data)]

    CALL = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(Arg), ctypes.c_uint64, ctypes.POINTER(Arg))
    # (the message's address: a buffer kept here, as a library's static text)
    ERROR = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)

    class Native(ctypes.Structure):
        _fields_ = [("abi", ctypes.c_uint32), ("signature", ctypes.c_char_p), ("call", CALL), ("state", ctypes.c_void_p), ("error", ERROR)]

    def call(state, args, n, out):
        return fn(args, out)

    messages = {}

    def message(state, code):
        text = (error or (lambda c: b"failed"))(code)
        messages[code] = ctypes.create_string_buffer(text)
        return ctypes.addressof(messages[code])

    keep = [CALL(call), ERROR(message), messages]
    native = Native(1, signature.encode(), keep[0], None, keep[1])
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
    ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = ctypes.pythonapi.PyCapsule_New(ctypes.addressof(native), b"zrun.native.v1", None)
    # (alive as long as the capsule's used: the test's)
    _native.kept.append((native, keep))
    return capsule


_native.kept = []


def test_native_host():
    def count(args, out):
        data, byte = args[0].b, args[1].i
        out[0].i = sum(1 for k in range(data.len) if data.ptr[k] == byte)
        return 0

    def checked(args, out):
        if args[0].i < 0:
            return 7
        out[0].i = args[0].i * 2
        return 0

    native_lang = _engine_lang(("count", "checked"))
    native_lang.native_host("count", _native("bi:i", count))
    native_lang.native_host("checked", _native("i:i", checked, lambda c: b"negative input (code %d)" % c))
    src = "fn n(d, b) { return count(d, b); }\nfn twice(x) { return checked(x); }\n"
    results = {}
    for mode in MODES:
        program = native_lang.load(src, "native")
        got = [program.call("n", b"abcabca", 97, mode=mode), program.call("twice", 21, mode=mode)]
        for args in ((-1,), ("x",)):
            try:
                program.call("twice", *args, mode=mode)
            except zrun.Error as e:
                got.append(e.diagnostic.message)
        results[mode] = got
    assert results["python"] == results["compiled"]
    assert results["compiled"][:2] == [3, 42]
    assert "negative input (code 7)" in results["compiled"][2]
    assert "takes an int" in results["compiled"][3]


def test_the_mode_last_run():
    program = lang.load(SOURCE, "engine")
    # (never run: compiled)
    assert program.call("bump", 2) == 2
    program.run(mode="python")
    assert program.call("bump", 2) == 2
