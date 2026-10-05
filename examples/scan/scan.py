"""scan: a small YARA-like rule language, run by zrun as an engine.

    python scan.py rules.scan PATH... [--threads N]

A rule set is a function of a file's bytes:

    const MZ = 0x5a4d;

    ruleset scan(f) {
        rule elf   = size(f) >= 20 and u32le(f, 0) == 0x464c457f;
        rule elf64 = elf and u8(f, 4) == 2;
        rule nops  = count(f, 0x90) > 64;
    }

Its rules are checked in order (a rule's name, in the rules after it, is
whether it matched); a call returns the names of the rules that matched. The
rules are loaded once and called for every file: `program.map("scan",
files, threads=n)` scans them on native threads, each file memory-mapped
(read through zrun.Bytes, not copied). `count` and `entropy` are native host
functions (scanlib.zig), called without Python.
"""

import ctypes
import mmap
import os
import subprocess
import sys

import zgram
import zrun
from zrules import Rules, scopes

GRAMMAR = r"""
program    = ws (body:item ws)*                                       -> Program
@silent item = const_decl | ruleset
const_decl = 'const' kw ws name:ident ws '=' ws value:expr ws ';'    -> Const
ruleset    = 'ruleset' kw ws name:ident ws '(' ws params:ident ws ')' ws body:rules  -> Ruleset
rules      = '{' ws (items:rule ws)* '}'                             -> Rules
rule       = 'rule' kw ws name:ident ws '=' ws cond:expr ws ';'     -> Rule

@left expr "condition" = left:conj (ws op:orop ws right:conj)*       -> BinOp
@left conj "condition" = left:inv (ws op:andop ws right:inv)*        -> BinOp
@silent inv = not | cmp
not        = 'not' kw ws operand:inv                                 -> Not
@left cmp "condition"   = left:sum (ws op:cmpop ws right:sum)?       -> BinOp
@left sum "expression"  = left:term (ws op:addop ws right:term)*     -> BinOp
@left term "expression" = left:primary (ws op:mulop ws right:primary)*  -> BinOp
@silent primary = float | hex | number | call | ident | '(' ws expr ws ')'
call       = name:ident ws '(' ws (args:expr (ws ',' ws args:expr)*)? ws ')'  -> Call

float      = [0-9]+ '.' [0-9]+                                       -> float
hex        = '0x' [0-9a-fA-F]+                                       -> Hex
number     = [0-9]+                                                  -> int
ident "name"     = !keyword [a-zA-Z_] [a-zA-Z0-9_]*                  -> Name
orop "operator"  = 'or' kw                                           -> str
andop "operator" = 'and' kw                                          -> str
cmpop "operator" = '==' | '!=' | '<=' | '>=' | '<' | '>'             -> str
addop "operator" = [+\-]                                             -> str
mulop "operator" = [*%&]                                             -> str

@silent keyword = ('const' | 'ruleset' | 'rule' | 'and' | 'or' | 'not') kw
@silent kw      = ![a-zA-Z0-9_]
@silent ws      = ([ \t\n\r] | '#' [^\n]*)*
"""

PARSER = zgram.compile(GRAMMAR)

READERS = ("u8", "i8", "u16le", "u16be", "u32le", "u32be", "u64le", "i32le")
NATIVE = ("count", "entropy")

RULES = Rules(
    PARSER,
    [
        scopes(
            scope=("Program", "Ruleset"),
            define=("Const > .name", "Rule > .name", "Ruleset > .params"),
            define_outer="Ruleset > .name",
            use="Name",
            hoist="Ruleset > .name",
            after=("Const > .name", "Rule > .name"),
            builtins=READERS + NATIVE + ("size",),
        ),
    ],
)

lang = zrun.Language(PARSER, RULES)

# A rule set: a function of the file's data
lang.function("Ruleset")


@lang.exec("Const")
def const(node, rt):
    rt.store(node.name, rt.eval(node.value))


@lang.exec("Rules")
def rules(node, rt):
    matched = []
    for rule in node.items:
        hit = rt.eval(rule.cond)
        rt.store(rule.name, hit)
        if hit:
            matched.append(rule.name.text)
    raise rt.Return(matched)


@lang.eval("BinOp")
def binop(node, rt):
    op = node.op
    # (and, or: the right side only when needed, as a read past the end
    # guarded by a size check on the left mustn't run)
    if op == "and":
        if not rt.eval(node.left):
            return False
        return rt.eval(node.right)
    if op == "or":
        left = rt.eval(node.left)
        if left:
            return left
        return rt.eval(node.right)
    a = rt.eval(node.left)
    b = rt.eval(node.right)
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
    if op == ">=":
        return a >= b
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "%":
        return a % b
    return a & b


@lang.eval("Not")
def not_(node, rt):
    return not rt.eval(node.operand)


@lang.eval("Hex")
def hex_(node, rt):
    return int(node.text, 16)


@lang.eval("Call")
def call(node, rt):
    name = node.name.text
    args = node.args
    if name == "size":
        return len(rt.eval(args[0]))
    if name == "u8":
        return rt.u8(rt.eval(args[0]), rt.eval(args[1]))
    if name == "i8":
        return rt.i8(rt.eval(args[0]), rt.eval(args[1]))
    if name == "u16le":
        return rt.u16le(rt.eval(args[0]), rt.eval(args[1]))
    if name == "u16be":
        return rt.u16be(rt.eval(args[0]), rt.eval(args[1]))
    if name == "u32le":
        return rt.u32le(rt.eval(args[0]), rt.eval(args[1]))
    if name == "u32be":
        return rt.u32be(rt.eval(args[0]), rt.eval(args[1]))
    if name == "i32le":
        return rt.i32le(rt.eval(args[0]), rt.eval(args[1]))
    if name == "u64le":
        return rt.u64le(rt.eval(args[0]), rt.eval(args[1]))
    return rt.call(rt.eval(node.name), rt.eval(args))


# The native host functions: scanlib.zig, built next to this file
HERE = os.path.dirname(os.path.abspath(__file__))
CAPSULE = b"zrun.native.v1"


def _library():
    src = os.path.join(HERE, "scanlib.zig")
    out = os.path.join(HERE, "scanlib.so")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        subprocess.run(["zig", "build-lib", "-dynamic", "-OReleaseFast", "-femit-bin=" + out, src], check=True, cwd=HERE)
    return ctypes.CDLL(out)


def _capsule(lib, symbol):
    new = ctypes.pythonapi.PyCapsule_New
    new.restype = ctypes.py_object
    new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    return new(ctypes.addressof(ctypes.c_char.in_dll(lib, symbol)), CAPSULE, None)


_lib = _library()
for _name in NATIVE:
    lang.native_host(_name, _capsule(_lib, "scan_" + _name))


def load(path):
    with open(path) as f:
        return lang.load(f.read(), path)


def files(paths):
    """The regular files at or under `paths`."""
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in sorted(names):
                    full = os.path.join(root, name)
                    if os.path.isfile(full) and not os.path.islink(full):
                        yield full
        elif os.path.isfile(path):
            yield path


def scan(program, paths, threads=None, batch=1024):
    """(path, the names of the rules it matched) for every file, the files
    memory-mapped and scanned `batch` at a time on `threads` threads."""
    paths = list(paths)
    for start in range(0, len(paths), batch):
        part = paths[start : start + batch]
        maps = [_map(p) for p in part]
        try:
            results = program.map("scan", maps, threads=threads)
        finally:
            for m in maps:
                if isinstance(m, mmap.mmap):
                    m.close()
        yield from zip(part, results)


def _map(path):
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size == 0:
            return b""
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Scan files with a rule set.")
    ap.add_argument("rules")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    try:
        program = load(a.rules)
        found = list(files(a.paths))
        t = time.perf_counter()
        n = 0
        for path, matched in scan(program, found, a.threads):
            n += 1
            if matched:
                print(f"{path}: {' '.join(matched)}")
        print(f"{n} files in {time.perf_counter() - t:.3f}s", file=sys.stderr)
    except (zrun.LoadError, zrun.Error) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
