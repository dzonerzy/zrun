"""An engine (examples/scan, a YARA-like rule language): rules loaded once and
called over thousands of files, from several Python threads and through
map(), giving one thread's results (and the reference mode's)."""

import os
import random
import shutil
import struct
import sys
import threading

import pytest
import zrun

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "examples", "scan"))

if shutil.which("zig") is None:
    pytest.skip("scan's native library needs zig to build", allow_module_level=True)

import scan  # noqa: E402

RULES = os.path.join(HERE, "..", "examples", "scan", "rules.scan")


def _elf(bits, kind, rng):
    head = b"\x7fELF" + bytes([2 if bits == 64 else 1, 1, 1]) + bytes(9) + struct.pack("<HH", kind, 62)
    return head + bytes(rng.randrange(256) for _ in range(rng.randrange(0, 400)))


def _pe(rng, good=True):
    at = 0x80
    body = bytearray(rng.randrange(256) for _ in range(at + 64))
    body[0:2] = b"MZ"
    body[0x3C:0x40] = struct.pack("<I", at if good else 0xFFFF00)
    body[at : at + 4] = b"PE\0\0"
    return bytes(body)


def _files(tmp_path, n=3000):
    rng = random.Random(7)
    makers = [
        lambda: _elf(64, 3, rng),
        lambda: _elf(64, 2, rng),
        lambda: _elf(32, 2, rng),
        lambda: _pe(rng),
        lambda: _pe(rng, good=False),
        lambda: b"#!/bin/sh\necho hi\n" * rng.randrange(1, 5),
        lambda: b"\x1f\x8b\x08" + bytes(rng.randrange(256) for _ in range(50)),
        lambda: b"\x89PNG\r\n\x1a\n" + bytes(30),
        lambda: os.urandom(8192),
        lambda: b"\x90" * rng.randrange(50, 200) + b"\xcc",
        lambda: b"",
        lambda: b"x",
        lambda: bytes(rng.randrange(256) for _ in range(rng.randrange(1, 100))),
    ]
    paths = []
    for i in range(n):
        p = tmp_path / f"f{i:05}"
        p.write_bytes(makers[i % len(makers)]())
        paths.append(str(p))
    return paths


def test_scan_threads_and_map(tmp_path):
    paths = _files(tmp_path)
    datas = [open(p, "rb").read() for p in paths]
    program = scan.load(RULES)

    # (the reference: the semantics run as Python, every 7th file)
    reference = scan.load(RULES)
    some = range(0, len(datas), 7)
    expected = {i: reference.call("scan", datas[i], mode="python") for i in some}

    one = [program.call("scan", d) for d in datas]
    assert {i: one[i] for i in some} == expected
    assert any("pe" in r for r in one) and any("packed" in r for r in one) and any("nops" in r for r in one)
    assert one[paths.index(str(tmp_path / "f00010"))] == ["empty"]

    taken = program.report()["gil_taken"]
    for threads in (1, 4, 16):
        assert program.map("scan", datas, threads=threads) == one
    # (the files memory-mapped, through scan())
    assert [m for _, m in scan.scan(program, paths, threads=8, batch=500)] == one

    # Python threads calling the same program at once
    got = [None] * len(datas)

    def work(k):
        for i in range(k, len(datas), 4):
            got[i] = program.call("scan", datas[i])

    ts = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert got == one
    # (no call touched Python: all of them could run in parallel)
    assert program.report()["gil_taken"] == taken


def test_scan_errors():
    # (a read past the end, not guarded: the first failing file's error)
    program = scan.lang.load("ruleset scan(f) { rule a = u32le(f, 4) == 1; }", "unguarded.scan")
    items = [bytes(8)] * 20 + [b"abc"] + [bytes(8)] * 20 + [b""]
    with pytest.raises(zrun.Error) as e:
        program.map("scan", items, threads=8)
    assert e.value.diagnostic.message == "rt.u32le(): offset 4 out of range"
    # (a native host function failing: its library's message)
    program = scan.lang.load("ruleset scan(f) { rule a = count(f, size(f)) > 0; }", "native.scan")
    with pytest.raises(zrun.Error) as e:
        program.map("scan", [bytes(8), bytes(300)], threads=2)
    assert "count(): a byte is 0 to 255" in e.value.diagnostic.message
