"""Python bytes natively, and the library functions over them: struct's,
int.from_bytes and to_bytes, zlib's checksums. The same results and errors
as the reference mode; native in strict mode."""

import random
import struct
import zlib

import zrun
from test_intrinsics import _program, same_in_both


def basics(a, b):
    # (a program's data arrives as zrun.Bytes, in every mode: bytes of it)
    a = bytes(a)
    b = bytes(b)
    data = a + b"\x00\xff" + b
    return (len(data), data[0] if data else None, data[-1], data[1:3], data[::2], b"\xff" in data, 255 in data,
            data.find(b"\x00"), data.rfind(b"z"), data.count(b"a"), data.startswith(a), data.endswith((b"x", b)),
            data.hex(), data.upper(), data.strip(b"\x00ab"), data.split(b"\x00"), b"-".join([a, b]), data.replace(b"\xff", b"!"),
            a * 2, data == a + b"\x00\xff" + b, isinstance(data, bytes), type(data) is bytes)


def test_basics():
    same_in_both(_program(basics), [(b"ab", b"cd"), (b"", b"x"), (b"aaa", b"")])
    assert _program(basics, strict=True).call("f", b"ab", b"cd") == basics(b"ab", b"cd")


def text(a, b):
    b = bytes(b)
    return (a.encode(), a.encode("utf-8").decode(), b.decode("latin-1"), a.encode("utf-8")[1:].decode("utf-8", ) if a else "",
            bytes(3), bytes([1, 2, 255]), bytes(b), bytes.fromhex("de ad BE ef"), b"%d" if False else b"lit")


def test_text():
    same_in_both(_program(text), [("héllo", b"\xe9t\xe9"), ("x", b"")])
    assert _program(text, strict=True).call("f", "héllo", b"\xe9") == text("héllo", b"\xe9")


def errors(a, b):
    if a == 0:
        return b[10]
    if a == 1:
        return bytes(b).decode("utf-8")
    if a == 2:
        return "é".encode("ascii")
    if a == 3:
        return bytes(b).index(b"zz")
    if a == 4:
        return struct.unpack("<I", b)
    if a == 5:
        return (300).to_bytes(1, "big")
    if a == 6:
        return (-1).to_bytes(2, "little")
    return bytes([256])


def test_errors_are_pythons():
    same_in_both(_program(errors), [(i, b"\xff\xfe") for i in range(8)])


def numbers(a, b):
    return (int.from_bytes(a, "little"), int.from_bytes(a, "big"), int.from_bytes(a, byteorder="big", signed=True),
            b.to_bytes(8, "little", signed=True), (b & 0xFFFF).to_bytes(2, byteorder="big"))


def test_from_bytes_and_to_bytes():
    rng = random.Random(3)
    cases = []
    for n in (0, 1, 2, 3, 4, 7, 8, 9, 15, 16):
        for _ in range(4):
            cases.append((bytes(rng.randrange(256) for _ in range(n)), rng.randrange(-(2**63), 2**63)))
    same_in_both(_program(numbers), cases)
    assert _program(numbers, strict=True).call("f", b"\x01\x02", 5) == numbers(b"\x01\x02", 5)


def unpacking(a, b):
    return (struct.unpack("<hHiIqQ", a[:28]), struct.unpack(">bBfd?", a[:15]), struct.unpack_from("<2I3s", a, 3),
            struct.unpack("@bi", a[:8]), struct.unpack("!4xQ", a[:12]), struct.unpack("c", a[:1]))


def packing(a, b):
    return (struct.pack("<hHiI", a, b, -a, b), struct.pack(">qQ?", a * 1000, b, a), struct.pack("<fd", a / 3, b / 7),
            struct.pack("3s2xc", b"abcdef", b"z"), struct.pack("@bi", a, b))


def test_struct():
    rng = random.Random(5)
    datas = [bytes(rng.randrange(256) for _ in range(40)) for _ in range(20)]
    same_in_both(_program(unpacking), [(d, 0) for d in datas])
    same_in_both(_program(packing), [(rng.randrange(-100, 100), rng.randrange(0, 60000)) for _ in range(20)])
    assert _program(unpacking, strict=True).call("f", datas[0], 0) == unpacking(datas[0], 0)
    assert _program(packing, strict=True).call("f", 5, 7) == packing(5, 7)


def struct_errors(a, b):
    if a == 0:
        return struct.pack("<B", b)
    if a == 1:
        return struct.unpack_from("<Q", b"abc", 0)
    return struct.pack("<f", 1e300)


def test_struct_errors_are_pythons():
    same_in_both(_program(struct_errors), [(0, 300), (1, 0), (2, 0)])


def checksums(a, b):
    return (zlib.crc32(a), zlib.crc32(b, zlib.crc32(a)), zlib.adler32(a), zlib.adler32(b, 7))


def test_zlib():
    rng = random.Random(9)
    same_in_both(_program(checksums), [(bytes(rng.randrange(256) for _ in range(rng.randrange(60))), bytes(rng.randrange(256) for _ in range(rng.randrange(60)))) for _ in range(20)])
    assert _program(checksums, strict=True).call("f", b"abc", b"def") == checksums(b"abc", b"def")


def as_keys(a, b):
    a = bytes(a)
    b = bytes(b)
    d = {a: 1, b: 2}
    s = {a, b, a + b}
    return (d[a], d.get(b), list(s), a in s)


def test_bytes_as_keys_in_pythons_order():
    same_in_both(_program(as_keys), [(b"k%d" % i, b"v%d" % i) for i in range(20)])


def given_back(a, b):
    return (bytes(a), b"lit", a)


def test_bytes_given_back_to_python():
    r = _program(given_back).call("f", b"some data", 0, mode="compiled")
    assert type(r[0]) is bytes and r[0] == b"some data" and r[1] == b"lit"
    # (data given to a program: a zrun.Bytes, in every mode)
    assert isinstance(r[2], zrun.Bytes)
