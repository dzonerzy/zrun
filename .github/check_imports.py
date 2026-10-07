"""Check a Windows DLL for constants holding the address of an import slot.

Python's objects that are data of its DLL (None, True, False, the types) are
imported: code loads their address from the import table at run time. An
address the compiler takes for known while compiling (Zig's `&extern_var`)
can be put in a constant instead, a switch's table of results, and the
linker then fills it with the import slot's address: the code hands out the
slot as if it were the object, and writing its reference count crashes.

Usage: python check_imports.py zrun.pyd (exits 1 listing the constants)."""

import struct
import sys


def main(path):
    data = open(path, "rb").read()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    optional = struct.unpack_from("<H", data, pe + 20)[0]
    base = struct.unpack_from("<Q", data, pe + 24 + 24)[0]
    # (data directory 12: the import address table)
    iat, iat_size = struct.unpack_from("<II", data, pe + 24 + 112 + 12 * 8)
    lo, hi = base + iat, base + iat + iat_size
    found = []
    for i in range(sections):
        at = pe + 24 + optional + 40 * i
        name = data[at : at + 8].rstrip(b"\0").decode()
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, at + 8)
        if name in (".text", ".pdata", ".reloc"):
            continue
        for off in range(0, min(vsize, rsize) - 7, 8):
            where = base + va + off
            if lo <= where < hi:
                continue  # (the import table itself)
            value = struct.unpack_from("<Q", data, raw + off)[0]
            if lo <= value < hi:
                found.append((name, where, value))
    for name, where, value in found:
        print("%s: the constant at 0x%x is the import slot 0x%x" % (name, where, value))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
