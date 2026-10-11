"""src/unicode.zig from this Python's Unicode data: str.lower(),
str.upper() and repr() as Python does them, natively.

    python tools/gen_unicode.py > src/unicode.zig

Each character's full mapping (one to three characters: 'İ'.lower() is
'i̇', 'ß'.upper() 'SS'), and for lower()'s one rule with a context (a
capital sigma ending a word is ς) the characters Cased and Case_Ignorable.
Python doesn't give Case_Ignorable: it's read from lower() itself, a
character x being case-ignorable when a sigma after a cased letter and x
is still final ('AxΣ' and 'AΣx' end in ς). The characters repr() escapes
beyond ASCII: those not isprintable()."""

import sys
import unicodedata

SIGMA = "Σ"


def cased(c):
    # (Lowercase, Uppercase or titlecase: islower() of one character is
    # its Lowercase, istitle() its Lt or Uppercase)
    return c.islower() or c.istitle()


def case_ignorable(c):
    if not cased(c):
        return ("A" + c + SIGMA).lower().endswith("ς")
    return ("A" + SIGMA + c).lower()[1] == "ς"


def ranges(pred, surrogates=False):
    out = []
    start = None
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            ok = surrogates
        else:
            ok = pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def mapping(method):
    out = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        m = getattr(c, method)()
        if m != c:
            assert 1 <= len(m) <= 3, (hex(cp), m)
            out.append((cp, [ord(x) for x in m]))
    return out


def table(name, entries):
    lines = [f"pub const {name} = [_]Map{{"]
    for cp, to in entries:
        padded = to + [0] * (3 - len(to))
        lines.append(f"    .{{ .cp = 0x{cp:X}, .len = {len(to)}, .to = .{{ 0x{padded[0]:X}, 0x{padded[1]:X}, 0x{padded[2]:X} }} }},")
    lines.append("};")
    return "\n".join(lines)


def range_table(name, rs):
    lines = [f"pub const {name} = [_]Range{{"]
    for lo, hi in rs:
        lines.append(f"    .{{ .lo = 0x{lo:X}, .hi = 0x{hi:X} }},")
    lines.append("};")
    return "\n".join(lines)


def main():
    print(f"""//! str.lower(), str.upper() and repr() as Python {sys.version_info.major}.{sys.version_info.minor} does them (Unicode
//! {unicodedata.unidata_version}): each character's full mapping, the characters
//! Cased and Case_Ignorable (a capital sigma ending a word lowercases to ς), and
//! those not printable (repr() escapes them). Made by tools/gen_unicode.py: not
//! edited by hand.

/// A character's mapping: len characters (1 to 3)
pub const Map = struct {{ cp: u21, len: u8, to: [3]u21 }};

/// Characters lo..hi (both in)
pub const Range = struct {{ lo: u21, hi: u21 }};
""")
    print(table("lower", mapping("lower")))
    print()
    print(table("upper", mapping("upper")))
    print()
    print(range_table("cased", ranges(cased)))
    print()
    print(range_table("case_ignorable", ranges(case_ignorable)))
    print()
    # (surrogates aren't printable: repr() writes \udxxx)
    print(range_table("not_printable", ranges(lambda c: not c.isprintable(), surrogates=True)))


if __name__ == "__main__":
    main()
