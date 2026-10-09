"""More of Python's syntax compiled: `:=`, `del` (a local, an item),
`raise ... from ...`, `import` inside a semantic, `*args` in calls. Each the
same as the reference mode; those native in strict mode too."""

import pytest
import zrun
from test_intrinsics import _program, outcome, same_in_both


def walrus(a, b):
    out = []
    if (n := a * 2) > 4:
        out.append(n)
    i = 0
    while (i := i + 1) < b:
        out.append(i)
    total = [y for x in range(b) if (y := x * x) % 2 == 0]
    return (n, i, out, total, y)


def test_walrus():
    same_in_both(_program(walrus), [(1, 3), (5, 6), (3, 1)])
    assert _program(walrus, strict=True).call("f", 5, 6) == walrus(5, 6)


def deleting(a, b):
    d = {"x": 1, "y": 2, b: 3}
    del d["x"]
    items = [10, 20, 30, 40]
    del items[a]
    t = a * 2
    del t
    del d[b]
    return (d, items)


def test_del():
    # (a missing key, an index out of range: Python's errors, its words)
    same_in_both(_program(deleting), [(1, "z"), (-1, "w"), (9, "q"), (0, "x")])
    assert _program(deleting, strict=True).call("f", 1, "z") == deleting(1, "z")


def reading_after_del(a, b):
    t = a
    del t
    return t


def test_reading_after_del_is_refused():
    p = _program(reading_after_del, strict=True)
    with pytest.raises(zrun.CompileError):
        p.call("f", 1, 2)


def raising_from(a, b):
    try:
        if a > 0:
            raise ValueError("inner")
        return 0
    except ValueError as e:
        if b == 0:
            raise KeyError("outer") from e
        raise KeyError("outer") from None


def test_raise_from():
    same_in_both(_program(raising_from), [(0, 0), (1, 0), (1, 1)])
    # (its cause kept, as Python's)
    p = _program(raising_from)
    for mode in ("compiled", "python"):
        with pytest.raises(zrun.Error) as e:
            p.call("f", 1, 0, mode=mode)
        assert "outer" in str(e.value)


def importing(a, b):
    import math
    import os.path
    import os.path as osp
    from math import floor, sqrt
    from collections import OrderedDict as OD

    return (math.floor(a), floor(a) + sqrt(b), os.path.basename("a/b"), osp.basename("c/d"), OD.__name__)


def test_import_inside():
    same_in_both(_program(importing), [(2.5, 4), (7, 9)])
    assert _program(importing, strict=True).call("f", 2.5, 4) == importing(2.5, 4)


def other_constants(a, b):
    import os

    sep = b"/" if isinstance(a, bytes) else "/"
    kinds = (isinstance(a, (bytes, bytearray)), isinstance(b, (set, frozenset, complex)))
    return (sep, kinds, os.fspath(a), 2j, ...)


def test_other_constants():
    # (bytes, complex, `...`: Python's objects; isinstance() of their
    # classes and os.fspath() of a str native)
    same_in_both(_program(other_constants), [("x", 1), ("y", {1}), ("z", 2j)])
    assert _program(other_constants, strict=True).call("f", "x", 1) == other_constants("x", 1)


def add3(x, y, z):
    return x * 100 + y * 10 + z


def spreading(a, b):
    known = (1, 2)
    r1 = add3(*known, a)
    r2 = add3(a, *[b, 3])
    args = list(range(a, a + 3))
    r3 = max(*args)
    r4 = min(b, *args)
    return (r1, r2, r3, r4)


def test_star_args():
    same_in_both(_program(spreading), [(1, 2), (5, 0), (-3, 9)])
    assert _program(spreading).call("f", 1, 2) == spreading(1, 2)


def test_outcome_helper_still_there():
    # (the helpers this file shares)
    assert outcome(_program(spreading), "compiled", 1, 2)[0] == "ok"
