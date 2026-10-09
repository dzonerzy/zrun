"""More of Python's syntax compiled: `:=`, `del` (a local, an item),
`raise ... from ...`, `import` inside a semantic, `*args` in calls. Each the
same as the reference mode; those native in strict mode too."""

from dataclasses import dataclass, field

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


def unrolled_jumps(a, b):
    # (loops whose iterations are unrolled: a break or continue known when
    # compiling, and one only at run time)
    out = []
    for x in (1, 2, 3, 4):
        if x == 3:
            break
        out.append(x)
    for x in (1, 2, 3, 4):
        if x == 2:
            continue
        out.append(x * 10)
    else:
        out.append("else")
    i = 0
    while True:
        i += 1
        if i > 3:
            break
        out.append(i * 100)
    for x in (1, 2, 3):
        if x == a:
            break
        out.append(-x)
    else:
        out.append("no break")
    j = 0
    while j < 4:
        j += 1
        if j == b:
            continue
        out.append(j * 1000)
    return out


def test_break_and_continue_in_unrolled_loops():
    same_in_both(_program(unrolled_jumps), [(0, 0), (2, 3), (1, 1), (3, 4)])
    assert _program(unrolled_jumps, strict=True).call("f", 2, 3) == unrolled_jumps(2, 3)


@dataclass
class Tracker:
    log: list
    suppress: bool = False

    def __enter__(self):
        self.log.append("enter")
        return len(self.log)

    def __exit__(self, kind, exc, tb):
        self.log.append("exit " + ("none" if kind is None else kind.__name__))
        return self.suppress


def using(a, b):
    log = []
    with Tracker(log) as n:
        log.append(n)
    with Tracker(log, True):
        if a == 0:
            raise ValueError("swallowed")
        log.append("body")
    for i in range(3):
        with Tracker(log), Tracker(log) as m:
            if i == b:
                break
            log.append(m)
    if a == 1:
        with Tracker(log):
            raise KeyError("out")
    return log


def returning(a, b):
    log = []
    t = Tracker(log)
    with t:
        if a:
            return log
        log.append("after")
    return log


def test_with():
    same_in_both(_program(using), [(0, 1), (1, 0), (2, 5)])
    same_in_both(_program(returning), [(0, 0), (1, 0)])
    assert _program(returning).call("f", 1, 0, mode="compiled") == ["enter", "exit none"]


def test_with_a_record_in_strict_mode():
    def strict_with(a, b):
        log = []
        with Tracker(log) as n:
            log.append(n + a)
        return log

    same_in_both(_program(strict_with), [(1, 0)])
    assert _program(strict_with, strict=True).call("f", 1, 0) == strict_with(1, 0)


@dataclass
class Point:
    x: int
    y: int = 0
    tags: list = field(default_factory=list)


def records(a, b):
    p = Point(a)
    q = Point(a, y=b)
    r = Point(y=a, x=b)
    q.tags.append(1)
    return (p.x, p.y, p.tags, q.y, q.tags, r.x, r.y, p.tags is q.tags)


def test_dataclass_keywords_and_defaults():
    same_in_both(_program(records), [(1, 2), (5, -3)])
    assert _program(records, strict=True).call("f", 1, 2) == records(1, 2)


def python_manager(a, b):
    import contextlib

    out = []
    with contextlib.suppress(ZeroDivisionError):
        out.append(a / b)
    return out


def test_with_a_python_context_manager():
    same_in_both(_program(python_manager), [(1, 2), (1, 0)])


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
