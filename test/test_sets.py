"""Sets, natively (set.zig: CPython's table and hashes): the same items,
errors and order as the reference mode's sets."""

import random

import zrun
from test_intrinsics import _program, same_in_both


def basics(a, b):
    s = {a, b, 3}
    t = set(b) if isinstance(b, (list, str)) else {b}
    s.add(a)
    s.discard(99)
    u = s | t
    return (sorted(map(str, u)), len(s), a in s, 99 in s, s == {3, a, b}, list({x * 2 for x in range(5)}))


def test_basics():
    same_in_both(_program(basics), [(1, 2), (1, 1), ("x", "yz"), (2.5, [1, 2]), ((1, 2), "a")])


def native(a, b):
    s = {a, b}
    s.add(a + b)
    t = s | {b * 2}
    t -= {a}
    return (list(t), len(s & t), a in t, s.issubset(t), list({x % 3 for x in range(b)}))


def test_native_in_strict_mode():
    same_in_both(_program(native), [(1, 2), (3, 10)])
    assert _program(native, strict=True).call("f", 3, 10) == native(3, 10)


def ordered(a, b):
    # (the order Python's sets go over their items in: the same)
    s = set()
    for x in a:
        s.add(x)
    folded = {"alpha", "beta", "gamma", "delta", "epsilon"}
    t = {x for x in b}
    return (list(s), list(folded), list(t), list(s | t), list(s & t), list(s - t), list(s ^ t), list(t - s))


def test_order_is_pythons():
    p = _program(ordered)
    words = [f"w{i}" for i in range(40)]
    same_in_both(p, [
        (list(range(30)), list(range(20, 50))),
        ([-1, -2, 2**40, 2**62, 7.5, -0.25, 1e300], [7.5, 2**40, 3]),
        (words[:25], words[10:]),
        (["é", "日本", "a", "\U0001F600", "ab"], ["日本", "x"]),
        ([(1, 2), (2, 1), ("a", 1), ()], [(1, 2), (3,)]),
    ])


def operations(ops, b):
    # (a sequence of operations, run on a set: what it holds after each)
    s = set()
    out = []
    for op, x in ops:
        if op == 0:
            s.add(x)
        elif op == 1:
            s.discard(x)
        elif op == 2 and s:
            out.append(s.pop())
        elif op == 3:
            s.update(x)
        elif op == 4:
            s |= set(x)
        elif op == 5:
            s -= set(x)
        elif op == 6:
            s &= set(x) | s if x else s
        elif op == 7:
            s ^= set(x)
        elif op == 8:
            s = s.copy()
        out.append(list(s))
    return out


def test_random_operations_in_pythons_order():
    p = _program(operations)
    rng = random.Random(7)
    pool = list(range(-20, 60)) + [f"k{i}" for i in range(60)] + [i * 0.5 for i in range(20)]
    pool += [(i, "t") for i in range(10)] + [2**70 + i for i in range(5)] + [-(2**63) - i for i in range(3)] + [f"ü{i}" for i in range(5)]
    cases = []
    for _ in range(200):
        ops = []
        for _ in range(rng.randrange(1, 120)):
            op = rng.randrange(9)
            if op in (3, 4, 5, 6, 7):
                ops.append((op, [rng.choice(pool) for _ in range(rng.randrange(6))]))
            else:
                ops.append((op, rng.choice(pool)))
        cases.append((ops, 0))
    same_in_both(p, cases)


def methods(a, b):
    s = set(a)
    t = set(b)
    r = [s.issubset(t), s.issuperset(t), s.isdisjoint(t), s <= t, s < t, s >= t, s > t]
    r.append(sorted(s.union(b, [100])))
    r.append(sorted(s.intersection(b)))
    r.append(sorted(s.difference(b)))
    r.append(sorted(s.symmetric_difference(b)))
    c = s.copy()
    c.intersection_update(b)
    c.difference_update([1])
    c.symmetric_difference_update([2, 3])
    r.append(sorted(c))
    s.clear()
    r.append(len(s))
    return r


def test_methods():
    same_in_both(_program(methods), [([1, 2, 3], [2, 3]), ([], [1]), ([1, 2], [1, 2]), ([5], [])])


def errors(a, b):
    s = {1, 2}
    if a == 0:
        s.remove(b)
    if a == 1:
        set().pop()
    if a == 2:
        s.add([b])
    if a == 3:
        s |= [b]
    return s


def test_errors_are_pythons():
    same_in_both(_program(errors), [(0, 5), (1, 0), (2, 0), (3, 0), (4, 0)])


def given_to_python(a, b):
    s = {a}
    s.add(b)
    return s


def test_a_set_given_to_python():
    assert _program(given_to_python).call("f", 1, 2, mode="compiled") == {1, 2}


def inplace_lists(a, b):
    x = [a]
    y = x
    x += [b]
    x += (b,)
    return (y, x is y)


def test_inplace_operators_change_the_object():
    same_in_both(_program(inplace_lists), [(1, 2), ("a", "b")])


def test_iterating_and_type():
    def it(a, b):
        s = {a, b}
        total = []
        for x in s:
            total.append(x)
        return (total, type(s) is set, isinstance(s, set), bool(set()), list(set("hello")))

    same_in_both(_program(it), [(3, 4), ("q", "r")])


LITERALS = '''
def lits(a, b):
    return (list({"x", "y", "z"}), list({"p", "q", "r", "s"}), list({"k1", "k2", "k3", "k4", "k5", "k6", "k7"}),
            list({1.5, "a", (1, 2)}), list({b"a", b"b", b"c"}))
'''

CHECK = """
import sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import lits_mod
from test_intrinsics import _program
p = _program(lits_mod.lits)
print(p.call("f", 1, 2, mode="compiled") == lits_mod.lits(1, 2))
"""


def test_literals_in_the_order_of_a_pyc(tmp_path):
    # (a set literal of constants is a frozenset in the function's code: a
    # .pyc made by another process (another hash secret) lays it out as that
    # one's order of its items made it: the set the same as Python's either
    # way, made from source and loaded from the .pyc)
    import os
    import subprocess
    import sys

    from conftest import HERE

    (tmp_path / "lits_mod.py").write_text(LITERALS)
    for seed in ("11", "12", "13", "14"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", CHECK, str(tmp_path), HERE], capture_output=True, text=True, env=env, timeout=120)
        assert r.stdout.strip() == "True", (seed, r.stdout, r.stderr[-2000:])


def test_report_has_no_python_for_sets():
    p = _program(ordered)
    p.call("f", [1, 2], [2], mode="compiled")
    assert zrun is not None
