"""sorted() and list.sort(), natively: the same order (stable, reverse,
key), the same errors as Python's sort."""

import math
import random

from test_intrinsics import _program, same_in_both


def sorting(a, b):
    s = sorted(a)
    r = sorted(a, reverse=True)
    k = sorted(a, key=lambda x: -x if isinstance(x, (int, float)) else x)
    c = list(a)
    c.sort(reverse=b)
    return (s, r, k, c)


def test_numbers_strings_tuples():
    rng = random.Random(1)
    cases = [
        ([rng.randrange(-50, 50) for _ in range(30)], False),
        ([rng.random() * 10 - 5 for _ in range(30)], True),
        ([1, 2.5, -3, 2**70, -(2**65), 0.0, True, False], False),
        ([], True),
    ]
    same_in_both(_program(sorting), cases)


def words(a, b):
    return (sorted(a), sorted(a, key=len), sorted(a, key=str.lower, reverse=True), sorted([(len(w), w) for w in a]))


def test_strings_and_keys_stable():
    same_in_both(_program(words), [(["pear", "Apple", "fig", "kiwi", "apple", "Fig", "é", "zz"], 0)])


def odd(a, b):
    return sorted(a)


def test_nan_and_mixed_types_as_python():
    same_in_both(_program(odd), [([3.0, math.nan, 1.0, 2.0], 0), ([1, "a", 2], 0), ([(1, "x"), (1, 2)], 0), ([None, 1], 0)])


def in_place(a, b):
    x = a
    y = x
    y.sort(key=lambda t: t[1])
    return (x, x is y)


def test_sort_in_place_seen_by_every_variable():
    same_in_both(_program(in_place), [([(1, "b"), (2, "a"), (3, "b"), (4, "a")], 0)])


def strict_sorting(a, b):
    # (a list made natively: one given by Python stays Python's)
    xs = [(i * a) % b for i in range(20)]
    ys = list(xs)
    ys.sort(key=lambda x: -x)
    return (sorted(xs), sorted(xs, key=lambda x: x % 3, reverse=True), ys)


def test_native_in_strict_mode():
    same_in_both(_program(strict_sorting), [(7, 11), (3, 5)])
    assert _program(strict_sorting, strict=True).call("f", 7, 11) == strict_sorting(7, 11)
