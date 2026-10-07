# Writing fast semantics

zrun compiles your semantics with each program: it runs them while compiling, with the program's tree known, and emits native code for what's left. Any Python works (a semantic that can't be compiled runs as Python), but semantics written with the compiler in mind run much faster. This guide says what compiles, how to see what didn't, and what makes compiled code fast.

## How zrun reads a semantic

A semantic is read from its Python source (`inspect.getsource` and `ast`), once, when it's registered. Then, for each program, the compiler *runs* it over the program's tree, with these known while compiling:

- the node, its fields, its kind, its text: `node.op`, `node.kind`, `node.name.text`
- constants: literals, module-level names that are never assigned while the program runs (a dict of operators, a table of escapes)
- what's derived from those: `node.op == "+"`, `OPS[node.op]`, `len(node.args)`, a loop over `node.children`

Everything known is decided while compiling; only what depends on values the program computes becomes code. So

```python
@lang.eval("BinOp")
def binop(node, rt):
    a, b = rt.eval(node.left), rt.eval(node.right)
    if node.op == "+":
        return a + b
    if node.op == "-":
        return a - b
    ...
```

compiles, for a `BinOp` node whose `op` is `"+"`, to the code of its two operands and one checked add: the comparisons on `node.op` are gone. Write semantics the natural way, branching on what the node says; the branching costs nothing.

## What compiles

| | Compiles |
|---|---|
| Values | `int`, `float`, `bool`, `str`, `None`, tuples, lists, dicts, records (dataclasses, classes with `__slots__`), the program's functions, nodes |
| Statements | assignment (including tuple unpacking and `x[i] = v`, `obj.f = v`, `+=`), `if`, `while`, `for` (over ranges, lists, tuples, dicts, strs, `zip`, `enumerate`), `break`, `continue`, `return`, `pass`, `assert`, `raise`, `try` / `except` / `else` / `finally` |
| Expressions | arithmetic, comparisons (chained too), `and` / `or` / `not`, `x if c else y`, indexing and slicing, list / dict comprehensions and generator expressions, f-strings, attribute reads, calls |
| Calls | other semantics, `rt`, helper functions of your module (compiled with them, inline when small), host functions, records' classes and methods |
| Builtins | `len`, `int`, `float`, `str`, `bool`, `abs`, `isinstance`, `type`, `range`, `zip`, `enumerate`, `print`; the others (`min`, `max`, `round`, `ord`, ...) are worked out while compiling when their arguments are known, and called through Python otherwise |
| Methods | of `str`: `upper`, `lower`, `strip`, `lstrip`, `rstrip`, `startswith`, `endswith`, `find`, `join`, `isdigit`, `isalpha`, `isspace`, `count`, `format`; of lists and dicts: `append`, `extend`, `insert`, `pop`, `index`, `count`, `sort`, `reverse`, `copy`, `get`, `keys`, `values`, `items`, `setdefault` |

Not compiled: `with`, `lambda`, `yield`, `global` / `nonlocal`, `del`, sets, `**` unpacking, keyword arguments to most builtins. A semantic using them is reported, and runs as Python:

```
>>> lang.python_semantics()
{'with_': "subset.py:10:5: in with_(): `with` can't be compiled"}
```

Mark a semantic `@lang.eval(kind, native=False)` to run it as Python on purpose (anything goes there: I/O, other libraries).

## Seeing what's slow

```python
program.run(mode="compiled", report=True)
program.report()
```

```
{'python_crossings': {'call_python ord(str)': 283, 'binary list add list': 300, ...},
 'module_state': {'G': 'native', 'ESCAPES': 'constant (only read)', ...},
 'speculated': {'line 92': 'int'},
 'gil_taken': 1, ...}
```

- **`python_crossings`**: each place compiled code went through Python, and how many times. Those are where the time goes: each is a call into CPython, and takes the GIL. `call_python ord(str)` is a builtin compiled code doesn't do itself; `binary list add list` an operation on values it handed to Python.
- **`module_state`**: your module's top-level tables. `constant` (only read: a constant in the code), `native` (changed by the semantics, but made a native table), or why it stayed a Python object (each access a crossing).
- **`speculated`**: functions of the program that got a typed entry for the kinds of their arguments.
- **`gil_taken`**: how many times compiled runs and calls took the GIL back. For an engine, 0 is what lets calls run in parallel.

## What makes compiled code fast

**Branch on the node, not on values.** Anything computed from the node is free; a test on a value costs a check at run time. Prefer `if node.op == "+"` to looking the operator up in a dict of Python functions and calling one.

**Let the values be zrun's.** Ints, floats, bools, strs, lists, dicts, tuples and records are native in compiled code. A Python object (an instance of an ordinary class, a set, a `bytes`) is a *host value*: every operation on it goes through Python. Use records (dataclasses or `__slots__` classes) for your language's objects: compiled code makes them natively and reads their fields directly.

**Declare types when the language has them.** With a zrules `types()` rule and `lang.types({...})`, a node's value is checked against its type once, where it's made; after that compiled code knows its kind: ints and floats in registers, calls between typed functions without argument lists.

**Keep lists of one kind.** A list whose items are all ints (or all floats) is marked so, and compiled code reads its items without checking them. Storing a value of another kind into it clears the mark for good.

**Integers are 64-bit and checked.** `a + b` of two ints is one add and an overflow check; an overflow is the program's error, at the node. If your language's ints wrap around (Lua's, hashes), use `rt.wrapping_add(a, b)`, `rt.wrapping_sub`, `rt.wrapping_mul`, `rt.wrapping_shl`, `rt.wrapping_shr`, `rt.wrapping_ushr`: native, where wrapping by hand (`(a + b) & MASK`, then fixing the sign) makes Python big ints first.

**Small helpers are free.** A helper function of your module called from a semantic is compiled with it: inline when it's small (or its first `if` is: the common case inline, the rest out of line), specialized for what the call knows. Factor freely.

**Host functions: small is fast, Python-only is slow.** A host function the program calls is compiled when it's a plain Python function of what compiles (`def len(x): return builtins.len(x)` runs inline); one that does real I/O or uses other libraries is a call into Python. For a hot one that can't be Python, write it in Zig or C and register it with `lang.native_host(name, capsule)`: compiled code calls it directly, without the GIL.

**Read data with `rt.u8` and friends.** For bytes a program inspects (files, packets), pass them as `zrun.Bytes` (or `bytes`, `mmap`: they become one) and read with `rt.u8(data, i)`, `rt.u32le(data, i)`, ...: inline, bounds-checked loads, no copies.

**Engines: load once, call many times.** Compiling is the expensive part; `program.call()` and `program.map()` reuse the compiled code. Calls that don't touch Python run in parallel; check `report()["gil_taken"]`.

## Same results, always

Whatever compiles must behave exactly as it does run as Python: the same output, the same errors at the same nodes, with the same call stacks. The test suite of a language should run its programs in both modes and compare (zrun's own `test/test_modes.py` does this for everything). When compiled code and the reference disagree, that's a zrun bug: please report it with the program and the semantics.
