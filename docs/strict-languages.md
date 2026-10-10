# Strict languages

A language made with `zrun.Language(..., strict=True)` promises that its compiled code calls nothing in Python. zrun holds it to that: whatever would run as Python is an error, saying where and why, instead of a slow path you didn't notice. In return the program's code runs with no Python at all: as fast as it gets, on threads without the GIL (`program.map`), and as a standalone executable with no Python in it (`zrun.build_native`).

This guide says what strict mode refuses, how to read its errors, and how to fix what they point at. The Lua example (`examples/lua`) is strict without a change: all of it, its library included, compiles natively.

## Turning it on

```python
lang = zrun.Language(PARSER, RULES, strict=True)
```

Errors come at three moments, each at the earliest point it can be known:

| When | Error | What |
|---|---|---|
| A semantic is registered | `zrun.CompileError` | its source isn't in the compilable subset (`yield`, `async`, a nested class...), or it's `native=False` |
| A program is compiled | `zrun.CompileError` | its code would call into Python: a Python function that can't be compiled, a Python object used as a value, a module variable the code rebinds |
| A program runs | `zrun.StrictError` (a `CompileError`) | it went into Python anyway: a host function given only at run time that can't be compiled |

Each message names the place: the semantic's file and line for the first two, the program's for the third, and why:

```
lang.py:42:12: in call_(): strict: calls the Python function shout() (it would run in Python)
strict: fib.tiny:7:1: the compiled code went into Python while it ran, calling shout(), which isn't compiled: **kwargs can't be compiled (...)
```

Errors themselves are allowed: an exception raised by compiled code is still a Python exception where Python catches it, and its words are Python's.

## What compiles natively

Everything in [Writing fast semantics](writing-fast-semantics.md#what-compiles), and with no Python:

- values: ints (64-bit, checked), floats, strs, bytes, `None`, tuples, lists, dicts, sets, records (dataclasses, classes with `__slots__`), the language's functions, closures, nodes
- their operators and methods (`str`'s common ones, `partition`, `split`, `encode`...; lists', dicts', sets' all), `**`, `%` formatting, f-strings and their format specs
- builtins: `len`, `int` (a base too), `float`, `str`, `repr`, `bool`, `abs`, `min`, `max`, `sorted`, `chr`, `ord`, `isinstance`, `type`, `range`, `zip`, `enumerate`, `print`
- `math`'s functions, `struct.pack` / `unpack`, `int.from_bytes` / `to_bytes`, `zlib.crc32` / `adler32`, `bytes.fromhex`
- output: `print(...)`, `sys.stdout.write(s)`, `sys.stderr.write(s)` (to whatever `sys.stdout` is when they run)
- exceptions of Python's builtin classes and zrun's: raised, caught, their `args` and `str()`
- the module's tables and records the semantics read and change: made native when the program is compiled (`program.report()["module_state"]` says which)

## Fixing what strict mode refuses

**"calls the Python function f()"** — f is called from compiled code but can't be compiled itself. The message says why (`**kwargs can't be compiled`, `the source of f() isn't available`...). Make f compilable: plain parameters and `*args` are fine, `**kwargs` and keyword-only parameters aren't. A function of a C library (`re.match`, `json.loads`) has no Python source to compile: write what you need of it in the subset (the Lua example's pattern matcher is ~300 lines of plain Python), or give a native library's function with `lang.native_host`.

**"uses X, a Python object, as a value"** — the code holds a Python object at run time: a class, a module, an object of a class that isn't a record. Use values compiled code knows: a record (a dataclass) for structured data, a str or an int for a tag. A record class used to construct records or check them (`isinstance(v, Table)`) is fine (`isinstance` of a class that isn't a record is refused: "asks isinstance() of the class X"); `type(v)` compared with `type(w)` or with a class is fine; keeping `type(v)` in a variable isn't.

**"calls the host function f()"** — a host function given to the program (`lang.host(...)`) that can't be compiled: the same fixes as above.

**"assigns the module variable x (`global x`)"**, **"reads the module variable x, which a function rebinds"** — Python's variable, Python's to change. Keep the state in a module-level record or dict and change its contents instead (`STATE.count += 1`): that's made native and shared.

**A `StrictError` at run time** — something given to the code only as it runs: a host function called through a variable, a Python function stored in a table. It's compiled when first called; one that can't be is the error, naming it and why.

## Finding them before they bite

```python
program.run(mode="compiled", report=True)   # not strict: everything runs, crossings counted
program.report()["python_functions"]          # Python functions that couldn't be compiled, and why
program.report()["python_crossings"]          # every place compiled code went into Python, how often
lang.python_semantics()                       # semantics that run as Python, and why
```

A language not strict yet runs with all of that counted: make the crossings go away one by one, then turn strict mode on to keep them away.

## The Lua example

`examples/lua/lua.py` is a strict language as it is. A few of its choices are what strict mode asks for:

- its library is a table of records (`Builtin(name, fn)`) holding plain functions of the module: the table is made native when a program is compiled, each function compiled when the program calls it (for a standalone program, when it's built)
- its values are records (`Table`, `Method`) and Python's own ints, floats, strs, `None`: no class of its own for numbers or strings
- `io.write` and `print` end in `sys.stdout.write(text)`: native output
- `string.gmatch` returns a closure, `table.sort` a merge sort written in the module: no Python function of a C library anywhere
- errors are `rt.Throw(value, message)`: the language's errors, natively

## Standalone programs

A strict language's program builds into an executable with no Python:

```python
zrun.build_native(lang, "script.lua", "script")                  # everything the program may call
zrun.build_native(lang, "script.lua", "script", prune=True)      # only what it names: smaller, faster to build
```

See [Standalone programs](../README.md#standalone-programs) for what's in one and what isn't.
