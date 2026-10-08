<div align="center">

<img src="https://raw.githubusercontent.com/dzonerzy/zrun/main/docs/assets/logo.svg" alt="zrun Logo" width="150">

# zrun

**Execution for languages built with [zgram](https://github.com/dzonerzy/zgram) and [zrules](https://github.com/dzonerzy/zrules): write an interpreter in Python, run programs as native code.**

You say what each kind of node *does*, as ordinary Python functions: an interpreter, the most natural way to define a language. zrun runs those functions, and it also compiles them: because a program's tree is known before it runs, zrun specializes your interpreter for that program and hands the result to LLVM. `node.op == "+"` is decided while compiling, evaluating a child becomes the child's own code, and with zrules' types an `a + b` of two ints becomes one machine add.

[![GitHub Stars](https://img.shields.io/github/stars/dzonerzy/zrun?style=flat)](https://github.com/dzonerzy/zrun)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![Zig](https://img.shields.io/badge/zig-0.16+-orange)](https://ziglang.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/dzonerzy/zrun/blob/main/LICENSE)

Built with [PyOZ](https://github.com/pyozig/PyOZ)

</div>

---

- **Semantics in Python, programs native.** The functions you write are compiled with each program (partial evaluation); a semantic outside the compilable subset runs as Python, and zrun tells you which and why.
- **The same behavior in every mode.** A program runs with its semantics as Python (the reference) or compiled; output, errors, error locations and call stacks are the same.
- **Scripts and engines.** Run a program from its start, or load it once and call its functions millions of times: from several Python threads, or on native threads with `map()`, over data passed without copies.
- **Ahead of time.** Save a program's compiled code as a module and load it in another process without compiling.
- **One LLVM.** zrun builds its code in memory through the LLVM zgram already ships, through zgram's capsule: no second copy, nothing to install.

Part of zsuite: zgram (syntax), zrules (semantics), zrun (execution), zlsp (editor support).

## Performance

### Typed code against C

[examples/typed](https://github.com/dzonerzy/zrun/tree/main/examples/typed), a statically typed language whose types zrules works out, against the same code in C (`gcc -O2`, with the same overflow checks, since zrun's ints raise on overflow):

| | zrun | C |
|---|---|---|
| `fib(30)`, recursive | 0.0043 s | 0.0022 s |
| summing a 1000-item list, `len()` in the loop's condition | 0.27 ns an item | 0.22 ns an item |

### A real language

[examples/lua](https://github.com/dzonerzy/zrun/tree/main/examples/lua) is Lua 5.4: a grammar in zgram, its static rules in zrules, and about 2,300 lines of semantics in Python (values, metatables, closures, varargs, the string, table and math libraries, `pcall`). It passes its test programs in both modes. Compiled against the same semantics run as Python, and against Lua's own interpreter (PUC-Rio's, written in C):

| Program | zrun, compiled | zrun, semantics as Python | Lua 5.4 |
|---|---|---|---|
| `fib(30)` | 0.041 s | 15.8 s | 0.026 s |
| a numeric loop, 10 million times (`s = (s + i * i) % 1000003`) | 0.119 s | 47.2 s | 0.075 s |
| binary trees, depth 14, 8 times | 0.074 s | 3.7 s | 0.023 s |
| building a string of 200,000 numbers | 0.064 s | 1.1 s | 0.037 s |

Lua is dynamically typed and its semantics here are generic Python; zrun compiles them to 17-400x the speed of running them, within 1.6-3.2x of a hand-written C interpreter. (A loop LLVM can reduce, `s = s + i`, is worked out while compiling: no loop left to time.)

### An engine

[examples/scan](https://github.com/dzonerzy/zrun/tree/main/examples/scan) is a small YARA-like rule language: conditions over a file's bytes, two native host functions (`count`, `entropy`) called without Python. Loaded once, called for 6,000 memory-mapped files (467 MB, page cache warm, files already mapped) with `program.map()`:

| Threads | Time |
|---|---|
| 1 | 0.159 s |
| 8 | 0.023 s |

No call takes the GIL. Every way of running it (threads, `map()`, the reference mode) gives the same results.

### Compiling

Compiled code is kept between runs (on disk, by what it compiles). A program run compiled whose optimized code isn't there yet is compiled fast first and starts at once, while the optimized code is made on other threads; later runs use it. `examples/lua/tests/programs.lua` with an empty cache: first run 3.3 s (26 s optimized at once); once its optimized code is all cached, 0.75 s for a new process's first run and 0.009 s a run after that.

## Installation

```bash
pip install zrun-py
```

Like every zsuite package, it is named `<name>-py` on PyPI; the module is `zrun`. It installs `zgram-py` 0.5.0 and `zrules-py` 0.2.0 or newer with it. Prebuilt wheels cover CPython 3.10+ on **x86_64 Linux** and **x86_64 Windows**.

### From source

Requires [Zig](https://ziglang.org/) 0.16 and the zgram and zrules packages.

```bash
pip install pyoz
pyoz build --release     # builds the wheel into dist/
pip install dist/*.whl
python -m pytest test
```

`zig build -Doptimize=ReleaseFast` alone produces `zig-out/lib/zrun.so` for quick iteration.

## Quick start

A language is a zgram parser, zrules rules (for its names and scopes), and semantics:

```python
import builtins
import zrun

lang = zrun.Language(PARSER, RULES)      # a zgram parser, a zrules Rules
lang.function("FuncDef")                 # `fn name(params) { body }` defines functions

@lang.exec("While")
def while_(node, rt):
    while rt.eval(node.cond):
        if not rt.loop(node.body):       # False: the body broke out
            break

@lang.exec(["Let", "Assign"])
def assign(node, rt):
    rt.store(node.name, rt.eval(node.value))

@lang.exec("Return")
def return_(node, rt):
    raise rt.Return(rt.eval(node.value))

@lang.eval("BinOp")
def binop(node, rt):
    a, b = rt.eval(node.left), rt.eval(node.right)
    if node.op == "+":
        return a + b
    if node.op == "<":
        return a < b
    ...

@lang.eval("Call")
def call(node, rt):
    return rt.call(rt.eval(node.name), rt.eval(node.args))

@lang.host                               # plain Python: anything goes
def print(*args):
    builtins.print(*args)

program = lang.load(source, "fib.tiny")  # parse, check: zrun.LoadError lists the errors
program.run()                            # semantics as Python: the reference
program.run(mode="compiled")             # native code
```

[examples/tiny](https://github.com/dzonerzy/zrun/blob/main/examples/tiny/tiny.py) is this language, complete in 150 lines. Node fields are the grammar's labels (`node.cond`, `node.body`); variables are zrules' symbols, so `rt.load(name)` and `rt.store(name, v)` are slot accesses, not lookups.

## Writing semantics

- `@lang.eval(kind)`: an expression's value, `fn(node, rt) -> value`. `kind` is a node kind (a zgram `->` class or rule name) or a list of them.
- `@lang.exec(kind)`: a statement, `fn(node, rt)`.
- `lang.function(kind, params="params", body="body", name="name", hoist=True, missing="error", extra="error")`: nodes of `kind` define functions; zrun makes, calls and returns from them, frames and closures included.
- `@lang.host` / `@lang.host("name")`: a Python function the program calls through the builtin of that name.
- `lang.native_host(name, capsule)`: a function of a native library (a `zrun.native.v1` capsule), called by compiled code directly.

What `rt` offers:

| | |
|---|---|
| `rt.eval(x)`, `rt.exec(x)` | run a child node (or each of a list) |
| `rt.load(name)`, `rt.store(name, v)` | the variable a name node refers to |
| `rt.function(node)`, `rt.call(f, args, receiver=None)` | make and call functions (the program's or a host function) |
| `rt.tail_call(f, args, receiver=None)` | return what calling `f` returns, the function's frame given up first: chains of tail calls take no depth |
| `rt.loop(body)` | run a loop's body once: `False` if it broke out |
| `raise rt.Return(v)`, `rt.Break()`, `rt.Continue()` | control flow |
| `raise rt.Throw(value)`, `rt.error(node, message)` | the language's errors |
| `rt.receiver`, `rt.varargs`, `rt.context`, `rt.path` | the method's object, extra arguments, the host's context, the program's name |
| `rt.scope(node)`, `rt.symbol(node)`, `rt.type_of(node)`, `rt.fresh(node)` | zrules' view of names and types; new loop variables each time round |
| `rt.u8(data, i)` ... `rt.i64be(data, i)` | integers read from data, bounds checked |
| `rt.wrapping_add(a, b)`, `_sub`, `_mul`, `_shl`, `_shr`, `_ushr` | 64-bit arithmetic that wraps around |

Tail calls are the language's to decide: a semantic calls `rt.tail_call(f, args)` where the language returns a call's result as it is (Lua's `return f(x)`), and the function is left before the call is made, in every mode, so `return loop(n - 1)` runs in constant stack and errors' call stacks keep one frame for the chain. The semantics' `finally` blocks run before the call; an `except` catching `zrun.TailCall` (or `Exception`) takes it, the call not made. At the top level, outside any function, it's a call whose result is returned.

Integers are 64-bit and checked: an overflow is a runtime error at the node (`zrun.IntegerOverflow`), not a silent wraparound; the `wrapping_` functions are for languages that want one (Lua, hashes).

Semantics are compiled from their Python source. What compiles, and how to write semantics that compile to fast code, is in [the guide to writing fast semantics](https://github.com/dzonerzy/zrun/blob/main/docs/writing-fast-semantics.md); `lang.python_semantics()` lists the ones that run as Python, and why.

## Modes

`program.run(mode=...)`:

- `"python"` (the default): semantics run as Python. The reference: compiled code must behave exactly like it.
- `"compiled"`: native code. Compiled fast the first time if the optimized code isn't cached; the optimized code is made in the background and runs from the run after it's ready (`zrun.configure(tiers=False)`: optimized at once).
- `"auto"`: as Python until the optimized code is ready, then compiled.

`program.report()` says where compiled code still went through Python and why, which state of the semantics' modules stayed Python, what the cache did, and which functions got typed entries.

## Engines

A program can be a set of entry points, loaded once and called many times:

```python
rules = lang.load(source)
rules.call("match", data, context=scan)                 # ~0.15 us a call from Python
rules.map("match", [(f,) for f in files], threads=8)    # native threads, no GIL
```

- **Calls** compile the program once and release the GIL while they run, taking it back only to touch Python; `report()["gil_taken"]` counts those times.
- **`map()`** runs the calls on native threads without returning to Python between items; the first failing item's error is raised as one thread calling them in order would raise it.
- **Data** (`bytes`, `bytearray`, `memoryview`, `mmap`) reaches the language as `zrun.Bytes`, read in place: slices share the memory.
- **Context**: `rt.context` is what the caller passed, for results and options.

## Types

With a zrules `types()` rule, say what values each type has:

```python
lang.types({"int": int, "float": float, "bool": bool, "str": str, "list": list})
```

A value is then checked against its node's declared type once, where it is made, and compiled code knows its kind from there: ints and floats unboxed, calls between typed functions with plain arguments in registers. A value that isn't what its type says is a runtime error at the node, never a wrong result.

## Ahead of time

```python
program = lang.compile(source, "rules.zrc")   # or program.save(path), after running it
program = lang.load_compiled("rules.zrc")     # another process: nothing compiled
```

A compiled module holds the program's source and its compiled code. Loading it checks a hash of the language's definition (grammar, semantics, host functions): a module made for another version of the language, another zrun or another CPU is refused, not misrun.

The compiled-code cache is in the platform's cache directory (`%LOCALAPPDATA%\zrun\Cache`, `~/Library/Caches/zrun`, `$XDG_CACHE_HOME/zrun` or `~/.cache/zrun`); `zrun.configure(cache=False)` or `cache="dir"` changes that. It takes at most 1 GiB (`zrun.configure(cache_size=bytes)`, 0 for no limit): past it, the compiled code used least recently is deleted, down to 80% of the limit. `zrun.clear_cache()` empties it.

## Executables

```bash
pip install "zrun-py[exe]"        # with the ziglang package: Zig, to link the launcher
```

```python
zrun.build_executable("lua:lang", "fib.lua", "fib", setup="lua:set_args")    # ./fib: for this machine, `arg` its arguments
zrun.build_executable(lua.lang, "fib.lua", "fib", target="x86_64-windows")   # fib.exe, made on Linux
```

One file runs the program, on a machine with no Python: in it are a Python runtime ([python-build-standalone](https://github.com/astral-sh/python-build-standalone)'s, 3.10 to 3.14: `python="3.12"`, by default the one building), zrun, zgram and zrules, the language's module with the Python modules beside it (or its package), the program and its compiled code. The language is given as `"module:attribute"` or as the `Language` itself (an attribute of a module imported). The program's errors are found when building (`zrun.LoadError`).

Run, it unpacks itself once into the platform's cache directory (`zrun/exe/<its hash>`) and runs the program there with the Python it brought: the semantics run as Python, host functions and Python's modules work as they do anywhere. The program's arguments are `sys.argv[1:]`, and `setup="module:function"` names a function (of a module beside the language's) called with the program's path and its arguments before it runs: `setup="lua:set_args"` makes Lua's `arg`. A runtime error is printed and exits with 1, an error found loading with 2.

The targets are `x86_64-linux` (glibc 2.17 or newer) and `x86_64-windows`, from either. For this machine, the compiled code is made when building; for the other, zrun, zgram and zrules come from PyPI (the same versions' wheels), and the program is compiled the first time it runs, then cached. Building downloads the runtime once (about 30 MB, kept in the cache directory under `zrun/build`). An executable takes 30 to 40 MB, about 100 MB unpacked. Its first run unpacks it and compiles the grammar (about 1.2 s for Lua's); later runs start in about 0.16 s (zgram keeps compiled grammars on disk).

## Sessions and the REPL

```python
lang.repl()                       # reads entries with input(), shows expression values
s = lang.session()
s.run("let x = 1;")
s.run("fn inc(n) { x = x + n; return x; }")
s.run("inc(5);")                  # 6: entries share the variables earlier ones defined
```

`python examples/lua/lua.py` with no script is a Lua REPL.

## Errors

A runtime error raises `zrun.Error`: `e.diagnostic` is a `zgram.Diagnostic` at the failing node, `e.stack` the language's calls that led there, and `str(e)` renders them:

```
fib.tiny:3:12: error: integer overflow [runtime]
    3 |     return fib(n - 1) + fib(n - 2);
      |            ^^^^^^^^^^^^^^^^^^^^^^^
  in fib(), called at fib.tiny:5:7
```

A Python exception a semantic raises (a `ZeroDivisionError`, a `KeyError`) is the program's runtime error at the node being run, worded as Python words it ("division by zero"); a host function's failure is worded `name: Type: message`. Errors found before running (syntax, the rules) raise `zrun.LoadError` with all the diagnostics.

## API Reference

Every class and method has its documentation in `help()` (and in the `.pyi` stubs the wheel ships).

| | |
|---|---|
| `Language(parser, rules=None, max_depth=1000)` | a language: a zgram parser, zrules rules |
| `@lang.eval(kind)`, `@lang.exec(kind)` | the semantics of a node kind (`native=False`: run as Python) |
| `lang.function(kind, ...)`, `@lang.host`, `lang.native_host(name, capsule)` | functions of the language, of Python, of a native library |
| `lang.types(mapping)` | what values the language's types have |
| `lang.load(source, path=None)` | parse and check a program: a `Program` |
| `program.run(mode=..., report=False)` | run it from its start |
| `program.call(name, *args, context=None)`, `program.map(name, items, threads=None)` | an engine's entry points |
| `program.report()`, `lang.python_semantics()` | what to look at to make it faster |
| `lang.compile(source, output)`, `program.save(path)`, `lang.load_compiled(path)` | compiled modules |
| `zrun.build_executable(language, source, output, target=None, python=None)` | the program as one executable file |
| `lang.session()`, `lang.repl()` | programs run one after another, sharing their names |
| `zrun.Bytes(data)` | data read in place |
| `zrun.configure(cache=, cache_size=, tiers=, perf_map=)`, `zrun.clear_cache()` | process-wide settings; the compiled-code cache |
| `zrun.collect()` | the cycle collector, now |
| `zrun.Error`, `zrun.LoadError`, `zrun.CompileError`, `zrun.IntegerOverflow` | errors |

## Architecture

```
zgram Tree + zrules Analysis (capsules, read in place)
     |
     v
[Program data]     -- nodes, fields, symbols, scopes as native arrays (program.zig)
     |
     v
[Front]            -- each semantic's Python source, read once (front.zig)
     |
     v
[Compiler]         -- the semantics run over the tree while compiling: what's known is
     |                decided, what isn't becomes LLVM IR (compile.zig, ir.zig)
     v
[LLVM]             -- zgram's, through its capsule: optimized, kept in the cache, loaded
     |                into its JIT (driver.zig, cache.zig, jit.zig)
     v
[Runtime]          -- tagged values, lists, dicts, records, frames; reference counts and
                      a cycle collector; Python's objects at the boundary (value.zig,
                      helpers.zig, gc.zig, proxies.zig, bridge.zig)
```

Key implementation details:

- **Partial evaluation**: compiling runs the semantics with the program's tree known. A node's fields, its kind, its operator text, a module's constant tables, a function's definition are values the compiler has; the code it emits is what remains. Parts it can't decide while compiling (a node evaluated only by a semantic run as Python) are compiled when first asked for, on their own.
- **Values**: 16-byte tagged words. Ints are 64-bit and checked; ints, floats and bools are never allocated. Lists, dicts and records are native and shared with Python through proxies when Python sees them, not copied.
- **Lists' elements kinds**: a list knows when all its items are ints or floats, so compiled code reads an item with its kind known.
- **Speculation**: a hot function gets a typed entry for the kinds of arguments it's called with (ints, floats, bools, lists), guarded where it's entered; calls from code that knows those kinds go to it directly, its arguments in registers.
- **Calls**: the program's functions are called through their code, the call stack kept as the reference mode's for errors; calls of host functions known while compiling jump to their compiled code.
- **Memory**: reference counting, with a cycle collector of CPython's design (generations, counted as containers are made). Functions read from the program's top level aren't counted on calls; one rebound while code runs is kept until the code returns.
- **Tiers**: a program whose optimized code isn't cached runs code LLVM compiled without optimizing, while the optimized code is made on other threads.

## Threads

- Compiled runs and calls release the GIL; code takes it back only to touch Python (host functions, semantics run as Python, Python values), and `report()["gil_taken"]` counts those times.
- `program.map()` runs calls on native threads; the program's variables and its module state are made immortal first, so the calls share them without counting. Calls mustn't change them.
- Each thread allocates values from lists of its own; LLVM's work in the background runs on worker threads, without the GIL.

## Known Issues

- Executables (a program with zrun's runtime, without Python) are planned, not built.
- x86_64 only: the LLVM zgram ships targets x86-64. No macOS wheels yet.
- Untyped code (Lua here) is 2.6-6x slower than a hand-written C interpreter; a program's top level isn't specialized as functions are.
- What compiles is a subset of Python (see the guide); the rest runs as Python, correctly but slower.
- On Windows with Python 3.12 or 3.13, `mode="python"` stops at a few hundred nested calls of the language's functions with "call stack too deep", before `max_depth`. Those Pythons allow 3000 calls nested through C on Windows (10000 elsewhere), and a call run as Python is several. Compiled code and other Pythons go to `max_depth`.

## Project Structure

```
src/
  lib.zig               # Python module: Language, Program, Runtime (rt), Session
  program.zig           # A loaded program's native data: tree, fields, symbols, scopes
  tree.zig              # zgram's parse tree as native code sees it (zgram.tree.v1)
  zrules_abi.zig        # zrules' analysis as native code sees it
  grammar.zig           # What zrun knows of a grammar: actions, kinds, labels
  front.zig             # Semantics read from their Python source
  compile.zig           # The compiler: semantics partially evaluated into LLVM IR
  ir.zig                # LLVM IR built in memory, through LLVM's C API
  jit.zig               # zgram's LLVM, through its zgram.llvm.v1 capsule
  driver.zig            # Compiled programs: IR to code, tiers, background compiling
  cache.zig             # Compiled code kept between runs
  aot.zig               # Compiled modules
  value.zig             # Values of compiled code and their runtime
  helpers.zig           # What compiled code calls: the execution context, the runtime
  bridge.zig            # Semantics run as Python inside compiled code
  proxies.zig           # Compiled lists, dicts and records seen from Python
  objects.zig           # Nodes, frames and functions of the reference mode
  types.zig             # zrun.I64, the errors and control-flow exceptions
  gc.zig                # The cycle collector
  pool.zig              # The values' allocator, per thread
  gil.zig               # Compiled code without the GIL
  adopt.zig             # Module state of the semantics made native
  bytes.zig             # zrun.Bytes: data read in place
  native.zig            # Native host functions (zrun.native.v1)
  wrapping.zig          # rt.wrapping_add and the other 64-bit wrapping operations
  pyhelp.zig            # Helpers over the Python C API
  exe/build.py          # zrun.build_executable(): the runtime, the packages, the payload
  exe/launcher.zig      # An executable's start: unpacks its payload, runs its Python
test/
  test_modes.py         # Differential tests: every program in every mode
  test_run.py           # Running programs: semantics as Python
  test_front.py         # Semantics read from their Python source
  test_typed.py         # The typed language: structs, methods, typed entries, errors
  test_lua.py           # Lua 5.4's test programs, like real Lua, in every mode
  test_engine.py        # program.call, map(), zrun.Bytes, native host functions
  test_scan.py          # A YARA-like engine: threads, map(), compiled modules
  test_boundary.py      # Values crossing between compiled code and Python
  test_state.py         # Module state the semantics change, made native
  test_scopes.py        # Block scopes and closures
  test_tail.py          # rt.tail_call: tail calls in every mode
  test_methods.py       # Records' methods called on values known only at run time
  test_gc.py            # The cycle collector, leaks, functions rebound while called
  test_cache.py         # The compiled-code cache between processes
  test_tiers.py         # Fast code first, optimized in the background
  test_aot.py           # Compiled modules
  test_exe.py           # Executables
  test_session.py       # Sessions and the REPL
examples/tiny/          # A small language: the quick start
examples/typed/         # A typed language: structs, methods, lists, optionals
examples/lua/           # Lua 5.4, with its test programs; a REPL
examples/scan/          # A YARA-like rule language: an engine, native host functions
docs/                   # The guide to writing fast semantics
build.zig               # Zig build configuration
pyproject.toml          # Python package configuration
```

## License

MIT
