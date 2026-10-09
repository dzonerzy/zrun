# Changelog

All notable changes to zrun are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2026-10-09

### Added
- **Strict mode: `zrun.Language(..., strict=True)`.** Compiled code calling nothing in Python, or the reason it can't: a semantic outside the compilable subset (or `native=False`) is a `CompileError` when it's registered; code that would call into Python (a Python function or builtin, a Python object as a value, `isinstance()` of a class not the language's, a module variable a function rebinds, a semantic run as Python) a `CompileError` when the program compiles, at the semantic's line; what only shows as the code runs (a host function in Python) a `zrun.StrictError` (a `CompileError`) at the program's line, saying what it called. Errors (still Python's exceptions) and code compiled while the program runs are allowed. In `run()`, `call()` and `map()`.

- **A `def` inside a semantic or helper is compiled** where it's called (in the function, in itself, in the others defined there): a function of its own, given what it reads of the functions around it as they are when it's called (what a closure reads then). Used as a value (returned, kept, passed), it isn't compiled yet. The Lua example's `table.sort` (a recursive merge sort defined inside it) is native.
- **`program.report()["python_functions"]`**: the Python functions compiled code called that couldn't be compiled, and why. Strict mode's message for a call to one says why too.
- **`@zrun.comptime`**: a function whose result depends only on its arguments, its author says. Compiled code calling it with values known when compiling (constants, the tree's fields, other such results) calls it then, once for those values in the process: its result is a constant of the code, whatever Python is in the function (the compilable subset or not). Lists, dicts and tuples in a result are native all through at run time and read-only (a semantic changing one runs as Python). With values known only at run time it's called as any function is; the reference mode calls it as ever. A table built from the spec while compiling, looked up by the code at run time: no Python.

### Performance
- **More of Python compiled natively**, the same results and errors as the reference mode (an error's words Python's own: its function called for them):
  - `int(s, base)` of an ASCII str, `chr()`, `ord()`, `id()` (of any value: an object's address, None's, True's and False's CPython's; a number's, odd and from its bits), `list()` of a list, tuple, dict or str, `min()` and `max()` of numbers and strs, `len(s.encode("utf-8"))` (no bytes made);
  - `math.floor`, `ceil`, `sqrt`, `fabs`, `degrees`, `radians`, `isnan`, `isinf`, `isfinite`, `copysign`, and on Linux `fmod`, `exp`, `log`, `log2`, `log10`, the trigonometric and hyperbolic functions, `expm1`, `log1p`, `atan2`, `pow` (from the process's libm: CPython's, the same to the last bit); math's functions of known numbers decided when compiling;
  - `list + list`, `tuple + tuple`, a list, tuple or str times an int;
  - a list's and tuple's `index()` and `count()`;
  - `for ... in d.items()` (`keys()`, `values()`) of a dict known only at run time (a dict changed while the loop runs isn't the RuntimeError Python's view raises);
  - `rt.load()` and `rt.store()` of an rt handed to other code as a value (out-of-line code, a function called with it);
  - a builtin given known values it refuses (a branch the code never takes): its error when the code runs, not a call to Python;
  - `s[i]` of a str that isn't ASCII (an index of its code points made the first time it's indexed: O(1), as Python's);
  - `for ... in enumerate(items, start)`, the start known when compiling;
  - `str(x)`, `f"{x}"`, `f"{x!r}"` of a float (its shortest digits in Python's layout: the same as repr()), a bool, None;
  - `fmt % args` (printf-style): `%d %i %u %o %x %X %c %s %r %%`, and on Linux `%e %E %f %F %g %G` (C's printf: correctly rounded, the same digits as CPython's), flags, widths and precisions (`*` too), as Python's str formats them (`0` with a precision, zeros for `inf`);
  - `float.hex(x)`, `x.hex()`; `int(s, base)` with a base known only at run time;
  - str methods of any str, not only ASCII ones: `find`, `rfind`, `index`, `rindex`, `count` (with a start and an end), `startswith` and `endswith` (a tuple too), `replace`, `split` (a separator, or whitespace; a maxsplit), `strip`, `lstrip`, `rstrip` (whitespace Python's, or the characters given), `join`; `float(s)` of a str (its syntax Python's, the number correctly rounded); a float's `is_integer()`.

  The Lua example's numeric loop (`math.floor`, `min`): 0.099 s, was 0.119. Of the Lua test programs in strict mode, seven go into Python only to print.
- **`except rt.Return as r` using `r.args` only is native**: the jump's value kept in the handler, no Python exception made (the same for rt.Break and rt.Continue). The Lua example's chunk (a `return` at the top level) needs no Python for it. A handler using the name otherwise gets the exception, as before.
- **`x is y` of two Python objects known when compiling is decided then**: `type(1.5) is float` makes no Python object at run time.
- **Helpers with native paths no longer take the GIL first**: unpacking a list or tuple, methods of strs, lists and dicts done natively, `int()`, `str()`, `abs()`, `bool()`, `len()` of native values. Code running them on several threads (`map()`) doesn't take turns for them.

### Fixed
- **A program loaded again in the same process was compiled again**, and its compiled code not found in the cache: each load's code had names of its own (two programs alive at once can't share the JIT's), so its IR, so its key. A load now runs the code an earlier load of the program (the same language, source and path) compiled, while that one lives and after it goes (the last eight kept); with tiers, the code compiled fast too, and the optimized code an earlier load was having made when it went. Lua's `fib(30)` loaded ten times in a process, an empty cache: 1.55 s for the first load, then 0.040 s each (was 1.5 s each).
- **A semantic calling a Python object it has as a value** (`fs[i](x)`) called it as `rt.call` does: its ints as I64s, its error worded as a host function's (`sqrt: ValueError: ...`). It's called as Python calls it now, as the reference mode does.
- **An int beyond 128 bits a Python function gave compiled code went into the program**: the reference mode's I64 refuses one beyond 64 bits (integer overflow), compiled code refused them only up to 128 (`math.floor(1e300)`). Both refuse them now.
- **An attribute of a module or class that may be rebound was decided when compiling**: `sys.stdout` read in a semantic was the one bound when the program compiled, so a second run with `sys.stdout` redirected wrote to the first run's. An attribute that isn't a function, class, module or value (an instance, a list...) is now read when the code runs.

## [0.4.1] - 2026-10-08

### Performance
- **f-strings format strs and ints natively** in compiled code (`f"{name}"`, `f"{n}"`, `f"{n:d}"`, with `!s` too): no call into Python; other values and format specs are Python's, as before. The Lua example's messages and number formatting use f-strings.

## [0.4.0] - 2026-10-08

### Added
- **`build_executable(..., setup="module:function")`**: a function of a module beside the language's, called with the program's path and its arguments before it runs in the executable. The Lua example's `lua:set_args` makes `arg` from them.

### Performance
- **Floor division and modulo of ints are inline** in compiled code (Python's semantics: the result rounded toward minus infinity, the remainder the divisor's sign), the helper only for a divisor of 0 or, for `//`, -1. The Lua example's integer `//` and `%` no longer wrap results to 64 bits through arithmetic beyond them (floor division and modulo of 64-bit ints fit, but for minint // -1, wrapped as Lua's). Lua's `s = (s + i * i) % 1000003` ten million times: 0.12 s, was 0.34 (Lua 5.4: 0.07).
- **Helpers compiled out of line count references inline**, as functions' and loops' code does: they run for each call of the code calling them. The Lua example's tables make their hash part when a key outside the array part is first set (an array or a constructor's table: two objects, not three). Binary trees, depth 14, 8 times: 0.088 s, was 0.141.
- **A Python function compiled code called before is found by the function** (no `__closure__` or `__code__` looked up for each call), and **a record's method called where its class isn't known while compiling** runs its compiled code directly, the method found once per class (no Python object made for the record, the bound method or the name). Lua building a string of 200,000 numbers: 0.066 s, was 0.093.

### Fixed
- A staticmethod of a record's class, called on a record (`r.helper(x)`), was given the record as its first argument: compiled code took it for a method, and a record seen from Python bound it to the record. Records' class attributes are now looked up as Python does (the attribute as the class defines it, its `__get__` applied): staticmethods, classmethods and properties as on the class's instances.

## [0.3.0] - 2026-10-08

### Added
- **Tail calls: `rt.tail_call(f, args, receiver=None)`.** Returns what calling `f` returns, the function being run left first: a chain of tail calls takes no depth, in every mode (compiled code gives the frame up and its caller makes the call; the reference mode raises `zrun.TailCall` for the call running it). The semantics' `finally` blocks run before the call, an `except` catching `zrun.TailCall` takes it; at the top level it's a call. The Lua example makes `return f(x)` one, as Lua does: `count(200000)` written as tail recursion runs (examples/lua/tests/tailcalls.lua, its output Lua 5.4's).

### Fixed
- A program in which a function got a typed entry (it got hot) and code ran out of line for a semantic run as Python could be refused by LLVM ("duplicate definition of symbol"): typed entries and that code were named alike.

## [0.2.0] - 2026-10-08

Requires zgram 0.5.0 (compiled grammars kept on disk: a program starts in a fraction of the time) and zrules 0.2.0.

### Added
- **The cache's size is limited**: 1 GiB by default, `zrun.configure(cache_size=bytes)` (0: no limit). A process compiling code looks the cache over when it first writes to it and after each tenth of the limit it writes: past the limit, the compiled code used least recently goes, down to 80% of the limit (code loaded from the cache counts as used). Files a stopped process left half-written are deleted after an hour.
- **`zrun.clear_cache()`**: the cache emptied.
- **Executables**: `zrun.build_executable(language, source, output, target=None, python=None)` makes one file running the program on a machine with no Python: a Python runtime (python-build-standalone's, 3.10 to 3.14), zrun, zgram, zrules, the language's module, the program and its compiled code, behind a small launcher that unpacks them once into the cache directory. For `x86_64-linux` and `x86_64-windows`, built from either. Needs the `ziglang` package (`pip install zrun-py[exe]`).

### Fixed
- A compiled module is loaded where the language's files have moved (another install, another directory): their paths were in the definition's hash.

### Performance
- **Typed functions taking lists get a typed entry** from their declaration, as those taking ints, floats and bools have: called directly from typed code, the list given borrowed (no reference counted per call, unless the function stores to the parameter). Arguments read from variables stay borrowed for the call. A typed function summing a 10-item list, called in a loop: 0.44 ns per item (C: 0.21), was 0.71.
- **The program's variables no function refers to are kept on the stack** while its code runs: in registers, their kinds known from what's stored, the checks on them folded. They are in the program's frame wherever code that sees the frames runs (semantics run as Python, code out of line, an `rt` handed out) and when the program ends. Lua's `for i = 1, 10000000 do s = s + i end` at the top level: 340 instructions an iteration before; LLVM now works the loop out.
- A function taking extra arguments (`extra="keep"`) called with none gets the empty tuple without a call.
- **Lists of numbers only aren't tracked by the cycle collector** (as CPython's containers of atoms): made and freed without its bookkeeping, tracked once they may hold a container. Collections go through fewer objects.
- **Lua example: a function returning one value returns it, not a list of one** (a call taking its first result makes no list; `nil` alone and several values stay a list). `fib(30)`: 0.040 s, was 0.068 (Lua 5.4: 0.025). From Python, `program.call()` of such a function gives the value.

## [0.1.0] - 2026-10-08

The first release.

### Added
- **Languages from Python semantics.** `zrun.Language(parser, rules)` over a zgram parser and zrules rules; `@lang.eval(kind)` and `@lang.exec(kind)` say what each kind of node does, as Python functions taking the node and `rt`; `lang.function(kind, ...)` makes nodes of a kind the language's functions (frames, closures, hoisting, missing and extra arguments); `@lang.host` adds Python functions the program calls by name. Variables are zrules' symbols: `rt.load` and `rt.store` are slot accesses.
- **Compiled execution.** `program.run(mode="compiled")`: the semantics are partially evaluated for the program's tree and compiled to native code with the LLVM zgram ships (through its `zgram.llvm.v1` capsule). What's known while compiling (node fields, kinds, operator texts, module constants) is decided then. A semantic outside the compilable subset runs as Python, and `lang.python_semantics()` says why. Output, errors, their locations and call stacks are the same as the reference mode's (`mode="python"`).
- **Tiers and `mode="auto"`.** A program whose optimized code isn't cached runs code compiled fast while the optimized code is made on other threads; `mode="auto"` runs as Python until it's ready. `zrun.configure(tiers=False)` optimizes at once.
- **Typed values.** `lang.types(mapping)` says what values a language's types (zrules' `types()`) have; values are checked once where they're made, and typed code keeps ints and floats unboxed, calling typed functions with plain arguments.
- **Speculation.** Hot functions get a typed entry for the kinds of arguments they're called with (ints, floats, bools, lists), guarded where they're entered.
- **Engines.** `program.call(name, *args, context=None)` calls a program's function (about 0.15 µs from Python), releasing the GIL; `program.map(name, items, threads=None)` runs calls on native threads; `zrun.Bytes` and `rt.u8(data, i)` ... `rt.i64be(data, i)` read data in place; `lang.native_host(name, capsule)` registers a native library's function, called by compiled code directly.
- **Compiled modules.** `program.save(path)`, `lang.compile(source, output)` and `lang.load_compiled(path)`: a program's compiled code saved and loaded in another process without compiling, refused for another definition of the language, another zrun or another CPU.
- **A cache** of compiled code between processes, in the platform's cache directory (`zrun.configure(cache=...)`).
- **Memory**: reference counting with a cycle collector of CPython's design (`zrun.collect()`). Compiled lists, dicts and records are shared with Python through proxies, not copied.
- **Sessions and a REPL.** `lang.session()` runs programs one after another, each seeing what the ones before it defined; `lang.repl()` reads entries from `input()`.
- **`rt.wrapping_add`, `_sub`, `_mul`, `_shl`, `_shr`, `_ushr`**: 64-bit arithmetic that wraps around, native in compiled code. Plain integer arithmetic is checked: an overflow is the program's error.
- **`program.report()`**: where compiled code went through Python, what became of the semantics' module state, the cache, the tier running, GIL acquisitions, typed entries.
- **Examples**: tiny (a small language), a typed language, Lua 5.4 (with its test programs and a REPL), and scan (a YARA-like rule engine with native host functions).
- **Documentation**: the README and a guide to writing fast semantics.
