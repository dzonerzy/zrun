# Changelog

All notable changes to zrun are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`build_native(..., prune=True)`**: only the library functions a program can name compiled (those in the language's tables under one of the program's words or a str the code uses): a 10-line Lua program 0.9 MB instead of 1.7 MB, built in under a second. One reached by a name made at run time stops the program, saying the build was pruned; `left_out=[]` gets the names of those left out. `program.native_objects(prune=..., left_out=...)` too.
- **`build_native(..., setup=...)`**: a standalone program's arguments. A function (or `'module:function'`) compiled with the program, called with its path and its arguments (a list of strs) as it starts: `setup="lua:set_args"` makes Lua's `arg`, as `build_executable(setup=...)` does. `program.native_objects(setup=...)` too.
- **Natively**: `a ** b` of numbers (ints exactly within 128 bits, floats as CPython's float_pow: the C library's pow, its special cases and errors; Linux and Windows), `str.partition` and `rpartition`, `int(s, base)` of ints up to 128 bits.
- **A float's digits natively on every platform** (`%e %f %g`, `{:.3f}`...): Windows' were Python's (and refused in strict mode). The digits of the double's binary value rounded half to even, as CPython's own conversion: glibc's printf on Linux, zrun's own exact conversion elsewhere (checked against glibc on 60,000 values, `zig build test`).
- **An f-string's format spec known only at run time** (`f"{x:{spec}}"`) compiled, natively for ints, floats and strs; it was a `CompileError` in strict mode.

### Performance
- **Standalone builds cached**: each module's object kept in zrun's cache by its IR (as the JIT's are), the code made the same text in every build (names of what it is, not of the order it's made in; a Python object's index read from the image), each helper in a module of its own (with the item calling it, all given up if one fails). A program built again is linked from the cache (the Lua example's: 0.7 to 1 s, was 1.4 to 2.4 s); another program of the same language compiles its own code only, the library's taken from the cache. `zrun.configure(cache=False)` builds without.
- **Standalone programs half the size, built 3.5x faster again, running faster** (the Lua example's 10-line program: 3.6 MB to 1.7 MB, 0.9 MB pruned; its test programs built in 7.7 s together, 27 s before; fib 20% faster, loops 14%): a library function's code made once, shared by the helper calls out of line that are the same code; a helper that never returns (an error's: `lua_error`) called out of line, not inline at every error site; a helper that can't leave by a jump (rt.Return, rt.Break, rt.Continue) called without code for one at each call, each releasing what's held; `while True:` returning in run-time control flow not unrolled; Python objects' counts (stand-ins, in a standalone program) no code at each count. The JIT's code as before.
- **Standalone builds 2.2x faster** (the Lua example's test programs: 60 s to 27 s together): each language function a module of its own and the top level's optimized alongside the rest, not before them; a module past 10000 blocks (Lua's pattern matcher) optimized a level down, its code as fast.
- **The JIT's huge modules optimized a level down too** (past 10000 blocks, as standalone builds do): the Lua example's test programs compiled optimized in 16% less time again, their code as fast.
- **Less code, compiled faster** (the JIT's optimized code: 21% less time): an error's way out gives counts back by calls, not inline; a helper inline 16 times in one function's code out of line after that; a field of a record a local holds read and written without taking a count of the record. Run time as before.

### Documentation
- **[Strict languages](docs/strict-languages.md)**: what `strict=True` refuses, its errors, how to fix what they point at, the Lua example as a case.

### Fixed
- **Format specs, natively as Python has them** (found with 120,000 random specs): zero padding with a separator puts separators in the zeros (`f"{1234:010,}"`: `00,001,234`); `0` with an alignment and no fill pads with zeros (`f"{-1:>08}"`: `000000-1`); a sign in a str's spec is Python's ValueError; `#` with no type gives a float in exponent form its point (`1.e-07`).
- **`type(a) is type(b)` with one side known while compiling and the other not** refused in strict mode (the known side a Python object), and could crash: both compared natively now.
- **CI runs every supported Python** (3.10 to 3.14), not 3.10, 3.12 and 3.14 only.
- **Code compiled as a program runs needing a function's frame** (a library function calling back into the program, a node evaluated at run time) failed instead of compiling the function again with its variables in a frame, as the program's own code does.

## [0.5.0] - 2026-10-10

Needs zgram 0.5.2 (its LLVM capsule's `LLVMAddAlias2`, for standalone programs).

### Added
- **Strict mode: `zrun.Language(..., strict=True)`.** Compiled code calling nothing in Python, or the reason it can't: a semantic outside the compilable subset (or `native=False`) is a `CompileError` when it's registered; code that would call into Python (a Python function or builtin, a Python object as a value, `isinstance()` of a class not the language's, a module variable a function rebinds, a semantic run as Python) a `CompileError` when the program compiles, at the semantic's line; what only shows as the code runs (a host function in Python) a `zrun.StrictError` (a `CompileError`) at the program's line, saying what it called. Errors (still Python's exceptions) and code compiled while the program runs are allowed. In `run()`, `call()` and `map()`.

- **A `def` inside a semantic or helper is compiled** where it's called (in the function, in itself, in the others defined there): a function of its own, given what it reads of the functions around it as they are when it's called (what a closure reads then). Used as a value (returned, kept, passed), it isn't compiled yet. The Lua example's `table.sort` (a recursive merge sort defined inside it) is native.
- **More of Python's syntax compiled**, as the reference mode runs it:
  - `x := value` (in an `if`'s or a `while`'s test, in a comprehension: the variable the function's);
  - `del x` of a local (reading it after is a `CompileError`), `del d[k]`, `del items[i]` (Python's errors for a missing key, an index out of range);
  - `raise E(...) from cause` and `from None` (the cause kept, as Python keeps it);
  - `import m`, `import a.b`, `import a.b as c`, `from m import x as y` inside a semantic or helper (the module's, imported when compiling);
  - `f(*args)` (the items unpacked when they're known, a list made at run time otherwise);
  - constants of other types (`b"..."`, `2j`, `...`): Python's objects, held; `isinstance()` of `bytes`, `bytearray`, `complex`, `set`, `frozenset` natively; `os.fspath()` of a str;
  - **closures**: a `lambda`, a nested `def` used as a value (returned, kept in a list, a dict or a record, passed), `nonlocal`: native functions of compiled code. The variables they read live in a heap frame of the run they're made in, shared as Python's cells are (a closure sees them change, `[lambda: i for i in ...]` the last `i`); a def only called where it's defined still gets them as arguments. Calling one with the wrong number of arguments is Python's TypeError, its words. One handed to Python is a TypeError (compiled code's only). The Lua example's `string.gmatch` (an iterator closure) is native: counting the numbers of a 200,000-number string with it, 0.73 s, was 48.7 s;
  - `global name` in a semantic: the module's variable assigned (read when the code runs; Python's, refused in strict mode);
  - **sets, natively**: `{a, b}`, set comprehensions, `set()`, `set(items)`, `in`, `len()`, iteration, `add`, `discard`, `remove`, `pop`, `clear`, `copy`, `update`, `union`, `intersection`, `difference`, `symmetric_difference` (and their `_update`s), `issubset`, `issuperset`, `isdisjoint`, `| & - ^` and their in-place forms, `<= < >= >`, `==`, `isinstance(x, set)`, `type(x)`. CPython's own table and hashes (its probing and resizing, SipHash of strs with the process's key, its int, float and tuple hashes): a set made by the same operations goes over its items in Python's order (checked against CPython 3.10, 3.11 and 3.13 on random operations). Errors are Python's (`KeyError` of a missing item, an unhashable one's `TypeError`). A set handed to Python is a Python set of its items (its order Python's for them);
  - **`with`**, as PEP 343 runs it: `__enter__`, `__exit__` given the exception (suppressing it if it returns true) or `None`s (on a return, a break, the body's end), several items. A record's methods compiled, as any: a `with` over a dataclass is native;
  - a dataclass made with keyword arguments, and with fields left to their defaults (`default`, `default_factory`); `list()`, `dict()`, `tuple()` of nothing;
  - **Python `bytes`, natively**: literals, `bytes()` (of nothing, a count, ints, data), `bytes.fromhex()`, `s.encode()` (UTF-8, ASCII, Latin-1), `b.decode()`, `b[i]`, slices (views, not copies), `+`, `*`, `in`, `len()`, `==`, as dict keys and set items (CPython's hash), `find`, `rfind`, `index`, `rindex`, `count`, `startswith`, `endswith`, `hex`, `upper`, `lower`, `strip`s, `split`, `replace`, `join`, `isinstance(x, bytes)`, `type(x)`. A bytes given back to Python is the object it was, or a bytes of its bytes;
  - **`struct.unpack`, `unpack_from`, `pack`** (`<`, `>`, `!`, `=`, `@` with native sizes and alignment; `x c b B ? h H i I l L q Q f d s`), **`int.from_bytes`** (`signed=` too), **`n.to_bytes`**, **`zlib.crc32`**, **`zlib.adler32`**, of bytes and of zrun.Bytes. Python's errors where it raises (`struct.error`, `OverflowError`...);
  - **`sorted()` and `list.sort()`**, `key=` (a lambda's, a builtin's) and `reverse=`: a stable sort of keys compared natively (numbers exactly, strs, bytes, tuples and lists of them); keys it can't compare (a NaN, mixed types) put in Python's order by Python's own sort, its errors.
- **Errors natively**: compiled code's exceptions of Python's builtin classes (in CPython's hierarchy), zrun.IntegerOverflow, zrun.Error, rt.Throw, struct.error, dataclasses' FrozenInstanceError are raised, matched by `except` and caught with no Python object: `raise ValueError(...)`, `raise rt.Throw(value, message)`, `raise ... from ...`, a bare `raise`, `except (KeyError, IndexError) as e`, `e.args`, `e.value`, `e.message`, a zrun.Error's `e.diagnostic.message`, `str(e)`, `type(e)`, `type(x).__name__`. An exception handed to Python is Python's (its class, its arguments, its cause); an uncaught one is the zrun.Error the reference mode gives. Exceptions of other classes are Python's, as before. With `rt.scope()` of an rt handed over, f-string format specs and `abs()` of the least 64-bit int natively, all ten of the Lua example's test programs run in strict mode without Python.
- **Standalone programs: `zrun.build_native(language, source, output)`.** A strict language's program compiled ahead of time, all of it, and linked with zrun's runtime into one executable with no Python in it (Linux, for this machine; about 180 KB for a small program: only the runtime it reaches is linked; starting in a millisecond). What the JIT compiles as a program runs is compiled when building: the code of nodes evaluated by code knowing them only at run time, the Python functions a language holds as values for every way the code calls them (closures too, their captured variables as they are then). The values the code refers to (module tables, records, constants, a closure's function) are in the executable, made as it starts. Output is `print()`'s and `sys.stdout`/`sys.stderr.write()`'s; a runtime error is written as the reference mode words it, exiting with 1. All ten of the Lua example's test programs run standalone, as real Lua runs them. `program.native_objects()` gives the object files (to link with `libzrun_rt.a`, `zig build rt`); the extension carries the runtime on Linux. Needs the ziglang package.
- **Output, natively**: `print(...)` (`sep=`, `end=`, `file=sys.stdout` or `sys.stderr`, `flush=`, `*args` known only at run time), `sys.stdout.write(s)` and `sys.stderr.write(s)` (`from sys import stdout` too), written to `sys.stdout` and `sys.stderr` as they are then (redirected ones included). `str()` and `repr()` of native values as Python writes them: lists, tuples, dicts, sets (`[...]` for one inside itself), bytes, strs (quotes and escapes), records (a dataclass's repr), exceptions. The Lua example runs in strict mode without Python, printing included.
- **A Python function taking `*args` is compiled** (its arguments after the others a tuple): the tiny example's `print(*args)` is native.
- **f-string format specs, natively**: `f"{x:.14g}"`, `f"{n:>8,}"`, `f"{n:#x}"`, `f"{s:*^10}"`: the mini-language (fill and alignment, sign, `#`, `0`, width, `,` and `_`, precision, `d x X o b e E f F g G %` and none) for ints, floats and strs, as Python formats them (on Linux for floats: the C library's correctly rounded digits).
- **zrun.Bytes has the buffer protocol** (read only): `struct.unpack(fmt, data)`, `int.from_bytes(data)`, `zlib.crc32(data)`, `memoryview(data)` of a program's data, in every mode.
- **`program.report()["python_functions"]`**: the Python functions compiled code called that couldn't be compiled, and why. Strict mode's message for a call to one says why too.
- **`@zrun.comptime`**: a function whose result depends only on its arguments, its author says. Compiled code calling it with values known when compiling (constants, the tree's fields, other such results) calls it then, once for those values in the process: its result is a constant of the code, whatever Python is in the function (the compilable subset or not). Lists, dicts and tuples in a result are native all through at run time and read-only (a semantic changing one runs as Python). With values known only at run time it's called as any function is; the reference mode calls it as ever. A table built from the spec while compiling, looked up by the code at run time: no Python.

### Performance
- **More of Python compiled natively**, the same results and errors as the reference mode (an error's words Python's own: its function called for them):
  - `int(s, base)` of an ASCII str, `chr()`, `ord()`, `id()` (of any value: an object's address, None's, True's and False's CPython's; a number's, odd and from its bits), `list()` of a list, tuple, dict or str, `min()` and `max()` of numbers and strs, `len(s.encode("utf-8"))` (no bytes made);
  - `math.floor`, `ceil`, `sqrt`, `fabs`, `degrees`, `radians`, `isnan`, `isinf`, `isfinite`, `copysign`, `modf`, and on Linux `fmod`, `exp`, `log`, `log2`, `log10`, the trigonometric and hyperbolic functions, `expm1`, `log1p`, `atan2`, `pow` (from the process's libm: CPython's, the same to the last bit); math's functions of known numbers decided when compiling;
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
- **A function of a frozen module (Python 3.12's `os.path`) couldn't be read** (`inspect` finds no source for `<frozen posixpath>`): its source is read from the module's file; one with no source at all is a function Python runs, not an error.
- **Errors worded as this Python words them**: from 3.14, `division by zero` for every division, `cannot use 'list' as a set element (unhashable type: 'list')`; 3.13's `float modulo by zero`.
- **A set literal of constants loaded from a .pyc** goes over its items as Python's does: the .pyc's frozenset is laid out as the process that wrote it made it (another hash secret), and compiled code makes its copy so. Sets of strs and bytes on Windows too (the hash secret read from Python's DLL).
- **`repr()` of a float in compiled code wrote past Zig's formatting buffer** (smaller than its formatter asks for): undefined behavior, harmless so far by luck, an endless loop in a standalone program.
- **A list appended to itself** (`l.append(l)`, or to a list it's in) crashed compiling (a known list holding itself without end) or held a copy: it's the list itself now, as in Python.
- **A host function's native error** (compiled code's `KeyError`...) went up from `rt.call` as itself: it's the zrun.Error the reference mode raises for a host function's exception (an rt.Throw going on as itself).
- **An error leaving a semantic was still the exception it was** in compiled code: an `except KeyError` around `rt.eval(node)` caught the KeyError the node's semantic raised, and an error under a semantic run as Python was reported at that semantic's node. The reference mode makes an exception leaving a semantic the zrun.Error it is, at the node it was raised at (an rt.Throw going on as itself); compiled code now does the same, whichever semantics run as Python.
- **`x += y` of a list made a new list**: a variable referring to the same list didn't see it grow, as it does in Python (`list.__iadd__` extends it). `+=` of a list extends it now; `|=`, `&=`, `-=`, `^=` of a set and `|=` of a dict change them in place; a Python object's in-place operator is its own. Numbers' arithmetic stays inline.
- **A `break` or `continue` in a loop whose iterations were unrolled was ignored**: `for x in (1, 2, 3, 4): if x == 3: break` went over every item, and `while True: ... break` gave the code after it a state of none of its paths (`return out` returned None). A jump known when compiling now ends the unrolling (or goes on to the next iteration) there; a loop with one only at run time is compiled as a loop at run time.
- **An error could be lost as a program was freed** (`SystemError: error return without exception set`): freeing a Program cleared the exception being raised (`lang.load(...).run()` of a temporary in strict mode, on a cold cache). Freeing a Program or Language keeps it now.
- **A message too long for zrun's buffer raised nothing** (cut inside a UTF-8 character, Python couldn't make it): cut at a character now, longer ones fit.
- **Code compiled after a module LLVM refused could call that module's helpers** (`Symbols not found`): what the refused module was to compile is forgotten, as for one that failed compiling.
- **A known list or dict a `try` body read could be refused by LLVM** ("Instruction does not dominate all uses"): made a run-time object in the body, it was released on the paths an error takes out of it, where it wasn't made yet. It's given a slot before the body now.
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
