# Lua 5.4 on zrun

A subset of Lua 5.4: its full grammar in zgram, its static rules in zrules
(`lualang.py`), and what each kind of node does as plain Python (`lua.py`,
about 2,300 lines), compiled by zrun.

```bash
python lua.py script.lua [args...]     # run a script (`arg` holds its arguments)
python lua.py                          # a REPL
```

It is a demonstration of what zrun compiles, not a replacement for Lua: the
language's core is here, its libraries in part.

## What's implemented

- **The language**: nil, booleans, 64-bit integers that wrap and floats,
  strings, tables (an array part and a hash part), closures and upvalues,
  varargs and multiple results, methods (`a:b()`), integer and generic `for`,
  `while`, `repeat`, `break`, `<const>` locals (checked statically), and tail
  calls (`return f(x)` takes no stack).
- **Metatables**: `__index`, `__newindex`, `__call`, `__tostring`, `__name`,
  `__metatable`, `__pairs`, `__len`, `__eq`, `__lt`, `__le`, `__concat`,
  `__unm`, the arithmetic ones (`__add`, `__sub`, `__mul`, `__div`, `__mod`,
  `__pow`, `__idiv`) and the bitwise ones (`__band`, `__bor`, `__bxor`,
  `__shl`, `__shr`, `__bnot`).
- **Errors**: `error` with any value and a level, `pcall`, `xpcall`, Lua's
  messages with their locations.
- **Base library**: `assert`, `error`, `getmetatable`, `ipairs`, `next`,
  `pairs`, `pcall`, `print`, `rawequal`, `rawget`, `rawlen`, `rawset`,
  `select`, `setmetatable`, `tonumber`, `tostring`, `type`, `unpack`, `xpcall`.
- **string**: `byte`, `char`, `find`, `format`, `gmatch`, `gsub`, `len`,
  `lower`, `match`, `rep`, `reverse`, `sub`, `upper`, with Lua's patterns;
  strings' methods (`s:upper()`).
- **table**: `concat`, `insert`, `pack`, `remove`, `sort`, `unpack`.
- **math**: `abs`, `atan`, `ceil`, `exp`, `floor`, `fmod`, `log`, `max`,
  `min`, `modf`, `random`, `randomseed`, `sqrt`, `tointeger`, `type`, `ult`,
  and its constants (`pi`, `huge`, `maxinteger`, `mininteger`).
- **os**: `clock`, `getenv`, `time`. **io**: `write`.

## What's missing

- Coroutines (the `coroutine` library).
- `goto` and labels: parsed and checked, an error when run.
- `load`, `loadstring`, `dofile`, `require`: a program is one file.
- `utf8`, `string.pack`, `string.unpack`, `string.dump`.
- Most of `io` and `os` (files, `io.read`, `os.date`, `os.execute`, ...).
- Garbage-collection metamethods and modes: `__gc`, `__mode` (weak tables),
  `__close` (a `<close>` variable is a plain local), `collectgarbage`.
- The `debug` library.

## Tests

`tests/` holds Lua programs and the output real Lua 5.4 gives for them
(`.expected`): closures, functions, numbers, strings, OOP with metatables,
errors and library errors, tail calls, chunk returns and a set of larger
programs. `../../test/test_lua.py` runs each in every mode and compares.
