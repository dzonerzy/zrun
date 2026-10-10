CPython 3.10's headers (the Stable ABI's floor), for building the runtime of
standalone programs (`libzrun_rt.a`) for each target `zrun.build_native`
makes programs for, on any machine: the runtime never calls Python (its
calls go to a stub), but shares its code, and so its declarations, with the
extension.

- `linux/include`: CPython 3.10.12's, as Ubuntu 22.04 ships them, with the
  x86-64 `pyconfig.h` in place of Debian's dispatching one
- `windows/include`: CPython 3.10.21's for x86-64 Windows, from
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone)

Both under the Python Software Foundation License
(https://docs.python.org/3/license.html).
