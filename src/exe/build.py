"""zrun.build_executable(): a program and what it needs to run, as one
executable file.

What it holds: a Python runtime (python-build-standalone's, for the target),
zrun, zgram and zrules (the installed ones for this machine's platform,
else their wheels for the target's, from PyPI), the language's module and
the other modules beside it, the program, and its compiled code (when the
target is this machine's: compiled ahead of time; else compiled where it
first runs, then cached there). The launcher (launcher.zig, linked with the
ziglang package's Zig) unpacks them once into the platform's cache
directory and runs the program with the Python of the payload, which embeds
Python for what the program needs of it (semantics run as Python, host
functions).
"""

import hashlib
import importlib
import importlib.util
import os
import platform
import posixpath
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zlib

# python-build-standalone: the release used, and its CPython for each minor
# version (the one building's by default)
RELEASE = "20261003"
PYTHONS = {10: "3.10.22", 11: "3.11.17", 12: "3.12.15", 13: "3.13.16", 14: "3.14.8"}
TRIPLES = {"x86_64-linux": "x86_64-unknown-linux-gnu", "x86_64-windows": "x86_64-pc-windows-msvc"}
WHEEL_PLATFORMS = {"x86_64-linux": "manylinux_2_17_x86_64", "x86_64-windows": "win_amd64"}
ZIG_TARGETS = {"x86_64-linux": "x86_64-linux-musl", "x86_64-windows": "x86_64-windows-gnu"}
PACKAGES = (("zrun", "zrun-py"), ("zgram", "zgram-py"), ("zrules", "zrules-py"))

MAGIC = b"ZRUNEXE1"

MAIN = r'''"""The program of this executable (zrun.build_executable() wrote it)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "site"))

import zrun  # noqa: E402

# (compiled code kept where the program was unpacked: one compiling where
# the compiled module was made for another CPU)
zrun.configure(cache=os.path.join(HERE, "cache"))
lang = getattr(__import__({module!r}, fromlist=["_"]), {attr!r})
SOURCE = os.path.join(HERE, "program", {name!r})
COMPILED = os.path.join(HERE, "program.zrc")
SETUP = {setup!r}


def load():
    # (the language's setup, given the program's path and arguments: Lua's
    # `arg`)
    if SETUP is not None:
        module, _, func = SETUP.partition(":")
        getattr(__import__(module, fromlist=["_"]), func)({name!r}, sys.argv[1:])
    if os.path.exists(COMPILED):
        try:
            return lang.load_compiled(COMPILED)
        except ValueError:
            pass
    with open(SOURCE, encoding="utf-8") as f:
        return lang.load(f.read(), {name!r})


try:
    load().run(mode="compiled")
except zrun.LoadError as e:
    print(e, file=sys.stderr)
    sys.exit(2)
except zrun.Error as e:
    print(e, file=sys.stderr)
    sys.exit(1)
'''


def host_target():
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise ValueError("executables are made for x86-64: this machine is %s" % machine)
    return "x86_64-windows" if sys.platform == "win32" else "x86_64-linux"


def cache_dir():
    """Downloads kept between builds: zrun/build in the platform's cache."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "")
        base = xdg if os.path.isabs(xdg) else os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, "zrun", "build")
    os.makedirs(path, exist_ok=True)
    return path


def fetch(url, name):
    """A file downloaded once into the cache: its path."""
    path = os.path.join(cache_dir(), name)
    if os.path.exists(path):
        return path
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, path)
    return path


# What of the runtime a program doesn't use: its tests, the tools (IDLE,
# pip's installer, 2to3), Tk, the headers, the commands besides Python's
UNUSED_DIRS = {"test", "tests", "idle_test", "idlelib", "tkinter", "turtledemo", "ensurepip", "lib2to3",
               "site-packages", "include", "share", "Scripts", "tcl", "pkgconfig"}
UNUSED_DIR_PREFIXES = ("config-", "tcl", "tk", "itcl", "thread")
UNUSED_FILES = ("_tkinter", "libtcl", "libtk", "tcl8", "tk8", "tcl86", "tk86")


def unused(parts, target):
    """Whether the runtime's file python/`parts` is one a program doesn't
    use."""
    dirs, name = parts[1:-1], parts[-1]
    if any(d in UNUSED_DIRS or d.startswith(UNUSED_DIR_PREFIXES) for d in dirs) or name.startswith(UNUSED_FILES):
        return True
    if target.endswith("linux"):
        # (bin/python3 has libpython in it: the libraries are for embedding)
        return (dirs == ["bin"] and name != "python3") or (dirs == ["lib"] and name.startswith("libpython"))
    return False


def python_runtime(target, version, into):
    """python-build-standalone's CPython for the target (checked against the
    release's SHA256SUMS), what a program uses of it unpacked at
    `into`/python, links as the files they lead to (made the same on every
    platform: Windows' file names and links aren't Linux's). Returns the
    executables' paths (`python/...`)."""
    name = "cpython-%s+%s-%s-install_only_stripped.tar.gz" % (version, RELEASE, TRIPLES[target])
    base = "https://github.com/astral-sh/python-build-standalone/releases/download/%s/" % RELEASE
    path = fetch(base + name, name)
    sums = fetch(base + "SHA256SUMS", "SHA256SUMS-" + RELEASE)
    with open(sums, encoding="utf-8") as f:
        wanted = {line.split()[1]: line.split()[0] for line in f if len(line.split()) == 2}
    with open(path, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if wanted.get(name) != got:
        os.remove(path)
        raise ValueError("the Python runtime downloaded (%s) isn't the release's: try again" % name)
    files, links, executables = {}, {}, set()
    with tarfile.open(path) as t:
        for m in t.getmembers():
            parts = m.name.split("/")
            # (the archive's own paths only: python/...)
            if parts[0] != "python" or ".." in parts:
                continue
            if m.isfile():
                files[m.name] = m
            elif m.issym():
                links[m.name] = posixpath.normpath(posixpath.join(posixpath.dirname(m.name), m.linkname))
            elif m.islnk():
                links[m.name] = m.linkname
        for name, real in links.items():
            for _ in range(40):
                if real not in links:
                    break
                real = links[real]
            if real in files:
                files[name] = files[real]
        for name, m in files.items():
            if unused(name.split("/"), target):
                continue
            out = os.path.join(into, *name.split("/"))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with t.extractfile(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            if m.mode & 0o100:
                executables.add(name)
    return executables


def packages(target, site):
    """zrun, zgram and zrules into `site`: this Python's, for this machine's
    platform; for another, the same versions' wheels for it, from PyPI."""
    os.makedirs(site, exist_ok=True)
    if target == host_target():
        for module, _ in PACKAGES:
            m = importlib.import_module(module)
            shutil.copy2(m.__file__, site)
        return
    import zgram
    import zrules
    import zrun

    import json
    import zipfile

    versions = {"zrun-py": zrun.version(), "zgram-py": zgram.version(), "zrules-py": zrules.version()}
    for _, dist in PACKAGES:
        # (PyPI's list of the version's files: its abi3 wheel for the platform)
        with urllib.request.urlopen("https://pypi.org/pypi/%s/%s/json" % (dist, versions[dist])) as r:
            files = json.load(r)["urls"]
        wheels = [f for f in files if f["filename"].endswith("-abi3-%s.whl" % WHEEL_PLATFORMS[target])]
        if not wheels:
            raise ValueError("PyPI has no %s %s wheel for %s" % (dist, versions[dist], target))
        wheel = fetch(wheels[0]["url"], wheels[0]["filename"])
        with open(wheel, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != wheels[0]["digests"]["sha256"]:
                os.remove(wheel)
                raise ValueError("the wheel downloaded (%s) isn't PyPI's: try again" % wheels[0]["filename"])
        with zipfile.ZipFile(wheel) as z:
            for item in z.namelist():
                # (the modules, not the wheel's metadata)
                if "/" not in item and not item.endswith(".pyi"):
                    z.extract(item, site)


def language_of(language):
    """(module name, attribute, files to bring) of a language given as
    'module:attribute' or as the Language object itself."""
    import zrun

    if isinstance(language, str):
        module, _, attr = language.partition(":")
        attr = attr or "lang"
        importlib.import_module(module)
    elif isinstance(language, zrun.Language):
        found = [(name, a) for name, m in list(sys.modules.items()) if m is not None
                 for a, v in list(getattr(m, "__dict__", {}).items()) if v is language and name != "__main__"]
        if not found:
            raise ValueError("the language isn't an attribute of an imported module: give it as 'module:attribute'")
        module, attr = found[0]
    else:
        raise TypeError("language must be a zrun.Language or 'module:attribute'")
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ValueError("can't find the module %s" % module)
    return module, attr, spec


def bring_module(spec, site):
    """The language's module into `site`: its package, or itself and the
    Python modules beside it (those it imports from its folder)."""
    if spec.submodule_search_locations:
        top = spec.name.split(".")[0]
        root = os.path.dirname(spec.origin)
        for _ in spec.name.split(".")[1:]:
            root = os.path.dirname(root)
        shutil.copytree(root, os.path.join(site, top), ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
        return
    folder = os.path.dirname(spec.origin)
    for name in os.listdir(folder):
        if name.endswith(".py"):
            shutil.copy2(os.path.join(folder, name), site)


def pack(root, executables):
    """The payload: every file under `root`, compressed (launcher.zig says
    how); those in `executables` (paths from `root`, `/` between names)
    made executable where they're unpacked."""
    out = bytearray()
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(folder, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/").encode("utf-8")
            with open(full, "rb") as f:
                data = f.read()
            mode = 1 if rel.decode("utf-8") in executables else 0
            packed = zlib.compress(data, 6)
            out += struct.pack("<I", len(rel)) + rel + bytes([mode]) + struct.pack("<QQ", len(packed), len(data)) + packed
    out += struct.pack("<I", 0)
    return bytes(out)


def launcher(target, source, into):
    """launcher.zig built for the target, with the ziglang package's Zig."""
    try:
        import ziglang  # noqa: F401
    except ImportError:
        raise RuntimeError("making executables needs Zig: pip install ziglang (or zrun-py[exe])") from None
    src = os.path.join(into, "launcher.zig")
    with open(src, "w", encoding="utf-8") as f:
        f.write(source)
    out = os.path.join(into, "launcher.exe" if target.endswith("windows") else "launcher")
    subprocess.run(
        [sys.executable, "-m", "ziglang", "build-exe", src, "-OReleaseSmall", "-fstrip", "-target", ZIG_TARGETS[target],
         "-femit-bin=" + out, "--cache-dir", os.path.join(into, "zig-cache"), "--global-cache-dir", os.path.join(cache_dir(), "zig")],
        check=True,
        cwd=into,
    )
    with open(out, "rb") as f:
        return f.read()


def build_executable(language, source, output, target=None, path=None, python=None, setup=None, launcher_source=None):
    """The executable: see the module's doc. Returns its path."""
    import zrun

    target = target or host_target()
    if target not in TRIPLES:
        raise ValueError("target must be one of %s, not %r" % (", ".join(sorted(TRIPLES)), target))
    minor = int(str(python).split(".")[1]) if python else sys.version_info[1]
    if minor not in PYTHONS:
        raise ValueError("no Python 3.%d runtime: 3.10 to 3.14" % minor)
    module, attr, spec = language_of(language)
    lang = getattr(importlib.import_module(module), attr)
    if setup is not None:
        if not isinstance(setup, str) or ":" not in setup:
            raise TypeError("setup must be 'module:function' (a function of a module beside the language's)")
        smod, _, sfunc = setup.partition(":")
        if not callable(getattr(importlib.import_module(smod), sfunc, None)):
            raise ValueError("setup: %s has no function %s" % (smod, sfunc))
    if os.path.exists(source):
        name = os.path.basename(source)
        with open(source, encoding="utf-8") as f:
            text = f.read()
    else:
        name, text = path or "program", source
    name = path or name
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "root")
        os.makedirs(os.path.join(root, "program"))
        # (compiled here: this machine's platform; the CPU's, checked where
        # it runs. Another's: checked here, compiled where it first runs)
        if target == host_target():
            lang.compile(text, os.path.join(root, "program.zrc"), path=name)
        else:
            lang.load(text, name)
        with open(os.path.join(root, "program", name), "w", encoding="utf-8") as f:
            f.write(text)
        executables = python_runtime(target, PYTHONS[minor], root)
        packages(target, os.path.join(root, "site"))
        bring_module(spec, os.path.join(root, "site"))
        with open(os.path.join(root, "main.py"), "w", encoding="utf-8") as f:
            f.write(MAIN.format(module=module, attr=attr, name=name, setup=setup))
        payload = pack(root, executables)
        head = launcher(target, launcher_source, tmp)
        digest = hashlib.sha256(payload).digest()
        footer = MAGIC + struct.pack("<QQ", len(head), len(payload)) + digest
        if target.endswith("windows") and not output.lower().endswith(".exe"):
            output += ".exe"
        out = "%s.%d.tmp" % (output, os.getpid())
        with open(out, "wb") as f:
            f.write(head)
            f.write(payload)
            f.write(footer)
        os.chmod(out, 0o755)
        os.replace(out, output)
    return output
