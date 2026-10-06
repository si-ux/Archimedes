# Python console and scripts

Like the Abaqus command line: everything the gestures and the mouse can do is also a Python command, and
everything you do, by any route, is journaled as the command that repeats it.

Open it with **🐍 Python** (bottom left of the viewport) or the **`** key. Lines run in the server's Python, in your
modeller session, so the viewport updates as you type.

```python
>>> beam(b=300, h=500, L=6000)
>>> material('steel')
>>> pinned(at=0); roller(at=1)
>>> uniform(10, dir='-z'); point(30, at=0.5)
>>> solve()
>>> peak('U3')
{'field': 'U3', 'value': -1.57, 'at_mm': (3000, 150, 0), ...}
>>> show('S11'); cut('x', 0.5)
```

Units: **mm, kN, kN/m, MPa**. Positions along the member (`at`, `start`, `end`) go from 0 (start) to 1 (end).
Directions: `'+x' '-x' '+y' '-y' '+z' '-z'` (world axes, z up). `help()` lists the commands, `help(point)` explains one.

## Commands

| Group | Command | Does |
|---|---|---|
| Geometry | `beam(b, h, L)` · `column(b, h, L)` | rectangular member along +x / +z (`name=` optional) |
| | `round_beam(d, L)` · `round_column(d, L)` | circular member |
| | `material('steel')`, `material('concrete')` | S355 / C30 presets |
| | `material('C40', E=35000, nu=0.2, rho=2.5e-9, strength=40)` | your own (MPa, t/mm³) |
| | `demo('continuous')` | `continuous`, `cantilever`, `long_column`, `short_column` |
| | `increments(length=100, section=25, load=1)` | snap steps for values entered by hand |
| Supports | `fixed(at)` · `pinned(at)` · `roller(at)` | boundary conditions (a new one at the same spot replaces the old) |
| Loads | `point(P, at=1, dir='-z')` | point load, kN |
| | `uniform(w, start=0, end=1, dir='-z')` | uniform line load, kN/m |
| | `trapezoidal(w0, w1, start=0, end=1, dir='-z')` | linearly varying, kN/m |
| | `clear('loads' / 'supports' / 'all')` · `undo()` | |
| Solve & results | `solve()` | static + 6 modes; returns the results (also in `results`) |
| | `fields()` · `field('S11')` | field names · nodal values (numpy array) |
| | `peak('von_mises')` | largest \|value\|, where, min and max |
| | `at(0.5, 'U3')` | mean over the cross-section at a position |
| | `reactions()` · `modes()` · `summary()` | support reactions (kN) · frequencies (Hz) · section / buckling summary |
| View | `show('U3')` · `cut('x', 0.4)` · `cut((1, 0, 1), 0.5)` · `uncut()` · `view(azimuth, elevation)` | |
| Scripts | `run('file.py')` · `journal()` · `help()` | |

Also in the namespace: `model` (the live `Model`; edits to it are noticed and journaled), `results`, `np`, `math`,
and the classes `Model`, `Member`, `Section`, `Material`, `Load`, `Support`.

The console is a normal Python prompt: blocks (`for`, `def`, `if`) continue with `...` until a blank line, the value
of an expression is printed, errors show a traceback, and pasted multi-line code runs as one script. Tab completes
command names, ↑ ↓ walk the history, Esc cancels a block (or closes the panel), Ctrl+L clears.

## Scripts and modules

| Way | How |
|---|---|
| **Import script…** (console bar) | pick one or more `.py` files on your computer; they run in the console, then `run('name.py')` repeats them |
| `run('column_study.py')` | runs a file from `fe_scripts/`, or a path relative to the project |
| `import my_tools` | `fe_scripts/` is on the import path. Modules call the commands through `from archimedes_fe.scripting import api` (`api.point(...)`, `api.model`) |
| headless | `python -m archimedes_fe.scripting fe_scripts/column_study.py` (like `abaqus cae noGUI=`), or no argument for a terminal prompt (`make console`) |

`fe_scripts/` has examples: a beam-theory check, a parametric column study (a loop over heights printed as a table) and
a helper module (`self_weight()`, `span_loads(w, n)`).

## The journal (replay file)

Every edit is recorded as a command, wherever it came from. Edits made by gesture or mouse also appear in the
console as `# from gesture / mouse`, so you can learn the command for anything you do by hand. **Save journal**
downloads the session as `archimedes_journal.py`. Replay it with *Import script…* or `run(...)`. It rebuilds the same
model: geometry, material, supports, loads.

## Safety

The console runs arbitrary Python in the server process, with your permissions: that is what makes custom scripts
possible. Anyone who can open the page can use it. Codespaces ports are private by default, so only you can. If you
make the port **public** (for example to open `/xr` on a headset), start the server with the console off:
`ARCHIMEDES_CONSOLE=0 make demo`. A script runs to completion: an endless loop keeps that browser session busy
until the server restarts.
