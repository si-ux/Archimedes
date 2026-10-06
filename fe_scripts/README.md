# fe_scripts

Scripts and modules for the modeller's Python console (🐍 Python, or the ` key).

| File | What it shows |
|---|---|
| `simply_supported_check.py` | build, load, solve, compare with beam theory: `run('simply_supported_check.py')` |
| `column_study.py` | a parametric loop over column heights, printed as a table |
| `my_tools.py` | a module of your own helpers: `from my_tools import self_weight` then `self_weight()` |

This folder is on the console's import path. A saved journal (*Save journal* in the console) can go here
and be replayed with `run('archimedes_journal.py')`.
