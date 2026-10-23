# Contributing to Archimedes

Thanks for your interest in improving Archimedes. This guide covers how to set up a development environment, how
the code is organised, and what we look for in a pull request.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Contributions are accepted under the
[Apache License 2.0](LICENSE), the project's license (section 5 of the license).

## Ways to contribute

- **Report a bug:** open an issue with the *Bug report* form. Include steps to reproduce, what you expected, and
  your browser / OS / Python version. For gesture problems, say what the on-screen quality hint showed.
- **Suggest a feature:** open an issue with the *Feature request* form and describe the problem it solves.
- **Improve the docs:** typo fixes and clearer explanations are always welcome.
- **Fix or build something:** comment on an issue first, so that nobody duplicates work, then open a pull request.

Security problems should **not** go in public issues. See [SECURITY.md](SECURITY.md).

## Development setup

You need Python 3.10 to 3.12, Git, and Node.js 18+ for the CAD kernel tests. A webcam is optional: everything has a
mouse path and a scripted gesture tour.

```bash
git clone https://github.com/<your-fork>/archimedes.git
cd archimedes
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
make test && make lint
make demo                                            # http://localhost:8000 and /workbench/
```

GitHub Codespaces works out of the box: the dev container runs `.devcontainer/setup.sh`.

The DOLFINx solver core is optional and runs in Docker (`cd Archimedes/backend && docker compose up`).

## Where things live

| Area | Path | Notes |
|---|---|---|
| Gesture pipeline | `archimedes_gestures/` | Pure Python and NumPy; MediaPipe is imported lazily in `tracker.py` only |
| FE core | `archimedes_fe/` | `solver.py` (member brick solver), `meshsolver.py` (general hex8), `scripting.py` (console) |
| Web server | `webdemo/server.py` | FastAPI routes, WebSocket protocol, `/api/fe/*` |
| Gesture modeller UI | `webdemo/static/` | Plain ES modules, three.js from a CDN, no build step |
| Workbench UI | `Archimedes/Archimedes Workbench.dc.html` | Single-page app, no build step |
| CAD kernel | `Archimedes/cadkernel.js` | UMD module: runs in the browser and in Node for tests |
| DOLFINx backend | `Archimedes/backend/` | Separate Python project with its own `pyproject.toml` |
| Tests | `tests/` | pytest; `cadkernel_test.js` runs in Node |

Read [`docs/DESIGN.md`](docs/DESIGN.md) before changing the gesture vocabulary, and
[`Archimedes/docs/cad-workflow.md`](Archimedes/docs/cad-workflow.md) before changing Workbench behaviour.

## Making a change

1. **Branch from `main`:** `git checkout -b feature/short-description` (or `fix/…`, `docs/…`).
2. **Keep changes focused.** One topic per pull request is much easier to review.
3. **Add or update tests.**
   - Solver changes need a check against a closed-form result (beam theory, Euler buckling, a patch test).
   - Gesture changes need a test that uses the synthetic hand model (`archimedes_gestures/synthetic.py`).
   - CAD kernel changes need a case in `tests/cadkernel_test.js`.
4. **Run the checks** locally:
   ```bash
   make lint          # ruff, line length 120
   make test          # pytest (runs the Node kernel tests too when Node is installed)
   node tests/cadkernel_test.js
   ```
5. **Check UI changes in a browser.** Open the page, look for console errors, and attach a screenshot to the PR.
6. **Update the docs** when behaviour changes: the README, the relevant guide in `docs/` or `Archimedes/docs/`, and
   the *Unreleased* section of [CHANGELOG.md](CHANGELOG.md).

## Style

- **Python:** formatted and linted with [Ruff](https://docs.astral.sh/ruff/) (`pyproject.toml`). Use type hints on
  public functions and docstrings that state units (mm, N, MPa).
- **JavaScript:** match the surrounding code; no build step and no new runtime dependencies without discussion.
- **Units:** the FE code works in mm, N, MPa and t/mm³ throughout. Convert at the edges, never inside.
- **Comments:** explain *why*, not *what*.

## Commit messages

Write a short summary in the imperative mood (≤ 72 characters), a blank line, then the reasons for the change:

```
Snap rim nodes onto both faces in body-fitted meshing

Nodes on the edge between a cylinder and its cap were projected onto
one surface only, leaving a jagged rim.
```

## Pull requests

- Fill in the pull request template and link the issue it closes (`Closes #123`).
- CI must be green: lint, tests on Python 3.10 to 3.12, and the CAD kernel tests.
- A maintainer reviews every pull request. Expect questions; they are about the code, not about you.
- Pull requests are merged with a merge commit, so keep your branch history readable.

## Datasets and models

Recorded clips, personal calibration samples and trained models stay out of the repository (`data/` and
`models/*.joblib` are git-ignored). If you train on a public dataset such as HaGRID, respect that dataset's license
and say which data you used in your pull request.
