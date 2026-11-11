# Tech

- Python 3.13. One virtual environment at the repository root: `.venv` (`source .venv/Scripts/activate` in Git Bash, `.venv\Scripts\python` in PowerShell). Never install into the global interpreter.
- Runtime dependencies: `pandas`, `numpy`, `scikit-learn`. Config files are TOML, read with the standard-library `tomllib`. CLI uses `argparse`. Add no other runtime dependency without recording why in `docs/DECISIONS.md`.
- Dev dependencies: `pytest`, `hypothesis` (property-based tests), `ruff`.
- CPU only. Models are classic tabular models (scikit-learn, e.g. `HistGradientBoostingClassifier`, `LogisticRegression`). Every training run in the example apps must finish in under 60 seconds on a laptop CPU.
- Style: type hints on public functions, small functions, dataclasses for results, no global state. Pure functions for checks; file I/O only at the edges (`audit` writing outputs, the CLI).
- Errors: invalid input raises `hexperiment.HexInputError` with a message that says what is wrong and how to fix it. Never pass silently.
- Commands (run from the repository root):
  - Install: `.venv/Scripts/python -m pip install -e "hexperiment[dev]"`
  - Test: `.venv/Scripts/python -m pytest hexperiment/tests -q`
  - Lint: `.venv/Scripts/python -m ruff check hexperiment`
