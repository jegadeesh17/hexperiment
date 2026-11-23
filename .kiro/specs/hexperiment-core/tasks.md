# Implementation Plan: hexperiment-core

## How to use this plan

- Work the tasks in order. Each task ends with the whole suite green, so a later task never has to repair an earlier one.
- Run all commands from the repository root: `.venv/Scripts/python -m pytest hexperiment/tests -q` and `.venv/Scripts/python -m ruff check hexperiment`. Both must pass at the end of every task.
- Tests live next to the module they cover (`tests/test_<module>.py`) and are written in the same task as the module. Each check has example-based tests (a leaking case and a clean case) and at least one `hypothesis` test.
- Signatures, algorithms and file formats are in `design.md`. This plan names what to build and where, not how.
- The only runtime dependencies are `pandas`, `numpy` and `scikit-learn` (`tech.md`). If a task seems to need another, stop and record the reason in `docs/DECISIONS.md` first.

## Tasks

- [x] 1. Project scaffold, errors and shared models
  - Create `hexperiment/pyproject.toml`: src layout, `requires-python = ">=3.13"`, runtime dependencies `pandas`, `numpy`, `scikit-learn`, extra `dev` with `pytest`, `hypothesis`, `ruff`, console script `hexperiment = "hexperiment.cli:main"` (the target module arrives in task 16; installing does not import it), `[tool.ruff]` selecting `E`, `F`, `I`, `UP`, `B`, and `[tool.pytest.ini_options]` with `testpaths = ["tests"]`.
  - Create `hexperiment/src/hexperiment/errors.py` with `HexInputError(problem, fix)`, where `str(error)` is the problem followed by the fix.
  - Create `hexperiment/src/hexperiment/models.py` with the enums (`Status`, `Severity`, `RunMode`, `Outcome`) and the frozen dataclasses (`Finding`, `CheckResult`, `CheckSummary`, `MetricResult`, `WaiverRecord`, `SplitSignature`, `DataSummary`, `DataAuditResult`, `Verdict`), plus `to_dict` / `from_dict` helpers for the ones that are written to JSON.
  - Create `hexperiment/src/hexperiment/__init__.py` exporting `HexInputError`, `Finding` and `Verdict` for now. Tasks 7 and 14 add the other names.
  - Create `hexperiment/tests/conftest.py` that registers and loads the Hypothesis profile (`max_examples=60`, `deadline=None`).
  - Create `hexperiment/tests/test_package.py`: an empty test that only imports `hexperiment`.
  - Create `hexperiment/tests/test_errors.py` (message contains problem and fix) and `hexperiment/tests/test_models.py` (dictionary round trip for each serialized dataclass, as examples and as a Hypothesis test for `Finding`).
  - Extend `.gitignore` with `.venv/`, `.hex/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `.hypothesis/` and `*.egg-info/`. Do not ignore `.kiro/`.
  - Install into the root virtual environment with `.venv/Scripts/python -m pip install -e 'hexperiment[dev]'`, then run the test and lint commands.
  - _Requirements: 13.2_

- [x] 2. Configuration (`hex.toml`) and execution mode
  - Create `hexperiment/src/hexperiment/config.py`: `ExecMode`, the config dataclasses (`AppConfig` to `HexConfig`, `Waiver`), `load_config` (reads the file with `tomllib`), pure `parse_config`, `current_mode` and `run_id_from_env`, with the defaults from the schema table in `design.md`.
  - Validation raises `HexInputError` naming the key and the file: missing `data.path`, `target.column`, `metric.name` or `metric.threshold`; unreadable file; `metric.name` other than `roc_auc`; `metric.threshold` and `split.test_fraction` not a number strictly between 0 and 1 (booleans rejected); out-of-range values for the other numeric keys; unknown tables or keys.
  - Waiver validation: `check` must be `row_overlap`, `target_proxy` or `temporal` (so `canary`, `determinism`, `split_order` and `input` are rejected); `reason` must have at least 10 characters after stripping; `column` is allowed only with `target_proxy`.
  - Write `hexperiment/tests/test_config.py`: one example per key and default, one rejection example per rule above (asserting the message names the key and the path), `current_mode` with unset, valid and invalid `HEX_MODE`, and a Hypothesis test that any `threshold` or `test_fraction` outside the open interval (0, 1) is rejected.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 10.7, 11.1, 11.3_

- [x] 3. Data module, part 1: hash split and time split
  - Create `hexperiment/src/hexperiment/data.py` with `TimeCutoff`, `hash_fraction` (SplitMix64 finalizer in `uint64` numpy arithmetic, no Python `hash`), `parse_times` (`utc=True`, unparseable or missing values raise `HexInputError` naming column and frame), `compute_cutoff` (order by `(time, row id)`, `n_test = min(n - 1, max(1, floor(n * fraction + 0.5)))`, fewer than 2 rows raises), `test_mask` and `split_frame`.
  - `split_frame` returns copies that keep the row-id index. With a stored cutoff, a row is a test row when its `(time, row id)` is at or after the cutoff, so rows removed before the split cannot move the boundary.
  - Create `hexperiment/tests/test_data.py` with examples: known fraction ranges, equal timestamps at the boundary, mixed timezones, unparseable timestamp, fewer than 2 rows, and a frame with rows dropped before the split.
  - Add Hypothesis tests: without a time column, membership of row id `i` depends only on `i`, the seed and the fraction (equal indexes with different values give the same mask; the first `k` ids get the same decision for any total `n >= k`); with a time column, every training timestamp is at or before every test timestamp and the sizes follow `n_test`; dropping any subset of rows never moves a remaining row to the other side of a stored cutoff.
  - _Requirements: 1.2, 1.3, 1.4, 17.2_

- [x] 4. Data module, part 2: the canary scramble (tilted resample)
  - In `hexperiment/src/hexperiment/data.py` add `tilt_column` and `scramble_test_rows(df, mask, time_column, seed)` with the fixed `CANARY_SALT` and one `np.random.default_rng([seed, CANARY_SALT])` shared across columns in column order. The signature takes no target column or positive value.
  - Follow the algorithm in `design.md`: missing values stay in place; one distinct value is unchanged; two distinct values are swapped; three or more draw with replacement from the upper half (numeric: sorted list with repeats, pool start `min(floor(m/2), index of the first copy of the second-largest distinct value)`; other dtypes: the top `ceil(k/2)` categories by count, ties by ascending `str(value)`); if every draw is equal, repair the first draw with the highest-ranked different pool value. The time column and non-test rows are untouched.
  - Extend `hexperiment/tests/test_data.py` with the examples listed under "Example-based tests" in `design.md`: binary 0/1 target, boolean and two-category string columns swapped; one-valued column unchanged; ten distinct numeric values; `[1, 2, 3, 3, 3, 3, 3]` widened to `[2, 3, 3, 3, 3, 3]`; categories with counts 5, 3, 1, 1 and a tie at the cutoff; missing values in numeric, string and nullable-integer columns; constant-draw repair with three test rows; time column unchanged; same seed gives the same frame and a different seed a different one; categorical dtype keeps its categories.
  - Add the Hypothesis tests from the property table: index, shape, non-test rows and time column unchanged; no value absent from the column's original test rows (numeric, string, boolean and nullable columns with missing values); no column with two or more distinct test values becomes constant, including three-valued columns with very few test rows; a target with exactly two distinct test values keeps both classes and swaps the class counts; `tilt_column` writes only values at or above the pool minimum, and on a two-valued column is its own inverse.
  - _Requirements: 1.5, 17.2_

- [x] 5. Canary module, part 1: snapshots, comparison and the canary decision
  - Create `hexperiment/src/hexperiment/canary.py` with the pure layer only: `SCORE_TOLERANCE`, `Snapshot`, `LoadMarker`, `SnapshotDiff`, `CanaryOutcome`, `ExecutionReport`, `build_snapshot` (rows sorted by id, `hash_pandas_object(..., index=False)` per training row), `compare_snapshots`, `hint_locations`, `skipped_outcome` and `decide_canary` with its six ordered steps, the "weaker without training predictions" warning, and the evidence and fix text of The Peeking Spirit.
  - Do not import `subprocess` or touch the file system in this task. Canary findings are created with the severity they keep, using check names that waivers cannot name.
  - Create `hexperiment/tests/test_canary.py` with: `build_snapshot` and `compare_snapshots` examples (equal, one changed feature row, score change just above and just below `1e-9`, different id sets, missing scores); one example per row of the `decide_canary` table (marker missing, scrambled ids in train, id sets differ, no difference, difference with a matching control, difference with a differing control); the missing-`y_score_train` warning in steps 4 to 6; hint line numbers in the fix text.
  - Add a Hypothesis test: for any combination of equal or different normal, canary and control snapshots, the control runner is called at most once and only when the canary differs and the id sets are equal.
  - _Requirements: 2.3, 2.6, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

- [x] 6. Canary module, part 2: the `.hex/` exchange protocol
  - In `hexperiment/src/hexperiment/canary.py` add the private `ExecutionFailed`, `hex_dir`, `clear_mode_dir`, `write_snapshot` / `read_snapshot` (`meta.json` plus `snapshot.npz`, loaded with `allow_pickle=False`, `format_version` 1), `write_load_marker` / `read_load_marker` (`loaded.json`, `scrambled_ids.npy`) and `write_result` / `read_result` (`result.json`).
  - Every reader rejects a file whose `run_id` differs from the caller's or whose `format_version` is not 1 by raising `ExecutionFailed` with the message from the failure table in `design.md`. `read_load_marker` returns `None` when the marker is absent.
  - Extend `hexperiment/tests/test_canary.py`: round trips on `tmp_path` for snapshots (with and without `y_score_train`), markers (normal and canary) and results (including a result that carries an input error); `run_id` mismatch; wrong `format_version`; `clear_mode_dir` removes stale files and leaves other modes alone.
  - Add a Hypothesis test: `write_snapshot` then `read_snapshot` returns equal arrays.
  - _Requirements: 2.3, 2.4, 10.6_

- [x] 7. Data module, part 3: `load` and `split`
  - In `hexperiment/src/hexperiment/data.py` add `load` and `split`. `load` resolves `app_dir` (caller's directory when `None`; a caller without `__file__` raises asking for `app_dir`), reads the CSV at `data.path`, sets the `RangeIndex` in file order, computes the cutoff from the full frame when `time_column` is set, scrambles the test rows only when `HEX_MODE=canary`, writes the load marker through `canary.write_load_marker` only when `HEX_RUN_ID` is set, and stores `{"app_dir", "cutoff"}` in `df.attrs["hexperiment"]`.
  - `split` reads the app directory and cutoff from `df.attrs`; if `attrs` was lost it falls back to the caller's directory and recomputes the cutoff. It returns copies and raises `HexInputError` when either side is empty.
  - Update `hexperiment/src/hexperiment/__init__.py` to export `load` and `split`.
  - Extend `hexperiment/tests/test_data.py` with tmp-path apps: missing and empty CSV raise with the resolved path in the message; index is `0..n-1` in file order; normal and control mode return the file values; canary mode changes exactly the rows in the test mask by `scramble_test_rows`, never the time column or training rows; marker files written only when `HEX_RUN_ID` is set, with the scrambled ids in canary mode; `split` uses the stored cutoff after `dropna()`-style filtering; `split` with cleared `attrs` recomputes the cutoff; empty side raises.
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 10.6_

- [x] 8. Data module, part 4: input validation for `audit`
  - In `hexperiment/src/hexperiment/data.py` add `ValidatedInputs`, `SnapshotInputs`, `validate_audit_inputs` (rules in the order given in `design.md`, each message naming the offender and the fix, the single-class message naming the positive value and the target's distinct values) and `validate_snapshot_inputs` (training-side rules only).
  - Extend `hexperiment/tests/test_data.py`: one example per Requirement 4 case for `validate_audit_inputs` (empty train or test, missing target in either frame, single-class test target, wrong score length, NaN or out-of-range score, unknown feature, missing or unparseable time column); for `validate_snapshot_inputs`, it raises only for an empty train, a feature missing from train or a bad `y_score_train`, and accepts a single-class test target, a `y_score` of the wrong length and a missing time column.
  - _Requirements: 2.4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

- [x] 9. Checks, part 1: The Doppelganger, The Possession and The Prophet
  - Create `hexperiment/src/hexperiment/checks/__init__.py` (package marker only for now), `checks/row_overlap.py` (`check_row_overlap`), `checks/target_proxy.py` (`standalone_score`, `check_target_proxy`) and `checks/temporal.py` (`check_temporal`). Each is a pure function that returns a `CheckResult`, reads no file, environment or clock, and does not modify its arguments.
  - Create `hexperiment/tests/test_row_overlap.py`, `test_target_proxy.py` and `test_temporal.py` with the leaking case, the clean case and the boundaries from the example table in `design.md`: `fraction == max_fraction` not flagged, identical NaN rows equal, `score == auc_max` flagged, constant test feature skipped, unseen category, NaN values, equal timestamps not a leak, no `time_column` gives `not applicable`, mixed timezones.
  - Add Hypothesis tests: any test row copied from train is detected; a feature equal to the target is always detected (both classes present, at least 2 unique values); reordering rows of `train` or `test` never changes any of the three results.
  - _Requirements: 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 17.1, 17.2_

- [x] 10. Checks, part 2: split-order hint, metric gate and the check registry
  - Create `hexperiment/src/hexperiment/checks/split_order.py` (`check_split_order`: `ast` only, never executes the script; first call to `split`, `hexperiment.split` or `train_test_split`; warnings for earlier `fit`, `fit_transform`, `fit_resample` and the listed methods; bare `min` / `max` and calls rooted at an imported module alias ignored; unparseable script or no split call gives one `inconclusive` warning) and `checks/metric.py` (`compute_metric`, `check_metric` with the "too good to be true" warning).
  - Fill in `checks/__init__.py` with `run_data_checks`, the only place that lists the checks, in the fixed order `row_overlap`, `target_proxy`, `temporal`, `split_order`, metric. It also builds the `DataSummary` (row counts, positive rates, time range of each split when a time column is set).
  - Create `hexperiment/tests/test_split_order.py` and `test_metric.py` with the leaking and clean cases and boundaries from the example table (syntax error, no split call, `hexperiment.split` and `train_test_split` recognized, `np.mean` ignored, `min(a, b)` ignored, `value == threshold` passes, AUC above `threshold + margin`).
  - Create `hexperiment/tests/test_checks.py`: result order of `run_data_checks`, input arrays unchanged, and the clean-data test (2,000 rows of independent random numeric and categorical features and a random binary target, several fixed seeds, no `error` finding).
  - Add a Hypothesis test that reordering rows (scores permuted the same way) never changes `check_metric`.
  - _Requirements: 2.3, 2.5, 3.2, 8.1, 8.2, 8.3, 9.1, 9.2, 9.3, 14.1, 17.1, 17.2, 17.4_

- [x] 11. Verdict assembly
  - Create `hexperiment/src/hexperiment/verdict.py` with `FINDING_ORDER`, `assemble`, `apply_waivers` (only `row_overlap`, `target_proxy` and `temporal` findings are candidates; an unmatched waiver becomes a `waiver` warning), `decide_status`, `to_json` (sorted keys, two-space indentation, UTF-8, no NaN), `from_dict` and `format_summary`. The module has no file, clock or process access.
  - `assemble` handles the input-error case (`REJECTED`, single `input` finding), adds the `test_set_uses` warning when the count exceeds `checks.max_test_uses`, builds the six check summaries, and sorts findings so equal inputs give equal findings.
  - Create `hexperiment/tests/test_verdict.py`: the full truth table of `decide_status` (metric pass or fail, error present or not, canary outcome, fast or full); `apply_waivers` severity change, untouched canary findings, stale waiver warning with location `<check>` or `<check>:<column>`; `assemble` for the input-error case and the `test_set_uses` warning; `format_summary` against the example in `design.md`.
  - Add Hypothesis tests: `to_json` then `from_dict` returns an equal `Verdict` with sorted keys and no NaN; waived findings never change the status; `assemble` called twice with the same inputs but a different `run_id` and `created_at` gives equal `status`, `metric` and `findings`.
  - _Requirements: 4.7, 10.7, 11.2, 11.4, 12.2, 13.1, 13.2, 13.4, 13.5, 16.4_

- [x] 12. Model card
  - Create `hexperiment/src/hexperiment/card.py` with `render(verdict, history)`: the ten sections in the order given in `design.md`, the "Generated by hexperiment. Do not edit by hand." notice, and the `_cell` helper that escapes `|`, replaces newlines and trims. The module is pure.
  - Create `hexperiment/tests/test_card.py`: one example per section (counts and positive rates, time ranges only when configured, metric against threshold, checks table, findings with fix hints, waivers with reasons, test-set uses, history table and the "none" case, the notice).
  - Add a Hypothesis test: for arbitrary text in app names, descriptions, column names and evidence, every table row in the rendered card has the same number of cells as its header.
  - _Requirements: 11.4, 14.1, 14.2, 14.3_

- [x] 13. Run store and history
  - Create `hexperiment/src/hexperiment/runs.py` with `format_run_id`, `new_run_id`, pure `count_test_set_uses` (same split block; input-error verdicts excluded), `load_history` (reads `runs/*/verdict.json`, sorts by `(created_at, run_id)`, a malformed file raises `HexInputError` naming it, a missing `runs/` returns an empty list), `clear_app_verdict` and `finalize_run`.
  - `finalize_run` adds 1 to the use count unless the run is an input error, takes the version from `importlib.metadata`, creates `runs/<run_id>/` with `exist_ok=False` (an existing folder raises and is left untouched), writes `verdict.json` and `model_card.md` as UTF-8, and copies `verdict.json` to the app directory through a temporary file and `os.replace`.
  - Create `hexperiment/tests/test_runs.py`: run id format and uniqueness; history ordering; never overwriting an earlier run folder; the app-level copy equals the run copy; `count_test_set_uses` counts fast, full, `REJECTED` and `UNVERIFIED` runs with the same split block and ignores input-error runs; `clear_app_verdict` with and without a file; malformed history file.
  - Add Hypothesis tests: `format_run_id` always matches `^\d{8}-\d{6}-[0-9a-f]{6}$`; adding any number of input-error verdicts to the history never changes the count.
  - _Requirements: 12.1, 13.2, 13.3, 15.1, 15.2, 16.5_

- [x] 14. The `audit` entry point, the public API and the architecture test
  - Create `hexperiment/src/hexperiment/auditor.py` with `audit`: locate the app from `script`, load the config, read the mode and run id, validate (full validation in normal mode, `validate_snapshot_inputs` only in canary and control mode), write the snapshot in every mode, return after the snapshot in canary and control mode, run `run_data_checks` and write `result.json` in normal mode, and finalize a `fast` run itself when `HEX_RUN_ID` is not set. An input error in normal mode is written to `result.json` (and, when standalone, to a `REJECTED` verdict) and then re-raised. `audit` never modifies its arguments.
  - Finish `hexperiment/src/hexperiment/__init__.py` so that `__all__` is exactly `audit`, `load`, `split`, `load_history`, `HexInputError`, `Verdict`, `Finding`.
  - Create `hexperiment/tests/test_auditor.py` with tmp-path apps and the environment set through `monkeypatch`: normal mode writes snapshot, `result.json` and no verdict when `HEX_RUN_ID` is set; standalone mode writes a `fast` verdict and counts a test-set use; canary and control mode write only the snapshot; input errors in each mode; inputs unchanged after the call; `y_score_train` absent.
  - Add a Hypothesis test: in canary or control mode `audit` writes a snapshot for any test target (including one class), any `y_score` and any test-frame columns, as long as the training frame, features and `y_score_train` are valid.
  - Create `hexperiment/tests/test_architecture.py` (it parses the package with `ast`): `__all__` equals the seven names; only `canary.py` imports `subprocess`; `checks/*`, `verdict.py`, `card.py`, `models.py` and `errors.py` import none of `os`, `pathlib`, `shutil`, `subprocess`, `tomllib`, `datetime` and never call `open`; no module defines a module-level mutable variable. It walks the package directory, so modules added in tasks 15 and 16 are checked automatically.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.7, 15.2_

- [x] 15. Canary module, part 3: running `train.py` and orchestrating a run
  - In `hexperiment/src/hexperiment/canary.py` add `Runner`, `run_train_script` (`sys.executable`, `train.py`, `cwd=<app_dir>`, no shell, `HEX_MODE` and `HEX_RUN_ID` set, captured output, the configured timeout; no `PYTHONHASHSEED`; a timeout or non-zero exit raises `ExecutionFailed` with the last 40 lines of stderr) and `run_executions`.
  - `run_executions` runs `normal` and, in a full run, `canary`, then calls `decide_canary` with a `run_control` that runs and reads `control`. It clears the mode folder before each execution, checks that the snapshot exists with this `run_id` (otherwise `ExecutionFailed`), accepts a non-zero normal execution only when `result.json` carries an input error (then it skips the other executions), and never touches `<app_dir>/verdict.json`. A fast run executes `normal` only and reports the canary as `skipped`.
  - Extend `hexperiment/tests/test_canary.py` using a fake `runner` that writes prepared snapshots (the helper goes in `tests/conftest.py`): fast run runs once; full run with equal snapshots runs twice; a differing canary triggers exactly one control run; control matching normal gives The Peeking Spirit; control differing gives The Shapeshifter; differing id sets skip the canary; scrambled ids in train give `not applied`; missing marker; input error stops after the first execution; missing snapshot; `run_id` mismatch.
  - Add one test that runs `run_train_script` on a tiny script: a successful run, a non-zero exit with the stderr tail, and a timeout of 1 second.
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 16.2, 16.3, 16.5_

- [x] 16. Command line and the scenario harness (failure-path scenarios 8, 9 and 10)
  - Create `hexperiment/src/hexperiment/cli.py` with `build_parser` (`run <app_dir>` with a mutually exclusive `--fast` / `--full`, full by default), `exit_code_for` (0, 1, 3) and `main`. `main` first checks `app_dir` and calls `runs.clear_app_verdict`, then loads the config, generates the run id, calls `canary.run_executions`, calls `runs.finalize_run`, prints `format_summary`, and returns the exit code. A `HexInputError` (printed as `hexperiment: <problem> <fix>`), `ExecutionFailed` (with the stderr tail) or `OSError` returns 2 with no new verdict. `main` is the only function that prints.
  - Create `hexperiment/tests/test_cli.py`: argument parsing and the default mode; `exit_code_for` for each status; the console script entry in `pyproject.toml` points at `hexperiment.cli:main`; output lines for the status, the metric against its threshold and one line per finding; missing `app_dir`, missing `train.py` and bad `hex.toml` return 2.
  - Create the scenario harness: `hexperiment/tests/app_builder.py` (writes a `hex.toml`, the shared 600-row CSV described in `design.md`, a `train.py` rendered from `tests/fixtures/scripts/<scenario>/*.py.tmpl`, and `helpers.py` where noted) and `hexperiment/tests/test_canary_scripts.py`, which runs `cli.main(["run", str(app)])` once per scenario at module scope and shares the result across its assertions.
  - Add scenarios 8 (`input_error`: wrong `y_score` length gives `REJECTED`, exit 1, one `input` finding, only the normal execution ran), 9 (`timeout`: `time.sleep` longer than `timeout_seconds = 1`; exit 2, no new `runs/` folder) and 10 (`no_audit`: exit 2). Scenarios 9 and 10 first place an `APPROVED` `verdict.json` in the app folder and assert that it no longer exists afterwards.
  - _Requirements: 4.7, 13.3, 16.1, 16.2, 16.3, 16.4, 16.5_

- [x] 17. Canary detection scenarios (1 to 7 and 11)
  - Add the script templates under `hexperiment/tests/fixtures/scripts/<scenario>/` and the scenarios to `hexperiment/tests/test_canary_scripts.py`. This is the first point at which they can run, because they need the scramble, the canary decision, the audit and the command together.
  - Leak scenarios, each expected `REJECTED` with The Peeking Spirit: 1 `mean_impute_global` (mean over all rows before the split, plus a `split_order` warning at that line), 2 `scaler_global` (scaler fitted on all rows), 3 `helper_leak` (per-category target means computed in `helpers.py`, and no `split_order` warning), 11 `target_encoding_global` (`groupby(...).transform("mean")` of the target over all rows in `train.py`, plus a `split_order` warning at that line).
  - Clean and degraded scenarios: 4 `train_only_fit` (`APPROVED`, exit 0, canary `passed`), 5 `unrelated_fit` (a scaler fitted on a separate synthetic dataset; `APPROVED`, canary `passed`, `split_order` warning only), 6 `no_seed` (`UNVERIFIED`, exit 3, The Shapeshifter and no The Peeking Spirit), 7 `direct_read` (`pandas.read_csv` instead of `hexperiment.load`; `UNVERIFIED`, canary `not applied`, warning).
  - Add a repeat-run test: running scenario 4 twice gives identical `status`, `metric` and `findings`.
  - Use `LogisticRegression` with a fixed `random_state` in every template. If a leak scenario is not detected, fix the cause in `data.tilt_column` or the canary comparison, not the fixture.
  - _Requirements: 1.5, 10.1, 10.4, 10.5, 10.6, 13.5, 17.3_

- [x] 18. Final verification
  - Run the whole suite with `.venv/Scripts/python -m pytest hexperiment/tests -q --durations=10` and confirm it passes in under 90 seconds on a laptop CPU. The suite runs in parallel through `pytest-xdist` (`addopts` in `pyproject.toml`); do not add a separate test stage and do not lower Hypothesis settings.
  - Run `.venv/Scripts/python -m ruff check hexperiment` and confirm it reports no problems.
  - Confirm that the `.venv/Scripts/python -m pip install -e 'hexperiment[dev]'` command still works from a clean checkout and that `.venv/Scripts/hexperiment run --help` prints the usage.
  - Confirm that each check module has example-based tests for a leaking and a clean case and at least one Hypothesis test, that the clean-data test passes, and that no runtime dependency beyond `pandas`, `numpy` and `scikit-learn` was added.
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_

- [x] 19. Canary target check against bypass
  - In `hexperiment/src/hexperiment/canary.py`, record the hash of `test[target]` in each snapshot.
  - In `decide_canary`, when the canary execution's test-target hash equals the normal execution's test-target hash, mark the canary outcome as `not applied` with a finding that the scramble did not reach the audited frames.
  - Add a scenario test in `hexperiment/tests/test_canary_scripts.py` next to Scenario 7: a script calls `load()`, discards it, reads the CSV directly, and fits a scaler on train+test. It must end `UNVERIFIED`, never `APPROVED`.
  - _Requirements: 10.7_

- [x] 20. Check overrides and test-set use-count normalization
  - In `hexperiment/src/hexperiment/verdict.py` and `hexperiment/src/hexperiment/card.py`, record every non-default `[checks]` configuration threshold in `verdict.json` under `checks_overrides` (empty object `{}` when none) and display them in `model_card.md`.
  - In `hexperiment/src/hexperiment/runs.py`, normalize the test-set use-count key so that `data.path` resolves relative to the app folder.
  - Add `hexperiment/constraints.txt` pinning tested exact versions, and add unit tests for `checks_overrides` and normalized path keys.
  - _Requirements: 12.1, 13.2, 14.1_

- [x] 21. Per-column scramble check
  - In `hexperiment/src/hexperiment/canary.py`, replace `Snapshot.test_target_hash` with `test_columns` (the target plus the audited features, without the time column), `test_column_hashes` (uint64, one row per test row in sorted row-id order, one column per entry of `test_columns`, from `pd.util.hash_pandas_object(test[col], index=False)`) and `test_column_varies` (one bool per column: at least two distinct non-missing test values). `write_snapshot` and `read_snapshot` carry the new fields; `build_snapshot` takes `time_column`, and `auditor.py` passes `config.split.time_column` in every mode.
  - Replace `_scramble_missed` with `_missed_columns(normal, canary)`: the columns present in both snapshots that vary in `normal` and hold equal hashes on every test row id present in both. In `decide_canary`, a non-empty list gives `not applied`, determinism `skipped`, and a warning naming the columns (at most five, then "and N more"). The warning cannot be waived.
  - Tests: a scenario next to the load-bypass scenario in `hexperiment/tests/test_canary_scripts.py`, with a new template under `hexperiment/tests/fixtures/scripts/`: the target comes from `load()`, the features from the script's own CSV read, and a scaler is fitted on train+test; it must end `UNVERIFIED`. A second scenario with a constant test target and features from `load()` must have the canary applied. Unit tests in `hexperiment/tests/test_canary.py` for `_missed_columns`: varies and equal, varies and changed, constant, a column missing from one snapshot, and test-id subsets.
  - _Requirements: 10.6, 10.7_
  - Result: `Snapshot` now carries `test_columns`, `test_column_hashes` and `test_column_varies` (files and `build_snapshot` updated, `auditor.py` passes `split.time_column`); `_missed_columns` replaces `_scramble_missed`. New scenarios `partial_load_bypass` (UNVERIFIED, warning names `x1`, `x2`, not the target) and `constant_column` (APPROVED, canary applied). A constant test *target* cannot be built, because the audit rejects a single-class test target as an input error, so the second scenario uses a constant feature instead. 3201 tests pass in 158 s; ruff reports no problems.

- [x] 22. Relative data paths in the use count, and the MCP pin
  - In `hexperiment/src/hexperiment/runs.py`, give a data file outside the app folder as `os.path.relpath` to the app folder, so the key never holds an absolute path. Add a test.
  - Add `mcp` at its installed version to `hexperiment/constraints.txt`.
  - Result: `normalize_data_path` uses `os.path.relpath` (POSIX separators); a file on another Windows drive has no relative path and stays absolute. The old test that expected an absolute path outside the app now expects `../shared/churn.csv`; a second test covers an absolute input. `mcp==2.3.0` added to `constraints.txt`.
  - _Requirements: 12.1_

- [x] 23. Watch for the data file being opened outside `load`
  - In `canary.run_train_script`, start the script through a small bootstrap (a new module, for example `hexperiment/src/hexperiment/child.py`, run as `python -m hexperiment.child train.py` or `python -c`) that installs a `sys.addaudithook` before running `train.py` with `runpy.run_path(..., run_name="__main__")`. The script must behave as before: `__file__`, `sys.argv[0] == "train.py"`, `sys.path[0]` is the app folder, the exit code and the stderr tail (with `train.py` line numbers) are unchanged, and `import hexperiment` still does not load `mcp`.
  - The hook handles the `open` event: it resolves `str`, `bytes` and path-like paths against the working directory (`os.path.realpath`, `os.path.normcase`), ignores integer file descriptors, and counts opens of the app's resolved data file. `data.load` marks its own read so it is not counted. The hook must never raise and must not recurse into itself.
  - The count goes into the snapshot (`data_reads_outside_load`, written by `write_snapshot`, required by `read_snapshot`). In `decide_canary`, a count above zero in any execution gives `not applied`, determinism `skipped`, and a warning naming the file (Requirement 10.9); it cannot be waived. Check it before the per-column comparison.
  - Tests: a scenario `side_read` in `hexperiment/tests/test_canary_scripts.py` (frames from `load()`, fill values and a scaler fitted on `pd.read_csv` of the data file) ends `UNVERIFIED`; a scenario that reads a separate lookup CSV in the app folder keeps the canary applied and ends `APPROVED`; a scenario that copies the data file with `shutil.copy` and reads the copy ends `UNVERIFIED`; the existing `partial_load_bypass` and `load_bypass` scenarios still end `UNVERIFIED`. Unit tests for the path matching (relative, absolute, bytes, path-like, a file descriptor, a different file) and for the snapshot round trip.
  - Result: new module `hexperiment/child.py` (standard library only) is started by `run_train_script` as `python -c` and installs the audit hook from the `[data] path` of `hex.toml`; `data._read_csv` marks its read with `own_read`. The count is in `Snapshot.data_reads_outside_load` (`meta.json`), and `decide_canary` checks it, for the normal, canary and control snapshots, before the column comparison (new keyword `data_file` names the file). Because `runpy.run_path` sets `sys.argv[0]` to the path it is given, the script runs as `run_path("train.py")`: `sys.argv == ["train.py"]` and `__file__` is `train.py` (relative), not absolute as in a plain run. The warning of `load_bypass` and `partial_load_bypass` is now the read-outside-load warning, so those tests assert it; the column message stays covered by the unit tests, since the usual reads of the data file raise an `open` event first. New scenarios: `side_read`, `copy_read` (UNVERIFIED), `lookup_read` (APPROVED). 3538 tests pass (1 skipped: no `socket.sendmsg` on this platform), ruff reports no problems.
  - _Requirements: 10.6, 10.9, 16.2_

- [x] 24. Watch copies, links and renames of the data file
  - In `hexperiment/src/hexperiment/child.py`, besides `open`, count these audit events when their source path resolves to the data file: `shutil.copyfile` (args[0]), `_winapi.CopyFile2` (args[0]), `os.link` (args[0]), `os.symlink` (args[0]), `os.rename` and `os.replace` (args[0]). Resolve paths exactly as for `open`; ignore events whose arguments are not paths. `data.load`'s own read stays uncounted.
  - Make the hook's re-entry flag and the own-read depth per thread (`threading.local`), so an open in another thread while `load` runs or while the hook is busy is still counted.
  - Tests: scenarios next to `copy_read` in `hexperiment/tests/test_canary_scripts.py` for `shutil.copy2` and for `os.link` (skip where hard links are not supported), each ending `UNVERIFIED`; unit tests for each event name, for a different source file, and for a data-file open in a second thread while `load` runs.
  - Result: the hook in `child.py` now also counts `shutil.copyfile`, `_winapi.CopyFile2`, `os.link`, `os.symlink`, `os.rename` and `os.replace` when `args[0]` resolves to the data file. The re-entry flag and own-read depth live in a `threading.local`. Probe on Python 3.13.2 (Windows): `shutil.copy2` raises `_winapi.CopyFile2` (not `shutil.copyfile`), and `os.replace` raises `os.rename`; `os.link`, `os.symlink` and `os.rename` carry the source in `args[0]`. New scenarios `copy2_read` and `link_read` end `UNVERIFIED`; unit tests cover each event, a different source, non-path arguments, real calls, and an open in a second thread during `own_read`. 3548 tests pass (1 skipped), ruff reports no problems. Not tested: an open in another thread while the hook is busy (the hook runs to completion before any other hook, so the case cannot be staged from a script).
  - _Requirements: 10.9_

- [ ] 25. Name every watched action in the read-outside-load warning
  - In `hexperiment/src/hexperiment/canary.py`, `_side_read_warning` says the data file "was opened outside hexperiment.load", but the count also covers copies, links and renames (task 24). Change the message to "{data_file} was opened, copied, linked or renamed outside hexperiment.load", and keep the rest of the message, the fix text and the `details` unchanged.
  - In the `build_snapshot` docstring, describe `data_reads_outside_load` as the count of opens, copies, links and renames of the data file outside `hexperiment.load`.
  - Update the assertions in `hexperiment/tests/test_canary.py` and `hexperiment/tests/test_canary_scripts.py` that match the old phrase to the new one.
  - _Requirements: 10.9_
