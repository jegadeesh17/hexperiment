# Requirements Document: hexperiment-core

## Introduction

`hexperiment` is the reusable core of the kit: a Python package that audits training runs of a binary classifier. A training script loads its data with `hexperiment.load()`, splits it with `hexperiment.split()`, trains, and ends with one call to `hexperiment.audit(...)`. The audit checks the run for data leakage, compares the test metric with the threshold set in the app's `hex.toml`, and writes a machine-readable verdict and a human-readable model card.

The strongest check is the **canary test**: the kit re-runs the training script with the test rows scrambled at load time. In a leak-free pipeline nothing about training depends on the test rows, so the predictions on training rows must not change. If they change, test information reached training, wherever in the code that happened.

A run is APPROVED only when the metric meets the threshold, no leak is found, and the canary test has run.

Scope of this spec: data loading and splitting, configuration, the checks, the canary test, the metric gate, waivers, the verdict, the model card, run history and the `hexperiment run` command. The Kiro hooks, workflow loop, power packaging, example apps, gallery and leak benchmark are covered by later specs.

## Glossary

- **App:** a folder in `apps/` containing `hex.toml`, `train.py` and `runs/`.
- **Run:** one invocation of `hexperiment run`, which executes the app's `train.py` one or more times and produces one verdict.
- **Execution mode:** how a single execution of `train.py` behaves, set by the environment variable `HEX_MODE`: `normal` (the default), `canary` (test rows scrambled at load) or `control` (identical to normal, used to test determinism).
- **Fast mode / full mode:** a fast run executes `train.py` once and skips the canary test; a full run also executes the canary (and, if needed, the control).
- **Finding:** one problem found by a check, with a ghost name, severity (`error`, `warning` or `waived`), location, evidence and fix hint.
- **Ghost:** the display name of a finding: The Doppelganger (row overlap), The Possession (target proxy), The Peeking Spirit (test information reached training), The Prophet (temporal leak), The Shapeshifter (non-deterministic training).
- **Waiver:** a human-written exception in `hex.toml` that downgrades a matching finding to `waived`, with a recorded reason.
- **Verdict:** the result of a run: `APPROVED`, `REJECTED` or `UNVERIFIED`, plus the metric and all findings.

## Requirements

### Requirement 1: Loading and splitting

**User Story:** As a data scientist, I want the kit to load and split my data, so that it knows which rows are test rows before any of my code runs and can test whether my pipeline leaks.

#### Acceptance Criteria

1. THE package SHALL expose `load(app_dir=None)` that reads the CSV at `data.path` from `hex.toml` (relative to the app directory; `app_dir` defaults to the directory of the calling script) and returns a DataFrame whose index is the stable row id `0..n-1` in file order.
2. THE package SHALL expose `split(df)` that returns `(train, test)` DataFrames, preserving the row-id index.
3. WHERE `split.time_column` is configured, `split` SHALL put the latest `split.test_fraction` of rows (by time) in `test` and the rest in `train`.
4. WHERE `split.time_column` is not configured, `split` SHALL assign each row to `test` when a seeded hash of its row id falls below `split.test_fraction`, so that membership depends only on the row id and the seed, never on the row's values.
5. WHEN `load` runs with `HEX_MODE=canary` THEN the system SHALL compute the test row ids exactly as `split` would and, using a fixed seed, replace the values in those rows of every column except `split.time_column` (a "tilted resample"), before returning the DataFrame:
   - a column with exactly two distinct non-missing test values (including a binary target and boolean columns) SHALL have the two values swapped;
   - a column with three or more distinct non-missing test values SHALL have each non-missing test value replaced by a draw, with replacement, from the upper half of its non-missing test values (by value order for numeric columns; the more frequent half of its categories for non-numeric columns), with that pool widened downward until it holds at least two distinct values;
   - a column with one distinct non-missing test value SHALL be unchanged, and missing values SHALL stay missing;
   - the replacement SHALL never introduce a value that the column did not already contain in its test rows.
   The tilt moves each column's mean, quantiles and category frequencies systematically, so leaks through column-level statistics (for example mean imputation or a scaler fitted on all rows) change training and are detected.
6. IF `data.path` is missing or the file is empty THEN `load` SHALL raise `HexInputError` naming the path.

### Requirement 2: Audit entry point

**User Story:** As a data scientist, I want one call at the end of my training script that audits the run, so that no result leaves the script unchecked.

#### Acceptance Criteria

1. THE package SHALL expose `audit(train, test, y_score, features, script, y_score_train=None)` where `train` and `test` are DataFrames containing the target column (and the time column, if configured), `y_score` is the predicted probability of the positive class for each test row, `y_score_train` is the same for each training row, `features` is the list of columns the model used, and `script` is the path of the calling training script.
2. WHEN `audit` is called THEN the system SHALL locate the app directory as the parent folder of `script` and load `hex.toml` from it.
3. WHEN `audit` runs in `normal` mode THEN the system SHALL run the data checks (Requirements 5 to 9) and write a snapshot to `<app_dir>/.hex/normal/` containing the train and test row ids, a hash of `train[features]` per row, and `y_score_train`.
4. WHEN `audit` runs in `canary` or `control` mode THEN the system SHALL write only the snapshot (to `<app_dir>/.hex/<mode>/`) and SHALL NOT write a verdict or model card.
5. THE audit SHALL NOT modify `train`, `test`, `y_score` or `y_score_train`.
6. IF `y_score_train` is not given THEN the canary comparison SHALL use only the per-row hashes of `train[features]`; a difference SHALL still be reported as in Requirement 10, but no difference SHALL make the canary outcome `inconclusive` (so the run is at best `UNVERIFIED`), with a `warning` that shows how to pass `y_score_train`. A pipeline that fits the model on test rows without changing `train[features]` is visible only through `y_score_train`.

### Requirement 3: Configuration (`hex.toml`)

**User Story:** As a data scientist, I want the target, threshold and split rules in one file, so that the contract for the model is explicit and reviewable.

#### Acceptance Criteria

1. THE system SHALL read `hex.toml` with these keys: `[app] name, description`; `[data] path`; `[target] column, positive` (positive defaults to `1`); `[metric] name, threshold`; `[split] test_fraction` (default `0.2`), `seed` (default `0`), `time_column` (optional); `[checks] proxy_auc_max` (default `0.95`), `overlap_max_fraction` (default `0.0`), `too_good_margin` (default `0.15`), `max_test_uses` (default `5`); `[run] timeout_seconds` (default `120`); and zero or more `[[waivers]]` tables.
2. THE system SHALL support `metric.name = "roc_auc"`.
3. IF `hex.toml` is missing or unreadable, or lacks `data.path`, `target.column`, `metric.name` or `metric.threshold` THEN the system SHALL raise `HexInputError` naming the missing key and the file path.
4. IF `metric.threshold` or `split.test_fraction` is not a number strictly between 0 and 1 THEN the system SHALL raise `HexInputError`.

### Requirement 4: Input validation

**User Story:** As a data scientist, I want bad input to fail loudly, so that a broken run can never look like a passing one.

#### Acceptance Criteria

1. IF `train` or `test` is empty THEN the system SHALL raise `HexInputError` stating which one is empty.
2. IF the target column is absent from `train` or `test` THEN the system SHALL raise `HexInputError` naming the column and the frame.
3. IF the target in `test` contains a single class THEN the system SHALL raise `HexInputError`, because the metric is undefined.
4. IF the length of `y_score` differs from the number of test rows (or `y_score_train` from the number of training rows), or a score is NaN or outside [0, 1] THEN the system SHALL raise `HexInputError`.
5. IF a name in `features` is not a column of both `train` and `test` THEN the system SHALL raise `HexInputError` naming it.
6. IF `split.time_column` is configured but absent from `train` or `test`, or cannot be parsed as datetimes THEN the system SHALL raise `HexInputError`.
7. WHEN `HexInputError` is raised in `normal` mode THEN the system SHALL still write `verdict.json` with status `REJECTED` and a single finding of check `input` describing the error, so that loops and hooks see a failed run.

### Requirement 5: Row overlap check (The Doppelganger)

**User Story:** As a reviewer, I want to know if test rows also appear in training, so that the metric is not inflated by memorized rows.

#### Acceptance Criteria

1. WHEN the audit runs THEN the system SHALL count test rows whose values over `features` plus the target are identical to at least one training row.
2. IF the overlapping fraction of test rows is greater than `checks.overlap_max_fraction` THEN the system SHALL report an `error` finding with ghost "The Doppelganger", the count and fraction as evidence, and the fix hint "after the split, drop every test row that also appears in the training rows".
3. THE result of this check SHALL NOT change when the rows of `train` or `test` are reordered.

### Requirement 6: Target proxy check (The Possession)

**User Story:** As a reviewer, I want to know if a single feature gives away the target, so that a leaked column cannot pass as a strong model.

#### Acceptance Criteria

1. WHEN the audit runs THEN the system SHALL compute, for each feature, a standalone score on the test rows: for numeric, boolean, datetime and timedelta features (datetimes and timedeltas as their integer value; a text feature whose non-missing values in both frames all parse as ISO 8601 dates or datetimes counts as a datetime feature), `max(AUC, 1 - AUC)` of the raw value against the target, computed once with missing values ranked lowest and once ranked highest, keeping the higher score; for other features, the AUC of the training-set target mean of each category applied to the test rows (unseen categories and missing values get the training positive rate); and for numeric features with at most 50 distinct non-missing values in the test rows, also the category score, keeping the higher of the two.
2. IF a feature's standalone score is greater than or equal to `checks.proxy_auc_max` THEN the system SHALL report an `error` finding with ghost "The Possession", the column name as location, the score as evidence, and the fix hint "check whether this column is recorded after the outcome; if so, drop it".
3. IF a feature has a single unique value in the test rows, counting a missing value as a value, THEN the system SHALL skip it for this check. No other feature SHALL be skipped.
4. THE result of this check SHALL NOT change when the rows of `train` or `test` are reordered.

### Requirement 7: Temporal leak check (The Prophet)

**User Story:** As a reviewer, I want to know if the model trained on the future, so that a time-ordered problem is evaluated honestly even when a script splits the data itself.

#### Acceptance Criteria

1. WHERE `split.time_column` is configured, WHEN the audit runs THEN the system SHALL compare the latest training timestamp with the earliest test timestamp.
2. IF any training timestamp is later than the earliest test timestamp THEN the system SHALL report an `error` finding with ghost "The Prophet", the number of test rows earlier than the latest training row as evidence, and the fix hint "split by time with hexperiment.split".
3. WHERE `split.time_column` is not configured THEN the system SHALL list this check as "not applicable" in the verdict.

### Requirement 8: Split-order hint

**User Story:** As a data scientist, I want a fast pointer to the line that probably leaks, so that I (or the agent) know where to look before the slower canary test runs.

#### Acceptance Criteria

1. WHEN the audit runs THEN the system SHALL parse `script` with Python's `ast` module without executing it, and find the first call to `split`, `hexperiment.split` or `train_test_split`, also through import aliases. Calls nested in the arguments of that split call count as earlier calls.
2. IF a call to `fit`, `fit_transform` or `fit_resample`, or a call to `mean`, `median`, `std`, `quantile`, `min`, `max` or `transform` on a DataFrame or Series expression, appears on an earlier line than that first split call THEN the system SHALL report a `warning` finding (never an `error`) with the file and line number as location and the hint "this runs before the split; make sure it does not use test rows".
3. IF the script cannot be parsed or no split call is found THEN the system SHALL report a `warning` finding saying the hint could not be applied.
4. IF the script calls a data reader (`read_csv`, `read_parquet`, `read_excel`, `read_json`, `read_feather`, `read_table`, `read_fwf`, `read_pickle`, `read_sql`, `read_orc`, `loadtxt` or `genfromtxt`) THEN the system SHALL report a `warning` finding at that line saying the data is read without `hexperiment.load()`, so the canary test cannot scramble this copy.

### Requirement 9: Metric gate and the "too good" warning

**User Story:** As a data scientist, I want the threshold in my spec enforced, so that "done" means the model meets the bar honestly.

#### Acceptance Criteria

1. WHEN the audit runs THEN the system SHALL compute ROC AUC from the test target and `y_score`.
2. IF the metric is below `metric.threshold` THEN the system SHALL record the metric as failed.
3. IF the metric exceeds the lower of `metric.threshold + checks.too_good_margin` and `0.99` THEN the system SHALL report a `warning` finding "too good to be true" with the metric and threshold as evidence.

### Requirement 10: Canary test (The Peeking Spirit) and determinism (The Shapeshifter)

**User Story:** As a reviewer, I want proof that no test information reached training, wherever in the code it might happen, so that leaks in helper functions, other files or global statistics are caught without false alarms.

#### Acceptance Criteria

1. WHEN a full run executes THEN the system SHALL execute `train.py` in `normal` mode and then in `canary` mode, and compare the two snapshots.
2. IF the set of training row ids differs between the two snapshots THEN the system SHALL report a `warning` that the canary test is inconclusive because the split depends on data values, and record the canary as `skipped`.
3. IF, for any training row, `y_score_train` differs by more than `1e-9` or the hash of `train[features]` differs between `normal` and `canary` THEN the system SHALL execute `train.py` once more in `control` mode.
4. IF the `control` snapshot also differs from `normal` THEN the system SHALL report a `warning` finding with ghost "The Shapeshifter" ("training is not deterministic; set random_state everywhere"), and record the canary as `inconclusive`.
5. IF the `control` snapshot matches `normal` but the `canary` snapshot does not THEN the system SHALL report an `error` finding with ghost "The Peeking Spirit", with evidence giving the number of training rows whose predictions or features changed and the largest change, and the fix hint "test rows influenced training; fit preprocessing on training rows only, after the split". The fix hint SHALL list the line numbers reported by the split-order hint, if any.
6. IF the script did not load its data through `hexperiment.load` THEN the system SHALL record the canary as `not applied` with a `warning`.
7. WHEN the test target, or any audited feature that has at least two distinct non-missing test values in the normal execution, holds the same values in the canary execution on every test row id present in both executions THEN the kit SHALL record the canary as `not applied`, with a `warning` finding that names those columns (at most five, then "and N more") and says the scramble did not reach the audited frames. The time column SHALL be excluded from this comparison. The comparison SHALL cover only the target and the audited features whose names are columns of the frame returned by `hexperiment.load` (the raw data-file columns, recorded in the load marker); a feature the script derives under a new name, such as a part of the time column or a missing-value flag, SHALL NOT be compared, because the scramble cannot change it. A load marker that records no column names SHALL be read as covering every audited column.
8. Findings from this requirement SHALL NOT be waivable.
9. WHILE `train.py` runs in `normal`, `canary` or `control` mode, THE kit SHALL watch Python's audit events from before the script's first line. IF the app's data file (the resolved `[data] path`) is opened, copied (`shutil.copyfile`, `_winapi.CopyFile2`), linked (`os.link`, `os.symlink`) or renamed (`os.rename`, `os.replace`) outside `hexperiment.load` in any of these executions THEN the kit SHALL record the canary as `not applied`, with a `warning` finding that names the file and says that statistics computed from a copy read outside `hexperiment.load` cannot be checked by the scramble. Files other than the data file SHALL NOT trigger this. The watch catches mistakes and common workarounds, not a script written to evade it: native readers that raise no audit event (such as pyarrow), subprocesses and worker processes, and code that changes the watch itself are not seen.

### Requirement 11: Waivers

**User Story:** As a data scientist, I want to accept a specific finding with a written reason, so that a legitimate strong feature or natural duplicate does not block me, while the exception stays visible.

#### Acceptance Criteria

1. THE system SHALL read `[[waivers]]` entries with keys `check` (one of `row_overlap`, `target_proxy`, `temporal`), `column` (required for `target_proxy`, not allowed for the other checks) and `reason` (required, at least 10 characters).
2. WHEN a finding matches a waiver THEN the system SHALL change its severity to `waived` and attach the waiver's reason.
3. IF a waiver lacks a valid `reason`, names a check that cannot be waived, is a `target_proxy` waiver without a `column`, or gives a `column` for another check THEN the system SHALL raise `HexInputError`.
4. THE verdict and model card SHALL list every applied waiver with its reason.

### Requirement 12: Test-set use count

**User Story:** As a reviewer, I want to know how many times the test set was evaluated, so that I can judge how optimistic the reported metric is after repeated retraining.

#### Acceptance Criteria

1. THE verdict SHALL include `test_set_uses`: the number of runs in the app's history (including this one) that used the same resolved `data.path` relative to the app folder, `split.test_fraction`, `split.seed` and `split.time_column`. A run that ended on an input error never computed a metric, so it is not counted.
2. IF `test_set_uses` exceeds `checks.max_test_uses` THEN the system SHALL report a `warning` finding that the metric may be optimistic because the test set has been evaluated that many times.

### Requirement 13: Verdict

**User Story:** As a Kiro hook or workflow, I want one machine-readable verdict per run, so that I can decide whether to stop or loop.

#### Acceptance Criteria

1. THE verdict status SHALL be `REJECTED` if the metric fails or any `error` finding exists; otherwise `APPROVED` if the canary test passed in a full run; otherwise `UNVERIFIED` (a fast run, or a canary that was skipped, inconclusive or not applied).
2. THE `verdict.json` file SHALL contain: `status`, `mode` (`fast` or `full`), `app`, `run_id` (form `YYYYMMDD-HHMMSS-<6 hex chars>`, unique per run), `created_at` (ISO 8601 UTC), `hexperiment_version`, `metric` (`name`, `value`, `threshold`, `passed`), `checks` (each check's name, ghost and outcome: `passed`, `failed`, `skipped`, `inconclusive`, `not applied` or `not applicable`), `findings` (each with `check`, `ghost`, `severity`, `location`, `evidence`, `fix`, and `waiver_reason` when waived), `waivers`, `test_set_uses` and `checks_overrides` (an object recording every non-default `[checks]` configuration value; empty `{}` when none).
3. WHEN a run completes THEN the system SHALL write `verdict.json` and `model_card.md` into `runs/<run_id>/` and copy `verdict.json` into the app directory.
4. THE system SHALL write `verdict.json` as UTF-8 JSON with sorted keys and two-space indentation.
5. Running the same full run twice on the same inputs SHALL produce identical `status`, `metric` and `findings`.

### Requirement 14: Model card

**User Story:** As a reviewer, I want a short model card for every run, so that I can sign off a model without reading code.

#### Acceptance Criteria

1. WHEN a run completes THEN the system SHALL write `model_card.md` containing: app name and description, verdict status and mode, run id and date, train and test row counts and positive rates, the time range of each split (if a time column is configured), the metric against its threshold, a table of all checks with ghost name and outcome, every non-default check threshold from `checks_overrides` (if any), every finding with its fix hint, every waiver with its reason, the test-set use count, and a run history table of earlier runs of the same app (run id, status, metric).
2. THE model card SHALL state that it was generated by `hexperiment` and must not be edited by hand.
3. THE model card SHALL be valid GitHub-flavoured Markdown.

### Requirement 15: Run history

**User Story:** As a data scientist, I want every run kept, so that the path from a haunted result to an honest one is visible.

#### Acceptance Criteria

1. THE system SHALL never overwrite or delete an earlier `runs/<run_id>/` folder.
2. THE package SHALL expose `load_history(app_dir)` that returns the app's verdicts ordered by `created_at`.

### Requirement 16: Command line

**User Story:** As a Kiro hook or workflow, I want one command that runs an app and reports its verdict, so that it can be called on save and at the final gate.

#### Acceptance Criteria

1. THE package SHALL install a console command `hexperiment` with the subcommand `run <app_dir> [--fast | --full]` (default `--full`).
2. WHEN `hexperiment run` executes THEN the system SHALL run `<app_dir>/train.py` with the current Python interpreter, with `<app_dir>` as the working directory and `HEX_MODE` set per execution, perform the executions required by the mode (Requirement 10), then write the verdict and model card.
3. IF one execution of `train.py` exceeds `run.timeout_seconds` THEN the system SHALL stop it and treat the run as failed.
4. WHEN the run finishes THEN the system SHALL print the status, the metric against its threshold and one line per finding (ghost, location, fix).
5. THE command SHALL exit with code `0` for `APPROVED`, `1` for `REJECTED`, `3` for `UNVERIFIED`, and `2` if `train.py` fails, times out or does not produce a snapshot.

### Requirement 17: Quality

**User Story:** As a maintainer, I want the checks proven by tests, so that the kit itself can be trusted.

#### Acceptance Criteria

1. EACH check SHALL have example-based tests for a leaking case and a clean case.
2. EACH data check SHALL have at least one property-based test written with `hypothesis`, including: any test row copied from train is detected by the overlap check; a feature equal to the target is always detected by the proxy check; reordering rows never changes any check result; `split` membership never depends on row values; the canary scramble never introduces a value absent from a column's original test rows, never makes a column with two or more distinct test values constant, and always leaves both classes in a binary target.
3. THE canary test SHALL have tests with small training scripts proving: imputing with the mean of all rows before the split is detected; a scaler fitted on all rows before the split is detected; a target encoding computed over all rows before the split is detected; a leak inside a helper function in another module is detected; a model fitted on training and test rows together, with the training features unchanged, is detected; a clean pipeline that omits `y_score_train` ends `UNVERIFIED`; fitting on training rows only is not flagged; fitting on an unrelated dataset before the split is not flagged; a script without fixed seeds is reported as The Shapeshifter, not The Peeking Spirit.
4. A synthetic clean dataset (independent random features) SHALL produce no `error` findings.
5. THE full test suite SHALL pass with `.venv/Scripts/python -m pytest hexperiment/tests -q` and the time it took SHALL be reported. There is no time limit: the time depends on the machine's background load. On the reference laptop (AMD Ryzen 7 3750H, 4 cores, 8 threads), on the power plan it is left on (Power saver) with the Kiro IDE, a browser and an editor open, the suite took 135 to 213 s on 2026-11-12; 84 tests that train real models take most of it.
6. THE package SHALL pass `ruff check hexperiment`.
