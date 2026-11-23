# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The `hexperiment` package: `load`, `split` (seeded hash split and cutoff-based time split) and `audit`, configured by each app's `hex.toml`.
- The canary test: the training script runs again with the test rows scrambled (a tilted resample), plus a control run to rule out randomness, and the run is rejected when training changed.
- Data checks: row overlap between train and test, target proxies (single-column AUC), temporal leaks, a split-order hint and the metric gate, each named after a ghost.
- Verdicts (`APPROVED`, `REJECTED`, `UNVERIFIED`) with waivers, a model card for every run, and a run history that never overwrites earlier runs.
- The `hexperiment run` command with fast and full modes and exit codes, `hexperiment init` for new apps and for the Kiro files, and `hexperiment hook` for Kiro hooks.
- The `hexperiment-mcp` server with `list_apps`, `run_audit`, `get_verdict`, `get_history` and `explain_ghost`, and model cards as resources.
- Kiro files: a check-on-save hook, a Stop hook that tells the agent which changed apps are not approved, the `exorcise` workflow, the `ghost-hunter` agent (no shell; runs audits through `run_audit`) and the ML-integrity steering.
- The Kiro power in `power/hexperiment/`, with the skill and the MCP server.
- Two example apps on public data, `hotel-cancellations` and `flight-delays`, each with a haunted script, an honest script and a recorded repair by the `exorcise` workflow.
- The `hexperiment gallery` command, which builds a static site of the recorded runs, and a workflow that publishes it to GitHub Pages.
- A leak benchmark of 13 mechanisms from the Kapoor and Narayanan taxonomy, on synthetic data with three models, with its results table.
- Non-default check thresholds are recorded in `verdict.json` and the model card.
- A constraints file with the tested dependency versions.
- The gallery shows who made each recorded run (from an optional one-line `note.txt` in the run folder) and the benchmark totals above its table.

### Changed

- The ghost hunter runs audits only through the MCP `run_audit` tool; its shell was removed.
- Both example repairs were re-recorded from haunted scripts that carry no hints about the leaks.

### Fixed

- The canary is marked not applied, instead of passing, when the scramble never reaches the audited frames, when any audited column misses the scramble, and when `train.py` opens, copies, links or renames the data file outside `hexperiment.load`.
- Features a script derives under a new name, such as a part of the time column or a missing-value flag, no longer stop an honest run from being approved: the scramble check compares only the target and the raw data-file columns.
- An app file named like a standard-library module (for example `tomllib.py`) can no longer switch off the data-file watch.
- The gallery's benchmark section states that the data is synthetic and that the cases were written by the authors of the checks, and each case link's accessible name starts with its visible text.
- The canary fails closed on mixed predictions, bad load markers and feature-selection leaks, and needs training scores for a pass.
- Whole process trees are stopped on timeout, and run folders are written atomically.
- Target proxies hidden in missing-value patterns, dates and codes are caught.
- A preToolUse write guard refuses ghost hunter writes outside the app folders' Python files.
- The apps' test guard refuses network calls, including DNS and UDP, and stays inside the apps tests.
- Data paths in the test-set use count stay relative to the app folder.
