# Requirements Document: hexperiment-benchmark

## Introduction

Two example apps show that Hexperiment can catch a leak; they do not show how often it does. This spec adds a leak benchmark: a fixed set of small training scripts, each with one planted leak or none, run through the real `hexperiment run` command. Its results state, per kind of leak, how many leaky scripts were rejected and by which ghosts, and how many clean scripts were wrongly rejected.

A benchmark written by the authors of the checks can easily test only what the checks were built for. To avoid that, the kinds of leak come from a published taxonomy, the eight leakage types of Kapoor and Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 4(9), 100804, 2023 (doi:10.1016/j.patter.2023.100804). Types that Hexperiment is not designed to catch are measured too, and reported as misses.

Scope: the benchmark cases, the runner, the committed results file, its format (which the gallery reads, see `hexperiment-gallery`, Requirement 6), a short README for the benchmark, and tests. Changing a check because of a benchmark result is out of scope for this spec.

## Glossary

- **Leakage type:** one of the eight types of the taxonomy, with its ID (`L1.1` to `L3.3`) and name as the paper writes it.
- **Mechanism:** one concrete way to cause a leakage type in code, for example "scaler fitted on all rows before the split" for `L1.2`.
- **Case:** one mechanism with one model and one data seed: a leaky script and its clean twin.
- **Clean twin:** the leaky script with the leak removed and nothing else changed.
- **Rejected:** the run's status is `REJECTED`.

## Requirements

### Requirement 1: Cases from the taxonomy

**User Story:** As a judge, I want the benchmark to cover the kinds of leakage the research literature names, so that its catch rate is not chosen to flatter the kit.

#### Acceptance Criteria

1. THE benchmark SHALL contain at least the following mechanisms, each tagged with its leakage type:
   - `L1.1` No test set: the model is scored on its own training rows.
   - `L1.2` Pre-processing on training and test set: a scaler fitted on all rows; an imputer fitted on all rows; random over-sampling of the minority class before the split; a target (mean) encoding computed from all rows' labels.
   - `L1.3` Feature selection on training and test set: univariate selection with the labels of all rows.
   - `L1.4` Duplicates in datasets: a source file in which part of the rows appear more than once, split at random.
   - `L2` Model uses features that are not legitimate: a column recorded after the outcome that copies it; a noisy proxy of the target; a weaker proxy (stand-alone AUC between 0.85 and 0.92).
   - `L3.1` Temporal leakage: the script splits time-ordered data at random instead of with `hexperiment.split`; the time column is not declared in `hex.toml`, so `hexperiment.split` splits at random.
   - `L3.2` Nonindependence between training and test samples: several rows per entity with small differences, split at random.
2. THE benchmark SHALL list `L3.3` (sampling bias in test distribution) as not measured, with the reason: it depends on the population a claim is about, which neither the code nor the data shows.
3. EACH mechanism SHALL have three cases, one per model (`LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, each with a fixed `random_state`), each with its own data seed.
4. EACH leaky case SHALL mark every line of its leak, in `train.py` or `hex.toml`, with a `# LEAK:` comment, and its clean twin SHALL differ from it only in those lines: in a line diff of the two apps, every line on the leaky side carries `# LEAK:`, and the clean side holds only the lines that replace them. Each leak SHALL be one that changes the result: on its own data, the leaky app's test score SHALL differ from its clean twin's (Requirement 4.2 records both). Every clean twin SHALL split with `hexperiment.split`, because the canary scrambles exactly the rows that `hexperiment.split` puts in the test set. The clean twin of `L1.1` scores on the test rows; of `L1.4`, drops from the test rows every row that also occurs in the training rows; of `L3.1`, declares the time column; of `L3.2`, drops from the test rows every entity that also occurs in the training rows.
5. THE cases SHALL use each app's default checks: no `[checks]` table and no waivers in any `hex.toml`.

### Requirement 2: Data

**User Story:** As a maintainer, I want the benchmark to run anywhere without a download, so that anyone can repeat it.

#### Acceptance Criteria

1. THE benchmark SHALL generate its data with numpy and scikit-learn from fixed seeds, so that the same seed gives the same file on every machine.
2. THE datasets SHALL be small enough that one full run of one case takes at most 30 seconds in everyday use (as defined in `hexperiment-kiro`, Requirement 7.4), and large enough that the clean twin's score is stable (at least 2,000 rows).
3. THE datasets of `L3.1` SHALL have a time column, declared in `[split] time_column` in every app except the leaky app of the undeclared-time-column mechanism; the datasets of `L3.2` SHALL have an entity column that is not used as a feature.

### Requirement 3: The runner

**User Story:** As a maintainer, I want one command that runs every case through the real product and writes the results, so that the numbers in the README and the gallery come from real runs.

#### Acceptance Criteria

1. THE command `python hexperiment/benchmark/run_benchmark.py --out FILE` SHALL write every case's leaky app and clean twin in a new temporary folder, run `hexperiment run <app>` (full mode) on each through the installed command, read each verdict and write the results file to `FILE`.
2. THE option `--only MECHANISM` SHALL run the cases of one mechanism, for checking a single mechanism quickly. A results file written with `--only` SHALL say so, and the gallery SHALL refuse it.
3. THE runner SHALL print one line per case as it finishes and a summary table at the end, delete its temporary folder, and exit `0` whether or not leaks were caught. IF a run cannot start (for example the command is missing) THEN it SHALL stop with exit code `2` and name the problem.
4. A run that ends with an input error (exit code `2`) or a timeout SHALL be recorded with the status `ERROR` and its message, and SHALL NOT be counted as caught.
5. THE runner SHALL record the versions of `hexperiment`, Python, numpy, pandas and scikit-learn, the operating system and the total run time in the results file.

### Requirement 4: Results file

**User Story:** As the author of the gallery and the README, I want one documented results format, so that both show the same numbers.

#### Acceptance Criteria

1. THE results file SHALL be a JSON object with: `format` (`"hexperiment-benchmark/1"`), `created_at` (ISO 8601 UTC), `complete` (false when written with `--only`), `environment` (the versions and run time of Requirement 3.5), `source` (the taxonomy citation), `mechanisms`, `clean` and `not_measured`.
2. EACH item of `mechanisms` SHALL have: `leak_type` (for example `"L1.2"`), `leak_type_name` (the paper's name), `mechanism` (one sentence), `designed_to_catch` (the ghost names that should catch it, an empty list when no check is designed for it), `scripts`, `rejected`, `by_ghost` (object from ghost name, or check name for a finding without a ghost, to the number of rejected scripts with an error finding from it), `statuses` (object from status to count, including `ERROR`), and `mean_score_leaky` and `mean_score_clean` (the mean test ROC AUC of the leaky apps and of their clean twins, over the runs that produced a score, or `null` when none did).
3. `clean` SHALL have `scripts`, `false_alarms` (clean twins that were rejected), `by_ghost` and `statuses`. `not_measured` SHALL be a list of objects with `leak_type`, `leak_type_name` and `reason`.
4. THE committed results file SHALL be `hexperiment/benchmark/results.json`, written by one complete run in everyday use.

### Requirement 5: Honest reporting

**User Story:** As a judge, I want to trust that the published numbers were not tuned after the fact, so that the benchmark means something.

#### Acceptance Criteria

1. THE benchmark SHALL NOT lower, raise or waive any check for its cases, and the runner SHALL NOT pass any option that changes a check.
2. THE cases SHALL NOT be changed after the first complete run, except to fix a script that does not do what its mechanism says. Each such change SHALL be listed in `hexperiment/benchmark/README.md` with the reason, and the benchmark SHALL be run again in full.
3. `hexperiment/benchmark/README.md` SHALL describe every mechanism in one sentence, cite the taxonomy, state that the data is synthetic and the cases are written by the authors of the checks, show the results table from the committed results file, and name every mechanism that was missed or only partly caught, with the reason as far as it is known.

### Requirement 6: Quality

**User Story:** As a maintainer, I want the benchmark checked by the test suite without slowing it down, so that it cannot silently break.

#### Acceptance Criteria

1. THE tests SHALL check, without running any case: that every case writes a valid app (its `hex.toml` loads and its scripts compile); that every leaky app and its clean twin differ only in `# LEAK:` lines; that the mechanisms cover every leakage type of Requirement 1.1 and list `L3.3` as not measured; the summary built from verdicts made in the test, including `ERROR`; and that the committed `results.json` matches the format of Requirement 4 and the current list of mechanisms.
2. THE benchmark SHALL add no runtime dependency to the package and SHALL NOT be importable as part of `hexperiment` (it is not installed with the package).
3. THE full test suite SHALL pass with `.venv/Scripts/python -m pytest hexperiment/tests -q` in under 2 minutes in everyday use, and `ruff` SHALL report no problems in the benchmark folder.
