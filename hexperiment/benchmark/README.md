# Hexperiment leakage benchmark

This folder holds a benchmark of Hexperiment's leakage checks. It builds small leaky ML apps, each with a clean twin, runs both through the real `hexperiment run` command, and records which leaky apps were rejected, by which ghost, and whether any clean twin was rejected by mistake.

## What the benchmark is and is not

- The data is synthetic. It is generated from a seed by `hexbench/datagen.py`, so the same case always gets the same file. It says nothing about real data sets.
- The cases are written by the authors of the checks. The leaks are chosen from a published list of leakage types, but the same people wrote the checks and the cases. A result here is evidence that the checks see the leaks we thought of. It is not an independent measurement of how often they find leaks in the wild.
- Nothing in a case is tuned toward a check. A case is repaired only when its script crashes, does not do what its mechanism says, or its leak does not move the score. A change made because a ghost fired or did not fire is not allowed.
- Every app is run exactly as a user runs it: `hexperiment run <app>`, with no extra option, no `[checks]` section and no waiver in `hex.toml`.
- A missed leak is a result. The benchmark exits with code 0 whether or not leaks were caught.

### What each check can see

- The Doppelganger sees identical rows in both train and test.
- The Possession sees one feature that alone predicts the target almost perfectly (a stand-alone ROC AUC above `checks.proxy_auc_max`).
- The Peeking Spirit (the canary test) sees any dependence of training on the test rows' values. It does not see a leak that does not depend on those values.
- The Prophet sees training rows that are later than test rows, but only when `time_column` is declared in `hex.toml`.
- The Shapeshifter warns when identical runs give different results.

A leak outside these views is not expected to be found. Such cases are in the benchmark on purpose, so that the limits are measured and not assumed.

## The taxonomy

The leak types come from the survey table of Kapoor and Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", *Patterns* 4(9), 100804, 2023, doi:10.1016/j.patter.2023.100804. The names below are the ones on the paper's PubMed Central page (PMC10499856), word for word:

| Type | Name |
|---|---|
| L1.1 | No test set |
| L1.2 | Pre-processing on training and test set |
| L1.3 | Feature selection on training and test set |
| L1.4 | Duplicates in datasets |
| L2 | Model uses features that are not legitimate |
| L3.1 | Temporal leakage |
| L3.2 | Nonindependence between training and test samples |
| L3.3 | Sampling bias in test distribution |

L3.3 is not measured. It depends on the population a claim is about, which neither the code nor the data shows.

## The 13 mechanisms

Each mechanism is run with three models, so there are 39 cases and 78 apps. The models are not tuned and each has `random_state=0`: `lr` is `LogisticRegression`, `rf` is `RandomForestClassifier` and `hgb` is `HistGradientBoostingClassifier`. A case id is `<key>-<model>`, for example `scaler-all-rows-rf`.

1. `no-test-set` (L1.1, No test set): the model is scored on the rows it was trained on.
2. `scaler-all-rows` (L1.2, Pre-processing on training and test set): a standard scaler is fitted on all rows before the split.
3. `imputer-all-rows` (L1.2, Pre-processing on training and test set): a mean imputer is fitted on all rows before the split.
4. `oversample-before-split` (L1.2, Pre-processing on training and test set): the rare class is randomly over-sampled before the split.
5. `target-encoding-all-rows` (L1.2, Pre-processing on training and test set): a category's mean label is computed from all rows and used as a feature.
6. `univariate-selection-all-rows` (L1.3, Feature selection on training and test set): the 15 best columns are chosen with the labels of all rows before the split.
7. `duplicated-rows` (L1.4, Duplicates in datasets): part of the rows appear twice in the source file, and the split is random.
8. `post-outcome-copy` (L2, Model uses features that are not legitimate): a column recorded after the outcome copies the label.
9. `noisy-proxy` (L2, Model uses features that are not legitimate): a noisy measurement of the label is used as a feature.
10. `weak-proxy` (L2, Model uses features that are not legitimate): a weaker proxy of the label is used as a feature.
11. `random-split-in-script` (L3.1, Temporal leakage): time-ordered data is split at random by the script instead of with `hexperiment.split`.
12. `time-column-undeclared` (L3.1, Temporal leakage): the time column is not declared in `hex.toml`, so `hexperiment.split` splits at random.
13. `entity-leak` (L3.2, Nonindependence between training and test samples): several rows of one entity, each differing by small noise, are split at random.

For each mechanism the results file also records the check it was designed to catch. That is the ghost whose definition covers the leak, decided before the first run. Mechanisms 10 and 13 have none: no check is designed for them. A check that fires on a leak it was not designed for is reported as such and is not counted as design.

## Things to know when reading the table

- **The clean twin can change the test set.** For `no-test-set` (L1.1), `duplicated-rows` (L1.4) and `entity-leak` (L3.2) the clean twin does not only remove a leak: it changes which rows are scored (the real test rows, the test rows without duplicates, the test rows without known entities). The score pair of those cases mixes the leak with that change. Read the pair as "the score the script reports", which is what the results file records. In `entity-leak` the effect is the largest, so the data has many single-row entities and a smaller number of entities with several rows, and the clean test set is not the same set of rows as the leaky one.
- **Over-sampling and row ids.** The leaky script of `oversample-before-split` concatenates with `ignore_index=True`. Without it the frame would have duplicate row ids, which `audit` refuses as an input error: the run would end without a score and the effect of the leak could not be shown. That refusal is a different behaviour of the kit, and it is not what this case measures.
- **Rejected by the `input` check.** If a run ends as `REJECTED` because the kit refused the input (an error finding of the check `input`), it is counted as rejected and shown under the key `input` in `by_ghost`. That is a refusal of bad input, not a leak finding, and it is not read as a catch. A run that exits with code 2 is an `ERROR`, is not counted as caught, and the benchmark goes on. `UNVERIFIED` and `ERROR` runs are counted on their own and never folded into "caught".
- **The proxy pair brackets the default.** `noisy-proxy` and `weak-proxy` differ only in the noise added to the label. Their stand-alone AUCs lie on either side of the `checks.proxy_auc_max` default of 0.95: the noisy one above it, the weak one between 0.86 and 0.92. No case lies between 0.92 and 0.95, so how the check behaves close to its edge is not measured here.
- **Thresholds differ between mechanisms.** The metric threshold in `hex.toml` is lower for the cases whose honest score is expected to be low (`univariate-selection-all-rows`, and the two temporal mechanisms). It is the same in the leaky app and its twin, and it is not chosen from a verdict.
- **A leak has to move the score.** After each case the runner compares the leaky and the clean score. If they are equal to four decimals it prints `NO EFFECT`, and it lists the cases whose effect could not be checked because a score is missing. This is a report for the maintainer, not a failure.

## What we expected before the first run

We wrote these predictions into the benchmark's design before any case was run (commit `8dd6b24`, 2026-11-12). The numbers refer to the list of mechanisms above.

| Mechanisms | Predicted rejected | By |
|---|---|---|
| 1, 4, 7 | 9 of 9 | The Doppelganger |
| 2, 3, 5, 6 | 12 of 12 | The Peeking Spirit (6 is the least certain) |
| 8, 9 | 6 of 6 | The Possession |
| 11 | 3 of 3 | The Prophet |
| 10, 12, 13 | 0 of 9 | none: misses by design or by construction |

Predicted 30 of 39 leaky apps rejected, 0 of 39 clean twins rejected. These are predictions, not targets: nothing in the cases was adjusted toward them. Before the complete run we ran each mechanism once to check that its scripts run and that its leak moves the score; no case needed a change, and those runs are not the result.

## How to run it

Run all commands from the repository root with the project's virtual environment.

The whole benchmark takes about 40 minutes on a 2019 four-core laptop on a power-saving plan, and less on a faster machine. It runs one case at a time, so that no run competes with another for the processor. It writes each case to a temporary folder, deletes the folder after the case, and writes the results file only at the end, so an interrupted run leaves no partial file.

```powershell
.venv/Scripts/python hexperiment/benchmark/run_benchmark.py --out hexperiment/benchmark/results.json
```

To run one mechanism (six apps), name its key. The file it writes says `"complete": false` and holds only that mechanism, and `update_readme.py` refuses it:

```powershell
.venv/Scripts/python hexperiment/benchmark/run_benchmark.py --only scaler-all-rows --out $env:TEMP\one.json
```

Exit codes: 0 when the run finished (whatever it found), 2 when it could not run or stopped on an error (no results file is written), 130 after Ctrl+C.

After a complete run, `update_readme.py` writes the table below from `results.json`. Nothing between the two marker lines is written by hand:

```powershell
.venv/Scripts/python hexperiment/benchmark/update_readme.py
```

<!-- BEGIN BENCHMARK TABLE (generated by update_readme.py; do not edit) -->

| Leak type | Mechanism | Designed to catch | Rejected | Caught by | Mean score, leaky | Mean score, clean |
|---|---|---|---|---|---:|---:|
| L1.1 No test set | The model is scored on the rows it was trained on. | The Doppelganger | 3 of 3 (100%) | The Doppelganger (3) | 0.9182 | 0.7120 |
| L1.2 Pre-processing on training and test set | A standard scaler is fitted on all rows before the split. | The Peeking Spirit | 3 of 3 (100%) | The Peeking Spirit (3) | 0.7389 | 0.7418 |
| L1.2 Pre-processing on training and test set | A mean imputer is fitted on all rows before the split. | The Peeking Spirit | 3 of 3 (100%) | The Peeking Spirit (3) | 0.6763 | 0.6784 |
| L1.2 Pre-processing on training and test set | The rare class is randomly over-sampled before the split. | The Doppelganger | 3 of 3 (100%) | The Doppelganger (3) | 0.9296 | 0.7742 |
| L1.2 Pre-processing on training and test set | A category's mean label is computed from all rows and used as a feature. | The Peeking Spirit | 3 of 3 (100%) | The Peeking Spirit (3) | 0.8292 | 0.6663 |
| L1.3 Feature selection on training and test set | The 15 best columns are chosen with the labels of all rows before the split. | The Peeking Spirit | 3 of 3 (100%) | The Peeking Spirit (3) | 0.5947 | 0.5564 |
| L1.4 Duplicates in datasets | Part of the rows appear twice in the source file, and the split is random. | The Doppelganger | 3 of 3 (100%) | The Doppelganger (3) | 0.8492 | 0.7362 |
| L2 Model uses features that are not legitimate | A column recorded after the outcome copies the label. | The Possession | 3 of 3 (100%) | The Possession (3) | 1.0000 | 0.7292 |
| L2 Model uses features that are not legitimate | A noisy measurement of the label is used as a feature. | The Possession | 3 of 3 (100%) | The Possession (3) | 0.9874 | 0.7345 |
| L2 Model uses features that are not legitimate | A weaker proxy of the label (stand-alone AUC about 0.89) is used as a feature. | No check is designed for it | 0 of 3 (0%) | none | 0.9062 | 0.7443 |
| L3.1 Temporal leakage | Time-ordered data is split at random by the script instead of with `hexperiment.split`. | The Prophet | 3 of 3 (100%) | The Prophet (3) | 0.7978 | 0.7273 |
| L3.1 Temporal leakage | The time column is not declared in `hex.toml`, so `hexperiment.split` splits at random. | The Prophet | 0 of 3 (0%) | none | 0.8234 | 0.7227 |
| L3.2 Nonindependence between training and test samples | Several rows of one entity, each differing by small noise, are split at random. | No check is designed for it | 0 of 3 (0%) | none | 0.8386 | 0.7788 |
| clean twins: 0 false alarms in 39 scripts |  |  |  | none |  |  |

Not measured:

- L3.3 Sampling bias in test distribution: It depends on the population a claim is about, which neither the code nor the data shows.

Source: Kapoor and Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4(9), 100804, 2023, doi:10.1016/j.patter.2023.100804

Run with hexperiment 0.1.0, Python 3.13.2 and scikit-learn 1.9.1 on 2026-11-13: 78 runs.

<!-- END BENCHMARK TABLE -->

## Changes after the first complete run

None yet.
