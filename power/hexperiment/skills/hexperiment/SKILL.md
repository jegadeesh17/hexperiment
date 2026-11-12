---
name: hexperiment
description: "Use when training, tuning or reviewing a tabular ML model and the result must be trusted: data leakage, train test split, target leakage, a model card, or a score that looks too good to be true. Audits the training script, writes a verdict (APPROVED, REJECTED or UNVERIFIED) and a model card."
---

# Hexperiment: leakage-proof ML

"If your model looks too good to be true, it's haunted."

Hexperiment checks every run of a training script and refuses to approve a result that is good only because it cheated. You write the training code. The kit decides whether the number counts.

## When to use it

Use it when any of these is true:

- You write or change a training script in an app folder (a folder with `hex.toml` and `train.py`).
- The user asks about data leakage, a train/test split, a model card, or a model that scores suspiciously well.
- You are about to report a metric. Run the audit first and report what the verdict says.

## Setup

1. Install: `pip install "hexperiment[mcp]"`. The `mcp` extra is needed only for the MCP server.
2. Start an app: `hexperiment init apps/<name>`. It writes a commented `hex.toml` and a `train.py` that follows the rules below. It never overwrites a file and lists what it created and skipped.
3. Add the Kiro files to the project: `hexperiment init --kiro`. It writes the save hook, the `exorcise` workflow, the `ghost-hunter` agent and the steering file into `.kiro/`.
4. Put the data at the `data.path` of `hex.toml` and edit the lines marked `CHANGE`.
5. Make sure `hexperiment` is on the `PATH` of the shell that starts Kiro. Hooks and the ghost hunter run the bare command `hexperiment`.

The ghost hunter and the workflow work on apps below `apps/`.

If `list_apps` returns no apps, the server was started in the power's install folder, not in your project. Run `hexperiment init --kiro` in the project (or add the entry it prints to `.kiro/settings/mcp.json`) and use the workspace-level `hexperiment` tools.

## The ML-integrity rules

These rules apply to every training script in `apps/`. They exist because an agent optimizing a metric will find a leak if one is available. A result that breaks a rule is not a result.

0. **Let the kit own the data.** Load with `hexperiment.load()` and split with `hexperiment.split(df)`. Fix every random seed (`random_state=...`) so runs are deterministic; the canary test depends on it.
1. **Split first.** Call the train/test split before any `fit`, `fit_transform` or resampling. Fit every scaler, encoder, imputer and selector on training rows only, preferably inside a scikit-learn `Pipeline`.
2. **No row in both worlds.** A row must never appear in both train and test. Remove the overlap after the split: drop every test row that also appears in the training rows. Duplicates within the training rows are not a leak and may stay. Do not deduplicate the whole frame before the split; in the canary run the scrambled test rows deduplicate differently, which changes the training set and leaves the run unverified.
3. **No features from the future.** Use only columns that are known at prediction time. If a column is recorded after the outcome (a discharge code, a refund flag, a "days until churn"), drop it and say why in a comment.
4. **Respect time.** If `hex.toml` sets `time_column`, split by time: every training row must be earlier than every test row. Never shuffle across time. Build date features from the one time column, not by combining separate year, month and day columns: the canary run scrambles each column on its own, so a combined date can be impossible.
5. **Suspicion is a signal.** If a single feature predicts the target almost perfectly, or the metric is far above the threshold, treat it as a leak until proven otherwise.
6. **Always end with an audit.** Every training script ends with `hexperiment.audit(...)`. Never edit `verdict.json` or a model card by hand, and never lower a threshold in `hex.toml` to get a pass.
7. **Fix the cause, not the check.** When the audit reports a ghost, change the data handling that caused it. Do not silence, skip or weaken a check.

## Running the audit

- `hexperiment run apps/<name>` is the full run. It runs the training script several times, including the canary test, and writes `verdict.json` in the app folder and `runs/<run_id>/` with `verdict.json` and `model_card.md`.
- `hexperiment run --fast apps/<name>` skips the canary test. It is quick feedback and can never give `APPROVED`.
- Every run deletes the old `verdict.json` first. If the file exists, it belongs to a run that finished.
- Every run counts as a use of the test set. Do not run the audit in a tight loop to tune against the test metric.

With the hook installed, saving a Python file of an app runs the fast check. It clears the app's verdict and is never `APPROVED`, so an earlier `APPROVED` stops counting once you change the code. The current Kiro versions do not show you the hook's output, so after you change a training script, run `hexperiment run` yourself (or call `get_verdict`) and read the result before you call the work done.

## Reading a verdict

Status:

| Status | Meaning |
|---|---|
| `APPROVED` | A full run finished, every check passed and the metric reached its threshold. |
| `REJECTED` | A check found a leak, or the metric is below its threshold. |
| `UNVERIFIED` | Nothing was found, but the run could not be fully verified (a fast run, or results that changed between identical runs). |

Exit codes of `hexperiment run`: `0` approved, `1` rejected, `2` the run failed and there is no verdict (read the message), `3` unverified.

Each finding in `verdict.json` has `check`, `ghost`, `severity` (`error`, `warning` or `waived`), `location`, `message`, `evidence` and `fix`. Fix an `error` at its cause using the `fix` and the `location`. A `warning` does not change the status.

Read the newest `runs/<run_id>/model_card.md` for the metric, the data summary and each finding in plain words.

## The ghosts and the other checks

Each check reports a named ghost, so a failed run reads like a story.

| Ghost | Check | What it means | How to fix it |
|---|---|---|---|
| The Doppelganger | `row_overlap` | Identical rows exist in both train and test. | Follow rule 2: after the split, drop every test row that also appears in the training rows. |
| The Possession | `target_proxy` | One feature alone predicts the target almost perfectly. | Find out when the column is recorded. If it is after the outcome, drop it and say why in a comment (rule 3). Only a human can accept a strong feature, with a waiver. |
| The Peeking Spirit | `canary` | Scrambling the test rows changed what training saw, so test information reached training. It cannot be waived. | Fit every imputer, scaler, encoder and selector on training rows only, after the split (rule 1). The finding lists suspect lines. |
| The Prophet | `temporal` | Some training rows are later than some test rows. | Set `split.time_column` in `hex.toml`, split with `hexperiment.split` and never shuffle across time (rule 4). |
| The Shapeshifter | `determinism` | Identical runs gave different results (a warning). Nothing can be verified. | Set `random_state` or a seed everywhere (rule 0). |

Checks without a ghost:

| Check | What it means |
|---|---|
| `split_order` | A fit or a statistic comes before the first split, or data is read without `hexperiment.load` (warnings). A hint to the likely cause of The Peeking Spirit. |
| `metric` | The metric is below its threshold, or far above it (a "too good to be true" warning). Below: improve the model, never the threshold. Far above: look for a column that encodes the target. |
| `test_set_uses` | The same test set was scored more than `checks.max_test_uses` times. Stop tuning against it. |
| `waiver` | A waiver in `hex.toml` matched no finding (a warning). A human removes it. |
| `input` | `audit` received invalid input, so no check ran and the verdict is `REJECTED`. The message says what to fix. |

## The loop

To hand an app to an agent until the full audit approves it, start the workflow in the Kiro CLI:

```text
/workflow run exorcise --app apps/<name>
```

The workflow repeats the `ghost-hunter` agent at most 5 times and stops when `apps/<name>/verdict.json` says `APPROVED`. Then it pauses for the human with the latest status and findings.

The ghost hunter can edit the Python files of the app and run `hexperiment run` and `hexperiment run --fast` on it. It cannot edit `hex.toml`, `verdict.json`, `model_card.md`, `fetch_data.py`, `runs/`, `.hex/`, `data/`, `haunted/`, `.kiro/` or the hexperiment package. If it cannot fix a finding honestly, it reports the finding and stops.

## MCP tools

The `hexperiment` MCP server uses the folder it starts in as the workspace, or `HEXPERIMENT_WORKSPACE` when that is set.

- `list_apps()`: every app with its name, path and current status (`none` when it has no verdict). The result names the workspace; check it first.
- `run_audit(app, mode="full")`: runs `hexperiment run` on the app. The mode is `full` or `fast`. This is the only tool that executes code.
- `get_verdict(app)`: the current `verdict.json`, or a message that the app has none.
- `get_history(app)`: the earlier runs, oldest first, with run id, status, metric value and ghosts.
- `explain_ghost(name)`: what a ghost or a check means, why it matters and how to fix it.

The newest model card of each app is also a resource.

## What no agent may do

- Only `hexperiment run` in full mode produces an APPROVED verdict. A hook run, a fast run, your own summary and a hand-written file do not.
- Nobody edits `verdict.json` or a model card.
- Thresholds and waivers belong to the human. Never lower a threshold in `hex.toml` and never add a waiver to get a pass. If an honest fix leaves the metric below its threshold, report that.
- Removing the `hexperiment.audit` call, or a feature that is not the cause of a finding, is not a fix.
- When the final reply reports a result, state the status, the metric against its threshold and one line per finding of the latest `verdict.json`.
