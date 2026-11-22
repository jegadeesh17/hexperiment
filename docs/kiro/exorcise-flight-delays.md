# Kiro fixing the haunted flight-delays app

Recorded on 2026-11-22 at 18:45 India time (the run ids use UTC: 2026-11-22 13:15) with Kiro CLI in headless mode (`kiro-cli chat --v3 --no-interactive`, model `auto`). Kiro ran the `exorcise` workflow (`.kiro/workflows/exorcise.workflow.json`) on a clean copy of this repository in which `apps/flight-delays/train.py` was replaced by the hint-free haunted script. The workflow repeats one step, run by the `ghost-hunter` agent (`.kiro/agents/ghost-hunter.json`), until `verdict.json` says APPROVED. Here one round was enough.

The four recorded runs in `apps/flight-delays/runs/`:

| Run | Mode | Status | ROC AUC | Made by |
| --- | --- | --- | --- | --- |
| 20261122-131518-e8db31 | full | REJECTED | 0.7971 | the ghost hunter's first `run_audit` call |
| 20261122-131620-2ed225 | fast | UNVERIFIED | 0.6193 | the save hook, after the agent edited `train.py` (the agent itself ran no fast check) |
| 20261122-131642-1933a1 | full | APPROVED | 0.6193 | the ghost hunter's second run, on its fixed script |
| 20261122-131840-3eec70 | full | APPROVED | 0.6193 | the authored honest script, restored afterwards |

The ghost hunter's own version of `train.py` passed all checks. The authored honest script was restored as `train.py` and recorded its approved run last, so the repository keeps one reference fix.

Below is the ghost hunter's session from the log: the step prompt, its tool calls with their inputs and outputs, and its replies. Only local paths were shortened; nothing else was edited. The model's private reasoning is not shown.

---
*Workflow `exorcise` started with input `app = apps/flight-delays`.*

## Round 1: step `hunt`, agent `ghost-hunter`

**Step prompt (from `.kiro/workflows/exorcise.workflow.json`):**

> Work on the Hexperiment app in the folder given as the workflow input `app` (apps/flight-delays), written with forward slashes relative to the workspace, for example apps/churn.
> 1. Run `hexperiment run apps/flight-delays` (full mode).
> 2. Read `apps/flight-delays/verdict.json` and the model card in the newest folder under `apps/flight-delays/runs/`.
> 3. If the status is APPROVED, stop here and report it.
> 4. Otherwise change the app's Python files so that each finding is fixed at its cause. Use the finding's fix hint and the suspect lines it lists. Then run `hexperiment run apps/flight-delays` again as the last thing you do in this step.
> Do not change thresholds, waivers or any file outside the app's Python code. If you cannot fix a finding, say so and stop; do not work around it.
> End your reply with the status, the metric against its threshold, and one line per finding of the latest verdict.json.

**Tool: Read File**
 `apps/flight-delays/hex.toml`

**Tool: Read File**
 `apps/flight-delays/train.py`

**Tool: List Directory**
 `apps/flight-delays`

**Tool: @hexperiment/run_audit**

```json
{"app": "apps/flight-delays", "mode": "full"}
```

Result:

```
REJECTED (full)  roc_auc 0.7971 >= 0.6000  run 20261122-131518-e8db31
  error    The Prophet  time_hour    split by time with hexperiment.split
  warning  split_order  train.py:71  this runs before the split; make sure it does not use test rows
  warning  canary       train.py     split with hexperiment.split(df), not with your own split
  warning  metric       metric       treat this as a leak until proven otherwise: look for a column that encodes the target
```

**Tool: Grep Search**
 `def (split|load)\b` in `hexperiment/src/hexperiment/data.py`

**Tool: Read File**
 `apps/flight-delays/verdict.json`

**Tool: Replace in File**

`apps/flight-delays/train.py`

```diff
-     df = hexperiment.load()
-     day = df["time_hour"].str[:10]
-     df["carrier_day_late_rate"] = df.groupby(["carrier", day])[TARGET].transform("mean")
-     train, test = train_test_split(df, random_state=0)
+     df = hexperiment.load()
+     train, test = hexperiment.split(df)
```

**Tool: Replace in File**

`apps/flight-delays/train.py`

```diff
-     "distance",
-     "carrier_day_late_rate",
- ]
+     "distance",
+ ]
+ # carrier_day_late_rate was removed: it averaged the target (late) over all flights of a
+ # carrier on a day, so each row carried the outcomes of the flights in the same day,
+ # including test flights. Those outcomes are not known at prediction time.
```

**Tool: Replace in File**

`apps/flight-delays/train.py`

```diff
- from sklearn.model_selection import train_test_split
- 
```

**Tool: @hexperiment/run_audit**

```json
{"app": "apps/flight-delays", "mode": "full"}
```

Result:

```
APPROVED (full)  roc_auc 0.6193 >= 0.6000  run 20261122-131642-1933a1
```

**Tool: Read File**
 `apps/flight-delays/verdict.json`

**Tool: Send Message**

> apps/flight-delays is APPROVED (full run 20261122-131642-1933a1): roc_auc 0.6193 >= 0.6000, no findings. Fixed in train.py: split with hexperiment.split, removed carrier_day_late_rate.

**Ghost hunter:**

`apps/flight-delays` is APPROVED. I checked the current `verdict.json` myself: it comes from a full run (`20261122-131642-1933a1`) with the canary test included.

- **Metric:** roc_auc is 0.6193, against a threshold of 0.6000.
- **Checks:** all six passed (row overlap, target proxy, temporal, split order, canary, determinism), with no findings and no waivers.
- **Fix:** the ghost-hunter changed `train.py` to split with `hexperiment.split` and dropped the `carrier_day_late_rate` feature.

*Loop `hunt-until-approved` round 1 ended; stop condition met: True.*

*Workflow finished with status `completed`.*
