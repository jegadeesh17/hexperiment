# Kiro fixing the haunted flight-delays app

Recorded on 2026-11-12 at 01:27 India time (the run ids use UTC: 2026-11-11 19:58) with Kiro CLI 3.2.0 in headless mode (`kiro-cli chat --v3 --no-interactive`, model `auto`). Kiro ran the `exorcise` workflow (`.kiro/workflows/exorcise.workflow.json`) on a clean copy of this repository in which `apps/flight-delays/train.py` was replaced by the haunted script from `apps/flight-delays/haunted/train.py`. The workflow repeats one step, run by the `ghost-hunter` agent (`.kiro/agents/ghost-hunter.json`), until `verdict.json` says APPROVED. Here one round was enough.

The four recorded runs in `apps/flight-delays/runs/`:

| Run | Mode | Status | ROC AUC | Made by |
| --- | --- | --- | --- | --- |
| 20261111-195811-190e10 | full | REJECTED | 0.7971 | the ghost hunter's first `hexperiment run` |
| 20261111-195934-464b02 | fast | UNVERIFIED | 0.6193 | the save hook, after the agent edited `train.py` (the agent itself ran no fast check) |
| 20261111-200013-a6caaf | full | APPROVED | 0.6193 | the ghost hunter's second run, on its fixed script |
| 20261111-200257-5f6e81 | full | APPROVED | 0.6193 | the authored honest script, restored afterwards |

The ghost hunter's own version of `train.py` is in the history at commit `62d3b5c`. The next commit, `01067c0`, put the authored honest script back as `train.py` and recorded its approved run, so the repository keeps one reference fix.

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

**Tool: Run full Hexperiment audit on app**

```
hexperiment run apps/flight-delays
```

Result:

```
Output:
REJECTED (full)  roc_auc 0.7971 >= 0.6000  run 20261111-195811-190e10
  error    The Prophet  time_hour    split by time with hexperiment.split
  warning  split_order  train.py:76  this runs before the split; make sure it does not use test rows
  warning  canary       train.py     split with hexperiment.split(df), not with your own split
  warning  metric       metric       treat this as a leak until proven otherwise: look for a column that encodes the target


Exit Code: 1
```

**Tool: Read File**
 `apps/flight-delays/train.py`

**Tool: Read File**
 `apps/flight-delays/verdict.json`

**Tool: Read File**
 `apps/flight-delays/hex.toml`

**Tool: List Directory**
 `apps/flight-delays`

**Tool: Code Intelligence**
 `hexperiment/src/hexperiment/data.py`

**Tool: Read File**
 `hexperiment/src/hexperiment/data.py`

**Tool: Read File**
 `apps/flight-delays/README.md`

**Tool: Replace in File**

`apps/flight-delays/train.py`

```diff
-     df = hexperiment.load()
-     # LEAK: same-day late rate of the carrier over all rows, test labels included.
-     day = df["time_hour"].str[:10]
-     df["carrier_day_late_rate"] = df.groupby(["carrier", day])[TARGET].transform("mean")
-     # LEAK: a random split on time-ordered data puts later flights in training.
-     train, test = train_test_split(df, random_state=0)  # LEAK: random split
-     test
+     df = hexperiment.load()
+     # Split by time with the kit: every training flight is earlier than every test flight.
+     train, test = hexperiment.split(df)
+     test
```

**Tool: Replace in File**

`apps/flight-delays/train.py`

```diff
-     "distance",
-     "carrier_day_late_rate",  # LEAK: The Peeking Spirit. Built from every row's label.
- ]
+     "distance",
+ ]
+ # Removed carrier_day_late_rate: it was the carrier's same-day late rate computed over
+ # all rows, so it was built from the labels of test flights, which are not known when
+ # a flight is scheduled.
```

**Tool: Replace in File**

`apps/flight-delays/train.py`

```diff
- from sklearn.model_selection import train_test_split
- 
+ 
```

**Tool: Replace in File**

`apps/flight-delays/train.py`

```diff
- # Haunted starting point of the demo (Requirements 3.2, 3.3). Do not fix it here.
- #
- # To run it, copy this file over ../train.py, then run hexperiment on the app.
- # It is the honest script plus the leaks marked "LEAK:" below.
- from
+ from
```

**Tool: Re-run full Hexperiment audit on app**

```
hexperiment run apps/flight-delays
```

Result:

```
Output:
APPROVED (full)  roc_auc 0.6193 >= 0.6000  run 20261111-200013-a6caaf


Exit Code: 0
```

**Tool: Confirm latest verdict status and findings**

```
Get-Content apps/flight-delays/verdict.json | Select-String '"status"|"findings"|"value"'
```

Result:

```
Output:
Blocked by a permission rule (deny shell matching "*&*, *;*, *|*, *>*, *<*, *`*, *$(*, *..*, git*, rm *, rmdir*, del *, Remove-Item*, curl*, wget*, Invoke-WebRequest*, Invoke-RestMethod*, iwr *, irm *, pip*, python*, hexperiment init*, hexperiment hook*"); the user did not reject it. Source: agent-profile. Triggered by: "Select-String '"status"|"findings"|"value"'".

Exit Code: -1
```

**Tool: Read File**
 `apps/flight-delays/verdict.json`

**Tool: Send Message**

> apps/flight-delays is APPROVED: roc_auc 0.6193 >= 0.6000, no findings. Fixed in train.py: split with hexperiment.split, removed the label-derived carrier_day_late_rate feature.

**Ghost hunter:**

`hexperiment run apps/flight-delays` now ends APPROVED. The first run was REJECTED, with The Prophet as the only error. Both fixes are in `apps/flight-delays/train.py`:

- **The Prophet (`time_hour`):** the script split with scikit-learn's `train_test_split`, which put later flights in training. It now calls `hexperiment.split(df)`, so every training flight is earlier than every test flight.
- **Warnings (`split_order`, `canary`, `metric`):** the `split_order` warning pointed at `carrier_day_late_rate`. That feature was each carrier's same-day late rate computed over all rows, so it carried test labels into training. I removed it because it is built from the target and cannot be known when a flight is scheduled. This is also why the AUC was 0.7971 before and is 0.6193 now.

I also deleted the stale "haunted starting point" header comment and the unused `train_test_split` import. I didn't touch `hex.toml` or any other file.

Status: APPROVED
Metric: roc_auc 0.6193 against a threshold of 0.6000 (passed)
Findings in the latest `verdict.json`: none. All six checks passed: The Doppelganger, The Possession, The Prophet, split_order, The Peeking Spirit and The Shapeshifter.

*Loop `hunt-until-approved` round 1 ended; stop condition met: True.*

*Workflow finished with status `completed`.*
