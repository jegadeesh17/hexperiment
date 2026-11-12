# Kiro fixing the haunted hotel-cancellations app

Recorded on 2026-11-12 at 01:19 India time (the run ids use UTC: 2026-11-11 19:49) with Kiro CLI 3.2.0 in headless mode (`kiro-cli chat --v3 --no-interactive`, model `auto`). Kiro ran the `exorcise` workflow (`.kiro/workflows/exorcise.workflow.json`) on a clean copy of this repository in which `apps/hotel-cancellations/train.py` was replaced by the haunted script from `apps/hotel-cancellations/haunted/train.py`. The workflow repeats one step, run by the `ghost-hunter` agent (`.kiro/agents/ghost-hunter.json`), until `verdict.json` says APPROVED. Here one round was enough.

The four recorded runs in `apps/hotel-cancellations/runs/`:

| Run | Mode | Status | ROC AUC | Made by |
| --- | --- | --- | --- | --- |
| 20261111-194957-38a52f | full | REJECTED | 1.0000 | the ghost hunter's first `hexperiment run` |
| 20261111-195114-c55aa8 | fast | UNVERIFIED | 0.9090 | the save hook, after the agent edited `train.py` (the agent itself ran no fast check) |
| 20261111-195154-f34364 | full | APPROVED | 0.9090 | the ghost hunter's second run, on its fixed script |
| 20261111-195610-68b4d6 | full | APPROVED | 0.9096 | the authored honest script, restored afterwards |

The ghost hunter's own version of `train.py` is in the history at commit `0344b90`. The next commit, `05b8faa`, put the authored honest script back as `train.py` and recorded its approved run, so the repository keeps one reference fix.

Below is the ghost hunter's session from the log: the step prompt, its tool calls with their inputs and outputs, and its replies. Only local paths were shortened; nothing else was edited. The model's private reasoning is not shown.

---
*Workflow `exorcise` started with input `app = apps/hotel-cancellations`.*

## Round 1: step `hunt`, agent `ghost-hunter`

**Step prompt (from `.kiro/workflows/exorcise.workflow.json`):**

> Work on the Hexperiment app in the folder given as the workflow input `app` (apps/hotel-cancellations), written with forward slashes relative to the workspace, for example apps/churn.
> 1. Run `hexperiment run apps/hotel-cancellations` (full mode).
> 2. Read `apps/hotel-cancellations/verdict.json` and the model card in the newest folder under `apps/hotel-cancellations/runs/`.
> 3. If the status is APPROVED, stop here and report it.
> 4. Otherwise change the app's Python files so that each finding is fixed at its cause. Use the finding's fix hint and the suspect lines it lists. Then run `hexperiment run apps/hotel-cancellations` again as the last thing you do in this step.
> Do not change thresholds, waivers or any file outside the app's Python code. If you cannot fix a finding, say so and stop; do not work around it.
> End your reply with the status, the metric against its threshold, and one line per finding of the latest verdict.json.

**Tool: Run Command**

```
hexperiment run apps/hotel-cancellations
```

Result:

```
Output:
REJECTED (full)  roc_auc 1.0000 >= 0.8500  run 20261111-194957-38a52f
  error    The Possession      reservation_status  check whether this column is recorded after the outcome; if so, drop it
  warning  split_order         train.py:100        this runs before the split; make sure it does not use test rows
  error    The Peeking Spirit  train.py            test rows influenced training; fit preprocessing on training rows only, after the split. Suspect lines: train.py:100.
  warning  metric              metric              treat this as a leak until proven otherwise: look for a column that encodes the target


Exit Code: 1
```

**Tool: Read File**
 `apps/hotel-cancellations/train.py`

**Tool: List Directory**
 `apps/hotel-cancellations`

**Tool: Replace in File**

`apps/hotel-cancellations/train.py`

```diff
-     "customer_type",
-     "reservation_status",  # LEAK: The Possession. Filled in when the booking ends.
- ]
+     "customer_type",
+     # reservation_status is dropped: it is recorded when the booking ends
+     # (Check-Out, Canceled, No-Show), so it encodes the outcome and is unknown at
+     # prediction time.
+ ]
```

**Tool: Replace in File**

`apps/hotel-cancellations/train.py`

```diff
-     "country_cancel_rate",  # LEAK: The Peeking Spirit. Built from every row's label.
- ]
+     "country_cancel_rate",  # Built from training rows only, after the split.
+ ]
```

**Tool: Replace in File**

`apps/hotel-cancellations/train.py`

```diff
-     # LEAK: The Peeking Spirit. The rate uses every row, so test labels reach training.
-     df["country_cancel_rate"] = df.groupby("country", dropna=False)[TARGET].transform(
-         "mean"
-     )
-     train, test = hexperiment.split(df)
-     test = drop_seen_test_rows
+     train, test = hexperiment.split(df)
+     train, test = add_country_cancel_rate(train, test)
+     test = drop_seen_test_rows
```

**Tool: Replace in File**

`apps/hotel-cancellations/train.py`

```diff
- def build_model(
+ def add_country_cancel_rate(
+     train: pd.DataFrame, test: pd.DataFrame
+ ) -> tuple[pd.DataFrame, pd.DataFrame]:
+     """Add the cancel rate per country, computed from the training rows only.
+ 
+     Test rows look the rate up; countries unseen in training get NaN.
+     """
+     rate = train.groupby("country", dropna=False)[TARGET].mean()
+     train = train.assign(country_cancel_rate=train["country"].map(rate))
+     test = test.assign(country_cancel_rate=test["country"].map(rate))
+     return train, test
+ 
+ 
+ def build_model(
```

**Tool: Replace in File**

`apps/hotel-cancellations/train.py`

```diff
- # Haunted starting point of the demo (Requirements 2.2, 2.3). Do not fix it here.
- #
- # To run it, copy this file over ../train.py, then run hexperiment on the app.
- # It is the honest script plus the leaks marked "LEAK:" below.
- from
+ from
```

**Tool: Run Command**

```
hexperiment run apps/hotel-cancellations
```

Result:

```
Output:
APPROVED (full)  roc_auc 0.9090 >= 0.8500  run 20261111-195154-f34364
  warning  split_order  train.py:73  this runs before the split; make sure it does not use test rows


Exit Code: 0
```

**Tool: Read File**
 `apps/hotel-cancellations/verdict.json`

**Tool: Send Message**

> apps/hotel-cancellations is APPROVED (roc_auc 0.9090 >= 0.8500, run 20261111-195154-f34364). I dropped reservation_status and computed country_cancel_rate from training rows only, after the split.

**Ghost hunter:**

**Status: APPROVED** (full mode, run 20261111-195154-f34364)

The first run was REJECTED with roc_auc 1.0000, which is far too high to trust. I made two changes in `apps/hotel-cancellations/train.py`:
- **The Possession:** I dropped `reservation_status`. It is filled in when the booking ends (Check-Out, Canceled or No-Show), so it encodes the outcome and isn't known at prediction time.
- **The Peeking Spirit:** `country_cancel_rate` was computed from every row's label before the split. A new function, `add_country_cancel_rate`, now computes it from training rows only, after the split. Test rows look the rate up, and countries not seen in training get NaN.

I also removed the stale "haunted starting point, do not fix" comment at the top of the file.

**Metric:** roc_auc 0.9090 against a threshold of 0.8500, passed.

**Findings in the latest `verdict.json`:**
- **Warning, `split_order`, `train.py:73`:** `mean()` is called before the first split. The call sits in the new helper function, which is defined above `main()` but runs only after the split, on training rows. The canary check passed, so I left it.

All checks passed: The Doppelganger, The Possession, The Peeking Spirit and The Shapeshifter. The Prophet is not applicable because no time column is set.

*Loop `hunt-until-approved` round 1 ended; stop condition met: True.*

*Workflow finished with status `completed`.*
