# flight-delays

Will a 2013 New York flight arrive more than 15 minutes late? The model decides from the schedule alone: carrier, origin, destination, scheduled departure and arrival time, distance, and the date.

This app starts **haunted**: its training script splits the flights at random although they are ordered in time, and adds a statistic computed over all rows. Hexperiment flags it as `REJECTED`. The honest script splits by time and reaches an AUC of 0.6193 with the verdict `APPROVED`.

## The data

- **Dataset:** nycflights13, every flight that left New York City (EWR, JFK, LGA) in 2013. The source has 336,776 flights and 20 columns.
- **Source:** <https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv> (33,206,996 bytes), via Rdatasets. `fetch_data.py` checks the file against a pinned SHA-256 (`b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42`).
- **Licence and attribution:** nycflights13, CC0, original data from the US Bureau of Transportation Statistics.
- The data is downloaded, not committed (`apps/*/data/` is ignored by git).
- **Preparation.** `fetch_data.py` drops the row-number column, drops the 9,430 flights without an arrival delay (cancelled or diverted), and adds the target `late` (1 when `arr_delay` is above 15 minutes, otherwise 0). It drops `dep_time`, `dep_delay`, `arr_time`, `arr_delay` and `air_time`, which are known only after departure. The prepared file has 327,346 rows and 15 columns, and 23.71% of the flights are late. `time_hour` is the scheduled hour as an ISO 8601 UTC timestamp, for example `2013-01-01T10:00:00Z`.

## Run it

From the repository root, with the root virtual environment:

```
.venv/Scripts/python apps/flight-delays/fetch_data.py
.venv/Scripts/hexperiment run apps/flight-delays
```

`fetch_data.py` downloads `flights.csv` into `apps/flight-delays/data/` unless the file is already there with the pinned checksum, then writes `data/prepared.csv`. `hexperiment run` executes `train.py`, runs the canary test, and writes `verdict.json` and a model card. With the honest `train.py` the expected verdict is `APPROVED` with exit code 0.

The target is `late`. The metric is ROC AUC with a threshold of 0.60. `hex.toml` sets `time_column = "time_hour"`, so the latest 20% of the flights form the test set: 261,877 training rows and 65,469 test rows.

**Run time.** One full run of the honest script (`hexperiment run`, which runs `train.py` normally and once more with the test rows scrambled, so two executions) took 21.1 s on 2026-11-12. A second run took 22.3 s and gave the same verdict and AUC. The machine was an AMD Ryzen 7 3750H laptop (4 cores, 8 threads) on Windows 11 with the Balanced power plan, CPU only, with pandas 3.1.1, in everyday use: the Kiro IDE (with both hexperiment MCP servers running), Chrome and VS Code were open. The limit in `hex.toml` is `timeout_seconds = 120`. The run with the split fixed and the leak kept needs a third execution (the control run, started when the canary finds a difference) and took 31.1 s.

## The haunted script

`haunted/train.py` is the honest script plus the leaks marked `# LEAK:`. When it is run, `hexperiment run` rejects it, and the two leaks show up one after the other.

| Leak | Where | Ghost | How it is caught |
|---|---|---|---|
| A random split with scikit-learn's `train_test_split` instead of `hexperiment.split`. Flights from after the earliest test flight are in the training set. | the line ending in `# LEAK: random split` | The Prophet | The check compares the times of the training and test rows. The error names `time_hour`. The canary test reports `not applied`, because the scrambled test rows are mostly inside the random training set. |
| `carrier_day_late_rate`, the late rate of each carrier on each day, computed over all rows before the split. The test labels reach the training features. | `main()`, on the lines marked `# LEAK: The Peeking Spirit` | The Peeking Spirit | It appears once the split is fixed. For days wholly inside the test period, each flight's own label sits in its own `carrier_day_late_rate`, and the canary cannot see that. The canary catches the leak through the cutoff day, where flights of the same carrier fall on both sides of the split: scrambling the test rows changes the rate of those carrier-days, so the training predictions change. The finding lists the `transform` line among the suspect lines. |

The haunted run also reports two warnings: `split_order` (a statistic computed over all rows before the split, at the `transform` line) and `metric` (the AUC is above 0.75, too good to be true).

The haunted script splits with `train_test_split` and its default test share of 25%. The honest script's test set is the latest 20% by time (`hex.toml`).

## The honest script

`train.py` differs from the haunted script in these ways:

- It splits with `hexperiment.split`, so no training flight is later than any test flight. Flights in the same scheduled hour at the cutoff are divided by row id.
- It has no `carrier_day_late_rate`.
- The features are known when the flight is scheduled: `carrier`, `origin`, `dest`, `month`, `day`, `sched_dep_time`, `sched_arr_time` and `distance`.
- `flight` and `tailnum` are identifiers and are left out. `hour` and `minute` are already inside `sched_dep_time`.
- `month` and `day` stay on purpose. A scheduled flight repeats every day with the same other columns, so without a date identical rows would sit in both train and test and The Doppelganger would reject the model. The script does not combine `year`, `month` and `day` into one date.
- Every learned step sits in one scikit-learn `Pipeline` fitted on training rows only. The model is a `HistGradientBoostingClassifier` with `random_state=0` and no early stopping, so two runs give the same scores.
- After the split, `drop_seen_test_rows` removes every test row that also occurs in the training rows. A test row counts as occurring in the training rows when a training row has the same value in every feature and the same `late`. Repeated rows inside the test set stay. On the real data it removes 0 of the 65,469 test rows. It stays in the script as a guard.

## Results

Measured on 2026-11-12 on an AMD Ryzen 7 3750H laptop (4 cores, 8 threads), Windows 11, Balanced power plan, with the Kiro IDE, Chrome and VS Code open, in full mode (`hexperiment run` through `apps/verify.py` in a scratch copy of the app). These runs are not in `runs/`; the recorded history is below.

| Script | Verdict | AUC | Threshold | Findings | Executions | Wall time |
|---|---|---|---|---|---|---|
| `haunted/train.py` | `REJECTED` (exit 1) | 0.7971 | 0.60 | The Prophet on `time_hour`. Warnings: `split_order`, `canary` (not applied), `metric` | 2 | 24.4 s |
| Haunted with only the split fixed | `REJECTED` (exit 1) | 0.7605 | 0.60 | The Peeking Spirit. No The Prophet. Warnings: `split_order`, `metric` | 3 | 31.1 s |
| `train.py` (honest) | `APPROVED` (exit 0) | 0.6193 | 0.60 | none | 2 | 21.1 s |

The second row is not a file in the repository. `apps/verify.py` builds it from the haunted script by replacing the line ending in `# LEAK: random split` with `train, test = hexperiment.split(df)`. It is the state an agent reaches after fixing the first finding.

The haunted AUC is 0.1778 higher than the honest one. The honest model is only 0.0193 above the threshold. That is what the schedule alone can tell you about delays.

## Recorded history

The committed `runs/` folder holds the runs of one `exorcise` workflow on the haunted script (2026-11-22), followed by one run of the honest script. The ghost hunter's session is in [docs/kiro/exorcise-flight-delays.md](../../docs/kiro/exorcise-flight-delays.md).

| Run | Started by | Mode | Verdict | AUC | Error findings |
|---|---|---|---|---|---|
| `20261122-131518-e8db31` | the ghost hunter, on the haunted script | full | `REJECTED` | 0.7971 | The Prophet |
| `20261122-131620-2ed225` | the save hook, after the ghost hunter's edit | fast | `UNVERIFIED` | 0.6193 | none |
| `20261122-131642-1933a1` | the ghost hunter, on its fix | full | `APPROVED` | 0.6193 | none |
| `20261122-131840-3eec70` | a manual run, on the honest script restored | full | `APPROVED` | 0.6193 | none |

The ghost hunter fixed the split and removed `carrier_day_late_rate` in one round, so the recorded history never shows The Peeking Spirit; the "split fixed" row of the results table shows that stage. The honest script was put back and run last, and that run is the current `verdict.json`.

## Replay the demo

1. Copy the haunted script over the honest one.

   Git Bash:

   ```
   cp apps/flight-delays/haunted/train.py apps/flight-delays/train.py
   ```

   PowerShell:

   ```
   Copy-Item apps/flight-delays/haunted/train.py apps/flight-delays/train.py
   ```

2. Run the check. This replaces any stale verdict and shows `REJECTED`:

   ```
   .venv/Scripts/hexperiment run apps/flight-delays
   ```

3. In the Kiro CLI, start the workflow and let the ghost hunter fix the script until the full audit gives `APPROVED`:

   ```
   /workflow run exorcise --app apps/flight-delays
   ```

4. To return to the packaged honest script, run:

   ```
   git checkout -- apps/flight-delays/train.py
   ```

Every run adds a folder to `runs/`, so replaying adds to the history. The first verdict is `REJECTED` by The Prophet. If the agent fixes only the split first, the next verdict shows The Peeking Spirit; in the recorded run it fixed both leaks at once. Do not lower the threshold in `hex.toml` and do not edit `verdict.json`.

## Limits

- One year (2013) and three New York airports. The model says nothing about other years or airports.
- The model sees only the schedule. It does not see weather, aircraft or the delay of the previous flight, so its AUC is modest.
- The test set is the latest 20% of the flights by scheduled hour, so it covers only the latest part of the year, and the AUC may differ for other periods.
- The AUC is measured on the 65,469 test rows, the latest 20% of the flights; `drop_seen_test_rows` removed none of them.

## Files

| File | Purpose |
|---|---|
| `hex.toml` | target, metric, threshold, time split and timeout |
| `fetch_data.py` | downloads and checks `flights.csv`, writes `data/prepared.csv` |
| `train.py` | the honest training script |
| `haunted/train.py` | the haunted starting point of the demo |
| `data/` | downloaded data (not committed) |
| `runs/` | one folder per run with `verdict.json` and `model_card.md` (created by real runs) |
| `verdict.json` | the latest verdict (created by real runs) |
