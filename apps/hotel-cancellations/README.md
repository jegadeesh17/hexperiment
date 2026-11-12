# hotel-cancellations

Will a hotel booking be cancelled? The model reads what is known about a booking (lead time, deposit type, country, price per night and so on) and scores the chance of cancellation.

This app starts **haunted**: its training script has two realistic leaks and reports an AUC of 1.0000. Hexperiment flags it as `REJECTED`. The honest script fixes the leaks and reaches an AUC of 0.9096 with the verdict `APPROVED`.

## The data

- **Dataset:** hotel booking demand, two Portuguese hotels (a city hotel and a resort hotel), 119,390 bookings and 32 columns. 37.04% of the bookings are cancelled.
- **Source:** <https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv> (16,855,599 bytes), redistributed through TidyTuesday. `fetch_data.py` checks the file against a pinned SHA-256 (`7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`).
- **Licence and attribution:** N. Antonio, A. de Almeida, L. Nunes, "Hotel booking demand datasets", Data in Brief 22 (2019), CC BY 4.0.
- The data is downloaded, not committed (`apps/*/data/` is ignored by git).

## Run it

From the repository root, with the root virtual environment:

```
.venv/Scripts/python apps/hotel-cancellations/fetch_data.py
.venv/Scripts/hexperiment run apps/hotel-cancellations
```

`fetch_data.py` downloads `hotels.csv` into `apps/hotel-cancellations/data/` unless the file is already there with the pinned checksum, then writes `data/prepared.csv` (all 32 columns, unchanged). `hexperiment run` executes `train.py`, runs the canary test, and writes `verdict.json` and a model card. With the honest `train.py` the expected verdict is `APPROVED` with exit code 0.

The target is `is_canceled`. The metric is ROC AUC with a threshold of 0.85. The split is a 20% test set chosen by a hash of the row, with a fixed seed (`hex.toml`).

**Run time.** One full run of the honest script (`hexperiment run`, which runs `train.py` normally and once more with the test rows scrambled, so two executions) took 22.2 s on 2026-11-12. A second run took 20.7 s and gave the same verdict and AUC. The machine was an AMD Ryzen 7 3750H laptop (4 cores, 8 threads) on Windows 11 with the Balanced power plan, CPU only, with pandas 3.1.1, in everyday use: the Kiro IDE (with both hexperiment MCP servers running), Chrome and VS Code were open. The limit in `hex.toml` is `timeout_seconds = 120`. The haunted script needs a third execution (the control run, started when the canary finds a difference) and took 26.8 s.

## The haunted script

`haunted/train.py` is the honest script plus two leaks, each marked with a `# LEAK:` comment. When it is run, `hexperiment run` rejects it.

| Leak | Where | Ghost | How it is caught |
|---|---|---|---|
| `reservation_status` used as a feature. It is filled in when the booking ends (cancelled, checked out, no-show), so it contains the answer. | the `CATEGORICAL` list, on the line marked `# LEAK: The Possession` | The Possession | This one column alone predicts `is_canceled` almost perfectly. The finding names `reservation_status`. |
| `country_cancel_rate`, the cancellation rate of each country, computed over all rows before the split. The test labels reach the training features. | `main()`, on the lines marked `# LEAK: The Peeking Spirit` | The Peeking Spirit | The canary run scrambles the test rows. The training predictions then change, which they cannot do if training never saw test rows. The finding lists the `transform` line among the suspect lines. |

The same run also reports two warnings: `split_order` (a statistic computed over all rows before the split, at the `transform` line) and `metric` (the AUC is too good to be true).

Other columns that are recorded after the outcome, such as `reservation_status_date` and `assigned_room_type`, are not planted: no check can name them (each alone scores about 0.6), so a loop that stops at `APPROVED` could leave them in. The honest script leaves them out anyway.

## The honest script

`train.py` differs from the haunted script in these ways:

- It does not use `reservation_status`, `reservation_status_date` or `assigned_room_type`. The first two are filled in when the booking ends. `assigned_room_type` is assigned at check-in or after a change. `company` is also left out because 94% of its values are missing.
- It has no `country_cancel_rate`.
- Every learned step (the category encoder and the model) sits in one scikit-learn `Pipeline` that is fitted on training rows only.
- The model is a `HistGradientBoostingClassifier` with `random_state=0` and no early stopping, so two runs give the same scores.
- After the split, `drop_seen_test_rows` removes every test row that also occurs in the training rows. A test row counts as occurring in the training rows when a training row has the same value in every feature and the same `is_canceled`. Repeated rows inside the test set stay, and so do repeated rows inside the training set. Nothing is deduplicated before the split, because that would change the training set in the canary run.

**Why the honest AUC is measured on 16,023 rows.** The split gives 95,457 training rows and 23,933 test rows. 7,910 of the test rows (33%) also occur in the training rows, because the file contains many repeated bookings. `drop_seen_test_rows` drops them, which leaves 16,023 test rows. A model that has seen a booking in training would otherwise be scored on it again.

## Results

Measured on 2026-11-12 on an AMD Ryzen 7 3750H laptop (4 cores, 8 threads), Windows 11, Balanced power plan, with the Kiro IDE, Chrome and VS Code open, in full mode (`hexperiment run` through `apps/verify.py` in a scratch copy of the app).

| Script | Verdict | AUC | Threshold | Findings | Executions | Wall time |
|---|---|---|---|---|---|---|
| `haunted/train.py` | `REJECTED` (exit 1) | 1.0000 | 0.85 | The Possession on `reservation_status`; The Peeking Spirit. Warnings: `split_order`, `metric` | 3 | 26.8 s |
| Haunted with only `reservation_status` removed | `REJECTED` (exit 1) | 0.9101 | 0.85 | The Peeking Spirit. No The Possession. Warning: `split_order` | 3 | 34.1 s |
| `train.py` (honest) | `APPROVED` (exit 0) | 0.9096 | 0.85 | none | 2 | 22.2 s |

The second row is not a file in the repository. `apps/verify.py` builds it from the haunted script by removing the list item marked `# LEAK: The Possession`. It is the state an agent reaches after fixing the first finding.

The haunted AUC is 0.0904 higher than the honest one, and it is not a better model: it reads the answer. The `country_cancel_rate` leak adds only 0.0005 to the AUC (0.9101 against 0.9096), but the canary still catches it, because it looks at what training depends on, not at the metric.

## Replay the demo

1. Copy the haunted script over the honest one.

   Git Bash:

   ```
   cp apps/hotel-cancellations/haunted/train.py apps/hotel-cancellations/train.py
   ```

   PowerShell:

   ```
   Copy-Item apps/hotel-cancellations/haunted/train.py apps/hotel-cancellations/train.py
   ```

2. Run the check. This replaces any stale verdict and shows `REJECTED`:

   ```
   .venv/Scripts/hexperiment run apps/hotel-cancellations
   ```

3. In the Kiro CLI, start the workflow and let the ghost hunter fix the script until `hexperiment run` gives `APPROVED`:

   ```
   /workflow run exorcise --app apps/hotel-cancellations
   ```

4. To return to the packaged honest script, run:

   ```
   git checkout -- apps/hotel-cancellations/train.py
   ```

Every run adds a folder to `runs/`, so replaying adds to the history. Do not lower the threshold in `hex.toml` and do not edit `verdict.json`.

## Limits

- Two hotels in Portugal, bookings from 2015 to 2017. The model says nothing about other hotels or countries.
- The split is random, not by time.
- `booking_changes`, `adr`, `required_car_parking_spaces` and `total_of_special_requests` can be updated after the booking is made. The honest script keeps them, and no check can tell whether they were final at booking time, so they are a possible source of optimism in the 0.9096.
- The AUC is measured on the 16,023 test rows that remain after removing the rows that also occur in the training rows.

## Files

| File | Purpose |
|---|---|
| `hex.toml` | target, metric, threshold, split and timeout |
| `fetch_data.py` | downloads and checks `hotels.csv`, writes `data/prepared.csv` |
| `train.py` | the honest training script |
| `haunted/train.py` | the haunted starting point of the demo |
| `data/` | downloaded data (not committed) |
| `runs/` | one folder per run with `verdict.json` and `model_card.md` (created by real runs) |
| `verdict.json` | the latest verdict (created by real runs) |
