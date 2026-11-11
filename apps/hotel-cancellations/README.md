# hotel-cancellations

Will a hotel booking be cancelled? The model reads what is known about a booking (lead time, deposit type, country, price per night and so on) and scores the chance of cancellation.

This app starts **haunted**: its training script has two realistic leaks and reports an AUC of 1.0000. Hexperiment flags it as `REJECTED`. The honest script fixes the leaks and reaches an AUC of 0.9092 with the verdict `APPROVED`.

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

**Run time.** One full run of the honest script (`hexperiment run`, which runs `train.py` normally and once more with the test rows scrambled) took 47.6 s on 2026-11-11, on Windows, CPU only, with pandas 3.1.1. Another process was using the CPU during that run, so an idle machine may be faster. The limit in `hex.toml` is `timeout_seconds = 120`. These numbers are from that run on 2026-11-11 and will be replaced by the final measurements.

## The haunted script

`haunted/train.py` is the honest script plus the leaks marked `# LEAK:`. When it is run, `hexperiment run` rejects it.

| Leak | Where | Ghost | How it is caught |
|---|---|---|---|
| `reservation_status` used as a feature. It is filled in when the booking ends (cancelled, checked out, no-show), so it contains the answer. | `CATEGORICAL` list | The Possession | This one column alone predicts `is_canceled` almost perfectly. The finding names `reservation_status`. |
| `country_cancel_rate`, the cancellation rate of each country, computed over all rows before the split. The test labels reach the training features. | `train.py:94` | The Peeking Spirit | The canary run scrambles the test rows. The training predictions then change, which they cannot do if training never saw test rows. The finding lists `train.py:94` among the suspect lines. |
| `assigned_room_type` and `status_day` (the date of the final status, from `reservation_status_date`). Both are recorded after the outcome. | `CATEGORICAL` and `NUMERIC` lists, `main()` | none | The run reports no finding for them. The honest script leaves them out anyway, because they are not known when the cancellation happens. |

The same run also reports two warnings: `split_order` (a statistic computed over all rows before the split, at `train.py:94`) and `metric` (the AUC is too good to be true).

## The honest script

`train.py` differs from the haunted script in these ways:

- It does not use `reservation_status`, `reservation_status_date` or `assigned_room_type`. The first two are filled in when the booking ends. `assigned_room_type` is assigned at check-in or after a change. `company` is also left out because 94% of its values are missing.
- It has no `country_cancel_rate` and no `status_day`.
- Every learned step (the category encoder and the model) sits in one scikit-learn `Pipeline` that is fitted on training rows only.
- The model is a `HistGradientBoostingClassifier` with `random_state=0` and no early stopping, so two runs give the same scores.
- After the split, `drop_seen_test_rows` removes every test row that also occurs in the training rows. Duplicates inside the training rows stay. Nothing is deduplicated before the split, because that would change the training set in the canary run.

**Why the honest AUC is measured on 15,805 rows.** The split gives 95,457 training rows and 23,933 test rows. 8,128 of the test rows (34%) are copies of training rows, because the file contains many repeated bookings. `drop_seen_test_rows` drops them, which leaves 15,805 test rows. A model that has seen a booking in training would otherwise be scored on it again.

## Results

From a run on 2026-11-11 (full mode, `hexperiment run` through `apps/verify.py` in a scratch copy of the app). These numbers will be replaced by the final measurements.

| Script | Verdict | AUC | Threshold | Findings | Wall time |
|---|---|---|---|---|---|
| `haunted/train.py` | `REJECTED` (exit 1) | 1.0000 | 0.85 | The Possession on `reservation_status`; The Peeking Spirit. Warnings: `split_order`, `metric` | 66.2 s |
| `train.py` (honest) | `APPROVED` (exit 0) | 0.9092 | 0.85 | none | 47.6 s |

The haunted AUC is 0.0908 higher than the honest one, and it is not a better model: it reads the answer.

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

2. Run the check:

   ```
   .venv/Scripts/hexperiment run apps/hotel-cancellations
   ```

   The verdict is `REJECTED`.

3. Fix it. Either start the workflow in Kiro:

   ```
   /workflow run exorcise --app apps/hotel-cancellations
   ```

   or fix the script by hand, one finding at a time, and run `.venv/Scripts/hexperiment run apps/hotel-cancellations` again until the verdict is `APPROVED`. Do not lower the threshold in `hex.toml` and do not edit `verdict.json`.

Every run adds a folder to `runs/`, so replaying adds to the history. `git restore apps/hotel-cancellations/train.py` brings back the honest script.

## Limits

- Two hotels in Portugal, bookings from 2015 to 2017. The model says nothing about other hotels or countries.
- The split is random, not by time.
- `booking_changes`, `adr`, `required_car_parking_spaces` and `total_of_special_requests` can be updated after the booking is made. The honest script keeps them, and no check can tell whether they were final at booking time, so they are a possible source of optimism in the 0.9092.
- The AUC is measured on the 15,805 test rows that remain after removing copies of training rows.

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
