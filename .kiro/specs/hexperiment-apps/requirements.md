# Requirements Document: hexperiment-apps

## Introduction

Two example apps show the kit on real public data. Each one starts **haunted**: a training script with the leaks people really make, which looks better than it is. The kit **flags** it, an agent in the `exorcise` workflow fixes it one finding at a time, and the app ends **honest**: a lower but true score with an `APPROVED` verdict. The run history and model card of each app record that arc.

- **hotel-cancellations** predicts whether a hotel booking is cancelled. Its haunted script uses `reservation_status`, a column filled in only after the booking's outcome (The Possession), and a country cancellation rate computed over all rows before the split (The Peeking Spirit).
- **flight-delays** predicts whether a 2013 New York flight arrives more than 15 minutes late. Its haunted script splits at random although the data is ordered in time (The Prophet), and adds a same-day carrier late rate computed over all rows (The Peeking Spirit).

Scope: data fetching, the two apps (haunted and honest scripts, configuration, README), and their tests. The gallery and the leak benchmark are later specs.

## Glossary

- **Haunted script:** `apps/<app>/haunted/train.py`, the starting point of the demo, kept unchanged.
- **Honest script:** `apps/<app>/train.py`, the current script that `hexperiment run` executes.
- **Prepared CSV:** the file at `data.path` in `hex.toml`, written by `fetch_data.py`.

## Requirements

### Requirement 1: Fetching the data

**User Story:** As someone trying the kit, I want one command that downloads and prepares an app's data, so that I can run the app without hunting for files.

#### Acceptance Criteria

1. EACH app SHALL have `fetch_data.py` that downloads its source file from a pinned URL into `apps/<app>/data/`, verifies it against a pinned SHA-256, and writes the prepared CSV that `hex.toml` names.
2. THE pinned sources SHALL be: hotel-cancellations, `https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv` (SHA-256 `7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06`, 16,855,599 bytes); flight-delays, `https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv` (SHA-256 `b791a2590c9167aa0bd3cb45af3f64101180878cf088c183eae4011e3a654f42`, 33,206,996 bytes).
3. IF the prepared CSV already exists and the downloaded source matches its checksum THEN `fetch_data.py` SHALL not download again.
4. IF the download fails or the checksum does not match THEN `fetch_data.py` SHALL exit with a non-zero code and a message naming the URL and the problem, and SHALL leave no partial prepared CSV.
5. THE preparation SHALL be a pure function of the source frame, separate from the download, so it can be tested without network access.
6. FOR flight-delays the preparation SHALL drop flights without an arrival delay (cancelled or diverted), add the target `late` (`1` when `arr_delay > 15`, else `0`), keep `time_hour` as an ISO 8601 UTC timestamp, and drop the columns that are only known after departure (`dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time`).
7. FOR hotel-cancellations the preparation SHALL keep every source column, including `reservation_status` and `reservation_status_date`, so that the haunted script can use them.
8. THE data SHALL never be committed (`apps/*/data/` is ignored by git).

### Requirement 2: hotel-cancellations

**User Story:** As a judge or reader, I want to see a realistic leak in a familiar business problem caught and fixed, so that I understand what the kit does in one example.

#### Acceptance Criteria

1. THE app SHALL have `hex.toml` with target `is_canceled` (positive `1`), metric `roc_auc` with threshold `0.85`, a hash split with `test_fraction = 0.2` and a fixed seed, and a `timeout_seconds` that a full run fits in.
2. THE haunted script SHALL use `reservation_status` as a feature and SHALL add a per-country cancellation rate computed over all rows before `hexperiment.split`, and otherwise follow the ML-integrity rules (load with `hexperiment.load`, split with `hexperiment.split`, fixed seeds, `y_score_train` passed to `audit`).
3. WHEN the haunted script runs under `hexperiment run` THEN the verdict SHALL be `REJECTED` with The Possession on `reservation_status`, The Peeking Spirit, and the "too good to be true" warning.
4. THE honest script SHALL fit every learned step on training rows only, SHALL NOT use `reservation_status`, `reservation_status_date` or `assigned_room_type` (recorded after the outcome), and SHALL reach `APPROVED` under `hexperiment run`.

### Requirement 3: flight-delays

**User Story:** As a judge or reader, I want to see why a random split lies on time-ordered data, so that I trust the lower honest score.

#### Acceptance Criteria

1. THE app SHALL have `hex.toml` with target `late` (positive `1`), metric `roc_auc` with threshold `0.60`, `split.time_column = "time_hour"`, `test_fraction = 0.2`, and a `timeout_seconds` that a full run fits in.
2. THE haunted script SHALL split with scikit-learn's `train_test_split` at random instead of `hexperiment.split`, and SHALL add a per-carrier, per-day late rate computed over all rows before the split.
3. WHEN the haunted script runs under `hexperiment run` THEN the verdict SHALL be `REJECTED` with The Prophet; and WHEN only the split is fixed THEN the verdict SHALL be `REJECTED` with The Peeking Spirit.
4. THE honest script SHALL split with `hexperiment.split`, use only features known when the flight is scheduled, and reach `APPROVED` under `hexperiment run`.

### Requirement 4: README per app

**User Story:** As a reader, I want each app to explain itself, so that I can follow the haunted, flagged and honest steps without reading code first.

#### Acceptance Criteria

1. EACH app SHALL have `README.md` stating the question, the dataset, its source and licence with the required attribution (hotel-cancellations: N. Antonio, A. de Almeida, L. Nunes, "Hotel booking demand datasets", Data in Brief 22 (2019), CC BY 4.0; flight-delays: nycflights13, CC0, original data from the US Bureau of Transportation Statistics), how to fetch the data and run the app, each leak in the haunted script and the ghost that catches it, and the honest score next to the haunted one.
2. EACH README SHALL show how to replay the demo: copy `haunted/train.py` over `train.py`, then run the `exorcise` workflow or fix it by hand.

### Requirement 5: Quality

**User Story:** As a maintainer, I want the apps tested without downloading anything, so that the test suite stays fast and offline.

#### Acceptance Criteria

1. THE tests SHALL live in `apps/tests/`, run with `.venv/Scripts/python -m pytest apps/tests -q`, never use the network, and cover: each preparation function on a small in-memory frame, the checksum check and the no-partial-file rule, and that each `hex.toml` loads with the core configuration loader.
2. A full `hexperiment run` of each honest app SHALL finish within its `timeout_seconds` on the reference laptop, and the measured time SHALL be stated in the app's README.
3. THE committed run history (`runs/`), `verdict.json` and `model_card.md` of each app SHALL come from real runs of the haunted and honest scripts, never be edited by hand, and show the haunted, flagged and honest steps in order.
4. `ruff check apps` SHALL report no problems.
5. WHEN the apps tests run in the same pytest command as the kit tests (`pytest hexperiment/tests apps/tests`), THE apps no-network guard SHALL apply only to tests whose file is inside `apps/tests/`, and every kit test SHALL pass as it does when the kit tests run alone.
