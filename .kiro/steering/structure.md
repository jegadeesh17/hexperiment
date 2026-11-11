# Structure

```
.kiro/                     Kiro steering, specs, hooks (committed; never gitignored)
hexperiment/               the skeleton: the reusable kit
  pyproject.toml
  src/hexperiment/
    __init__.py            public API: load, split, audit, load_history, Verdict, Finding, HexInputError
    errors.py              HexInputError
    models.py              shared dataclasses and enums (Finding, Verdict, outcomes)
    config.py              loads and validates hex.toml
    auditor.py             the in-script audit() (validation, data checks, snapshot)
    runs.py                run ids, run folders, history
    data.py                load() and split(), including canary scrambling
    checks/                one module per check, each a pure function returning findings
    canary.py              snapshots and the normal / canary / control comparison
    verdict.py             combines findings + metric into APPROVED / REJECTED
    card.py                writes the model card (Markdown)
    cli.py                 `hexperiment run <app_dir>`
    console.py             print_text(), the only caller of print
  tests/
apps/                      two applications built on the skeleton
  <app-one>/               hex.toml, fetch_data.py, train.py, runs/
  <app-two>/
power/                     the Kiro power packaging (POWER.md, steering, hooks)
```

- Each app is self-contained: its own `hex.toml`, data fetch script, training script and `runs/` history. Raw data is downloaded by `fetch_data.py`, not committed.
- An app's `train.py` calls `hexperiment.load()`, `hexperiment.split(df)`, trains, predicts, and ends with one call to `hexperiment.audit(...)`. Nothing else in the app knows how checks work.
- `<app>/.hex/` holds per-execution snapshots for the canary comparison (not committed).
- `runs/<run_id>/` holds `verdict.json` and `model_card.md` for every run (committed; they are the evidence). The latest verdict is also written to `<app_dir>/verdict.json`.
- Tests mirror the package: `tests/test_<module>.py`. Each check has example-based tests and at least one property-based test.
