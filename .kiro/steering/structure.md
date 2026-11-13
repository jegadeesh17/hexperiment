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
    ghosts.py              the ghost catalogue (names, meanings, fix hints)
    canary.py              snapshots and the normal / canary / control comparison
    verdict.py             combines findings + metric into APPROVED / REJECTED / UNVERIFIED
    card.py                writes the model card (Markdown)
    cli.py                 `hexperiment run`, `hook`, `init` and `gallery`
    console.py             print_text(), the only caller of print
    hook_event.py          reads Kiro hook events (pure)
    hook.py                the save hook and the stop check
    workspace.py           finds apps in a workspace
    init.py                `hexperiment init`: a new app, or the Kiro files of a project
    templates/             the files that `init` writes (app and .kiro/)
    mcp_server/            the MCP server (`hexperiment-mcp`); only server.py imports the SDK
    gallery/               the static gallery site (`hexperiment gallery`)
  tests/
apps/                      two applications built on the skeleton
  hotel-cancellations/     hex.toml, fetch_data.py, train.py, haunted/train.py, runs/
  flight-delays/
power/                     the Kiro power (Agent Plugins format: plugin.json, mcp.json, skills/)
docs/                      the documentation index and the Kiro notes (docs/kiro/)
.github/workflows/         builds the gallery and deploys it to GitHub Pages
```

- Each app is self-contained: its own `hex.toml`, data fetch script, training script and `runs/` history. Raw data is downloaded by `fetch_data.py`, not committed.
- An app's `train.py` calls `hexperiment.load()`, `hexperiment.split(df)`, trains, predicts, and ends with one call to `hexperiment.audit(...)`. Nothing else in the app knows how checks work.
- `<app>/.hex/` holds per-execution snapshots for the canary comparison (not committed).
- `runs/<run_id>/` holds `verdict.json` and `model_card.md` for every run (committed; they are the evidence). The latest verdict is also written to `<app_dir>/verdict.json`.
- Tests mirror the package: `tests/test_<module>.py`. Each check has example-based tests and at least one property-based test.
