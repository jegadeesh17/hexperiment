---
inclusion: fileMatch
fileMatchPattern: "apps/**/*.py"
---

# ML integrity rules (model info sheet)

These rules apply to every training script in `apps/`. They exist because an agent optimizing a metric will find a leak if one is available. A result that breaks a rule is not a result.

0. **Let the kit own the data.** Load with `hexperiment.load()` and split with `hexperiment.split(df)`. Fix every random seed (`random_state=...`) so runs are deterministic; the canary test depends on it.
1. **Split first.** Call the train/test split before any `fit`, `fit_transform` or resampling. Fit every scaler, encoder, imputer and selector on training rows only, preferably inside a scikit-learn `Pipeline`.
2. **No row in both worlds.** Drop exact duplicate rows before splitting. A row must never appear in both train and test.
3. **No features from the future.** Use only columns that are known at prediction time. If a column is recorded after the outcome (a discharge code, a refund flag, a "days until churn"), drop it and say why in a comment.
4. **Respect time.** If `hex.toml` sets `time_column`, split by time: every training row must be earlier than every test row. Never shuffle across time.
5. **Suspicion is a signal.** If a single feature predicts the target almost perfectly, or the metric is far above the threshold, treat it as a leak until proven otherwise.
6. **Always end with an audit.** Every training script ends with `hexperiment.audit(...)`. Never edit `verdict.json` or a model card by hand, and never lower a threshold in `hex.toml` to get a pass.
7. **Fix the cause, not the check.** When the audit reports a ghost, change the data handling that caused it. Do not silence, skip or weaken a check.
