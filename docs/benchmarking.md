# Benchmarking Handoff Notes

`hellinger-tree` currently provides the estimator and lightweight validation helpers needed by a downstream benchmark suite. It does not include benchmark orchestration, result storage, plotting, or dataset download code.

## Stable Entry Points

- `hellinger_tree.HellingerDecisionTreeClassifier`
- `hellinger_tree.make_gaussian_skew_dataset`
- `HellingerDecisionTreeClassifier.predict`
- `HellingerDecisionTreeClassifier.predict_proba`
- `HellingerDecisionTreeClassifier.export_tree`
- `HellingerDecisionTreeClassifier.get_depth`
- `HellingerDecisionTreeClassifier.get_n_leaves`

## Suggested Benchmark Comparisons

- Single HDDT against `sklearn.tree.DecisionTreeClassifier`.
- Bagged HDDT against `sklearn.ensemble.BaggingClassifier` using CART.
- Synthetic skew sweeps using `make_gaussian_skew_dataset` for smoke validation before external datasets.
- Metrics appropriate for imbalance, including AUROC, average precision, balanced accuracy, recall at fixed precision, and calibration summaries.

## Reproducibility

- Set `random_state` on `HellingerDecisionTreeClassifier` when `max_features` is not `None`.
- Set `random_state` on `BaggingClassifier` and the base estimator.
- Persist package versions from `hellinger_tree.__version__`, NumPy, and scikit-learn with benchmark outputs.
- Store `export_tree()` for representative runs when auditing split behavior.

## Scope Boundaries

The estimator intentionally does not include native categorical splitting, missing-value routing, pruning, sampling methods, or multiclass behavior. Benchmark code should handle preprocessing explicitly before fitting HDDT.
