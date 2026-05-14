# hellinger-tree

## Purpose

`hellinger-tree` provides a small, auditable, scikit-learn-compatible implementation of Hellinger Distance Decision Trees for binary imbalanced classification research.

The package exists to support modern benchmarking of skew-insensitive tree learning, especially comparisons against CART, RandomForest, BalancedRandomForest, XGBoost, LightGBM, and neural tabular models.

## Scope

This project implements:

- Hellinger-distance-based binary decision tree induction
- Numeric feature threshold splitting
- Binary classification
- Laplace-smoothed leaf probabilities
- `fit`, `predict`, and `predict_proba`
- scikit-learn estimator compatibility
- compatibility with `sklearn.ensemble.BaggingClassifier`

This project does not initially implement:

- native categorical splitting
- multiclass Hellinger splitting
- pruning
- missing-value routing
- SMOTE or other sampling methods
- benchmark experiment orchestration

## Design Principles

- Research-first, not production-first.
- Small and inspectable implementation.
- Deterministic behavior under fixed `random_state`.
- Favor correctness and testability over speed.
- Make Bagged HDDT reproducible enough for modern imbalance benchmarking.