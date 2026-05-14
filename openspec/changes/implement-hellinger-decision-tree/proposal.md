# Implement Hellinger Decision Tree

## Summary

Implement a scikit-learn-compatible `HellingerDecisionTreeClassifier` for binary imbalanced classification research.

The classifier will use Hellinger distance as its split criterion and will expose Laplace-smoothed leaf probabilities through `predict_proba`.

## Motivation

The original Hellinger Distance Decision Tree work showed that Hellinger-based splitting is robust under class imbalance and that bagged HDDTs are a practical alternative to sampling-heavy approaches.

Modern imbalance benchmarking requires a clean, auditable implementation that can be used alongside scikit-learn, XGBoost, LightGBM, and neural tabular baselines.

Scikit-learn does not provide Hellinger distance as a tree split criterion and does not natively expose Laplace-smoothed leaf probabilities for decision trees.

## Scope

This change adds:

- `HellingerDecisionTreeClassifier`
- binary classification support
- numeric feature support
- recursive binary tree induction
- Hellinger split selection
- Laplace-smoothed leaf probabilities
- `fit`, `predict`, and `predict_proba`
- scikit-learn estimator compatibility
- BaggingClassifier compatibility
- unit tests

## Out of Scope

This change does not add:

- native categorical split handling
- multiclass Hellinger splitting
- pruning
- missing value routing
- SMOTE or sampling methods
- benchmark orchestration
- paper result generation

## Impacted Specs

- `hellinger-decision-tree`
- `laplace-leaf-probabilities`
- `sklearn-estimator-compatibility`
- `bagging-compatibility`