# hellinger-tree

`hellinger-tree` is a small, auditable, scikit-learn-compatible implementation of a Hellinger decision tree classifier for binary imbalanced classification research.

## Scope

Included in this initial implementation:

- Binary classification only.
- Numeric feature thresholds only.
- Recursive binary tree induction without pruning.
- Hellinger-distance split scoring.
- Deterministic tie-breaking by lower feature index, then lower threshold.
- Laplace-smoothed leaf probabilities by default.
- Compatibility with `sklearn.ensemble.BaggingClassifier`.

Not included yet:

- Native categorical splitting.
- Missing-value routing.
- Pruning.
- Benchmarking code.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Single HDDT

```python
import numpy as np
from hellinger_tree import HellingerDecisionTreeClassifier

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([0, 0, 1, 1])

clf = HellingerDecisionTreeClassifier(random_state=0)
clf.fit(X, y)

print(clf.predict([[1.5]]))
print(clf.predict_proba([[1.5]]))
```

### BaggingClassifier With HDDT

```python
import numpy as np
from sklearn.ensemble import BaggingClassifier
from hellinger_tree import HellingerDecisionTreeClassifier

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([0, 0, 1, 1, 1, 1])

bagged = BaggingClassifier(
    estimator=HellingerDecisionTreeClassifier(random_state=0),
    n_estimators=10,
    random_state=42,
)
bagged.fit(X, y)

print(bagged.predict([[2.5]]))
print(bagged.predict_proba([[2.5]]))
```

### Export a Learned Tree

```python
from hellinger_tree import HellingerDecisionTreeClassifier

X = [[0.0], [1.0], [2.0], [3.0]]
y = [0, 0, 1, 1]

clf = HellingerDecisionTreeClassifier(random_state=0).fit(X, y)

print(clf.get_depth())
print(clf.get_n_leaves())
print(clf.export_tree())
```

`export_tree()` returns nested dictionaries with node sample counts, class counts, probabilities, split thresholds, Hellinger scores, and split diagnostics.

### Synthetic Skew Helper

```python
from hellinger_tree import HellingerDecisionTreeClassifier, make_gaussian_skew_dataset

X, y = make_gaussian_skew_dataset(
    n_majority=1000,
    n_minority=100,
    separation=2.5,
    random_state=0,
)

clf = HellingerDecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
print(clf.predict_proba(X[:5]))
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

## API Stability

The package is currently versioned as `0.x` and should be treated as research-alpha software. The constructor parameters and public methods documented in this README are intended to remain stable within the `0.1.x` line, but internal node representation and exported diagnostic fields may evolve before a `1.0` release.

## Notes

Laplace smoothing affects only `predict_proba`; split selection uses raw class counts in the Hellinger score.

Benchmarking orchestration is intentionally out of scope for the estimator package. See `docs/benchmarking.md` for handoff notes for downstream benchmark suites.
