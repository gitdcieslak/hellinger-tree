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

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Notes

Laplace smoothing affects only `predict_proba`; split selection uses raw class counts in the Hellinger score.
