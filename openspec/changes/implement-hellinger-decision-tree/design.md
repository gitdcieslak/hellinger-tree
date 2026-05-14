# Design: HellingerDecisionTreeClassifier

## Overview

`HellingerDecisionTreeClassifier` will be implemented as a small recursive binary tree learner.

The implementation prioritizes readability, determinism, and testability over speed. It is intended for research benchmarking, not as a drop-in optimized replacement for scikit-learn's internal tree implementation.

## Public API

```python
class HellingerDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        laplace=True,
        positive_class=None,
    ):
        ...
Internal Tree Representation

Use a lightweight internal node structure.

@dataclass
class _Node:
    is_leaf: bool
    depth: int
    n_samples: int
    class_counts: np.ndarray
    proba: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
Class Handling

The first implementation supports binary classification only.

self.classes_ will be inferred from y.

If positive_class is provided, it identifies the positive class for Hellinger scoring.

If positive_class is not provided, the positive class defaults to self.classes_[1].

Predicted probability columns must follow self.classes_ order.

Split Search

For each considered feature:

Sort samples by feature value.
Identify adjacent distinct values.
Generate midpoint thresholds.
Reject thresholds violating min_samples_leaf.
Compute Hellinger score.
Select the best split.

Tie-breaking:

Higher Hellinger score.
Lower feature index.
Lower threshold.
Hellinger Criterion

For a candidate split:

H = sqrt(
    (sqrt(pos_left / total_pos) - sqrt(neg_left / total_neg))^2
  + (sqrt(pos_right / total_pos) - sqrt(neg_right / total_neg))^2
)

If the current node contains only one class, no split is attempted.

Leaf Probabilities

When laplace=True:

P(class_k | leaf) = (count_k + 1) / (n_leaf + n_classes)

For binary classification:

P(class_k | leaf) = (count_k + 1) / (n_leaf + 2)

When laplace=False, use empirical class frequencies.

Laplace smoothing affects only leaf probabilities. It does not affect split scoring.

Prediction

predict_proba(X) routes each row to a leaf and returns the leaf probability vector.

predict(X) calls predict_proba(X) and returns the class with maximum probability.

Ties are resolved by choosing self.classes_[0].

Scikit-learn Compatibility

The implementation will inherit from:

BaseEstimator
ClassifierMixin

It will use scikit-learn validation helpers where appropriate:

check_X_y
check_array
check_is_fitted

The implementation will expose:

classes_
n_features_in_
Bagging Compatibility

The estimator must work with:

BaggingClassifier(
    estimator=HellingerDecisionTreeClassifier(),
    n_estimators=100,
    random_state=42,
)

Older base_estimator usage may be documented but does not need first-class support unless required by the local scikit-learn version.

Known Limitations
Numeric features only.
Binary classification only.
No pruning.
No native missing-value handling.
No categorical-native splits.
Performance may be slower than compiled scikit-learn trees.