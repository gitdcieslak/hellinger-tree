from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils.validation import check_is_fitted


@dataclass
class _Node:
    counts: np.ndarray
    proba: np.ndarray
    prediction: int
    feature_index: int | None = None
    threshold: float | None = None
    hellinger_score: float | None = None
    pos_left: int | None = None
    neg_left: int | None = None
    pos_right: int | None = None
    neg_right: int | None = None
    n_left: int | None = None
    n_right: int | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.laplace = laplace
        self.positive_class = positive_class

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype="numeric")
        self._validate_hyperparameters(X.shape[1])

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        if self.classes_.shape[0] != 2:
            raise ValueError("HellingerDecisionTreeClassifier supports binary classification only.")

        if self.positive_class is None:
            self.positive_class_ = self.classes_[1]
            self.positive_index_ = 1
        else:
            matches = np.flatnonzero(self.classes_ == self.positive_class)
            if matches.size != 1:
                raise ValueError("positive_class must be one of the two fitted classes.")
            self.positive_class_ = self.positive_class
            self.positive_index_ = int(matches[0])

        self.negative_index_ = 1 - self.positive_index_
        self.n_features_in_ = X.shape[1]
        self._rng = check_random_state(self.random_state)
        self.tree_ = self._grow(X, y_encoded, depth=0)
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X, dtype="numeric")
        self._check_n_features(X)
        leaves = (self._predict_leaf(row) for row in X)
        return self.classes_[np.fromiter((leaf.prediction for leaf in leaves), dtype=int)]

    def predict_proba(self, X):
        check_is_fitted(self, "tree_")
        X = check_array(X, dtype="numeric")
        self._check_n_features(X)
        proba = np.empty((X.shape[0], 2), dtype=float)
        for i, row in enumerate(X):
            proba[i] = self._predict_leaf(row).proba
        return proba

    def get_depth(self):
        check_is_fitted(self, "tree_")
        return self._node_depth(self.tree_)

    def get_n_leaves(self):
        check_is_fitted(self, "tree_")
        return self._node_n_leaves(self.tree_)

    def export_tree(self):
        check_is_fitted(self, "tree_")
        return self._export_node(self.tree_, depth=0)

    def _validate_hyperparameters(self, n_features):
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be None or a non-negative integer.")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        self.max_features_ = self._resolve_max_features(n_features)

    def _resolve_max_features(self, n_features):
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features as an integer must be at least 1.")
            return min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("max_features as a float must be in (0, 1].")
            return max(1, int(np.ceil(self.max_features * n_features)))
        raise ValueError("max_features must be None, an integer, or a float in (0, 1].")

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=2).astype(float)
        node = _Node(
            counts=counts,
            proba=self._leaf_proba(counts),
            prediction=int(np.argmax(counts)),
        )

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, diagnostics = split
        left_mask = X[:, feature_index] <= threshold
        node.feature_index = feature_index
        node.threshold = threshold
        node.hellinger_score = diagnostics["hellinger_score"]
        node.pos_left = diagnostics["pos_left"]
        node.neg_left = diagnostics["neg_left"]
        node.pos_right = diagnostics["pos_right"]
        node.neg_right = diagnostics["neg_right"]
        node.n_left = diagnostics["n_left"]
        node.n_right = diagnostics["n_right"]
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _should_stop(self, y, depth):
        if np.unique(y).size == 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if y.shape[0] < self.min_samples_split:
            return True
        counts = np.bincount(y, minlength=2)
        return counts[self.positive_index_] == 0 or counts[self.negative_index_] == 0

    def _best_split(self, X, y):
        best_score = -np.inf
        best_feature = None
        best_threshold = None
        best_diagnostics = None
        feature_indices = self._candidate_features(X.shape[1])

        for feature_index in feature_indices:
            for threshold, score, diagnostics in self._feature_splits(X[:, feature_index], y):
                if self._is_better_split(score, feature_index, threshold, best_score, best_feature, best_threshold):
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold
                    best_diagnostics = diagnostics

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_diagnostics

    def _candidate_features(self, n_features):
        if self.max_features_ == n_features:
            return np.arange(n_features)
        return np.sort(self._rng.choice(n_features, size=self.max_features_, replace=False))

    def _feature_splits(self, values, y):
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_y = y[order]
        total_pos = np.sum(sorted_y == self.positive_index_)
        total_neg = np.sum(sorted_y == self.negative_index_)
        left_pos_cumsum = np.cumsum(sorted_y == self.positive_index_)
        left_neg_cumsum = np.cumsum(sorted_y == self.negative_index_)
        n_samples = y.shape[0]

        for i in range(n_samples - 1):
            if sorted_values[i] == sorted_values[i + 1]:
                continue
            left_n = i + 1
            right_n = n_samples - left_n
            if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                continue

            pos_left = left_pos_cumsum[i]
            neg_left = left_neg_cumsum[i]
            pos_right = total_pos - pos_left
            neg_right = total_neg - neg_left
            threshold = float((sorted_values[i] + sorted_values[i + 1]) / 2.0)
            score = self._hellinger_score(
                pos_left,
                neg_left,
                pos_right,
                neg_right,
                total_pos,
                total_neg,
            )
            yield threshold, score, {
                "hellinger_score": score,
                "pos_left": int(pos_left),
                "neg_left": int(neg_left),
                "pos_right": int(pos_right),
                "neg_right": int(neg_right),
                "n_left": int(left_n),
                "n_right": int(right_n),
            }

    def _hellinger_score(self, pos_left, neg_left, pos_right, neg_right, total_pos, total_neg):
        return float(
            np.sqrt(
                (np.sqrt(pos_left / total_pos) - np.sqrt(neg_left / total_neg)) ** 2
                + (np.sqrt(pos_right / total_pos) - np.sqrt(neg_right / total_neg)) ** 2
            )
        )

    def _is_better_split(self, score, feature_index, threshold, best_score, best_feature, best_threshold):
        if score > best_score:
            return True
        if not np.isclose(score, best_score, rtol=0.0, atol=1e-12):
            return False
        if best_feature is None or feature_index < best_feature:
            return True
        return feature_index == best_feature and threshold < best_threshold

    def _leaf_proba(self, counts):
        if self.laplace:
            return (counts + 1.0) / (np.sum(counts) + 2.0)
        return counts / np.sum(counts)

    def _predict_leaf(self, x):
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def _check_n_features(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but HellingerDecisionTreeClassifier "
                f"is expecting {self.n_features_in_} features as input."
            )

    def _node_depth(self, node):
        if node.is_leaf:
            return 0
        return 1 + max(self._node_depth(node.left), self._node_depth(node.right))

    def _node_n_leaves(self, node):
        if node.is_leaf:
            return 1
        return self._node_n_leaves(node.left) + self._node_n_leaves(node.right)

    def _export_node(self, node, depth):
        exported = {
            "is_leaf": node.is_leaf,
            "depth": depth,
            "n_samples": int(np.sum(node.counts)),
            "class_counts": node.counts.astype(int).tolist(),
            "proba": node.proba.tolist(),
        }
        if node.is_leaf:
            return exported

        exported.update(
            {
                "feature_index": int(node.feature_index),
                "threshold": float(node.threshold),
                "hellinger_score": float(node.hellinger_score),
                "pos_left": int(node.pos_left),
                "neg_left": int(node.neg_left),
                "pos_right": int(node.pos_right),
                "neg_right": int(node.neg_right),
                "n_left": int(node.n_left),
                "n_right": int(node.n_right),
                "left": self._export_node(node.left, depth + 1),
                "right": self._export_node(node.right, depth + 1),
            }
        )
        return exported
