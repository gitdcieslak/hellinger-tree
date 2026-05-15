import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from hellinger_tree import HellingerDecisionTreeClassifier, __version__, make_gaussian_skew_dataset


def test_version_is_exposed():
    assert __version__ == "0.1.0"


def test_basic_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = HellingerDecisionTreeClassifier(random_state=0)
    result = clf.fit(X, y)

    assert result is clf
    assert np.array_equal(clf.classes_, np.array([0, 1]))
    assert clf.n_features_in_ == 1
    assert clf.predict(X).shape == (4,)


def test_predict_proba_shape_and_rows_sum_to_one():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = HellingerDecisionTreeClassifier(random_state=0).fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_laplace_smoothing_on_leaf_counts_zero_and_three():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 1])

    clf = HellingerDecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y)

    assert np.allclose(clf.predict_proba([[2.0]]), [[1 / 5, 4 / 5]])


def test_laplace_false_gives_empirical_probabilities():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 1])

    clf = HellingerDecisionTreeClassifier(max_depth=1, laplace=False, random_state=0).fit(X, y)

    assert np.allclose(clf.predict_proba([[2.0]]), [[0.0, 1.0]])


def test_pure_node_stopping():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 1, 1])

    clf = HellingerDecisionTreeClassifier(random_state=0).fit(X, y)

    assert clf.tree_.right.is_leaf
    assert np.array_equal(clf.tree_.right.counts, np.array([0.0, 3.0]))


def test_binary_only_validation():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])

    clf = HellingerDecisionTreeClassifier()

    with pytest.raises(ValueError, match="binary classification only"):
        clf.fit(X, y)


def test_bagging_classifier_compatibility():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 0, 1, 1, 1, 1])
    bag = BaggingClassifier(
        estimator=HellingerDecisionTreeClassifier(random_state=0),
        n_estimators=3,
        random_state=42,
    )

    bag.fit(X, y)

    assert bag.predict(X).shape == (6,)
    assert bag.predict_proba(X).shape == (6, 2)


def test_tree_introspection_methods():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = HellingerDecisionTreeClassifier(random_state=0).fit(X, y)

    exported = clf.export_tree()

    assert clf.get_depth() == 1
    assert clf.get_n_leaves() == 2
    assert exported["is_leaf"] is False
    assert exported["depth"] == 0
    assert exported["n_samples"] == 4
    assert exported["class_counts"] == [2, 2]
    assert exported["feature_index"] == 0
    assert exported["threshold"] == 1.5
    assert exported["left"]["is_leaf"] is True
    assert exported["right"]["is_leaf"] is True


def test_export_tree_contains_split_diagnostics():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    clf = HellingerDecisionTreeClassifier(random_state=0).fit(X, y)

    exported = clf.export_tree()

    assert exported["hellinger_score"] == pytest.approx(np.sqrt(2.0))
    assert exported["pos_left"] == 0
    assert exported["neg_left"] == 2
    assert exported["pos_right"] == 2
    assert exported["neg_right"] == 0
    assert exported["n_left"] == 2
    assert exported["n_right"] == 2


def test_synthetic_skew_dataset_helper():
    X1, y1 = make_gaussian_skew_dataset(
        n_majority=20,
        n_minority=5,
        separation=3.0,
        random_state=42,
    )
    X2, y2 = make_gaussian_skew_dataset(
        n_majority=20,
        n_minority=5,
        separation=3.0,
        random_state=42,
    )

    assert X1.shape == (25, 2)
    assert y1.shape == (25,)
    assert np.bincount(y1).tolist() == [20, 5]
    assert np.allclose(X1, X2)
    assert np.array_equal(y1, y2)


def test_cart_comparison_smoke_on_synthetic_data():
    for n_majority, n_minority in ((80, 20), (50, 50)):
        X, y = make_gaussian_skew_dataset(
            n_majority=n_majority,
            n_minority=n_minority,
            separation=2.0,
            random_state=7,
        )
        hddt = HellingerDecisionTreeClassifier(max_depth=3, random_state=11).fit(X, y)
        cart = DecisionTreeClassifier(max_depth=3, random_state=11).fit(X, y)

        hddt_proba = hddt.predict_proba(X)[:, 1]
        cart_proba = cart.predict_proba(X)[:, 1]

        for scores in (hddt_proba, cart_proba):
            assert scores.shape == (n_majority + n_minority,)
            assert 0.0 <= roc_auc_score(y, scores) <= 1.0
            assert 0.0 <= average_precision_score(y, scores) <= 1.0

        assert hddt.get_depth() > 0
        assert hddt.get_n_leaves() > 1
        assert "hellinger_score" in hddt.export_tree()


def test_export_tree_is_deterministic_with_fixed_random_state():
    X, y = make_gaussian_skew_dataset(n_majority=40, n_minority=10, random_state=5)

    first = HellingerDecisionTreeClassifier(max_features=1, random_state=123).fit(X, y)
    second = HellingerDecisionTreeClassifier(max_features=1, random_state=123).fit(X, y)

    assert first.export_tree() == second.export_tree()


def test_max_features_behavior_is_deterministic_under_fixed_seed():
    X, y = make_gaussian_skew_dataset(n_majority=50, n_minority=10, random_state=17)

    first = HellingerDecisionTreeClassifier(max_depth=3, max_features=0.5, random_state=99).fit(X, y)
    second = HellingerDecisionTreeClassifier(max_depth=3, max_features=0.5, random_state=99).fit(X, y)

    assert first.max_features_ == 1
    assert first.export_tree() == second.export_tree()
