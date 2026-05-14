# Scikit-learn Estimator Compatibility Specification

## Requirements

### Requirement: Estimator interface

The classifier SHALL follow the scikit-learn estimator interface.

#### Scenario: Constructor has no required positional arguments

When `HellingerDecisionTreeClassifier()` is instantiated
Then construction SHALL succeed without required arguments.

#### Scenario: Fit returns self

Given valid `X` and `y`
When `fit(X, y)` is called
Then the method SHALL return `self`.

#### Scenario: Predict after fit

Given a fitted classifier
When `predict(X)` is called
Then the returned array SHALL have shape `(n_samples,)`.

#### Scenario: Predict proba after fit

Given a fitted classifier
When `predict_proba(X)` is called
Then the returned array SHALL have shape `(n_samples, 2)`.

---

### Requirement: Learned attributes

The classifier SHALL expose standard learned attributes after fitting.

#### Scenario: Classes attribute

Given a fitted classifier
Then `classes_` SHALL exist
And `classes_` SHALL preserve the class ordering used by predictions.

#### Scenario: Feature count attribute

Given a fitted classifier
Then `n_features_in_` SHALL exist
And equal the number of input features.

---

### Requirement: Input validation

The classifier SHALL validate inputs using scikit-learn-compatible validation.

#### Scenario: Invalid feature matrix

Given an invalid feature matrix
When `fit(X, y)` is called
Then the classifier SHALL raise a clear validation error.

#### Scenario: Predict before fit

Given an unfitted classifier
When `predict(X)` or `predict_proba(X)` is called
Then the classifier SHALL raise a fittedness error.

---

### Requirement: Random state determinism

The classifier SHALL behave deterministically under a fixed `random_state`.

#### Scenario: Same random state

Given two classifiers with the same `random_state`
And the same training data
When both are fitted
Then they SHALL produce identical predictions.

---

### Requirement: Max features support

The classifier SHOULD support `max_features`.

#### Scenario: All features by default

Given `max_features=None`
When evaluating splits
Then all features SHALL be considered at each node.

#### Scenario: Integer max features

Given `max_features` is an integer
When evaluating splits
Then at most that many features SHALL be considered at each node.

#### Scenario: Fractional max features

Given `max_features` is a float in `(0, 1]`
When evaluating splits
Then that fraction of features SHALL be considered at each node, rounded to at least one feature.