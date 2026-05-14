# Bagging Compatibility Specification

## Requirements

### Requirement: Compatible with sklearn BaggingClassifier

The classifier SHALL work as a base estimator inside `sklearn.ensemble.BaggingClassifier`.

#### Scenario: Bagging fit

Given a `BaggingClassifier` using `HellingerDecisionTreeClassifier` as its estimator
When `fit(X, y)` is called
Then fitting SHALL complete successfully.

#### Scenario: Bagging predict

Given a fitted bagged HDDT classifier
When `predict(X)` is called
Then predictions SHALL be returned with shape `(n_samples,)`.

#### Scenario: Bagging predict_proba

Given a fitted bagged HDDT classifier
When `predict_proba(X)` is called
Then probabilities SHALL be returned with shape `(n_samples, 2)`.

---

### Requirement: Modern and older sklearn support

The implementation SHOULD support the modern `estimator=` argument.

The implementation MAY document fallback behavior for older scikit-learn versions that used `base_estimator=`.

#### Scenario: Modern sklearn

Given a supported modern scikit-learn version
When constructing:

```python
BaggingClassifier(
    estimator=HellingerDecisionTreeClassifier(),
    n_estimators=3,
    random_state=42,
)

Then construction and fitting SHALL succeed.