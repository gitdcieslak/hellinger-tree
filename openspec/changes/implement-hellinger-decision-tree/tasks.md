# Tasks

## 1. Package skeleton

- [ ] Create `src/hellinger_tree/`
- [ ] Add `src/hellinger_tree/__init__.py`
- [ ] Add `src/hellinger_tree/estimator.py`
- [ ] Add `src/hellinger_tree/tree.py` if separating node logic
- [ ] Add `src/hellinger_tree/criteria.py` if separating Hellinger scoring
- [ ] Configure `pyproject.toml`
- [ ] Add test dependencies

## 2. Implement core estimator

- [ ] Add `HellingerDecisionTreeClassifier`
- [ ] Inherit from `BaseEstimator` and `ClassifierMixin`
- [ ] Add constructor parameters:
  - [ ] `max_depth`
  - [ ] `min_samples_split`
  - [ ] `min_samples_leaf`
  - [ ] `max_features`
  - [ ] `random_state`
  - [ ] `laplace`
  - [ ] `positive_class`
- [ ] Add `fit`
- [ ] Add `predict`
- [ ] Add `predict_proba`

## 3. Implement tree induction

- [ ] Add internal `_Node` representation
- [ ] Implement recursive node growth
- [ ] Implement pure-node stopping
- [ ] Implement `max_depth` stopping
- [ ] Implement `min_samples_split` stopping
- [ ] Implement `min_samples_leaf` validation
- [ ] Implement no-valid-split fallback to leaf

## 4. Implement Hellinger split selection

- [ ] Implement numeric threshold generation
- [ ] Implement binary Hellinger score
- [ ] Skip invalid splits
- [ ] Select maximum-score split
- [ ] Implement deterministic tie-breaking:
  - [ ] higher score wins
  - [ ] lower feature index wins
  - [ ] lower threshold wins

## 5. Implement leaf probabilities

- [ ] Implement Laplace-smoothed probabilities
- [ ] Implement empirical probabilities when `laplace=False`
- [ ] Ensure probability columns follow `classes_`
- [ ] Ensure probability rows sum to 1

## 6. Implement sklearn compatibility

- [ ] Add input validation with `check_X_y`
- [ ] Add prediction validation with `check_array`
- [ ] Add fittedness checks
- [ ] Set `classes_`
- [ ] Set `n_features_in_`
- [ ] Ensure `fit` returns `self`
- [ ] Ensure deterministic behavior with `random_state`

## 7. Implement max_features behavior

- [ ] Support `max_features=None`
- [ ] Support integer `max_features`
- [ ] Support float `max_features`
- [ ] Ensure at least one feature is considered
- [ ] Use `random_state` for feature subsampling

## 8. Unit tests

- [ ] Test basic `fit`
- [ ] Test `predict` shape
- [ ] Test `predict_proba` shape
- [ ] Test probability row sums
- [ ] Test Laplace smoothing for counts `[0, 3]`
- [ ] Test empirical probabilities when `laplace=False`
- [ ] Test pure-node stopping
- [ ] Test `min_samples_leaf`
- [ ] Test deterministic tie-breaking
- [ ] Test binary-only validation
- [ ] Test BaggingClassifier compatibility

## 9. Documentation

- [ ] Add README usage example
- [ ] Document binary/numeric-only limitations
- [ ] Document Laplace smoothing behavior
- [ ] Document BaggingClassifier usage
- [ ] Add short note explaining that Laplace smoothing affects probabilities only

## 10. Acceptance

- [ ] `pytest` passes
- [ ] Package imports successfully
- [ ] Estimator fits numeric binary data
- [ ] `predict` works
- [ ] `predict_proba` works
- [ ] Bagged HDDT works with `BaggingClassifier`