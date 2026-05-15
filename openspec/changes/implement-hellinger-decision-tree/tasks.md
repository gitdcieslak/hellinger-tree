# Tasks

## Phase 1 — Core estimator

### 1. Package skeleton

- [x] Create `src/hellinger_tree/`
- [x] Add `src/hellinger_tree/__init__.py`
- [x] Add `src/hellinger_tree/estimator.py`
- [x] Configure `pyproject.toml`
- [x] Add test dependencies

### 2. Implement core estimator

- [x] Add `HellingerDecisionTreeClassifier`
- [x] Inherit from `BaseEstimator` and `ClassifierMixin`
- [x] Add constructor parameters
- [x] Add `fit`
- [x] Add `predict`
- [x] Add `predict_proba`

### 3. Implement tree induction

- [x] Add internal node representation
- [x] Implement recursive node growth
- [x] Implement stopping rules
- [x] Implement no-valid-split fallback

### 4. Implement Hellinger split selection

- [x] Implement numeric threshold generation
- [x] Implement binary Hellinger score
- [x] Skip invalid splits
- [x] Implement deterministic tie-breaking

### 5. Implement leaf probabilities

- [x] Implement Laplace-smoothed probabilities
- [x] Implement empirical probabilities when `laplace=False`
- [x] Ensure probability rows sum to 1

### 6. Implement sklearn compatibility

- [x] Add sklearn validation
- [x] Add fittedness checks
- [x] Set `classes_`
- [x] Set `n_features_in_`
- [x] Ensure deterministic behavior

### 7. Unit tests

- [x] Test basic fit/predict
- [x] Test `predict_proba`
- [x] Test Laplace smoothing
- [x] Test binary-only validation
- [x] Test BaggingClassifier compatibility

---

# Phase 2 — Validation and introspection

## 8. Tree introspection utilities

- [x] Add `get_depth()`
- [x] Add `get_n_leaves()`
- [x] Add `export_tree()`
- [ ] Add optional `print_tree()`

## 9. Split auditing support

- [x] Store node-level Hellinger scores
- [x] Store split class distributions
- [ ] Add optional debug/export mode

## 10. Synthetic skew validation

- [x] Add Gaussian synthetic dataset generator
- [x] Add controllable skew ratios
- [x] Add overlap/noise controls
- [x] Add synthetic smoke benchmarks

## 11. CART comparison smoke tests

- [x] Compare against sklearn CART
- [x] Evaluate balanced vs skewed scenarios
- [x] Measure AUROC and PR-AUC

## 12. Reproducibility hardening

- [ ] Pin sklearn version
- [ ] Pin numpy version
- [x] Add deterministic seed tests
- [x] Verify identical exported trees under fixed seeds

---

# Phase 3 — Research readiness

## 13. Original-paper validation

- [ ] Reproduce at least one synthetic result from the original HDDT paper
- [ ] Reproduce one bagged HDDT comparison against CART

## 14. Packaging cleanup

- [ ] Add versioning
- [ ] Add license
- [ ] Add CI workflow
- [ ] Add wheel build support

## 15. Benchmark handoff readiness

- [ ] Finalize estimator API stability
- [ ] Publish install instructions
- [ ] Prepare benchmark integration notes
