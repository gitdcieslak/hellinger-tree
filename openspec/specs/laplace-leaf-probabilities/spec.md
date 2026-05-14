# Laplace Leaf Probabilities Specification

## Requirements

### Requirement: Leaf probability estimates

The classifier SHALL expose class probability estimates through `predict_proba`.

#### Scenario: Probability shape

Given a fitted binary classifier
When `predict_proba(X)` is called
Then the returned array SHALL have shape `(n_samples, 2)`.

#### Scenario: Probability normalization

Given a fitted classifier
When `predict_proba(X)` is called
Then each row SHALL sum to 1.

---

### Requirement: Laplace smoothing by default

The classifier SHALL use Laplace-smoothed leaf probabilities by default.

For binary classification:

```text
P(class_k | leaf) = (count_k + 1) / (n_leaf + 2)
Scenario: Leaf with zero examples for one class

Given a leaf with class counts [0, 3]
And Laplace smoothing is enabled
When computing leaf probabilities
Then the probabilities SHALL be [1/5, 4/5].

Scenario: Leaf with both classes

Given a leaf with class counts [2, 3]
And Laplace smoothing is enabled
When computing leaf probabilities
Then the probabilities SHALL be [3/7, 4/7].

Requirement: Empirical probabilities option

The classifier SHALL allow Laplace smoothing to be disabled.

Scenario: Laplace disabled

Given a leaf with class counts [0, 3]
And Laplace smoothing is disabled
When computing leaf probabilities
Then the probabilities SHALL be [0.0, 1.0].

Requirement: Smoothing affects probabilities only

Laplace smoothing SHALL affect leaf probability estimates only.

Scenario: Split selection unaffected

Given laplace=True
When the classifier evaluates candidate splits
Then Laplace smoothing SHALL NOT alter the Hellinger split score.