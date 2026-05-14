# Hellinger Decision Tree Specification

## Requirements

### Requirement: Binary classification support

The classifier SHALL support binary classification.

#### Scenario: Fit binary labels

Given numeric feature matrix `X`
And binary label vector `y`
When `HellingerDecisionTreeClassifier.fit(X, y)` is called
Then the classifier SHALL fit successfully
And `classes_` SHALL contain exactly two labels.

#### Scenario: Reject non-binary labels

Given a label vector `y` containing more than two classes
When `fit(X, y)` is called
Then the classifier SHALL raise a clear `ValueError`.

---

### Requirement: Numeric feature splitting

The classifier SHALL support numeric feature threshold splits.

#### Scenario: Candidate thresholds

Given a feature with sorted distinct values
When evaluating candidate splits
Then thresholds SHALL be placed between adjacent distinct values
And each threshold SHALL be the midpoint of adjacent values.

#### Scenario: Constant feature

Given a feature with no distinct adjacent values
When evaluating splits
Then the feature SHALL produce no candidate thresholds.

---

### Requirement: Hellinger split criterion

The classifier SHALL select splits by maximizing Hellinger distance.

For a binary split:

```text
H = sqrt(
    (sqrt(pos_left / total_pos) - sqrt(neg_left / total_neg))^2
  + (sqrt(pos_right / total_pos) - sqrt(neg_right / total_neg))^2
)
Scenario: Best split chosen

Given multiple valid candidate splits
When growing a node
Then the classifier SHALL choose the split with the highest Hellinger score.

Scenario: Pure node

Given a node containing only one class
When growing the node
Then the node SHALL become a leaf.

Scenario: Missing class at node

Given a node where total_pos == 0 or total_neg == 0
When evaluating splits
Then the node SHALL become a leaf.

Requirement: Stopping rules

The classifier SHALL stop growing according to configured stopping rules.

Scenario: Maximum depth reached

Given max_depth is set
When a node reaches max_depth
Then the node SHALL become a leaf.

Scenario: Too few samples to split

Given a node has fewer than min_samples_split samples
When growing the node
Then the node SHALL become a leaf.

Scenario: Minimum leaf size

Given a candidate split would create a child with fewer than min_samples_leaf samples
When evaluating the candidate
Then that split SHALL be rejected.

Requirement: Deterministic tie-breaking

The classifier SHALL break split ties deterministically.

Scenario: Equal Hellinger score

Given two candidate splits have equal Hellinger score
When choosing the best split
Then the classifier SHALL prefer the lower feature index.

Scenario: Equal score and feature

Given two thresholds for the same feature have equal Hellinger score
When choosing the best split
Then the classifier SHALL prefer the lower threshold.


---

## `openspec/specs/laplace-leaf-probabilities/spec.md`

```markdown
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