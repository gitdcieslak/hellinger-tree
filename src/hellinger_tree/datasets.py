import numpy as np
from sklearn.utils import check_random_state


def make_gaussian_skew_dataset(
    n_majority=1000,
    n_minority=100,
    separation=2.5,
    random_state=None,
):
    """Generate a lightweight binary Gaussian dataset with class skew."""
    if n_majority < 1:
        raise ValueError("n_majority must be at least 1.")
    if n_minority < 1:
        raise ValueError("n_minority must be at least 1.")
    if separation < 0:
        raise ValueError("separation must be non-negative.")

    rng = check_random_state(random_state)
    majority = rng.normal(loc=0.0, scale=1.0, size=(n_majority, 2))
    minority = rng.normal(loc=separation, scale=1.0, size=(n_minority, 2))
    X = np.vstack([majority, minority])
    y = np.concatenate([
        np.zeros(n_majority, dtype=int),
        np.ones(n_minority, dtype=int),
    ])

    order = rng.permutation(X.shape[0])
    return X[order], y[order]
