"""Hellinger decision tree estimators."""

from .datasets import make_gaussian_skew_dataset
from .estimator import HellingerDecisionTreeClassifier

__version__ = "0.1.0"

__all__ = ["HellingerDecisionTreeClassifier", "__version__", "make_gaussian_skew_dataset"]
