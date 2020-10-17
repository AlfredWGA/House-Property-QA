"""Precision for ranking."""
import numpy as np

from matchzoo.engine.base_metric import BaseMetric
from sklearn.metrics import  f1_score

class F1(BaseMetric):
    """Precision metric."""

    ALIAS = 'f1'

    def __init__(self, k: int = 1, threshold: float = 0.5):
        """
        :class:`PrecisionMetric` constructor.

        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        self.k = k
        self.threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:

        y_pred[y_pred >= self.threshold] = 1.0
        y_pred[y_pred <  self.threshold] = 0.0

        return f1_score(y_true=y_true, y_pred=y_pred)
