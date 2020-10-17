"""Precision for ranking."""
import numpy as np

from matchzoo.engine.base_metric import BaseMetric
from sklearn.metrics import  recall_score

class Recall(BaseMetric):
    """Precision metric."""

    ALIAS = 'recall'

    def __init__(self, k: int = 1, threshold: float = 0.):
        """
        :class:`PrecisionMetric` constructor.

        :param k: Number of results to consider.
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:

        return recall_score(y_true=y_true, y_pred=y_pred)
