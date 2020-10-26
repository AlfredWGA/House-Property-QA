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

        # best, best_t = self.search_f1(y_true, y_pred)
        #
        # return best


    def search_f1(self, y_true, y_pred):
        best = 0
        best_t = 0
        for i in range(30, 60):
            tres = i / 100
            y_pred_bin = (y_pred > tres).astype(int)
            score = f1_score(y_true, y_pred_bin)
            if score > best:
                best = score
                best_t = tres

        # y_pred_bin = (y_pred > best_t).astype(int)
        # best = f1_score(y_true, y_pred_bin)

        return best, best_t
