import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EarlyStopping(object):
    def __init__(self, patience=10, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.stop_flag = False

    def _update_counter(self):
        self.counter += 1
        logger.info(
            f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
            self.stop_flag = True

    def reset_counter(self):
        self.counter = 0

    def step(self, score):
        # No initial best score.
        if self.best_score is None:
            self.best_score = score

        # Check if score is worse than best score, or is out of patience.
        elif (self.mode == 'min' and self.best_score <= score) or \
                (self.mode == 'max' and score <= self.best_score):
            self._update_counter()

        # Achieve a better score.
        else:
            logger.info('A best score! Reset EarlyStopping counter.')
            self.best_score = score
            self.counter = 0
        return self.stop_flag
