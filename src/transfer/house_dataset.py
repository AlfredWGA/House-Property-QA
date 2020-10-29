import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class HouseDataset(Dataset):
    def __init__(self, x_set, y_set):
        # assert len(x_set) == len(y_set)
        self.x_set = np.asarray(x_set).transpose((1, 0, 2))
        self.y_set = np.asarray(y_set)
        self._length = self.x_set.shape[0]

    def __len__(self):
        # Return the size of the dataset.
        return self._length

    def __getitem__(self, idx):
        #  Fetching a data sample for a given key.
        sample_x = self.x_set[idx]
        sample_y = self.y_set[idx]

        return sample_x, sample_y