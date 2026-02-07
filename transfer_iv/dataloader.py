#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Qishuo 

Transfer-IV: Dataloader for PyTorch models, updated for fine-tuning.
"""


import torch
from torch.utils.data import Dataset, DataLoader

class TorchDataset(Dataset):
    """A class to transform numpy arrays into a PyTorch Dataset."""
    def __init__(self, features, labels, s_hat=None):
        """
        Parameters
        ----------
        features : numpy.ndarray
            (n, p) matrix of input feature data.
        labels : numpy.ndarray
            (n, 1) matrix of input target data.
        s_hat : numpy.ndarray, optional
            (n, 1) matrix of pre-trained signals for fine-tuning. Defaults to None.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.s_hat = torch.tensor(s_hat, dtype=torch.float32) if s_hat is not None else None

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves a data tuple by index. If s_hat is present, it's included.
        """
        if self.s_hat is not None:
            return self.features[idx], self.labels[idx], self.s_hat[idx]
        else:
            return self.features[idx], self.labels[idx]

def create_dataloader(features, labels, batch_size, s_hat=None, shuffle=True):
    """
    Creates a PyTorch DataLoader from numpy arrays.
    
    Parameters
    ----------
    features, labels, batch_size, shuffle : as before
    s_hat : numpy.ndarray, optional
        Pre-trained signals. Defaults to None.

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader instance for the provided data.
    """
    dataset = TorchDataset(features, labels, s_hat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader