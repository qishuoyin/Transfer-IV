#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Qishuo

FAN_Lasso: Model structure for VanillaNN, with fine-tuning support.
"""


import torch
from torch import nn
from collections import OrderedDict

# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VanillaNN(nn.Module):
    """
    Standard deep ReLU network with support for fine-tuning.
    In 'finetune' mode, it concatenates the input X with a pretrained signal s_hat.
    """
    def __init__(self, input_dim, depth, width, mode='base'):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the main input X (i.e., p).
        depth : int
            The number of hidden layers.
        width : int
            The number of neurons in each hidden layer.
        mode : str, optional
            'base' for standard training, 'finetune' to accept an additional signal.
        """
        super(VanillaNN, self).__init__()
        self.mode = mode

        # Adjust network input dimension based on the mode
        net_input_dim = input_dim
        if self.mode == 'finetune':
            net_input_dim += 1  # Add dimension for s_hat

        layers = [('linear1', nn.Linear(net_input_dim, width)), ('relu1', nn.ReLU())]
        for i in range(depth - 1):
            layers.append((f'linear{i+2}', nn.Linear(width, width)))
            layers.append((f'relu{i+2}', nn.ReLU()))
        layers.append((f'linear{depth+1}', nn.Linear(width, 1)))

        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, x, s_hat=None):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            (batch_size, p) input tensor.
        s_hat : torch.Tensor, optional
            (batch_size, 1) pretrained signal tensor, required for 'finetune' mode.

        Returns
        -------
        torch.Tensor
            (batch_size, 1) output tensor.
        """
        if self.mode == 'finetune':
            if s_hat is None:
                raise ValueError("s_hat must be provided in 'finetune' mode.")
            # Concatenate original input X with the pretrained signal
            net_input = torch.cat((x, s_hat), dim=-1)
        else:
            net_input = x
            
        return self.network(net_input)