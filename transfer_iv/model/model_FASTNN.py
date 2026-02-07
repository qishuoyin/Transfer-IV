#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Qishuo (Adapted for FINDER)

FINDER: Model structure for Factor Augmented Sparse Throughput NN.
This version includes the full original architecture and fine-tuning support.
"""


import numpy as np
import torch
from torch import nn
from collections import OrderedDict

# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FASTNN(nn.Module):
    """
    A class for the Factor Augmented Sparse Throughput deep ReLU neural network.

    This model supports both a standard 'base' mode for training from scratch and a
    'finetune' mode, where it accepts an additional pre-trained signal `s_hat`
    as input to the main network. The architecture includes an optional reconstruction
    layer to remove the estimated factor effects from the input to the sparse
    throughput layer, enhancing its ability to capture idiosyncratic noise.

    Methods
    -------
    __init__(...)
        Initializes the model layers and architecture.
    forward(x, s_hat=None)
        Implements the forward pass through the network.
    regularization_loss(tau)
        Calculates the clipped L1 regularization penalty for the variable selection layer.
    """

    def __init__(self, input_dim, r_bar, depth, width, dp_mat, sparsity=None, rs_mat=None, mode='base'):
        """
        Initializes the FASTNN model.

        Parameters
        ----------
        input_dim : int
            The input dimension of the covariates (p).
        r_bar : int
            The number of diversified projection weights.
        depth : int
            The number of hidden layers in the main ReLU network.
        width : int
            The number of neurons in each hidden layer.
        dp_mat : numpy.ndarray
            The pre-computed (p, r_bar) diversified projection matrix (W). This is fixed.
        sparsity : int, optional
            The output dimension of the sparse throughput (variable selection) layer.
            If None, defaults to `width`.
        rs_mat : numpy.ndarray, optional
            The pre-computed (r_bar, p) reconstruction matrix. If provided, a
            reconstruction layer is added. Defaults to None.
        mode : str, optional
            The operational mode of the network. Can be 'base' or 'finetune'.
            Defaults to 'base'.
        """
        super(FASTNN, self).__init__()
        self.mode = mode

        # Layer 1: Diversified Projection (fixed weights)
        self.diversified_projection = nn.Linear(input_dim, r_bar, bias=False)
        # dp_tensor = torch.tensor(dp_mat.T, dtype=torch.float32)
        dp_tensor = torch.tensor(dp_mat.T, dtype=torch.float32) / input_dim
        self.diversified_projection.weight = nn.Parameter(dp_tensor, requires_grad=False)

        # Optional Reconstruction Layer (fixed weights)
        if rs_mat is not None:
            self.reconstruct = nn.Linear(r_bar, input_dim, bias=False)
            rs_tensor = torch.tensor(rs_mat.T, dtype=torch.float32) / input_dim # 
            self.reconstruct.weight = nn.Parameter(rs_tensor, requires_grad=False)
        else:
            self.reconstruct = None

        # Layer 2: Variable Selection (learnable weights)
        if sparsity is None:
            sparsity = width
        self.variable_selection = nn.Linear(input_dim, sparsity, bias=False)

        # Determine input dimension for the main ReLU stack
        relu_input_dim = r_bar + sparsity
        if self.mode == 'finetune':
            relu_input_dim += 1  # Add dimension for s_hat

        # Main ReLU Network
        relu_layers = [('linear1', nn.Linear(relu_input_dim, width)), ('relu1', nn.ReLU())]
        for i in range(depth - 1):
            relu_layers.append((f'linear{i+2}', nn.Linear(width, width)))
            relu_layers.append((f'relu{i+2}', nn.ReLU()))
        relu_layers.append((f'linear{depth+1}', nn.Linear(width, 1)))
        self.relu_stack = nn.Sequential(OrderedDict(relu_layers))

    def forward(self, x, s_hat=None):
        """
        Defines the forward pass of the FASTNN model.

        Parameters
        ----------
        x : torch.Tensor
            A (batch_size, p) tensor of input covariates.
        s_hat : torch.Tensor, optional
            A (batch_size, 1) tensor of pre-trained signals. Required if mode is 'finetune'.
            Defaults to None.

        Returns
        -------
        torch.Tensor
            A (batch_size, 1) tensor of model predictions.
        """
        factors = self.diversified_projection(x)

        if self.reconstruct is not None:
            x_residual = x - self.reconstruct(factors)
            idiosyncratic = self.variable_selection(x_residual)
        else:
            idiosyncratic = self.variable_selection(x)

        if self.mode == 'finetune':
            if s_hat is None:
                raise ValueError("s_hat must be provided in 'finetune' mode.")
            combined_input = torch.cat((factors, idiosyncratic, s_hat), dim=-1)
        else:
            combined_input = torch.cat((factors, idiosyncratic), dim=-1)
            
        return self.relu_stack(combined_input)

    def regularization_loss(self, tau):
        """
        Computes the clipped L1 regularization loss for the variable selection layer.

        Parameters
        ----------
        tau : float
            The clipping threshold hyperparameter from the manuscript.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the regularization loss.
        """
        l1_penalty = torch.abs(self.variable_selection.weight) / tau
        clipped_l1 = torch.clamp(l1_penalty, max=1.0)
        return torch.sum(clipped_l1)