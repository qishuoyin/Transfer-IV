#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Qishuo 

Transfer-IV: Utility functions for data handling and matrix computation.
"""


import numpy as np
import pandas as pd
import os
import torch
# from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh as largest_eigsh

# GPU device for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_all_sim_data(data_path, sim_num):
    """
    Loads all 8 pre-split datasets for a given simulation number.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the simulation data files.
    sim_num : int
        The simulation number to load.

    Returns
    -------
    dict
        A dictionary containing all data arrays, keyed by their role
        (e.g., 'X_p_train', 'y_q_test').
    """
    data_files = {
        'p_train': f'data_source_train_sim_{sim_num}.csv',
        'p_val':   f'data_source_val_sim_{sim_num}.csv',
        'p_test':  f'data_source_test_sim_{sim_num}.csv',
        'p_unlabeled': f'data_source_unlabeled_sim_{sim_num}.csv',
        'q_train': f'data_target_train_sim_{sim_num}.csv',
        'q_val':   f'data_target_val_sim_{sim_num}.csv',
        'q_test':  f'data_target_test_sim_{sim_num}.csv',
        'q_unlabeled': f'data_target_unlabeled_sim_{sim_num}.csv',
    }
    
    loaded_data = {}
    for key, filename in data_files.items():
        full_path = os.path.join(data_path, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found: {full_path}")
        df = pd.read_csv(full_path).to_numpy()
        # Unlabeled data only has X
        if 'unlabeled' in key:
            loaded_data[f'X_{key}'] = df
        else:
            loaded_data[f'X_{key}'] = df[:, :-1]
            loaded_data[f'y_{key}'] = df[:, -1:]

    return loaded_data

def get_projection_matrix(X, r_bar):
    """
    Computes the diversified projection matrix W from covariates X using PCA.

    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor
        (n, p) matrix of covariates.
    r_bar : int
        Number of principal components (diversified weights) to compute.

    Returns
    -------
    numpy.ndarray
        The (p, r_bar) diversified projection matrix.
    """
    # Convert to torch tensor if numpy array
    if isinstance(X, np.ndarray):
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X_torch = X.to(device)
    
    n, p = X_torch.shape
    
    # Compute covariance matrix on GPU
    cov_mat = (X_torch.T @ X_torch) / n
    
    # Compute eigenvalues and eigenvectors on GPU
    # torch.linalg.eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_mat)
    
    # Select top r_bar eigenvectors (largest eigenvalues, which are at the end)
    top_eigenvectors = eigenvectors[:, -r_bar:]
    
    # Multiply by sqrt(p) - the /p normalization happens when using the projection
    dp_mat = top_eigenvectors * np.sqrt(p)
    
    # Convert back to numpy for compatibility with existing code
    return dp_mat.cpu().numpy()

def get_reconstruction_matrix(X, W):
    """
    Computes the reconstruction matrix R for the projection residuals.
    
    R = (F^T F)^{-1} F^T X, where F = X @ W.
    
    Parameters
    ----------
    X : numpy.ndarray or torch.Tensor
        (n, p) matrix of covariates.
    W : numpy.ndarray or torch.Tensor
        (p, r_bar) projection matrix.
        
    Returns
    -------
    numpy.ndarray
        (r_bar, p) reconstruction matrix.
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(X, np.ndarray):
        X_torch = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X_torch = X.to(device)
    
    if isinstance(W, np.ndarray):
        W_torch = torch.tensor(W, dtype=torch.float32, device=device)
    else:
        W_torch = W.to(device)
    
    # Compute on GPU
    estimate_f = X_torch @ W_torch
    cov_f_mat = estimate_f.T @ estimate_f
    cov_fx_mat = estimate_f.T @ X_torch
    rs_matrix = torch.linalg.pinv(cov_f_mat) @ cov_fx_mat
    
    # Convert back to numpy for compatibility
    return rs_matrix.cpu().numpy()

def compute_transfer_projection_matrix(X_q, X_p, r_bar, delta):
    """
    Computes the transfer projection matrix W^TL based on the model selection rule.

    Parameters
    ----------
    X_q : numpy.ndarray or torch.Tensor
        Target domain covariates for matrix computation.
    X_p : numpy.ndarray or torch.Tensor
        Source domain covariates for matrix computation.
    r_bar : int
        Number of diversified weights.
    delta : float
        Threshold for the model selection rule.

    Returns
    -------
    dict
        A dictionary containing W_tl, W_q, and W_a.
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(X_q, np.ndarray):
        X_q_torch = torch.tensor(X_q, dtype=torch.float32, device=device)
    else:
        X_q_torch = X_q.to(device)
    
    if isinstance(X_p, np.ndarray):
        X_p_torch = torch.tensor(X_p, dtype=torch.float32, device=device)
    else:
        X_p_torch = X_p.to(device)
    
    n_q, p = X_q_torch.shape
    n_p = X_p_torch.shape[0]
    
    # Compute covariance matrices on GPU
    cov_p = (X_p_torch.T @ X_p_torch) / n_p
    cov_q = (X_q_torch.T @ X_q_torch) / n_q
    cov_a = (n_q * cov_q + n_p * cov_p) / (n_q + n_p)
    
    # Use the updated get_projection_matrix (now GPU-accelerated)
    W_p = get_projection_matrix(X_p_torch, r_bar)
    W_q = get_projection_matrix(X_q_torch, r_bar)
    
    # Compute eigenvalues and eigenvectors on GPU for cov_a
    eigenvalues_a, eigenvectors_a = torch.linalg.eigh(cov_a)
    top_eigenvectors_a = eigenvectors_a[:, -r_bar:]
    W_a = (top_eigenvectors_a * np.sqrt(p)).cpu().numpy()
    
    # Compute Frobenius norm on GPU
    frob_norm_diff = torch.linalg.norm(cov_q - cov_a, ord='fro').cpu().item()
    
    if (frob_norm_diff / p) <= delta:
        W_tl = W_a
    else:
        W_tl = W_q
        
    return {'W_tl': W_tl, 'W_q': W_q, 'W_a': W_a, 'W_p': W_p}