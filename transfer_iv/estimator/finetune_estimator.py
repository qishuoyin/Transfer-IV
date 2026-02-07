#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Qishuo

Transfer-IV: Main estimator for the fine-tuning task.
"""


import numpy as np
import torch
from transfer_iv.model.model_FASTNN import FASTNN
from transfer_iv.model.model_VanillaNN import VanillaNN
from transfer_iv.estimator.base_trainer import BaseTrainer
from transfer_iv.utility.utility_functions import get_projection_matrix, compute_transfer_projection_matrix

# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FineTuningEstimator:
    """
    A class to orchestrate the two-stage pre-training and fine-tuning process.

    This estimator first trains a model (g^P) on a source dataset, then uses its
    predictions as an additional feature to train a second model (h) on a
    target dataset.

    Methods
    -------
    __init__(...)
        Initializes the estimator with model and training parameters.
    fit(data)
        Executes the full two-stage fitting process.
    predict(X_q_test)
        Makes final predictions on new target data.
    """
    
    def __init__(self, model_params, train_params, finetune_arch='fast'):
        """
        Initializes the FineTuningEstimator.

        Parameters
        ----------
        model_params : dict
            Parameters for the neural network models (e.g., depth, width, r_bar).
        train_params : dict
            Parameters for the training process (e.g., epochs, lr, batch_size).
        finetune_arch : str, optional
            Architecture for the fine-tuning ('h') network.
            Can be 'fast' for FAN_Lasso or 'vanilla' for FT-Vanilla-NN.
            Defaults to 'fast'.
        """
        self.model_params = model_params
        self.train_params = train_params
        self.finetune_arch = finetune_arch
        self.pretrained_model = None
        self.finetuned_h_model = None
        self.W_tl = None

    def fit(self, data):
        """
        Executes the full two-stage fitting process using pre-split data.
        
        Parameters
        ----------
        data : dict
            A dictionary containing all necessary numpy data arrays:
            'X_p_train', 'y_p_train', 'X_p_val', 'y_p_val', 'X_p_unlabeled',
            'X_q_train', 'y_q_train', 'X_q_val', 'y_q_val', 'X_q_unlabeled'.
        """
        # --- Stage 1: Pre-train g^P on source data ---
        print("--- Stage 1: Pre-training g^P on Source Data ---")
        W_p = get_projection_matrix(data['X_p_unlabeled'], self.model_params['r_bar'])
        
        source_model = FASTNN(input_dim=data['X_p_train'].shape[1], dp_mat=W_p, 
                              mode='base', **self.model_params)
        # Save trainer as attribute to access stats later
        self.pretrainer = BaseTrainer(source_model, self.train_params)
        self.pretrained_model = self.pretrainer.fit(data['X_p_train'], data['y_p_train'],
                                                   data['X_p_val'], data['y_p_val'])
        
        # --- Stage 2: Fine-tune h on target data ---
        print("\n--- Stage 2: Fine-tuning h on Target Data ---")
        s_hat_train = self.pretrainer.predict(data['X_q_train'])
        s_hat_val = self.pretrainer.predict(data['X_q_val'])
        
        X_combined_unlabeled = np.vstack([data['X_p_unlabeled'], data['X_q_unlabeled']])
        matrices = compute_transfer_projection_matrix(data['X_q_unlabeled'], data['X_p_unlabeled'],
                                                      self.model_params['r_bar'], delta=0.1)
        self.W_tl = matrices['W_tl']
        
        p_q = data['X_q_train'].shape[1]
        if self.finetune_arch == 'fast':
            h_network = FASTNN(input_dim=p_q, dp_mat=self.W_tl, mode='finetune', **self.model_params)
        elif self.finetune_arch == 'vanilla':
            # Filter out FASTNN-specific parameters for VanillaNN
            vanilla_params = {k: v for k, v in self.model_params.items() if k in ['depth', 'width']}
            h_network = VanillaNN(input_dim=p_q, mode='finetune', **vanilla_params)
        else:
            raise ValueError(f"Unknown finetune_arch: {self.finetune_arch}")

        # Save trainer as attribute to access stats later
        self.finetuner = BaseTrainer(h_network, self.train_params)
        self.finetuned_h_model = self.finetuner.fit(data['X_q_train'], data['y_q_train'],
                                                    data['X_q_val'], data['y_q_val'],
                                                    s_hat_train=s_hat_train, s_hat_val=s_hat_val)

    def predict(self, X_q_test):
        """
        Makes final predictions on new target data using the fitted two-stage model.

        Parameters
        ----------
        X_q_test : numpy.ndarray
            The (n_test, p) covariate matrix for the target domain test set.

        Returns
        -------
        numpy.ndarray
            The (n_test, 1) matrix of predictions.
        """
        if self.pretrained_model is None or self.finetuned_h_model is None:
            raise RuntimeError("Model has not been fitted. Call .fit() first.")

        # Predict with the pretrained model to get s_hat
        s_hat_test = self.pretrained_model(torch.tensor(X_q_test, dtype=torch.float32).to(device)).cpu().detach().numpy()
        
        # Predict using the fine-tuned h_network
        self.finetuned_h_model.eval()
        x_tensor = torch.tensor(X_q_test, dtype=torch.float32).to(device)
        s_tensor = torch.tensor(s_hat_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = self.finetuned_h_model(x_tensor, s_hat=s_tensor)
        
        return predictions.cpu().numpy()