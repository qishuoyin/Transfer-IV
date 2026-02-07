#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Qishuo 

Transfer-IV: Reusable trainer class, updated for fine-tuning.
"""


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transfer_iv.dataloader import create_dataloader
from transfer_iv.model.model_FASTNN import FASTNN

# run script on gpu if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseTrainer:
    """
    Handles training, validation, and prediction for a PyTorch model.
    """
    
    def __init__(self, model, train_params):
        self.model = model.to(device)
        self.train_params = train_params
        self.loss_fn = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []} # Added history tracking

    def fit(self, X_train, y_train, X_val, y_val, s_hat_train=None, s_hat_val=None):
        """
        Trains the model with validation and early stopping.
        Now accepts pre-split data and optional s_hat signals.
        """
        train_loader = create_dataloader(X_train, y_train, self.train_params['batch_size'], s_hat=s_hat_train)
        val_loader = create_dataloader(X_val, y_val, self.train_params['batch_size'], s_hat=s_hat_val, shuffle=False)
        
        optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                         lr=self.train_params['learning_rate'])
        
        # Scheduler to reduce LR when validation loss stops improving
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        patience = self.train_params.get('patience', 30)
        min_delta = self.train_params.get('min_delta', 0.0)

        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        print(f"Starting training on {device}...")
        for epoch in range(self.train_params['epochs']):
            self.model.train()
            train_loss_sum = 0
            
            for batch in train_loader:
                # Unpack batch, handling both base and finetune cases
                if len(batch) == 3:
                    x_batch, y_batch, s_batch = batch
                    x_batch, y_batch, s_batch = x_batch.to(device), y_batch.to(device), s_batch.to(device)
                else:
                    x_batch, y_batch = batch
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    s_batch = None
                
                optimizer.zero_grad()
                pred = self.model(x_batch, s_hat=s_batch)
                loss = self.loss_fn(pred, y_batch)
                
                # --- Tau Annealing Implementation (Matches Paper) ---
                if isinstance(self.model, FASTNN) and 'lambda_reg' in self.train_params:
                    # Default base_tau is 0.005 if not provided
                    base_tau = self.train_params.get('tau', 0.005)
                    
                    # Annealing logic: Start high (20*tau), end at tau
                    start_tau = base_tau * 20
                    end_tau = base_tau
                    anneal_rate = (start_tau - end_tau) / self.train_params['epochs']
                    
                    # Calculate current tau for this epoch
                    current_tau = max(end_tau, start_tau - (epoch * anneal_rate))
                    
                    loss += self.train_params['lambda_reg'] * self.model.regularization_loss(current_tau)
                
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()
            
            avg_train_loss = train_loss_sum / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation loop
            self.model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x_batch, y_batch, s_batch = batch
                        x_batch, y_batch, s_batch = x_batch.to(device), y_batch.to(device), s_batch.to(device)
                    else:
                        x_batch, y_batch = batch
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        s_batch = None
                    pred = self.model(x_batch, s_hat=s_batch)
                    val_loss_sum += self.loss_fn(pred, y_batch).item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            self.history['val_loss'].append(avg_val_loss)
            
            # Step the scheduler
            scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.train_params['epochs']}, Val Loss: {avg_val_loss:.5f}")

            # Track best epoch for debugging
            if avg_val_loss < best_val_loss:
                # print(f"New best val loss: {avg_val_loss:.5f} at epoch {epoch+1}")
                pass

            # Early stopping check
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"Training completed at epoch {epoch+1}. Best val loss: {best_val_loss:.5f}")
        self.last_epoch = epoch + 1
        self.best_val_loss = best_val_loss

        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return self.model
    
    def predict(self, X):
        """
        Makes predictions on new data.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input features for prediction
            
        Returns
        -------
        numpy.ndarray
            Predictions
        """
        self.model.eval()
        # Ensure input is float32
        X_tensor = torch.FloatTensor(X).to(device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
