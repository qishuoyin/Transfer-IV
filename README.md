# Transfer-IV

Transfer Learning with Instrumental Variables - A Python package for factor-augmented sparse throughput neural networks with transfer learning capabilities.

## Overview

Transfer-IV implements advanced transfer learning techniques for neural networks with instrumental variables. 

## Features

- 🚀 GPU-accelerated computations with PyTorch
- 📊 Factor analysis with diversified projection matrices
- 🔄 Transfer learning with pre-training and fine-tuning
- 🎯 Sparse variable selection with clipped L1 regularization
- 📈 Early stopping and learning rate scheduling
- 🔧 Flexible architecture supporting both base and fine-tune modes

## Installation

### From source

```bash
git clone https://github.com/yourusername/Transfer-IV.git
cd Transfer-IV
pip install -e .
```

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- pandas >= 1.3.0
- PyTorch >= 1.9.0
- SciPy >= 1.7.0

## Quick Start

### Basic Usage

```python
from transfer_iv import FineTuningEstimator

# Define model and training parameters
model_params = {
    'r_bar': 5,      # Number of diversified projection weights
    'depth': 3,      # Number of hidden layers
    'width': 64,     # Neurons per hidden layer
}

train_params = {
    'epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
    'patience': 30,
    'lambda_reg': 0.01,  # Regularization strength
    'tau': 0.005,        # Clipping threshold
}

# Initialize estimator
estimator = FineTuningEstimator(
    model_params=model_params,
    train_params=train_params,
    finetune_arch='fast'  # or 'vanilla'
)

# Prepare your data dictionary
data = {
    'X_p_train': X_source_train,
    'y_p_train': y_source_train,
    'X_p_val': X_source_val,
    'y_p_val': y_source_val,
    'X_p_unlabeled': X_source_unlabeled,
    'X_q_train': X_target_train,
    'y_q_train': y_target_train,
    'X_q_val': X_target_val,
    'y_q_val': y_target_val,
    'X_q_unlabeled': X_target_unlabeled,
}

# Fit the model (two-stage training)
estimator.fit(data)

# Make predictions
predictions = estimator.predict(X_target_test)
```

### Factor Matrix Estimation

```python
from transfer_iv import FactorMatrixEstimator

# Initialize estimator
factor_estimator = FactorMatrixEstimator(r_bar=5, delta=0.1)

# Estimate and evaluate projection matrices
scores = factor_estimator.estimate_and_evaluate(
    X_q_unlabeled=X_target_unlabeled,
    X_p_unlabeled=X_source_unlabeled,
    B_q_true=B_target_true
)

print(f"W_tl score: {scores['W_tl']:.4f}")
print(f"W_q score: {scores['W_q']:.4f}")
print(f"W_a score: {scores['W_a']:.4f}")
```

## Architecture

### FASTNN Architecture

The FASTNN model consists of:

1. **Diversified Projection Layer**: Fixed PCA-based projection to capture factor structure
2. **Variable Selection Layer**: Learnable sparse layer with clipped L1 regularization
3. **Optional Reconstruction Layer**: Removes estimated factor effects
4. **Main ReLU Network**: Deep neural network for final prediction

In fine-tuning mode, the model accepts an additional pre-trained signal `s_hat` from the source domain.

### Two-Stage Training Process

1. **Stage 1 - Pre-training**: Train FASTNN on source domain data to learn `g^P(X)`
2. **Stage 2 - Fine-tuning**: Train model `h` on target domain using both target features and pre-trained signal

### LATE Estimator (NEED TO WRITE)

There are two types of LATE estimator we may need to compare. We may need to write up the LATE estimator and use a if-else loop to decide whether we want to FAST-NN model structure or Vanilla-NN structure. 

### ATE Estimator by transfer (NEED TO WRITE)

This is the second step of our project. We will use this python file to modulize the functions to run ATE estimation by transfer learnining. 

## Project Structure

```
Transfer-IV/
├── transfer_iv/
│   ├── __init__.py
│   ├── dataloader.py                  # PyTorch Dataset and DataLoader
│   ├── estimator/
│   │   ├── __init__.py
│   │   ├── base_trainer.py            # Training loop with early stopping
│   │   ├── factor_estimator.py        # Factor matrix estimation
│   │   ├── finetune_estimator.py      # Two-stage fine-tuning
│   │   ├── late_estimator.py          # LATE estimator (Need to be written)
│   │   └── ate_transfer_estimator.py  # ATE by transfer estimator (Need to be written)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model_FASTNN.py            # FASTNN architecture
│   │   └── model_VanillaNN.py         # Vanilla neural network
│   └── utility/
│       ├── __init__.py
│       └── utility_functions.py       # Matrix computations and data loading
├── setup.py
├── requirements.txt
└── README.md
```

## Key Components

### Models

- **FASTNN**: Factor-Augmented Sparse Throughput Neural Network with diversified projection and variable selection
- **VanillaNN**: Standard deep ReLU network for baseline comparisons

### Estimators

- **BaseTrainer**: Handles training loop, validation, and early stopping
- **FactorMatrixEstimator**: Computes and evaluates projection matrices (W^TL, W^Q, W^A)
- **FineTuningEstimator**: Orchestrates two-stage pre-training and fine-tuning

### Utilities

- **get_projection_matrix**: Computes PCA-based diversified projection
- **compute_transfer_projection_matrix**: Computes transfer learning projection with model selection
- **get_reconstruction_matrix**: Computes factor reconstruction matrix
- **load_all_sim_data**: Loads pre-split simulation datasets

## GPU Support

The package automatically uses GPU acceleration when available:

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

All computations are optimized for GPU execution, including:
- Covariance matrix computation
- Eigenvalue decomposition
- Neural network training

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue on GitHub.

