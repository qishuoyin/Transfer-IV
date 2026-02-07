#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transfer-IV: Transfer Learning with Instrumental Variables
A package for factor-augmented sparse throughput neural networks with transfer learning.
"""

__version__ = "0.1.0"
__author__ = "Qishuo"

from transfer_iv.dataloader import TorchDataset, create_dataloader
from transfer_iv.estimator.finetune_estimator import FineTuningEstimator
from transfer_iv.estimator.factor_estimator import FactorMatrixEstimator
from transfer_iv.estimator.base_trainer import BaseTrainer
from transfer_iv.model.model_FASTNN import FASTNN
from transfer_iv.model.model_VanillaNN import VanillaNN

__all__ = [
    'TorchDataset',
    'create_dataloader',
    'FineTuningEstimator',
    'FactorMatrixEstimator',
    'BaseTrainer',
    'FASTNN',
    'VanillaNN',
]

