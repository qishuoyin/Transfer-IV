#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Qishuo 

Transfer-IV: Estimator for the factor projection matrices (for Experiment 1).
"""


import numpy as np
from transfer_iv.utility.utility_functions import compute_transfer_projection_matrix

class FactorMatrixEstimator:
    """
    A class to estimate and evaluate the quality of factor projection matrices.

    This class is designed specifically for Experiment 1. Its primary function is to
    compute the three diversified projection matrices (W^TL, W^Q, W^A) and then
    evaluate them against the ground truth target loading matrix (B^Q) using the
    minimum singular value metric defined in the manuscript.

    Methods
    -------
    __init__(r_bar, delta)
        Initializes the estimator with hyperparameters.
    estimate_and_evaluate(X_q_unlabeled, X_p_unlabeled, B_q_true)
        Performs the full estimation and evaluation pipeline.
    """

    def __init__(self, r_bar, delta):
        """
        Initializes the FactorMatrixEstimator.

        Parameters
        ----------
        r_bar : int
            The number of diversified projection weights to compute for each matrix.
        delta : float
            The threshold for the model selection rule used in computing W^TL.
        """
        self.r_bar = r_bar
        self.delta = delta

    def estimate_and_evaluate(self, X_q_unlabeled, X_p_unlabeled, B_q_true):
        """
        Estimates the projection matrices and evaluates their performance.

        This method first calls a utility function to compute W^TL, W^Q, and W^A
        using the provided unlabeled source and target data. It then calculates the
        evaluation metric, vmin(p^-1 * (W*)^T * B^Q), for each of the
        three estimated matrices.

        Parameters
        ----------
        X_q_unlabeled : numpy.ndarray
            The (n_unlabeled_q, p) covariate matrix from the target domain,
            used for matrix computation.
        X_p_unlabeled : numpy.ndarray
            The (n_unlabeled_p, p) covariate matrix from the source domain,
            used for matrix computation.
        B_q_true : numpy.ndarray
            The ground truth (p, r) factor loading matrix for the target domain.
            This is required for evaluation.

        Returns
        -------
        dict
            A dictionary containing the evaluation score for each estimated matrix.
            Example: {'W_tl': 0.85, 'W_q': 0.75, 'W_a': 0.82}
        """
        # Step 1: Estimate the three W matrices
        estimated_matrices = compute_transfer_projection_matrix(
            X_q_unlabeled, X_p_unlabeled, self.r_bar, self.delta
        )

        # Step 2: Evaluate each estimated matrix
        p = B_q_true.shape[0]
        evaluation_scores = {}

        for name, W_est in estimated_matrices.items():
            # The matrix for which to find the minimum singular value
            M = (W_est.T @ B_q_true) / p
            # M = (W_est.T @ B_q_true)


            # The minimum singular value is the smallest value in the array 's'
            # returned by SVD.
            s = np.linalg.svd(M, compute_uv=False)
            score = np.min(s)
            evaluation_scores[name] = score

        return evaluation_scores