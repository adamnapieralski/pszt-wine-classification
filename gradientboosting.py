import numpy as np
import pandas as pd

class GradientBoostingClassifier:

    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3, loss_function="mse", verbose=False):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.loss_function = loss_function
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the gradient boosting model.

        Parameters:
        X - array with input samples, shape (n_samples, n_features) 
        Y - array with target values, correspnding to input samples, shape (n_samples,)
        """
        pass

    def compute_loss(self, y, y_hat):
        loss = []
        if self.loss_function == "mse":
            loss = (y - y_hat)**2
        return loss

    def compute_loss_gradient(self, y, y_hat):
        loss_gradient = []
        if self.loss_function == "mse":
            loss_gradient = -2 * (y - y_hat)
        return loss_gradient

    def initialize_with_const(self, y):
        y_uniques = np.unique(y)
        y_consts = np.repeat(y_uniques, y.shape[0]).reshape((y_uniques.shape[0], y.shape[0]))
        sums = np.sum(self.compute_loss(y, y_consts), axis=1)
        y_hat_init = y_consts[np.argmin(sums)]
        return y_hat_init