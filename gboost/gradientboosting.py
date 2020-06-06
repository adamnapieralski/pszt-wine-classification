"""Gradient Boosting Classifier class module.

Implements Gradient Boosting algorithm for classification.
"""
__author__ = "Napieralski Adam, Kostrzewa Lukasz"

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingClassifier:

    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3, loss_function="mse", verbosity=0):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.loss_function = loss_function
        self.verbosity = verbosity

    def fit(self, X, y):
        """Fit the gradient boosting model to the training input data.

        Parameters:
        X - array with input samples, shape (n_samples, n_features) 
        Y - array with target values, correspnding to input samples, shape (n_samples,)
        """
        self.f_models = []

        self.compute_initial_y_const(y)
        if self.verbosity:
            print("Initialized y const value: {}".format(self.y_const_initial))

        for i in range(self.n_estimators):
            residuals = self.compute_residuals(y, self.predict(X))
            regressor = DecisionTreeRegressor(max_depth=self.max_depth)
            regressor.fit(X, residuals)
            self.f_models.append(regressor)

            if self.verbosity:
                print("Step: {}".format(i))
                print(" Residuals: {}".format(residuals))
                print(" Predictions: {}".format(self.predict(X)))

    def predict(self, X):
        prediction = np.repeat(self.y_const_initial, X.shape[0]).astype(np.float64)
        # # if learning rate calculated at each iteration
        # for f, y in zip(self.f_models, self.gammas):
        #     prediction += y * f.predict(X)
        for f in self.f_models:
            prediction += self.learning_rate * f.predict(X)
        return prediction

    def score(self, X, y):
        y_predicted = self.predict(X)
        correct = 0
        for i in range(y.size):
            if round(y_predicted[i]) == y[i]:
                correct += 1
        return correct / y.size

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

    def compute_residuals(self, y, y_hat):
        return -1 * self.compute_loss_gradient(y, y_hat)

    def compute_initial_y_const(self, y):
        y_uniques = np.unique(y)
        y_consts = np.repeat(y_uniques, y.shape[0]).reshape((y_uniques.shape[0], y.shape[0]))
        sums = np.sum(self.compute_loss(y, y_consts), axis=1)
        self.y_const_initial = y_uniques[np.argmin(sums)]