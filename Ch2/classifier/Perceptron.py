"""
[Perceptron]

Input : X = [x1, ..., xm]
Output : y = 1 or -1

Weight : W = [w1, ..., wm] (Init : 0s or small random value)
Net Input:
    z = transpose(W) * X = (-threshold) * x0 + w1x1 + ... + wmxm (x0 = 1)
Activation function : y = o(z) = if z >= 0 then 1 else -1
    (Unit step function/Heaviside step function)

Step:
1. Init. weights
2. To all samples' x(i):
    (1) Count y_hat
    (2) Update weights : wj = wj + Δwj
        (Δwj = η * (y - y_hat) * xj [η : learning rate])
        if True (y = y_hat):
            Δwj = 0
        else:
            if y = 1 and y_hat = -1:
                Δwj = η * (1 - (-1)) * xj = η * 2 * xj
            else if y = -1 and y_hat = 1:
                Δwj = η * (-1 - 1) * xj = η * (-2) * xj

Constraint:
1. Linear separable
2. η must be very small to convergence
"""
import numpy as np
from numpy.random import RandomState


class Perceptron(object):
    """
    Perceptron Classifier

    Parameters:
    eta : float (Default : 0.01)
        Learning rate (0.0 ~ 1.0)
    n_iter : int (Default : 10)
        Passes over the training dateset
    random_seed : int (Default : 1)
        Random seed

    Attributes:
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        # of misclassifications in every epochs.
    """

    def __init__(self, eta=0.01, n_iter=50, random_seed=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.w_ = []
        self.errors_ = []

    def fit(self, X, y):
        """
        Fit training data

        Parameters:
        X : array of array-like (shape = [n_samples, n_features])
            Training vectors, where n_samples is # of samples &
            n_features is # of features
        y : array-like (shape = [n_samples])
            Target values

        Returns:
        self : object
        """
        rgen = RandomState(self.random_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net Input

        Paraters:
        X : array-like
            Training sample(s)

        Returns:
        Net input
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Predict class label after unit step

        Paraters:
        X : array-like
            Training sample(s)

        Returns:
        Class label
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
