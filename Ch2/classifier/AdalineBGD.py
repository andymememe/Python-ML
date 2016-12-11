"""
[Batch Gradient Descent]

Input : X = [x1, ..., xm]
Output : y = 1 or -1

Weight : W = [w1, ..., wm] (Init : 0s or small random value)
Net Input:
    z = transpose(W) * X = (-threshold) * x0 + w1x1 + ... + wmxm (x0 = 1)

Adaline rule (Widrow-Hoff rule):
Adaline : ADAptive LInear NEuron
Linear activation function:
    o(z) = z (Count errors and update weight; Differentiable)
Quantizer : Unit step function (Predict result)

Objective function (The cost function we want to minimize):
Sum of Squared Errors (SSE):
    J(W) = (1/2) * sum[for sample i]((y(i) - o(z(i))) ^ 2)
        (1/2 is for easy to get gradient)
    Convex function:
        Able to use gradient descent method to find argmin[W](J(W))

Update weights : W = W + ΔW (ΔW = -η * ∇J(W) [η : learning rate])
    => Δwj = η * sum[for sample i]((y(i) - o(z(i))) * xj(i))
"""
import numpy as np


class AdalineBGD(object):
    """
    Adaline Classifier(BGD)

    Parameters:
    eta : float (Default : 0.01)
        Learning rate (0.0 ~ 1.0)
    n_iter : int (Default : 50)
        Passes over the training dateset

    Attributes:
    w_ : 1d-array
        Weights after fitting
    cost_ : list
        Cost of misclassifications in every epochs.
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = []
        self.cost_ = []

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
        self.w_ = np.zeros(1 + X.shape[1])

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
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

    def activation(self, X):
        """
        Compute linear activation

        Paraters:
        X : array-like
            Training sample(s)

        Returns:
        Net input of X
        """
        return self.net_input(X)

    def predict(self, X):
        """
        Predict class label after unit step

        Paraters:
        X : array-like
            Training sample(s)

        Returns:
        Class label
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)
