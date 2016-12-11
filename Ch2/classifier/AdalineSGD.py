"""
[Stochastic Gradient Descent]
(Also called 'iterative' or 'online gradient descent')

Stochastic gradient descent (SGD) : Able to skip local minimum
    (Samples must in random order)

Input : X = [x1, ..., xm]
Output : y = 1 or -1

Weight : W = [w1, ..., wm] (Init : 0s or small random value)
Net Input:
    z = transpose(W) * X = (-threshold) * x0 + w1x1 + ... + wmxm (x0 = 1)

ΔW = η * (y(i) - o(z(i))) * x(i)
"""
import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    """
    Adaline Classifier(SGD)

    Parameters:
    eta : float
        Learning rate (0.0 ~ 1.0) (Default : 0.01)
    n_iter : int (Default : 10)
        Passes over the training dateset
    shuffle : bool (Default : True)
        Shuffles training data every epoch if True to prevent cycles
    random_state : int (Default : None)
        Set random state for shuffling and initializing the weights

    Attributes:
    w_ : 1d-array
        Weights after fitting
    cost_ : list
        Cost of misclassifications in every epochs.
    w_initialized : bool
        Weights is initialized
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.w_ = []
        self.cost_ = []

        if random_state:
            seed(random_state)

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
        self._initialize_weights(X.shape[1])
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """
        Fit training data w/o reinitializing the weights

        Parameters:
        X : array of array-like (shape = [n_samples, n_features])
            Training vectors, where n_samples is # of samples &
            n_features is # of features
        y : array-like (shape = [n_samples])
            Target values

        Returns:
        self : object
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """
        Shuffle training data

        Parameters:
        X : array-like
            Training vectors, where n_samples is # of samples &
            n_features is # of features
        y : array-like (shape = [n_samples])
            Target values

        Returns:
        shuffled X, y
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """
        Initialize weights to zeros

        Parameters:
        m : int
            # of features
        """
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """
        Apply Adaline learning rulte to update the weights

        Parameters:
        xi : array-like
            Training sample
        target : float
            Training target

        Returns:
        cost value
        """
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

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
