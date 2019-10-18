"""
Solver of non-negative least squares regression with Tikhonov (ridge)
regularization using generalized cross-validation (Golub et al DOI:10.2307/1268518)
w_pred = argmin_w ||y - Xw|| + alpha ||w||^2_2 s.t. w_i >= 0
"""
import numpy as np
import scipy
import scipy.optimize


class RidgeNNLS:
    def __init__(self, alphas):
        self.alphas = alphas
        self.alpha_ = None
        self.coef_ = None
        self.cv_values_ = None
        self.residual = None

    def gcv(self, X, y):
        N = X.shape[0]
        p = X.shape[1]

        self.cv_values_ = np.zeros(len(self.alphas))

        for i in range(len(self.alphas)):
            w = self.get_coef(X, y, self.alphas[i])
            rss = np.linalg.norm(y - X @ w) ** 2
            H = X @ np.linalg.inv(X.T @ X + self.alphas[i] * np.identity(p)) @ X.T
            self.cv_values_[i] = N * rss / (N - np.trace(H)) ** 2

        self.alpha_ = self.alphas[np.argmin(self.cv_values_)]

    def get_coef(self, X, y, alpha):
        p = X.shape[1]
        alpha_I = np.sqrt(alpha) * np.identity(p)
        X_ext = np.concatenate((X, alpha_I))
        y_ext = np.concatenate((y, np.zeros(p)))
        w_sol, rnorm = scipy.optimize.nnls(X_ext, y_ext)
        self.residual = rnorm
        return w_sol

    def fit(self, X, y, alpha=None):
        if alpha is None:
            self.gcv(X, y)
            alpha = self.alpha_

        self.coef_ = self.get_coef(X, y, alpha)
        return self

    def predict(self, X):
        return X @ self.coef_
