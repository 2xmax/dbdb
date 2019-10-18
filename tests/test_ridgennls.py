from unittest import TestCase

import numpy as np
import scipy

from ridge_nnls import RidgeNNLS


class TestRidgeNNLS(TestCase):

    def __init__(self, method_name: str = ...) -> None:
        super().__init__(method_name)

    def assertItemsAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        for a, b in zip(first, second):
            if msg is None:
                diff = np.argwhere(np.abs(np.array(first) - np.array(second)) > (1e-5 if delta is None else delta))
                msg = str(first) + "vs. \n " + str(second) + "\n Mismatch at " + str(diff)
            self.assertAlmostEqual(a, b, msg=msg, delta=delta)

    def test_zero_alpha(self):
        X = np.array([[1, 0],
                      [1, 0],
                      [0, 1]])
        y = np.array([2, 1, 1])
        w_nnls = scipy.optimize.nnls(X, y)[0]
        clf = RidgeNNLS(alphas=None).fit(X, y, alpha=0)
        w_ridge_nnls = clf.coef_
        self.assertItemsAlmostEqual(w_nnls, w_ridge_nnls, delta=1e-8)

        # add shrinkage penalty to the solution vector
        clf = RidgeNNLS(alphas=None).fit(X, y, alpha=1)
        w_ridge_nnls = clf.coef_
        self.assertLess(np.linalg.norm(w_ridge_nnls), np.linalg.norm(w_nnls))

    def test_gcv(self):
        X = np.array([[1, 0],
                      [1, 0],
                      [0, 1]])
        y = np.array([2, 1, 1])
        alphas = np.linspace(0, 10, 101)
        clf = RidgeNNLS(alphas=alphas).fit(X, y)
        self.assertAlmostEqual(0.4, clf.alpha_, delta=1e-8)
