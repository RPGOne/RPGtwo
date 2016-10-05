"""
Testing for grid search module (sklearn.grid_search)

"""
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import LinearSVC
from sklearn.svm.sparse import LinearSVC as SparseLinearSVC
from sklearn.metrics import f1_score


class MockClassifier(BaseEstimator):
    """Dummy classifier to test the cross-validation"""
    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        return self

    def predict(self, T):
        return T.shape[0]

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.
        else:
            score = 0.
        return score


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


def test_grid_search():
    """Test that the best estimator contains the right value for foo_param"""
    clf = MockClassifier()
    cross_validation = GridSearchCV(clf, {'foo_param': [1, 2, 3]})
    # make sure it selects the smallest parameter in case of ties
    assert_equal(cross_validation.fit(X, y).best_estimator.foo_param, 2)

    for i, foo_i in enumerate([1, 2, 3]):
        assert cross_validation.grid_scores_[i][0] == {'foo_param': foo_i}


def test_grid_search_error():
    """Test that grid search will capture errors on data with different
    length"""
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]})
    assert_raises(ValueError, cv.fit, X_[:180], y_)


def test_grid_search_sparse():
    """Test that grid search works with both dense and sparse matrices"""
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]})
    cv.fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator.C

    X_ = sp.csr_matrix(X_)
    clf = SparseLinearSVC()
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]})
    cv.fit(X_[:180], y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator.C

    assert np.mean(y_pred == y_pred2) >= .9
    assert_equal(C, C2)


def test_grid_search_sparse_score_func():
    X_, y_ = make_classification(n_samples=200, n_features=100, random_state=0)

    clf = LinearSVC()
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]}, score_func=f1_score)
    # XXX: set refit to False due to a random bug when True (default)
    cv.set_params(refit=False).fit(X_[:180], y_[:180])
    y_pred = cv.predict(X_[180:])
    C = cv.best_estimator.C

    X_ = sp.csr_matrix(X_)
    clf = SparseLinearSVC()
    cv = GridSearchCV(clf, {'C': [0.1, 1.0]}, score_func=f1_score)
    # XXX: set refit to False due to a random bug when True (default)
    cv.set_params(refit=False).fit(X_[:180], y_[:180])
    y_pred2 = cv.predict(X_[180:])
    C2 = cv.best_estimator.C

    assert_array_equal(y_pred, y_pred2)
    assert_equal(C, C2)
