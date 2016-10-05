"""
Maximum likelihood covariance estimator.

"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Virgile Fritsch <virgile.fritsch@inria.fr>
#
# License: BSD Style.

# avoid division truncation
from __future__ import division
import numpy as np
from scipy import linalg

from ..base import BaseEstimator
from ..utils.extmath import fast_logdet as exact_logdet


def log_likelihood(emp_cov, precision):
    """Computes the negative log_likelihood of the data

    Params
    ------
    emp_cov: 2D ndarray (n_features, n_features)
      Maximum Likelihood Estimator of covariance
    precision: 2D ndarray (n_features, n_features)
      The precision matrix of the covariance model to be tested

    """
    return -np.sum(emp_cov * precision) + exact_logdet(precision)


def empirical_covariance(X, assume_centered=False):
    """Computes the Maximum likelihood covariance estimator

    Parameters
    ----------
    X: 2D ndarray, shape (n_samples, n_features)
      Data from which to compute the covariance estimate

    assume_centered: Boolean
      If True, data are not centered before computation.
      Usefull to work with data whose mean is significantly equal to
      zero but is not exactly zero.
      If False, data are centered before computation.

    Returns
    -------
    covariance: 2D ndarray, shape (n_features, n_features)
      Empirical covariance (Maximum Likelihood Estimator)


    """
    X = np.asanyarray(X)
    if X.ndim == 1:
        X = np.atleast_2d(X).T

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    return covariance


class EmpiricalCovariance(BaseEstimator):
    """Maximum likelihood covariance estimator

    Parameters
    ----------
    store_precision : bool
        Specifies if the estimated precision is stored

    Attributes
    ----------
    `covariance_` : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix

    `precision_` : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    """
    def __init__(self, store_precision=True):
        self.store_precision = store_precision

    def _set_estimates(self, covariance):
        """Saves the covariance and precision estimates

        Storage is done accordingly to `self.store_precision`.
        Precision stored only if invertible.

        Params
        ------
        covariance: 2D ndarray, shape (n_features, n_features)
          Estimated covariance matrix to be stored, and from which
          the precision is computed.

        """
        covariance = np.atleast_2d(covariance)
        # set covariance
        self.covariance_ = covariance
        # set precision
        if self.store_precision:
            self.precision_ = linalg.pinv(covariance)
        else:
            self.precision_ = None

    def fit(self, X, assume_centered=False):
        """ Fits the Maximum Likelihood Estimator covariance model
        according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples
          and n_features is the number of features.

        assume_centered: Boolean
          If True, data are not centered before computation.
          Usefull to work with data whose mean is significantly equal to
          zero but is not exactly zero.
          If False, data are centered before computation.

        Returns
        -------
        self : object
            Returns self.

        """
        covariance = empirical_covariance(X, assume_centered=assume_centered)
        self._set_estimates(covariance)

        return self

    def score(self, X_test, assume_centered=False):
        """Computes the log-likelihood of a gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X_test : array-like, shape = [n_samples, n_features]
          Test data of which we compute the likelihood,
          where n_samples is the number of samples and n_features is
          the number of features.

        Returns
        -------
        res: float
          The likelihood of the data set with self.covariance_ as an estimator
          of its covariance matrix.

        """
        # compute empirical covariance of the test set
        test_cov = empirical_covariance(X_test,
                                        assume_centered=assume_centered)
        # compute log likelihood
        if self.store_precision:
            res = log_likelihood(test_cov, self.precision_)
        else:
            res = log_likelihood(test_cov, linalg.pinv(self.covariance_))

        return res

    def error_norm(self, comp_cov, norm='frobenius', scaling=True,
                   squared=True):
        """Computes the Mean Squared Error between two covariance estimators.
        (In the sense of the Frobenius norm)

        Parameters
        ----------
        comp_cov: array-like, shape = [n_features, n_features]
          The covariance which to be compared to.
        norm: str
          The type of norm used to compute the error. Available error types:
          - 'frobenius' (default): sqrt(tr(A^t.A))
          - 'spectral': sqrt(max(eigenvalues(A^t.A))
          where A is the error (comp_cov - self.covariance_).
        scaling: bool
          If True (default), the squared error norm is divided by n_features
          If False, the squared error norm is not rescaled
        squared: bool
          Whether to compute the squared error norm or the error norm.
          If True (default), the squared error norm is returned.
          If False, the error norm is returned.

        Returns
        -------
        The Mean Squared Error (in the sense of the Frobenius norm) between
        `self` and `comp_cov` covariance estimators.

        """
        # compute the error
        error = comp_cov - self.covariance_
        # compute the error norm
        if norm == "frobenius":
            squared_norm = np.sum(error ** 2)
        elif norm == "spectral":
            squared_norm = np.amax(linalg.svdvals(np.dot(error.T, error)))
        else:
            raise NotImplementedError(
                "Only spectral and frobenius norms are implemented")
        # optionaly scale the error norm
        if scaling:
            squared_norm = squared_norm / error.shape[0]
        # finally get either the squared norm or the norm
        if squared:
            result = squared_norm
        else:
            result = np.sqrt(squared_norm)

        return result
