"""
Generalized Linear models.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Vincent Michel <vincent.michel@inria.fr>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD Style.

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator
from ..base import RegressorMixin
from ..base import ClassifierMixin
from ..base import TransformerMixin
from .sgd_fast import Hinge, Log, ModifiedHuber, SquaredLoss, Huber
from ..utils.extmath import safe_sparse_dot
from ..utils import as_float_array, safe_asanyarray


###
### TODO: intercept for all models
### We should define a common function to center data instead of
### repeating the same code inside each fit method.
###
### Also, bayesian_ridge_regression and bayesian_regression_ard
### should be squashed into its respective objects.
###

class LinearModel(BaseEstimator, RegressorMixin):
    """Base class for Linear Models"""

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        X = safe_asanyarray(X)
        return safe_sparse_dot(X, self.coef_.T) + self.intercept_

    @staticmethod
    def _center_data(X, y, fit_intercept, normalize=False, copy=True):
        """
        Centers data to have mean zero along axis 0. This is here because
        nearly all linear models will want their data to be centered.

        If copy is False, modifies X in-place.
        """
        X = as_float_array(X, copy)

        if fit_intercept:
            if sp.issparse(X):
                X_mean = np.zeros(X.shape[1])
                X_std = np.ones(X.shape[1])
            else:
                X_mean = X.mean(axis=0)
                X -= X_mean
                if normalize:
                    X_std = np.sqrt(np.sum(X ** 2, axis=0))
                    X_std[X_std==0] = 1
                    X /= X_std
                else:
                    X_std = np.ones(X.shape[1])
            y_mean = y.mean()
            y = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            X_std = np.ones(X.shape[1])
            y_mean = 0.
        return X, y, X_mean, y_mean, X_std

    def _set_intercept(self, X_mean, y_mean, X_std):
        """Set the intercept_
        """
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_std
            self.intercept_ = y_mean - np.dot(X_mean, self.coef_.T)
        else:
            self.intercept_ = 0


class LinearRegression(LinearModel):
    """
    Ordinary least squares Linear Regression.

    Attributes
    ----------
    `coef_` : array
        Estimated coefficients for the linear regression problem.

    `intercept_` : array
        Independent term in the linear model.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (numpy.linalg.lstsq) wrapped as a predictor object.

    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

    def fit(self, X, y):
        """
        Fit linear model.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples]
            Target values
        fit_intercept : boolean, optional
            wether to calculate the intercept for this model. If set
            to false, no intercept will be used in calculations
            (e.g. data is expected to be already centered).
        normalize : boolean, optional
            If True, the regressors X are normalized

        Returns
        -------
        self : returns an instance of self.
        """
        X = np.asanyarray(X)
        y = np.asanyarray(y)

        X, y, X_mean, y_mean, X_std = self._center_data(X, y,
                self.fit_intercept, self.normalize, self.copy_X)

        self.coef_, self.residues_, self.rank_, self.singular_ = \
                np.linalg.lstsq(X, y)

        self._set_intercept(X_mean, y_mean, X_std)
        return self

##
## Stochastic Gradient Descent (SGD) abstract base classes
##


class BaseSGD(BaseEstimator):
    """Base class for dense and sparse SGD."""

    def __init__(self, loss, penalty='l2', alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, seed=0, learning_rate="optimal", eta0=0.0,
                 power_t=0.5):
        self.loss = str(loss)
        self.penalty = str(penalty)
        self.alpha = float(alpha)
        self.rho = float(rho)
        self.fit_intercept = bool(fit_intercept)
        self.n_iter = int(n_iter)
        if self.n_iter <= 0:
            raise ValueError("n_iter must be greater than zero.")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be either True or False")
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.verbose = int(verbose)
        self._set_loss_function(self.loss)
        self._set_penalty_type(self.penalty)

        self.learning_rate = str(learning_rate)
        self._set_learning_rate(self.learning_rate)
        self.eta0 = float(eta0)
        self.power_t = float(power_t)
        if self.learning_rate != "optimal":
            if eta0 <= 0.0:
                raise ValueError("eta0 must be greater than 0.0")

    def _set_learning_rate(self, learning_rate):
        learning_rate_codes = {"constant": 1, "optimal": 2, "invscaling": 3}
        try:
            self.learning_rate_code = learning_rate_codes[learning_rate]
        except KeyError:
            raise ValueError("learning rate %s"
            "is not supported. " % learning_rate)

    def _set_loss_function(self, loss):
        """Get concrete LossFunction"""
        raise NotImplementedError("BaseSGD is an abstract class.")

    def _set_penalty_type(self, penalty):
        penalty_types = {"l2": 2, "l1": 1, "elasticnet": 3}
        try:
            self.penalty_type = penalty_types[penalty]
            if self.penalty_type == 2:
                self.rho = 1.0
            elif self.penalty_type == 1:
                self.rho = 0.0
        except KeyError:
            raise ValueError("Penalty %s is not supported. " % penalty)

    def _set_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight == None:
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            sample_weight = np.asanyarray(sample_weight, dtype=np.float64,
                                          order="C")
        self.sample_weight = sample_weight
        if self.sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")

    def _allocate_parameter_mem(self, n_classes, n_features, coef_init=None,
                                intercept_init=None):
        """Allocate mem for parameters; initialize if provided."""
        if n_classes > 2:
            # allocate coef_ for multi-class
            if coef_init is not None:
                coef_init = np.asanyarray(coef_init)
                if coef_init.shape != (n_classes, n_features):
                    raise ValueError("Provided coef_ does not match dataset. ")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros((n_classes, n_features),
                                      dtype=np.float64, order="C")

            # allocate intercept_ for multi-class
            if intercept_init is not None:
                intercept_init = np.asanyarray(intercept_init)
                if intercept_init.shape != (n_classes, ):
                    raise ValueError("Provided intercept_init " \
                                     "does not match dataset.")
                self.intercept_ = intercept_init
            else:
                self.intercept_ = np.zeros(n_classes, dtype=np.float64,
                                           order="C")
        else:
            # allocate coef_ for binary problem
            if coef_init is not None:
                coef_init = np.asanyarray(coef_init, dtype=np.float64,
                                          order="C")
                coef_init = coef_init.ravel()
                if coef_init.shape != (n_features,):
                    raise ValueError("Provided coef_init does not " \
                                     "match dataset.")
                self.coef_ = coef_init
            else:
                self.coef_ = np.zeros(n_features, dtype=np.float64, order="C")

            # allocate intercept_ for binary problem
            if intercept_init is not None:
                intercept_init = np.asanyarray(intercept_init,
                                               dtype=np.float64)
                if intercept_init.shape != (1,) and intercept_init.shape !=():
                    raise ValueError("Provided intercept_init " \
                                 "does not match dataset.")
                self.intercept_ = intercept_init.reshape(1,)
            else:
                self.intercept_ = np.zeros(1, dtype=np.float64, order="C")


class BaseSGDClassifier(BaseSGD, ClassifierMixin):
    """Base class for dense and sparse classification using SGD.
    """

    def __init__(self, loss="hinge", penalty='l2', alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, n_jobs=1, seed=0, learning_rate="optimal",
                 eta0=0.0, power_t=0.5):
        super(BaseSGDClassifier, self).__init__(loss=loss, penalty=penalty,
                                                alpha=alpha, rho=rho,
                                                fit_intercept=fit_intercept,
                                                n_iter=n_iter, shuffle=shuffle,
                                                verbose=verbose, seed=seed,
                                                learning_rate=learning_rate,
                                                eta0=eta0, power_t=power_t)
        self.n_jobs = int(n_jobs)

    def _set_loss_function(self, loss):
        """Set concrete LossFunction."""
        loss_functions = {
            "hinge": Hinge(),
            "log": Log(),
            "modified_huber": ModifiedHuber(),
        }
        try:
            self.loss_function = loss_functions[loss]
        except KeyError:
            raise ValueError("The loss %s is not supported. " % loss)

    def _set_class_weight(self, class_weight, classes, y):
        """Estimate class weights for unbalanced datasets."""
        class_weight = {} if class_weight is None else class_weight
        if class_weight == {}:
            weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
        elif class_weight == 'auto':
            weight = np.array([1.0 / np.sum(y == i) for i in classes],
                              dtype=np.float64, order='C')
            weight *= classes.shape[0] / np.sum(weight)
        else:
            weight = np.ones(classes.shape[0], dtype=np.float64, order='C')
            if not isinstance(class_weight, dict):
                raise ValueError("class_weight must be dict, 'auto', or None.")
            for c in class_weight:
                i = np.searchsorted(classes, c)
                if classes[i] != c:
                    raise ValueError("Class label %d not present." % c)
                else:
                    weight[i] = class_weight[c]

        self.class_weight = weight

    def fit(self, X, y, coef_init=None, intercept_init=None,
            class_weight=None, sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        coef_init : array, shape = [n_classes,n_features]
            The initial coeffients to warm-start the optimization.

        intercept_init : array, shape = [n_classes]
            The initial intercept to warm-start the optimization.

        class_weight : dict, {class_label : weight} or "auto"
            Weights associated with classes. If not given, all classes
            are supposed to have weight one.

            The "auto" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : returns an instance of self.
        """

        # check only y because X might be dense or sparse
        y = np.asanyarray(y, dtype=np.float64, order='C')

        # make sure X has shape
        try:
            n_samples, n_features = X.shape
        except AttributeError:
            X = np.asanyarray(X)
            n_samples, n_features = X.shape

        if n_samples != len(y):
            raise ValueError("Shapes of X and y do not match.")

        # sort in asc order; largest class id is positive class
        self.classes = np.unique(y)
        n_classes = self.classes.shape[0]

        # Allocate datastructures from input arguments
        self._set_class_weight(class_weight, self.classes, y)
        self._set_sample_weight(sample_weight, n_samples)
        self._allocate_parameter_mem(n_classes, n_features,
                                     coef_init, intercept_init)

        # delegate to concrete training procedure
        if n_classes > 2:
            self._fit_multiclass(X, y)
        elif n_classes == 2:
            self._fit_binary(X, y)
        else:
            raise ValueError("The number of class labels must be "
                             "greater than one.")
        # return self for chaining fit and predict calls
        return self

    def _fit_binary(self, X, y):
        raise NotImplementedError("BaseSGDClassifier is an abstract class.")

    def _fit_multiclass(self, X, y):
        raise NotImplementedError("BaseSGDClassifier is an abstract class.")

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : array or scipy.sparse matrix of shape [n_samples, n_features]
           Whether the numpy.array or scipy.sparse matrix is accepted dependes
           on the actual implementation

        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """
        scores = self.decision_function(X)
        if self.classes.shape[0] == 2:
            indices = np.array(scores > 0, dtype=np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes[np.ravel(indices)]

    def predict_proba(self, X):
        """Predict class membership probability

        Parameters
        ----------
        X : array or scipy.sparse matrix of shape [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples] if n_classes == 2 else [n_samples,
        n_classes]
            Contains the membership probabilities of the positive class.

        """
        if (isinstance(self.loss_function, Log) and
            self.classes.shape[0] == 2):
            return 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        else:
            raise NotImplementedError("predict_(log_)proba only supported when"
                                      " loss='log' (%s given)" % self.loss)


class BaseSGDRegressor(BaseSGD, RegressorMixin):
    """Base class for dense and sparse regression using SGD.
    """
    def __init__(self, loss="squared_loss", penalty="l2", alpha=0.0001,
                 rho=0.85, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, p=0.1, seed=0, learning_rate="invscaling",
                 eta0=0.01, power_t=0.25):
        self.p = float(p)
        super(BaseSGDRegressor, self).__init__(loss=loss, penalty=penalty,
                                               alpha=alpha, rho=rho,
                                               fit_intercept=fit_intercept,
                                               n_iter=n_iter, shuffle=shuffle,
                                               verbose=verbose, seed=seed,
                                               learning_rate=learning_rate,
                                               eta0=eta0, power_t=power_t)

    def _set_loss_function(self, loss):
        """Get concrete LossFunction"""
        loss_functions = {
            "squared_loss": SquaredLoss(),
            "huber": Huber(self.p),
        }
        try:
            self.loss_function = loss_functions[loss]
        except KeyError:
            raise ValueError("The loss %s is not supported. " % loss)

    def fit(self, X, y, coef_init=None, intercept_init=None,
            sample_weight=None):
        """Fit linear model with Stochastic Gradient Descent.

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        coef_init : array, shape = [n_features]
            The initial coeffients to warm-start the optimization.

        intercept_init : array, shape = [1]
            The initial intercept to warm-start the optimization.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : returns an instance of self.
        """
        y = np.asanyarray(y, dtype=np.float64, order="C")

        # make sure X has shape
        try:
            n_samples, n_features = X.shape
        except AttributeError:
            X = np.asanyarray(X)
            n_samples, n_features = X.shape

        if n_samples != len(y):
            raise ValueError("Shapes of X and y do not match.")

        # Allocate datastructures from input arguments
        self._set_sample_weight(sample_weight, n_samples)
        self._allocate_parameter_mem(1, n_features,
                                     coef_init, intercept_init)

        self._fit_regressor(X, y)
        return self

    def _fit_regressor(self, X, y):
        raise NotImplementedError("BaseSGDRegressor is an abstract class.")

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : array or scipy.sparse matrix of shape [n_samples, n_features]
           Whether the numpy.array or scipy.sparse matrix is accepted dependes
           on the actual implementation

        Returns
        -------
        array, shape = [n_samples]
           Array containing the predicted class labels.
        """
        X = np.asanyarray(X)
        return np.dot(X, self.coef_) + self.intercept_


class CoefSelectTransformerMixin(TransformerMixin):
    """Mixin for linear models that can find sparse solutions.
    """

    def transform(self, X, threshold=1e-10):
        if len(self.coef_.shape) == 1 or self.coef_.shape[1] == 1:
            # 2-class case
            coef = np.ravel(self.coef_)
        else:
            # multi-class case
            coef = np.mean(self.coef_, axis=0)

        return X[:, coef > threshold]
