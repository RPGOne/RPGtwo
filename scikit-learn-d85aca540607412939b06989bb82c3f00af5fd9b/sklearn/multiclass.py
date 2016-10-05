"""
Multiclass algorithms
======================

This module implements multiclass learning algorithms:
    - one-vs-the-rest / one-vs-all
    - one-vs-one
    - error correcting output codes

The estimators provided in this module are meta-estimators: they require a base
estimator to be provided in their constructor. For example, it is possible to
use these estimators to turn a binary classifier or a regressor into a
multiclass classifier. It is also possible to use these estimators with
multiclass estimators in the hope that their accuracy or runtime performance
improves.
"""

# Author: Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD Style.

import numpy as np

from .base import BaseEstimator, ClassifierMixin, clone
from .preprocessing import LabelBinarizer
from .metrics.pairwise import euclidean_distances
from .utils import check_random_state


def fit_binary(estimator, X, y):
    """Fit a single binary estimator."""
    estimator = clone(estimator)
    estimator.fit(X, y)
    return estimator


def predict_binary(estimator, X):
    """Make predictions using a single binary estimator."""
    if hasattr(estimator, "decision_function"):
        return np.ravel(estimator.decision_function(X))
    else:
        # probabilities of the positive class
        return estimator.predict_proba(X)[:, 1]


def check_estimator(estimator):
    """Make sure that an estimator implements the necessary methods."""
    if not hasattr(estimator, "decision_function") and \
       not hasattr(estimator, "predict_proba"):
        raise ValueError("The base estimator should implement "
                         "decision_function or predict_proba!")


def fit_ovr(estimator, X, y):
    """Fit a one-vs-the-rest strategy."""
    check_estimator(estimator)

    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    estimators = [fit_binary(estimator, X, Y[:, i]) for i in range(Y.shape[1])]
    return estimators, lb


def predict_ovr(estimators, label_binarizer, X):
    """Make predictions using the one-vs-the-rest strategy."""
    Y = np.array([predict_binary(e, X) for e in estimators]).T
    return label_binarizer.inverse_transform(Y)


class OneVsRestClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-the-rest multiclass strategy

    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy and is a
    fair default choice.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        y : array-like, shape = [n_samples]
            Multi-class targets.

        Returns
        -------
        self
        """
        self.estimators_, self.label_binarizer_ = fit_ovr(self.estimator, X, y)
        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : array-like, shape = [n_samples]
            Predicted multi-class targets.
        """
        if not hasattr(self, "estimators_"):
            raise ValueError("The object hasn't been fitted yet!")

        return predict_ovr(self.estimators_, self.label_binarizer_, X)


def fit_ovo_binary(estimator, X, y, i, j):
    """Fit a single binary estimator (one-vs-one)."""
    cond = np.logical_or(y == i, y == j)
    y = y[cond]
    y[y == i] = 0
    y[y == j] = 1
    ind = np.arange(X.shape[0])
    return fit_binary(estimator, X[ind[cond]], y)


def fit_ovo(estimator, X, y):
    """Fit a one-vs-one strategy."""
    classes = np.unique(y)
    n_classes = classes.shape[0]
    estimators = [fit_ovo_binary(estimator, X, y, classes[i], classes[j])
                    for i in range(n_classes) for j in range(i + 1, n_classes)]

    return estimators, classes


def predict_ovo(estimators, classes, X):
    """Make predictions using the one-vs-one strategy."""
    n_samples = X.shape[0]
    n_classes = classes.shape[0]
    votes = np.zeros((n_samples, n_classes))

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            pred = estimators[k].predict(X)
            votes[pred == 0, i] += 1
            votes[pred == 1, j] += 1
            k += 1

    return classes[votes.argmax(axis=1)]


class OneVsOneClassifier(BaseEstimator, ClassifierMixin):
    """One-vs-one multiclass strategy

    This strategy consists in fitting one classifier per class pair.
    At prediction time, the class which received the most votes is selected.
    Since it requires to fit `n_classes * (n_classes - 1) / 2` classifiers,
    this method is usually slower than one-vs-the-rest, due to its
    O(n_classes^2) complexity. However, this method may be advantageous for
    algorithms such as kernel algorithms which don't scale well with
    `n_samples`. This is because each individual learning problem only involves
    a small subset of the data whereas, with one-vs-the-rest, the complete
    dataset is used `n_classes` times.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and `predict`.

    Attributes
    ----------
    estimators_ : list of `n_classes * (n_classes - 1) / 2` estimators
        Estimators used for predictions.

    classes_ : numpy array of shape [n_classes]
        Array containing labels.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        y : numpy array of shape [n_samples]
            Multi-class targets.

        Returns
        -------
        self
        """
        self.estimators_, self.classes_ = fit_ovo(self.estimator, X, y)
        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        if not hasattr(self, "estimators_"):
            raise ValueError("The object hasn't been fitted yet!")

        return predict_ovo(self.estimators_, self.classes_, X)


def fit_ecoc(estimator, X, y, code_size=1.5, random_state=None):
    """
    Fit an error-correcting output-code strategy.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    code_size: float, optional
        Percentage of the number of classes to be used to create the code book.

    random_state: numpy.RandomState, optional
        The generator used to initialize the codebook. Defaults to numpy.random.


    Returns
    --------
    estimators : list of `int(n_classes * code_size)` estimators
        Estimators used for predictions.

    classes : numpy array of shape [n_classes]
        Array containing labels.

    code_book_: numpy array of shape [n_classes, code_size]
        Binary array containing the code of each class.
    """
    check_estimator(estimator)
    random_state = check_random_state(random_state)

    classes = np.unique(y)
    n_classes = classes.shape[0]
    code_size = int(n_classes * code_size)

    # FIXME: there are more elaborate methods than generating the codebook
    # randomly.
    code_book = random_state.random_sample((n_classes, code_size))
    code_book[code_book > 0.5] = 1

    if hasattr(estimator, "decision_function"):
        code_book[code_book != 1] = -1
    else:
        code_book[code_book != 1] = 0

    cls_idx = dict((c, i) for i, c in enumerate(classes))

    Y = np.array([code_book[cls_idx[y[i]]] for i in xrange(X.shape[0])])

    estimators = [fit_binary(estimator, X, Y[:, i])
                                for i in range(Y.shape[1])]

    return estimators, classes, code_book


def predict_ecoc(estimators, classes, code_book, X):
    """Make predictions using the error-correcting output-code strategy."""
    Y = np.array([predict_binary(e, X) for e in estimators]).T
    pred = euclidean_distances(Y, code_book).argmin(axis=1)
    return classes[pred]


class OutputCodeClassifier(BaseEstimator, ClassifierMixin):
    """(Error-Correcting) Output-Code multiclass strategy

    Output-code based strategies consist in representing each class with a
    binary code (an array of 0s and 1s). At fitting time, one binary
    classifier per bit in the code book is fitted.  At prediction time, the
    classifiers are used to project new points in the class space and the class
    closest to the points is chosen. The main advantage of these strategies is
    that the number of classifiers used can be controlled by the user, either
    for compressing the model (0 < code_size < 1) or for making the model more
    robust to errors (code_size > 1). See the documentation for more details.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.

    code_size: float
        Percentage of the number of classes to be used to create the code book.
        A number between 0 and 1 will require fewer classifiers than
        one-vs-the-rest. A number greater than 1 will require more classifiers
        than one-vs-the-rest.

    random_state: numpy.RandomState, optional
        The generator used to initialize the codebook. Defaults to numpy.random.

    Attributes
    ----------
    estimators_ : list of `int(n_classes * code_size)` estimators
        Estimators used for predictions.

    classes_ : numpy array of shape [n_classes]
        Array containing labels.

    code_book_: numpy array of shape [n_classes, code_size]
        Binary array containing the code of each class.

    References
    ----------
     * [1] "Solving multiclass learning problems via error-correcting ouput
        codes",
        Dietterich T., Bakiri G.,
        Journal of Artificial Intelligence Research 2,
        1995.

     * [2] "The error coding method and PICTs",
        James G., Hastie T.,
        Journal of Computational and Graphical statistics 7,
        1998.

     * [3] "The Elements of Statistical Learning",
        Hastie T., Tibshirani R., Friedman J., page 606 (second-edition)
        2008.
    """

    def __init__(self, estimator, code_size=1.5, random_state=None):
        if (code_size <= 0):
            raise ValueError("code_size should be greater than 0!")

        self.estimator = estimator
        self.code_size = code_size
        self.random_state = random_state

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        y : numpy array of shape [n_samples]
            Multi-class targets.

        Returns
        -------
        self
        """
        self.estimators_, self.classes_, self.code_book_ = \
            fit_ecoc(self.estimator, X, y, self.code_size, self.random_state)
        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        y : numpy array of shape [n_samples]
            Predicted multi-class targets.
        """
        if not hasattr(self, "estimators_"):
            raise ValueError("The object hasn't been fitted yet!")

        return predict_ecoc(self.estimators_, self.classes_,
                            self.code_book_, X)
