""" Transformers to perform common preprocessing steps.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD

import numpy as np
import scipy.sparse as sp

from ..utils import check_arrays
from ..base import BaseEstimator, TransformerMixin

from ._preprocessing import inplace_csr_row_normalize_l1
from ._preprocessing import inplace_csr_row_normalize_l2


def _mean_and_std(X, axis=0, with_mean=True, with_std=True):
    """Compute mean and std dev for centering, scaling

    Zero valued std components are reset to 1.0 to avoid NaNs when scaling.
    """
    X = np.asanyarray(X)
    Xr = np.rollaxis(X, axis)

    if with_mean:
        mean_ = Xr.mean(axis=0)
    else:
        mean_ = None

    if with_std:
        std_ = Xr.std(axis=0)
        if isinstance(std_, np.ndarray):
            std_[std_ == 0.0] = 1.0
        elif std_ == 0.:
            std_ = 1.
    else:
        std_ = None

    return mean_, std_


def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    """Standardize a dataset along any axis

    Center to the mean and component wise scale to unit variance.

    Parameters
    ----------
    X : array-like
        The data to center and scale.

    axis : int (0 by default)
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : boolean, True by default
        If True, center the data before scaling.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    See also
    --------
    :class:`sklearn.preprocessing.Scaler` to perform centering and
    scaling using the ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`)
    """
    if sp.issparse(X):
        raise NotImplementedError(
            "Scaling is not yet implement for sparse matrices")
    X = np.asanyarray(X)
    mean_, std_ = _mean_and_std(
        X, axis, with_mean=with_mean, with_std=with_std)
    if copy:
        X = X.copy()
    Xr = np.rollaxis(X, axis)
    if with_mean:
        Xr -= mean_
    if with_std:
        Xr /= std_
    return X


class Scaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to unit variance

    Centering and scaling happen indepently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual feature do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    Parameters
    ----------
    with_mean : boolean, True by default
        If True, center the data before scaling.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    Attributes
    ----------
    mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.

    std_ : array of floats with shape [n_features]
        The standard deviation for each feature in the training set.

    See also
    --------
    :func:`sklearn.preprocessing.scale` to perform centering and
    scaling without using the ``Transformer`` object oriented API

    :class:`sklearn.decomposition.RandomizedPCA` with `whiten=True`
    to further remove the linear correlation across features.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        if sp.issparse(X):
            raise NotImplementedError(
                "Scaling is not yet implement for sparse matrices")
        self.mean_, self.std_ = _mean_and_std(
            X, axis=0, with_mean=self.with_mean, with_std=self.with_std)
        return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        if sp.issparse(X):
            raise NotImplementedError(
                "Scaling is not yet implement for sparse matrices")
        X = np.asanyarray(X)
        if copy:
            X = X.copy()
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.std_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        copy = copy if copy is not None else self.copy
        if sp.issparse(X):
            raise NotImplementedError(
                "Scaling is not yet implement for sparse matrices")
        X = np.asanyarray(X)
        if copy:
            X = X.copy()
        if self.with_std:
            X *= self.std_
        if self.with_mean:
            X += self.mean_
        return X


def normalize(X, norm='l2', axis=1, copy=True):
    """Normalize a dataset along any axis

    Parameters
    ----------
    X : array or scipy.sparse matrix with shape [n_samples, n_features]
        The data to normalize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    norm : 'l1' or 'l2', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix and if axis is 1).

    See also
    --------
    :class:`sklearn.preprocessing.Normalizer` to perform normalization
    using the ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`)
    """
    if norm not in ('l1', 'l2'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = 'csc'
    elif axis == 1:
        sparse_format = 'csr'
    else:
        raise ValueError("'%d' is not a supported axis" % axis)

    X = check_arrays(X, sparse_format=sparse_format, copy=copy)[0]
    if axis == 0:
        X = X.T

    if sp.issparse(X):
        if norm == 'l1':
            inplace_csr_row_normalize_l1(X)
        elif norm == 'l2':
            inplace_csr_row_normalize_l2(X)
    else:
        if norm == 'l1':
            norms = np.abs(X).sum(axis=1)[:, np.newaxis]
            norms[norms == 0.0] = 1.0
        elif norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1))[:, np.newaxis]
            norms[norms == 0.0] = 1.0
        X /= norms

    if axis == 0:
        X = X.T

    return X


class Normalizer(BaseEstimator, TransformerMixin):
    """Normalize samples individually to unit norm

    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1 or l2) equals one.

    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (use CSR format if you want to avoid the burden of
    a copy / conversion).

    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    Parameters
    ----------
    norm : 'l1' or 'l2', optional ('l2' by default)
        The norm to use to normalize each non zero sample.

    copy : boolean, optional, default is True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).

    Note
    ----
    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    See also
    --------
    :func:`sklearn.preprocessing.normalize` equivalent function
    without the object oriented API
    """

    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self

    def transform(self, X, y=None, copy=None):
        """Scale each non zero row of X to unit norm

        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        """
        copy = copy if copy is not None else self.copy
        return normalize(X, norm=self.norm, axis=1, copy=copy)


def binarize(X, threshold=0.0, copy=True):
    """Boolean thresholding of array-like or scipy.sparse matrix

    Parameters
    ----------
    X : array or scipy.sparse matrix with shape [n_samples, n_features]
        The data to binarize, element by element.
        scipy.sparse matrices should be in CSR format to avoid an
        un-necessary copy.

    threshold : float, optional (0.0 by default)
        The lower bound that triggers feature values to be replaced by 1.0.

    copy : boolean, optional, default is True
        set to False to perform inplace binarization and avoid a copy
        (if the input is already a numpy array or a scipy.sparse CSR
        matrix and if axis is 1).

    See also
    --------
    :class:`sklearn.preprocessing.Binarizer` to perform binarization
    using the ``Transformer`` API (e.g. as part of a preprocessing
    :class:`sklearn.pipeline.Pipeline`)
    """
    X = check_arrays(X, sparse_format='csr', copy=copy)[0]
    if sp.issparse(X):
        cond = X.data > threshold
        not_cond = np.logical_not(cond)
        X.data[cond] = 1
        # FIXME: if enough values became 0, it may be worth changing
        #        the sparsity structure
        X.data[not_cond] = 0
    else:
        cond = X > threshold
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0
    return X


class Binarizer(BaseEstimator, TransformerMixin):
    """Binarize data (set feature values to 0 or 1) according to a threshold

    The default threshold is 0.0 so that any non-zero values are set to 1.0
    and zeros are left untouched.

    Binarization is a common operation on text count data where the
    analyst can decide to only consider the presence or absence of a
    feature rather than a quantified number of occurences for instance.

    It can also be used as a pre-processing step for estimators that
    consider boolean random variables (e.g. modeled using the Bernoulli
    distribution in a Bayesian setting).

    Parameters
    ----------
    threshold : float, optional (0.0 by default)
        The lower bound that triggers feature values to be replaced by 1.0.

    copy : boolean, optional, default is True
        set to False to perform inplace binarization and avoid a copy (if
        the input is already a numpy array or a scipy.sparse CSR matrix).

    Notes
    -----
    If the input is a sparse matrix, only the non-zero values are subject
    to update by the Binarizer class.

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.
    """

    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.
        """
        return self

    def transform(self, X, y=None, copy=None):
        """Binarize each element of X

        Parameters
        ----------
        X : array or scipy.sparse matrix with shape [n_samples, n_features]
            The data to binarize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.
        """
        copy = copy if copy is not None else self.copy
        return binarize(X, threshold=self.threshold, copy=copy)


def _is_multilabel(y):
    return isinstance(y[0], tuple) or isinstance(y[0], list)


class LabelBinarizer(BaseEstimator, TransformerMixin):
    """Binarize labels in a one-vs-all fashion

    Several regression and binary classification algorithms are
    available in the scikit. A simple way to extend these algorithms
    to the multi-class classification case is to use the so-called
    one-vs-all scheme.

    At learning time, this simply consists in learning one regressor
    or binary classifier per class. In doing so, one needs to convert
    multi-class labels to binary labels (belong or does not belong
    to the class). LabelBinarizer makes this process easy with the
    transform method.

    At prediction time, one assigns the class for which the corresponding
    model gave the greatest confidence. LabelBinarizer makes this easy
    with the inverse_transform method.

    Attributes
    ----------
    classes_ : array of shape [n_class]
        Holds the label for each class.

    Examples
    --------
    >>> from sklearn import preprocessing
    >>> clf = preprocessing.LabelBinarizer()
    >>> clf.fit([1, 2, 6, 4, 2])
    LabelBinarizer()
    >>> clf.classes_
    array([1, 2, 4, 6])
    >>> clf.transform([1, 6])
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])

    >>> clf.fit_transform([(1, 2), (3,)])
    array([[ 1.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> clf.classes_
    array([1, 2, 3])
    """

    def fit(self, y):
        """Fit label binarizer

        Parameters
        ----------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Returns
        -------
        self : returns an instance of self.
        """
        self.multilabel = _is_multilabel(y)
        if self.multilabel:
            # concatenation of the sub-sequences
            self.classes_ = np.unique(reduce(lambda a, b: a + b, y))
        else:
            self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        """Transform multi-class labels to binary labels

        The output of transform is sometimes referred to by some authors as the
        1-of-K coding scheme.

        Parameters
        ----------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Returns
        -------
        Y : numpy array of shape [n_samples, n_classes]
        """

        if len(self.classes_) == 2:
            Y = np.zeros((len(y), 1))
        else:
            Y = np.zeros((len(y), len(self.classes_)))

        y_is_multilabel = _is_multilabel(y)

        if y_is_multilabel and not self.multilabel:
            raise ValueError("The object was not fitted with multilabel input!")

        elif self.multilabel:
            if not _is_multilabel(y):
                raise ValueError("y should be a list of label lists/tuples,"
                                 "got %r" % (y,))

            # inverse map: label => column index
            imap = dict((v, k) for k, v in enumerate(self.classes_))

            for i, label_tuple in enumerate(y):
                for label in label_tuple:
                    Y[i, imap[label]] = 1

            return Y

        elif len(self.classes_) == 2:
            Y[y == self.classes_[1], 0] = 1
            return Y

        elif len(self.classes_) >= 2:
            for i, k in enumerate(self.classes_):
                Y[y == k, i] = 1
            return Y

        else:
            # Only one class, returns a matrix with all 0s.
            return Y

    def inverse_transform(self, Y, threshold=0):
        """Transform binary labels back to multi-class labels

        Parameters
        ----------
        Y : numpy array of shape [n_samples, n_classes]
            Target values.

        threshold : float
            Threshold used to decide whether to assign the positive class or the
            negative class in the binary case. Use 0.5 when Y contains
            probabilities.

        Returns
        -------
        y : numpy array of shape [n_samples] or sequence of sequences
            Target values. In the multilabel case the nested sequences can
            have variable lengths.

        Note
        -----
        In the case when the binary labels are fractional
        (probabilistic), inverse_transform chooses the class with the
        greatest value. Typically, this allows to use the output of a
        linear model's decision_function method directly as the input
        of inverse_transform.
        """
        if self.multilabel:
            Y = np.array(Y > 0, dtype=int)
            return [tuple(self.classes_[np.flatnonzero(Y[i])])
                    for i in range(Y.shape[0])]

        if len(Y.shape) == 1 or Y.shape[1] == 1:
            y = np.array(Y.ravel() > threshold, dtype=int)

        else:
            y = Y.argmax(axis=1)

        return self.classes_[y]


class KernelCenterer(BaseEstimator, TransformerMixin):
    """Center a kernel matrix

    This is equivalent to centering phi(X) with
    sklearn.preprocessing.Scaler(with_std=False).
    """

    def fit(self, K):
        """Fit KernelCenterer

        Parameters
        ----------
        K : numpy array of shape [n_samples, n_samples]
            Kernel matrix.

        Returns
        -------
        self : returns an instance of self.
        """
        n_samples = K.shape[0]
        self.K_fit_rows_ = np.sum(K, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        return self

    def transform(self, K, copy=True):
        """Center kernel

        Parameters
        ----------
        K : numpy array of shape [n_samples1, n_samples2]
            Kernel matrix.

        Returns
        -------
        K_new : numpy array of shape [n_samples1, n_samples2]
        """

        if copy:
            K = K.copy()

        K_pred_cols = (np.sum(K, axis=1) /
                       self.K_fit_rows_.shape[0])[:, np.newaxis]

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K
