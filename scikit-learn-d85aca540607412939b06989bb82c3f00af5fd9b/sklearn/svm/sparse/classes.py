
from ...base import ClassifierMixin, RegressorMixin
from .base import SparseBaseLibSVM, SparseBaseLibLinear
from ...linear_model.sparse.base import CoefSelectTransformerMixin


class SVC(SparseBaseLibSVM, ClassifierMixin):
    """SVC for sparse matrices (csr).

    See :class:`sklearn.svm.SVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm.sparse import SVC
    >>> clf = SVC()
    >>> clf.fit(X, y)
    SVC(C=1.0, coef0=0.0, degree=3, gamma=0.5, kernel='rbf', probability=False,
      shrinking=True, tol=0.001)
    >>> print clf.predict([[-0.8, -1]])
    [ 1.]
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3):

        SparseBaseLibSVM.__init__(self, 'c_svc', kernel, degree, gamma, coef0,
                         tol, C, 0., 0.,
                         shrinking, probability)



class NuSVC (SparseBaseLibSVM, ClassifierMixin):
    """NuSVC for sparse matrices (csr).

    See :class:`sklearn.svm.NuSVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.array([1, 1, 2, 2])
    >>> from sklearn.svm.sparse import NuSVC
    >>> clf = NuSVC()
    >>> clf.fit(X, y)
    NuSVC(coef0=0.0, degree=3, gamma=0.5, kernel='rbf', nu=0.5, probability=False,
       shrinking=True, tol=0.001)
    >>> print clf.predict([[-0.8, -1]])
    [ 1.]
    """


    def __init__(self, nu=0.5, kernel='rbf', degree=3, gamma=0.0,
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3):

        SparseBaseLibSVM.__init__(self, 'nu_svc', kernel, degree,
                         gamma, coef0, tol, 0., nu, 0.,
                         shrinking, probability)




class SVR (SparseBaseLibSVM, RegressorMixin):
    """SVR for sparse matrices (csr)

    See :class:`sklearn.svm.SVR` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> from sklearn.svm.sparse import SVR
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = SVR(C=1.0, epsilon=0.2)
    >>> clf.fit(X, y)
    SVR(C=1.0, coef0=0.0, degree=3, epsilon=0.2, gamma=0.2, kernel='rbf', nu=0.5,
      probability=False, shrinking=True, tol=0.001)
    """

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=1e-3, C=1.0, nu=0.5, epsilon=0.1,
                 shrinking=True, probability=False):

        SparseBaseLibSVM.__init__(self, 'epsilon_svr', kernel,
                         degree, gamma, coef0, tol, C, nu,
                         epsilon, shrinking, probability)





class NuSVR (SparseBaseLibSVM, RegressorMixin):
    """NuSVR for sparse matrices (csr)

    See :class:`sklearn.svm.NuSVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).

    Examples
    --------
    >>> from sklearn.svm.sparse import NuSVR
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = NuSVR(nu=0.1, C=1.0)
    >>> clf.fit(X, y)
    NuSVR(C=1.0, coef0=0.0, degree=3, epsilon=0.1, gamma=0.2, kernel='rbf',
       nu=0.1, probability=False, shrinking=True, tol=0.001)
    """

    def __init__(self, nu=0.5, C=1.0, kernel='rbf', degree=3,
                 gamma=0.0, coef0=0.0, shrinking=True, epsilon=0.1,
                 probability=False, tol=1e-3):

        SparseBaseLibSVM.__init__(self, 'nu_svr', kernel,
                         degree, gamma, coef0, tol, C, nu,
                         epsilon, shrinking, probability)



class OneClassSVM (SparseBaseLibSVM):
    """NuSVR for sparse matrices (csr)

    See :class:`sklearn.svm.NuSVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).
    """

    def __init__(self, kernel='rbf', degree=3, gamma=0.0, coef0=0.0,
                 tol=1e-3, nu=0.5, shrinking=True,
                 probability=False):

        SparseBaseLibSVM.__init__(self, 'one_class', kernel, degree,
                         gamma, coef0, tol, 0.0, nu, 0.0,
                         shrinking, probability)

    def fit(self, X, class_weight={}, sample_weight=[]):
        super(OneClassSVM, self).fit(
            X, [], class_weight=class_weight, ample_weight=sample_weight)


class LinearSVC(SparseBaseLibLinear, ClassifierMixin,
                CoefSelectTransformerMixin):
    """
    Linear Support Vector Classification, Sparse Version

    Similar to SVC with parameter kernel='linear', but uses internally
    liblinear rather than libsvm, so it has more flexibility in the
    choice of penalties and loss functions and should be faster for
    huge datasets.

    See :class:`sklearn.svm.SVC` for a complete list of parameters

    Notes
    -----
    For best results, this accepts a matrix in csr format
    (scipy.sparse.csr), but should be able to convert from any array-like
    object (including other sparse representations).
    """
    pass
