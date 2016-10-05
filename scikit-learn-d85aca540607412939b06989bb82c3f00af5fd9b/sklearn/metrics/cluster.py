"""Utilities to evaluate the clustering performance of models

Functions named as *_score return a scalar value to maximize: the higher the
better.
"""

# Authors: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD Style.

from math import log
from scipy.misc import comb

import numpy as np


# the exact version if faster for k == 2: use it by default globally in
# this module instead of the float approximate variant
def comb2(n):
    return comb(n, 2, exact=1)


def check_clusterings(labels_true, labels_pred):
    """Check that the two clusterings matching 1D integer arrays"""
    labels_true = np.asanyarray(labels_true)
    labels_pred = np.asanyarray(labels_pred)

    # input checks
    if labels_true.ndim != 1:
        raise ValueError(
            "labels_true must be 1D: shape is %r" % (labels_true.shape,))
    if labels_pred.ndim != 1:
        raise ValueError(
            "labels_pred must be 1D: shape is %r" % (labels_pred.shape,))
    if labels_true.shape != labels_pred.shape:
        raise ValueError(
            "labels_true and labels_pred must have same size, got %d and %d"
            % (labels_true.shape[0], labels_pred.shape[0]))
    return labels_true, labels_pred


# clustering measures

def adjusted_rand_score(labels_true, labels_pred):
    """Rand index adjusted for chance

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are
    assigned in the same or different clusters in the predicted and
    true clusterings.

    The raw RI score is then "adjusted for chance" into the ARI score
    using the following scheme::

        ARI = (RI - Expected_RI) / (max(RI) - Expected_RI)

    The adjusted Rand index is thus ensured to have a value close to
    0.0 for random labeling independently of the number of clusters and
    samples and exactly 1.0 when the clusterings are identical (up to
    a permutation).

    ARI is a symmetric measure::

        adjusted_rand_score(a, b) == adjusted_rand_score(b, a)

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    Returns
    -------
    ari: float
       Similarity score between -1.0 and 1.0. Random labelings have an ARI
       close to 0.0. 1.0 stands for perfect match.

    Example
    -------

    Perfectly maching labelings have a score of 1 even

      >>> from sklearn.metrics.cluster import adjusted_rand_score
      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete be not always pure, hence penalized::

      >>> adjusted_rand_score([0, 0, 1, 2], [0, 0, 1, 1])  # doctest: +ELLIPSIS
      0.57...

    ARI is symmetric, so labelings that have pure clusters with members
    coming from the same classes but unnecessary splits are penalized::

      >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 2])  # doctest: +ELLIPSIS
      0.57...

    If classes members are completely split across different clusters, the
    assignment is totally incomplete, hence the ARI is very low::

      >>> adjusted_rand_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    References
    ----------
    - L. Hubert and P. Arabie, Comparing Partitions,
      Journal of Classification 1985
      http://www.springerlink.com/content/x64124718341j1j0/

    - http://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

    See also
    --------
    - ami_score: Adjusted Mutual Information (TODO: implement me!)

    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]

    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Special limit cases: no clustering since the data is not split.
    # This is a perfect match hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0):
        return 1.0

    # The cluster and class ids are not necessarily consecutive integers
    # starting at 0 hence build a map
    class_idx = dict((k, v) for v, k in enumerate(classes))
    cluster_idx = dict((k, v) for v, k in enumerate(clusters))

    # Build the contingency table
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    contingency = np.zeros((n_classes, n_clusters), dtype=np.int)

    for c, k in zip(labels_true, labels_pred):
        contingency[class_idx[c], cluster_idx[k]] += 1

    # Compute the ARI using the contingency data
    sum_comb_c = sum(comb2(n_c) for n_c in contingency.sum(axis=1))
    sum_comb_k = sum(comb2(n_k) for n_k in contingency.sum(axis=0))

    sum_comb = sum(comb2(n_ij) for n_ij in contingency.flatten())
    prod_comb = (sum_comb_c * sum_comb_k) / float(comb(n_samples, 2))
    mean_comb = (sum_comb_k + sum_comb_c) / 2.
    return ((sum_comb - prod_comb) / (mean_comb - prod_comb))


def homogeneity_completeness_v_measure(labels_true, labels_pred):
    """Compute the homogeneity and completeness and V-measure scores at once

    Those metrics are based on normalized conditional entropy measures of
    the clustering labeling to evaluate given the knowledge of a Ground
    Truth class labels of the same samples.

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.

    Those 3 metrics are independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score values in any way.

    V-Measure is furthermore symmetric: swapping `labels_true` and
    `label_pred` will give the same score. This does not hold for
    homogeneity and completeness.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    Returns
    -------
    homogeneity: float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

    completeness: float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    v_measure: float
        harmonic mean of the first two

    See also
    --------
    - homogeneity_score
    - completeness_score
    - v_measure_score
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]

    entropy_K_given_C = 0.
    entropy_C_given_K = 0.
    entropy_C = 0.
    entropy_K = 0.

    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    n_C = [float(np.sum(labels_true == c)) for c in classes]
    n_K = [float(np.sum(labels_pred == k)) for k in clusters]

    for i in xrange(len(classes)):
        entropy_C -= n_C[i] / n_samples * log(n_C[i] / n_samples)

    for j in xrange(len(clusters)):
        entropy_K -= n_K[j] / n_samples * log(n_K[j] / n_samples)

    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            # count samples at the intersection of class c and cluster k
            n_CK = float(np.sum((labels_true == c) * (labels_pred == k)))

            if n_CK != 0.0:
                # turn label assignments into contribution to entropies
                entropy_C_given_K -= n_CK / n_samples * log(n_CK / n_K[j])
                entropy_K_given_C -= n_CK / n_samples * log(n_CK / n_C[i])

    homogeneity = 1.0 - entropy_C_given_K / entropy_C if entropy_C else 1.0
    completeness = 1.0 - entropy_K_given_C / entropy_K if entropy_K else 1.0

    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = (2.0 * homogeneity * completeness
                           / (homogeneity + completeness))

    return homogeneity, completeness, v_measure_score


def homogeneity_score(labels_true, labels_pred):
    """Homogeneity metric of a cluster labeling given a ground truth

    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching `label_true` with `label_pred`
    will return the completeness_score which will be different in general.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    Returns
    -------
    homogeneity: float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

    References
    ----------
    V-Measure: A conditional entropy-based external cluster evaluation measure
    Andrew Rosenberg and Julia Hirschberg, 2007
    http://acl.ldc.upenn.edu/D/D07/D07-1043.pdf

    See also
    --------
    - completeness_score
    - v_measure_score

    Examples
    --------

    Perfect labelings are homegenous::

      >>> from sklearn.metrics.cluster import homogeneity_score
      >>> homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-pefect labelings that futher split classes into more clusters can be
    perfectly homogeneous:

      >>> homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2])
      1.0
      >>> homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3])
      1.0

    Clusters that include samples from different classes do not make for an
    homogeneous labeling::

      >>> homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1])
      0.0
      >>> homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0])
      0.0

    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[0]


def completeness_score(labels_true, labels_pred):
    """Completeness metric of a cluster labeling given a ground truth

    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is not symmetric: switching `label_true` with `label_pred`
    will return the homogeneity_score which will be different in general.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    Returns
    -------
    completeness: float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    References
    ----------
    V-Measure: A conditional entropy-based external cluster evaluation measure
    Andrew Rosenberg and Julia Hirschberg, 2007
    http://acl.ldc.upenn.edu/D/D07/D07-1043.pdf

    See also
    --------
    - homogeneity_score
    - v_measure_score

    Examples
    --------

    Perfect labelings are complete::

      >>> from sklearn.metrics.cluster import completeness_score
      >>> completeness_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Non-pefect labelings that assign all classes members to the same clusters
    are still complete:

      >>> completeness_score([0, 0, 1, 1], [0, 0, 0, 0])
      1.0
      >>> completeness_score([0, 1, 2, 3], [0, 0, 1, 1])
      1.0

    If classes members are splitted accross different clusters, the
    assignment cannot be complete::

      >>> completeness_score([0, 0, 1, 1], [0, 1, 0, 1])
      0.0
      >>> completeness_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[1]


def v_measure_score(labels_true, labels_pred):
    """V-Measure cluster labeling given a ground truth

    The V-Measure is the hormonic mean between homogeneity and completeness:

      v = 2 * (homogeneity * completeness) / (homogeneity + completeness)

    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.

    This metric is furthermore symmetric: switching `label_true` with
    `label_pred` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        cluster labels to evaluate

    Returns
    -------
    completeness: float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    References
    ----------
    V-Measure: A conditional entropy-based external cluster evaluation measure
    Andrew Rosenberg and Julia Hirschberg, 2007
    http://acl.ldc.upenn.edu/D/D07/D07-1043.pdf

    See also
    --------
    - homogeneity_score
    - completeness_score

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have score 1.0::

      >>> from sklearn.metrics.cluster import v_measure_score
      >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> v_measure_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    Labelings that assign all classes members to the same clusters
    are complete be not homogeneous, hence penalized::

      >>> v_measure_score([0, 0, 1, 2], [0, 0, 1, 1])     # doctest: +ELLIPSIS
      0.8...
      >>> v_measure_score([0, 1, 2, 3], [0, 0, 1, 1])     # doctest: +ELLIPSIS
      0.66...

    Labelings that have pure clusters with members coming from the same
    classes are homogeneous but un-necessary splits harms completeness
    and thus penalize V-measure as well::

      >>> v_measure_score([0, 0, 1, 1], [0, 0, 1, 2])     # doctest: +ELLIPSIS
      0.8...
      >>> v_measure_score([0, 0, 1, 1], [0, 1, 2, 3])     # doctest: +ELLIPSIS
      0.66...

    If classes members are completly splitted accross different clusters,
    the assignment is totally in-complete, hence the v-measure is null::

      >>> v_measure_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    Clusters that include samples from totally different classes totally
    destroy the homogeneity of the labeling, hence::

      >>> v_measure_score([0, 0, 1, 1], [0, 0, 0, 0])
      0.0

    """
    return homogeneity_completeness_v_measure(labels_true, labels_pred)[2]
