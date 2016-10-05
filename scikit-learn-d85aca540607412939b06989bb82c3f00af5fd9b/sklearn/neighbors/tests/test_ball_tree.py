import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from scipy.spatial import cKDTree

from sklearn import neighbors

# Note: simple tests of BallTree.query() and BallTree.query_radius()
# are contained within the tests of test_neighbors.py


def test_ball_tree_query_radius(n_samples=100, n_features=10):
    X = 2 * np.random.random(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1E-15  # roundoff error can cause test to fail
    bt = neighbors.BallTree(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind = bt.query_radius(query_pt, r + eps)[0]
        i = np.where(rad <= r + eps)[0]

        ind.sort()
        i.sort()

        assert np.all(i == ind)


def test_ball_tree_query_radius_distance(n_samples=100, n_features=10):
    X = 2 * np.random.random(size=(n_samples, n_features)) - 1
    query_pt = np.zeros(n_features, dtype=float)

    eps = 1E-15  # roundoff error can cause test to fail
    bt = neighbors.BallTree(X, leaf_size=5)
    rad = np.sqrt(((X - query_pt) ** 2).sum(1))

    for r in np.linspace(rad[0], rad[-1], 100):
        ind, dist = bt.query_radius(query_pt, r + eps, return_distance=True)

        ind = ind[0]
        dist = dist[0]

        d = np.sqrt(((query_pt - X[ind]) ** 2).sum(1))

        assert_array_almost_equal(d, dist)


def test_ball_tree_pickle():
    import pickle
    X = np.random.random(size=(10, 3))
    bt1 = neighbors.BallTree(X, leaf_size=1)
    ind1, dist1 = bt1.query(X)
    for protocol in (0, 1, 2):
        s = pickle.dumps(bt1, protocol=protocol)
        bt2 = pickle.loads(s)
        ind2, dist2 = bt2.query(X)
        assert np.all(ind1 == ind2)
        assert_array_almost_equal(dist1, dist2)

def test_ball_tree_p_distance():
    X = np.random.random(size=(100, 5))

    for p in (1, 2, 3, 4, np.inf):
        bt = neighbors.BallTree(X, leaf_size=10, p=p)
        kdt = cKDTree(X, leafsize=10)

        dist_bt, ind_bt = bt.query(X, k=5)
        dist_kd, ind_kd = kdt.query(X, k=5, p=p)

        assert_array_almost_equal(dist_bt, dist_kd)


if __name__ == '__main__':
    import nose
    nose.runmodule()
