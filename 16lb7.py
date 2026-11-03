"""Day 16 / Session 4 - Lab Assignment 1: K-Means from Scratch - Lloyd's Algorithm in numpy."""
import numpy as np
from sklearn.datasets import make_blobs


def make_data(seed=42):
    return make_blobs(n_samples=300, centers=4, n_features=2, cluster_std=1.0, random_state=seed)


def euclidean(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return float(np.sqrt(np.sum((p - q) ** 2)))


def assign_clusters(X, centroids):
    X, centroids = np.asarray(X, dtype=float), np.asarray(centroids, dtype=float)
    labels = np.empty(len(X), dtype=int)
    for i, x in enumerate(X):
        labels[i] = int(np.argmin([euclidean(x, c) for c in centroids]))
    return labels


def update_centroids(X, labels, k):
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    out = np.full((k, X.shape[1]), np.nan)
    for j in range(k):
        members = X[labels == j]
        if len(members):
            out[j] = members.mean(axis=0)
    return out  # an empty cluster stays NaN here; kmeans() keeps the old centroid for it


def compute_inertia(X, labels, centroids):
    X, labels, centroids = np.asarray(X, dtype=float), np.asarray(labels), np.asarray(centroids, dtype=float)
    return float(np.sum((X - centroids[labels]) ** 2))


def kmeans(X, k, init_centroids, max_iter=100, tol=1e-6):
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(init_centroids, dtype=float).copy()
    labels = assign_clusters(X, centroids)
    for _ in range(max_iter):
        new = update_centroids(X, labels, k)
        new = np.where(np.isnan(new), centroids, new)          # keep the old centroid if a cluster went empty
        shift = np.max(np.linalg.norm(new - centroids, axis=1))
        centroids = new
        labels = assign_clusters(X, centroids)
        if shift < tol:
            break
    return labels, centroids, compute_inertia(X, labels, centroids)


if __name__ == "__main__":
    print(euclidean([0, 0], [3, 4]))
    Xs = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    Cs = np.array([[0, 0.5], [10, 10.5]])
    print(assign_clusters(Xs, Cs).tolist())
    print(update_centroids(Xs, [0, 0, 1, 1], 2).tolist())
    print(round(compute_inertia(Xs, [0, 0, 1, 1], Cs), 3))
    X, y = make_data(42)
    init = X[[0, 75, 150, 225]].astype(float)
    labels, centroids, inertia = kmeans(X, 4, init)
    print(round(inertia, 1))
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    sk = KMeans(4, init=init, n_init=1, random_state=0).fit(X)
    print(round(sk.inertia_, 1), adjusted_rand_score(sk.labels_, labels))
