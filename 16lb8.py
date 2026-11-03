"""Day 16 / Session 4 - Lab Assignment 2: Choosing k - Elbow, From-Scratch Silhouette & z-Score Scaling."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def make_data(seed=0):
    X, y = make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=0.8, random_state=seed)
    X = X.astype(float)
    X[:, 1] = X[:, 1] * 50.0  # stretch feature 2 so scaling matters
    return X, y


def zscore_standardize(X):
    X = np.asarray(X, dtype=float)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # a constant column must not divide by zero
    return (X - X.mean(axis=0)) / std


def cluster_inertia(X, k, seed=0):
    return float(KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).inertia_)


def elbow_inertias(X, k_range, seed=0):
    return [cluster_inertia(X, k, seed) for k in k_range]


def silhouette_metric(X, labels):
    X, labels = np.asarray(X, dtype=float), np.asarray(labels)
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))   # pairwise Euclidean distances
    uniq = np.unique(labels)
    scores = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue  # a singleton cluster has silhouette 0
        a = dists[i, own].sum() / (own.sum() - 1)     # mean distance to the OTHER points of its own cluster
        b = min(dists[i, labels == c].mean() for c in uniq if c != labels[i])
        scores[i] = (b - a) / max(a, b)
    return float(scores.mean())


def best_k_by_silhouette(X, k_range, seed=0):
    best_k, best_s = None, -2.0
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        s = silhouette_metric(X, labels)
        if s > best_s:
            best_k, best_s = int(k), s
    return best_k


if __name__ == "__main__":
    X, y = make_data(0)
    Z = zscore_standardize(X)
    print(Z.mean(axis=0).round(6), Z.std(axis=0).round(6))
    print(np.isfinite(zscore_standardize(np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]))).all())
    ine = elbow_inertias(Z, range(1, 8))
    print([round(v, 1) for v in ine], all(a >= b - 1e-9 for a, b in zip(ine, ine[1:])))
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
    from sklearn.metrics import silhouette_score
    print(round(silhouette_metric(Z, km.labels_), 3), round(float(silhouette_score(Z, km.labels_)), 3))
    raw_km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    print("raw", round(silhouette_metric(X, raw_km.labels_), 3), "scaled", round(silhouette_metric(Z, km.labels_), 3))
    print(best_k_by_silhouette(Z, [2, 3, 4, 5, 6]))
