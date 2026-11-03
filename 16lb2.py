"""Day 16 / Session 1 - Lab Assignment 2: PCA - Explained Variance & Choosing k for 90%."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


def pca_ratios(X):
    return np.asarray(PCA().fit(X).explained_variance_ratio_)


def cumulative(ratios):
    return np.cumsum(ratios)


def choose_k(ratios, threshold=0.9):
    cum = cumulative(ratios)
    # >= with a tiny tolerance so an exact boundary (or a float sum like 0.9999999999) still counts
    hits = np.nonzero(cum >= threshold - 1e-9)[0]
    return int(hits[0] + 1) if len(hits) else len(cum)


def project(X, k):
    return PCA(n_components=k).fit_transform(X)


if __name__ == "__main__":
    X = load_digits().data
    r = pca_ratios(X)
    print(r[:3].round(3))
    print(cumulative(r)[:5].round(3))
    k = choose_k(r, 0.9)
    print(k)
    print(project(X, k).shape)
    print(choose_k(np.array([0.5, 0.3, 0.15, 0.05]), 0.9))
    print(choose_k(np.array([0.5, 0.25, 0.25]), 0.75))
    print(choose_k(r, 1.0), len(r))
