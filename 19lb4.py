"""Day 19 / Session 2 - Lab Assignment 2: Frozen Features Win with Few Labels - Transfer Learning, Simulated."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def make_data(seed=0, n=2000, D=500, k=5, classes=4, csep=3.0, onoise=16.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, csep, size=(classes, k))
    y = rng.integers(0, classes, size=n)
    Z = centers[y] + rng.normal(0, 1.0, size=(n, k))      # k-dim class signal
    W = rng.normal(0, 1.0, size=(k, D))
    X = Z @ W + rng.normal(0, onoise, size=(n, D))        # 500-dim noisy "pixels"
    return X.astype(float), y


def fit_backbone(X_pool, n_components=5):
    """The 'pre-trained backbone': a PCA fit once on a big pool, then frozen."""
    backbone = PCA(n_components=n_components, random_state=0)
    backbone.fit(X_pool)
    return backbone


def extract(backbone, X):
    return backbone.transform(X)


def transfer_score(backbone, Xtr, ytr, Xte, yte, k_neighbors=5):
    head = KNeighborsClassifier(k_neighbors)
    head.fit(extract(backbone, Xtr), ytr)                 # small head on the frozen features
    return round(float(head.score(extract(backbone, Xte), yte)), 4)


def raw_score(Xtr, ytr, Xte, yte, k_neighbors=5):
    head = KNeighborsClassifier(k_neighbors)
    head.fit(Xtr, ytr)                                    # same head, raw pixels
    return round(float(head.score(Xte, yte)), 4)


def sample_efficiency(backbone, Xtr, ytr, Xte, yte, sizes):
    table = {}
    for n in sizes:
        table[n] = {
            "transfer": transfer_score(backbone, Xtr[:n], ytr[:n], Xte, yte),
            "raw": raw_score(Xtr[:n], ytr[:n], Xte, yte),
        }
    return table


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    X, y = make_data(0)
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    bb = fit_backbone(X_pool, n_components=5)
    print(extract(bb, X_test).shape)                      # (600, 5)
    tbl = sample_efficiency(bb, X_pool, y_pool, X_test, y_test, [30, 60, 200])
    for n, r in tbl.items():
        print(n, r["transfer"], r["raw"])                 # 30 0.965 0.848 / 60 0.987 0.898 / 200 0.995 0.965
    print(tbl[200]["transfer"] >= tbl[30]["transfer"], tbl[60]["transfer"] - tbl[60]["raw"] > 0.02, tbl[60]["transfer"] > 0.85)
