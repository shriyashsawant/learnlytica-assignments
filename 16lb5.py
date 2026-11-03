"""Day 16 / Session 3 - Lab Assignment 1: Rescue F1 on an Imbalanced Set - class_weight + Threshold Tuning."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def make_data(seed=42):
    return make_classification(n_samples=2000, n_features=20, n_informative=6, n_redundant=4,
                               weights=[0.9, 0.1], random_state=seed)


def split_data(X, y, seed=0):
    return train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)


def fit_model(X_train, y_train, balanced=False):
    model = LogisticRegression(max_iter=5000, class_weight="balanced") if balanced else LogisticRegression(max_iter=5000)
    return model.fit(X_train, y_train)


def f1_at_threshold(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    return float(f1_score(y, (proba >= threshold).astype(int)))


def best_threshold(model, X, y, thresholds):
    best_t, best_f1 = None, -1.0
    for t in thresholds:
        f1 = f1_at_threshold(model, X, y, t)
        if f1 > best_f1:  # strict '>' keeps the first threshold on ties
            best_t, best_f1 = t, f1
    return best_t, best_f1


def f1_improvement(f1_before, f1_after):
    return float(f1_after - f1_before)


if __name__ == "__main__":
    X, y = make_data(42)
    Xtr, Xte, ytr, yte = split_data(X, y, 0)
    m = fit_model(Xtr, ytr, balanced=False)
    base = f1_at_threshold(m, Xte, yte)
    print(round(base, 3), round(float(f1_score(yte, m.predict(Xte))), 3))
    ths = list(np.round(np.linspace(0.05, 0.95, 19), 3))
    bt, bf = best_threshold(m, Xte, yte, ths)
    print(bt, round(bf, 3), round(f1_improvement(base, bf), 3), bf > base)
    mb = fit_model(Xtr, ytr, balanced=True)
    print(int(m.predict(Xte).sum()), int(mb.predict(Xte).sum()))
