"""Day 16 / Session 3 - Lab Assignment 2: Recursive Feature Elimination - Cut Noise, Measure the F1 Lift."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def _est():
    return LogisticRegression(max_iter=5000)


def make_data(seed=7):
    return make_classification(n_samples=500, n_features=60, n_informative=5, n_redundant=3,
                               n_clusters_per_class=1, shuffle=False, random_state=seed)


def rfe_support(X, y, n_select):
    return RFE(_est(), n_features_to_select=n_select).fit(X, y).support_


def rfe_ranking(X, y):
    return RFE(_est(), n_features_to_select=1).fit(X, y).ranking_


def rfecv_select(X, y):
    sel = RFECV(_est(), cv=5, scoring="f1").fit(X, y)
    return sel.support_, int(sel.n_features_)


def f1_cv(X, y):
    return float(cross_val_score(_est(), X, y, cv=5, scoring="f1").mean())


def f1_before_after(X, y, support):
    f1_all = f1_cv(X, y)
    f1_sel = f1_cv(X[:, support], y)
    return f1_all, f1_sel, f1_sel - f1_all


if __name__ == "__main__":
    X, y = make_data(7)
    sup = rfe_support(X, y, 8)
    print(int(sup.sum()), np.where(sup)[0])
    r = rfe_ranking(X, y)
    print(len(r), int(r.min()), int((r == 1).sum()))
    mask, n = rfecv_select(X, y)
    print(n)
    f_all, f_sel, imp = f1_before_after(X, y, mask)
    print(round(f_all, 3), round(f_sel, 3), round(imp, 3))
    print(round(f1_cv(X[:, sup], y), 3), round(f1_cv(X[:, -8:], y), 3))
