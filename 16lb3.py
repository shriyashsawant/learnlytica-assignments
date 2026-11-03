"""Day 16 / Session 2 - Lab Assignment 1: Voting Aggregators & Bagging Cuts Variance."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def hard_vote(preds):
    preds = np.asarray(preds, dtype=int)
    # argmax of bincount returns the first (smallest) label on ties
    return np.array([np.argmax(np.bincount(preds[:, j])) for j in range(preds.shape[1])])


def soft_vote(probas):
    return np.asarray(probas, dtype=float).mean(axis=0).argmax(axis=1)


def make_data(seed=42):
    return make_classification(n_samples=900, n_features=20, n_informative=8, n_redundant=4, random_state=seed)


def compare_single_vs_bagging(X, y, seed=0):
    base = DecisionTreeClassifier(random_state=seed)
    single = cross_val_score(base, X, y, cv=5).mean()
    bag = BaggingClassifier(base, n_estimators=50, random_state=seed)
    bagging = cross_val_score(bag, X, y, cv=5).mean()
    return float(single), float(bagging)


def rf_oob(X, y, n_estimators=300, seed=0):
    rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, bootstrap=True, random_state=seed)
    rf.fit(X, y)
    return float(rf.oob_score_)


if __name__ == "__main__":
    preds = np.array([[0, 1, 2, 2], [0, 1, 1, 2], [1, 1, 1, 0]])
    print(hard_vote(preds))
    print(hard_vote(np.array([[0, 1], [1, 0]])))  # tie -> smallest label
    probas = np.array([[[0.9, 0.1], [0.6, 0.4]], [[0.4, 0.6], [0.3, 0.7]], [[0.4, 0.6], [0.3, 0.7]]])
    print(soft_vote(probas))
    X, y = make_data(42)
    s, b = compare_single_vs_bagging(X, y)
    print(round(s, 3), round(b, 3), b > s)
    print(round(rf_oob(X, y, 200), 3))
