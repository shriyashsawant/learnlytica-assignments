"""Day 15 / Session 3 - Lab Assignment 1: Diagnose Under/Overfitting with a Validation Curve."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeClassifier


def make_data(seed=42):
    return make_classification(n_samples=800, n_features=20, n_informative=8, n_redundant=4, random_state=seed)


def depth_curves(X, y, depths):
    tr, va = validation_curve(DecisionTreeClassifier(random_state=0), X, y, param_name="max_depth",
                              param_range=list(depths), cv=5, scoring="accuracy")
    return tr.mean(axis=1), va.mean(axis=1)


def best_depth(depths, val_mean):
    return int(list(depths)[int(np.argmax(val_mean))])


def classify_fit(train_score, val_score, gap_thresh=0.1, low_thresh=0.8):
    if train_score < low_thresh and val_score < low_thresh:
        return "underfit"
    if train_score - val_score > gap_thresh:
        return "overfit"
    return "good_fit"


if __name__ == "__main__":
    X, y = make_data(42)
    depths = [1, 3, 5, 8, 12, 20]
    tr, va = depth_curves(X, y, depths)
    print([round(float(v), 2) for v in tr])
    print([round(float(v), 2) for v in va])
    print(best_depth(depths, va))
    print(classify_fit(1.0, 0.82))
    print(classify_fit(0.70, 0.68))
    print(classify_fit(0.90, 0.88))
