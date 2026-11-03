"""Day 15 / Session 3 - Lab Assignment 2: High-Bias vs High-Variance - Learning Curves & Regularization."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_data(seed=0):
    return make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, random_state=seed)


def lc_curves(estimator, X, y, train_sizes):
    sizes, tr, va = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=5, scoring="accuracy",
                                   shuffle=True, random_state=0)
    return sizes, tr.mean(axis=1), va.mean(axis=1)


def diagnose_bias_variance(train_mean, val_mean, gap_thresh=0.1, low_thresh=0.8):
    final_train, final_val = train_mean[-1], val_mean[-1]
    if final_train - final_val > gap_thresh:
        return "high_variance"
    if final_val < low_thresh:
        return "high_bias"
    return "good_fit"


def reg_sweep(X, y, C_values):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
    tr, va = validation_curve(pipe, X, y, param_name="clf__C", param_range=list(C_values), cv=5, scoring="accuracy")
    return tr.mean(axis=1), va.mean(axis=1)


def best_C(C_values, val_mean):
    return float(list(C_values)[int(np.argmax(val_mean))])


if __name__ == "__main__":
    X, y = make_data(0)
    ts, tr, va = lc_curves(LogisticRegression(max_iter=5000), X, y, [0.2, 0.5, 0.8, 1.0])
    print(ts, tr.round(3), va.round(3))
    print(diagnose_bias_variance([1.0, 1.0, 1.0], [0.78, 0.79, 0.80]))
    print(diagnose_bias_variance([0.72, 0.71, 0.70], [0.70, 0.69, 0.68]))
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0]
    trc, vac = reg_sweep(X, y, Cs)
    print(trc.round(3), vac.round(3), best_C(Cs, vac))
