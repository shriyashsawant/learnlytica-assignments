"""Day 16 / Session 2 - Lab Assignment 2: Boosting & Overfitting - staged_predict Peak vs Final."""
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def make_data(seed=0):
    X, y = make_friedman1(n_samples=600, noise=3.0, random_state=seed)
    return train_test_split(X, y, test_size=0.4, random_state=seed)  # X_train, X_test, y_train, y_test


def staged_test_errors(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    return np.array([mean_squared_error(y_test, pred) for pred in model.staged_predict(X_test)])


def best_n_estimators(errors):
    return int(np.argmin(errors) + 1)


def overfit_gap(errors):
    errors = np.asarray(errors, dtype=float)
    return float(errors[-1] - errors.min())


def is_overfitting(errors, tol=1e-9):
    errors = np.asarray(errors, dtype=float)
    return bool(errors[-1] - errors.min() > tol)


if __name__ == "__main__":
    errs = np.array([10.0, 6.0, 4.0, 3.0, 3.5, 5.0])
    print(best_n_estimators(errs), overfit_gap(errs), is_overfitting(errs))
    Xtr, Xte, ytr, yte = make_data(0)
    m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.3, max_depth=5, random_state=0)
    e = staged_test_errors(m, Xtr, ytr, Xte, yte)
    print(len(e), best_n_estimators(e), round(overfit_gap(e), 3), is_overfitting(e), e[0] > e[best_n_estimators(e) - 1])
