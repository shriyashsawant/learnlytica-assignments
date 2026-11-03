"""Day 15 / Session 4 - Lab Assignment 1: OLS from Scratch - the Normal Equation with R2 & MSE."""
import numpy as np
from sklearn.datasets import make_regression


def make_data(seed=0):
    return make_regression(n_samples=200, n_features=5, n_informative=5, noise=10.0, random_state=seed)


def add_bias(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_ols(X, y):
    Xb = add_bias(X)
    y = np.asarray(y, dtype=float)
    # normal equation theta = (Xb^T Xb)^-1 Xb^T y, solved without forming the explicit inverse
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


def predict(X, theta):
    return add_bias(X) @ np.asarray(theta, dtype=float)


def mse_scratch(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_scratch(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    Xk = rng.standard_normal((150, 4))
    true = np.array([5.0, 2.0, -3.0, 0.5, 1.5])
    yk = Xk @ true[1:] + true[0]
    theta = fit_ols(Xk, yk)
    print(np.round(theta, 3))
    print(round(mse_scratch(yk, predict(Xk, theta)), 8))
    print(round(r2_scratch(yk, predict(Xk, theta)), 8))
    print(round(mse_scratch([0.0, 0.0], [3.0, 4.0]), 3))
    # agreement with scikit-learn on the lab dataset
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    X, y = make_data(0)
    th = fit_ols(X, y)
    lr = LinearRegression().fit(X, y)
    print(np.allclose(th[1:], lr.coef_), np.isclose(th[0], lr.intercept_),
          np.isclose(r2_scratch(y, predict(X, th)), r2_score(y, lr.predict(X))),
          np.isclose(mse_scratch(y, predict(X, th)), mean_squared_error(y, lr.predict(X))))
