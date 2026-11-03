"""Day 15 / Session 4 - Lab Assignment 2: Gradient Descent & Ridge (L2) Regression from Scratch."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def make_data(seed=0):
    return make_regression(n_samples=300, n_features=4, n_informative=4, noise=8.0, random_state=seed)


def add_bias(X):
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def gd_fit(X, y, lr=0.1, n_iters=2000):
    Xb = add_bias(X)
    y = np.asarray(y, dtype=float)
    n = Xb.shape[0]
    theta = np.zeros(Xb.shape[1])
    loss_history = []
    for _ in range(n_iters):
        resid = Xb @ theta - y
        loss_history.append(float(np.mean(resid ** 2)))
        grad = (2 / n) * (Xb.T @ resid)
        theta = theta - lr * grad
    return theta, loss_history


def ridge_closed_form(X, y, alpha):
    Xb = add_bias(X)
    y = np.asarray(y, dtype=float)
    penalty = alpha * np.eye(Xb.shape[1])
    penalty[0, 0] = 0.0  # the intercept is not penalised
    return np.linalg.solve(Xb.T @ Xb + penalty, Xb.T @ y)


def train_val_diagnostic(X, y, alpha, test_size=0.3, seed=0):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=test_size, random_state=seed)
    theta = ridge_closed_form(X_tr, y_tr, alpha)
    mse = lambda Xs, ys: float(np.mean((add_bias(Xs) @ theta - ys) ** 2))
    return {"train_mse": mse(X_tr, y_tr), "val_mse": mse(X_va, y_va)}


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    Xk = rng.standard_normal((300, 4))
    true = np.array([5.0, 2.0, -3.0, 0.5, 1.5])
    yk = Xk @ true[1:] + true[0] + rng.standard_normal(300) * 0.1
    theta, hist = gd_fit(Xk, yk, lr=0.1, n_iters=3000)
    print(np.round(theta, 2))
    print(hist[0] > hist[-1], round(hist[-1], 3))
    print(np.round(ridge_closed_form(Xk, yk, 0.0), 2))
    print(train_val_diagnostic(Xk, yk, alpha=0.0))
    from sklearn.linear_model import Ridge
    X, y = make_data(0)
    th = ridge_closed_form(X, y, 5.0)
    r = Ridge(alpha=5.0).fit(X, y)
    print(np.allclose(th[1:], r.coef_), np.isclose(th[0], r.intercept_),
          np.linalg.norm(ridge_closed_form(X, y, 100)[1:]) < np.linalg.norm(ridge_closed_form(X, y, 0)[1:]))
