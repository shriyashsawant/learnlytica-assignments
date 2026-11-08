"""Day 22 / Session 3 - Lab Assignment 1: Framing a Forecast - Windows, Chronological Split & the Naive Baseline."""
import numpy as np


def make_windows(series, lookback, horizon=1):
    series = np.asarray(series, dtype=float)
    N = len(series) - lookback - horizon + 1
    X = np.array([series[i:i + lookback] for i in range(N)]).reshape(N, lookback)
    y = np.array([series[i + lookback:i + lookback + horizon] for i in range(N)]).reshape(N, horizon)
    return X, y


def chrono_split(X, y, test_frac=0.2):
    X = np.asarray(X)
    y = np.asarray(y)
    n_test = max(1, int(round(len(X) * test_frac)))          # the last rows are the test set - never shuffled
    cut = len(X) - n_test
    return X[:cut], X[cut:], y[:cut], y[cut:]


def naive_forecast(X, horizon=1):
    X = np.asarray(X, dtype=float)
    return np.repeat(X[:, -1:], horizon, axis=1)             # persistence: repeat each window's last value


def rmse(y_true, y_pred):
    d = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean(d ** 2)))


def mae(y_true, y_pred):
    d = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(d)))


if __name__ == "__main__":
    s = np.arange(20.)
    X, y = make_windows(s, 4, 1)
    print(X.shape, y.shape, X[0], y[0].ravel())                       # (16, 4) (16, 1) [0. 1. 2. 3.] [4.]
    Xtr, Xte, ytr, yte = chrono_split(X, y, 0.25)
    print(len(Xtr), len(Xte), Xte[0], Xtr[-1])                        # 12 4 ... order preserved
    print(naive_forecast(np.array([[1., 2., 3.]]), 2))                # [[3. 3.]]
    print(round(rmse([0, 0, 0], [3, 4, 0]), 4), round(mae([0, 0, 0], [3, 4, 0]), 4))   # 2.8868 2.3333
    X2, y2 = make_windows(np.arange(10.), 3, 2)
    print(X2.shape, y2.shape, y2[0])                                  # (6, 3) (6, 2) [3. 4.]
    a, b, c, d = chrono_split(np.arange(10).reshape(10, 1), np.arange(10).reshape(10, 1), 0.3)
    print(len(a), len(b), b.ravel())                                  # 7 3 [7 8 9]
    X3, y3 = make_windows(np.arange(6.), 4, 2)
    print(X3, y3)                                                     # [[0. 1. 2. 3.]] [[4. 5.]]
    print(rmse([0, 0], [2, 2]) == mae([0, 0], [2, 2]), rmse([0, 0], [0, 4]) > mae([0, 0], [0, 4]))   # True True
