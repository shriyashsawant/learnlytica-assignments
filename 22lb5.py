"""Day 22 / Session 3 - Lab Assignment 1: Framing a Forecast - Windows, Chronological Split & the Naive Baseline."""
import numpy as np
from sklearn.metrics import mean_absolute_error

series = np.arange(10, dtype=float)             # [0, 1, 2, ..., 9]


def make_windows(series, lookback):
    series = np.asarray(series, dtype=float)
    N = len(series) - lookback                  # a length-T series gives T - lookback windows
    X = np.array([series[i:i + lookback] for i in range(N)]).reshape(N, lookback)
    y = np.array([series[i + lookback] for i in range(N)])
    return X, y


def chrono_split(X, y, test_frac=0.2):
    X, y = np.asarray(X), np.asarray(y)
    n_test = max(1, int(round(len(X) * test_frac)))      # the last test_frac of the rows, at least 1
    cut = len(X) - n_test
    return X[:cut], X[cut:], y[:cut], y[cut:]            # no shuffling, order preserved


def naive_forecast(X):
    return np.asarray(X, dtype=float)[:, -1]             # persistence: predict the last observed value


def mae(y_true, y_pred):
    return float(mean_absolute_error(np.ravel(y_true), np.ravel(y_pred)))


if __name__ == "__main__":
    s = np.arange(10, dtype=float)
    X, y = make_windows(s, 3)
    print(X.shape, X[0], float(y[0]))                                              # (7, 3) [0. 1. 2.] 3.0
    Xtr, Xte, ytr, yte = chrono_split(np.arange(7).reshape(7, 1), np.arange(7), 0.3)
    print(len(Xtr), len(Xte), Xte.ravel())                                         # 5 2 [5 6]
    print(naive_forecast(np.array([[1., 2., 3.], [4., 5., 6.]])))                  # [3. 6.]
    print(round(mae([0, 0, 0], [3, 4, 0]), 4))                                     # 2.3333
    print(chrono_split(np.arange(3).reshape(3, 1), np.arange(3), 0.1)[1].ravel())  # [2]  (at least one test row)
