"""Day 22 / Session 3 - Lab Assignment 2: Beating the Baseline - Linear AR Forecaster & Recursive Multi-Step."""
import numpy as np
from sklearn.linear_model import LinearRegression

t = np.arange(400)
series = 0.05 * t + 3 * np.sin(2 * np.pi * t / 20)      # a trend plus a cycle


def make_lagged(series, lookback):
    series = np.asarray(series, dtype=float)
    N = len(series) - lookback
    X = np.array([series[i:i + lookback] for i in range(N)]).reshape(N, lookback)
    y = np.array([series[i + lookback] for i in range(N)])
    return X, y


def fit_ar(X, y):
    return LinearRegression().fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))


def recursive_forecast(model, history, steps):
    lookback = int(model.n_features_in_)
    window = list(np.asarray(history, dtype=float)[-lookback:])      # the seed window
    out = []
    for _ in range(steps):
        nxt = float(model.predict(np.array(window).reshape(1, -1))[0])
        out.append(nxt)
        window = window[1:] + [nxt]                                  # slide: each prediction becomes the next input
    return np.array(out)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.ravel(y_true) - np.ravel(y_pred))))


if __name__ == "__main__":
    s = 3.0 * np.arange(20) + 5.0
    X, y = make_lagged(s, 4)
    model = fit_ar(X, y)
    print(np.round(recursive_forecast(model, s[:4], 5), 3))                  # ~ [17. 20. 23. 26. 29.]
    print(round(mae([1, 2, 3], [1, 2, 5]), 4))                               # 0.6667
    Xa, ya = make_lagged([0, 1, 2, 3, 4], 3)
    print(Xa[0].tolist(), float(ya[0]), Xa.shape, ya.shape)                  # [0.0, 1.0, 2.0] 3.0 (2, 3) (2,)
    # held-out check on the trend + cycle series: the AR model should beat naive persistence
    Xs, ys = make_lagged(series, 20)
    cut = int(len(Xs) * 0.8)
    m = fit_ar(Xs[:cut], ys[:cut])
    print(mae(ys[cut:], m.predict(Xs[cut:])) < mae(ys[cut:], Xs[cut:, -1]))  # True
    print(len(recursive_forecast(m, series[:300], 60)))                      # 60
