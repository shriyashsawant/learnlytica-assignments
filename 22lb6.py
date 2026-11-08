"""Day 22 / Session 3 - Lab Assignment 2: Beating the Baseline - Linear AR Forecaster & Recursive Multi-Step."""
import numpy as np
from sklearn.linear_model import LinearRegression


def make_windows(series, lookback):
    series = np.asarray(series, dtype=float)
    N = len(series) - lookback
    X = np.array([series[i:i + lookback] for i in range(N)])
    y = series[lookback:lookback + N]
    return X, y


def fit_ar_forecaster(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def forecast_step(model, window):
    window = np.asarray(window, dtype=float).reshape(1, -1)
    return float(model.predict(window)[0])


def recursive_forecast(model, seed_window, steps):
    window = list(np.asarray(seed_window, dtype=float))
    preds = []
    for _ in range(steps):
        nxt = forecast_step(model, window)
        preds.append(nxt)
        window = window[1:] + [nxt]                      # slide: the prediction becomes the newest input
    return np.array(preds)


def rmse(y_true, y_pred):
    d = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean(d ** 2)))


if __name__ == "__main__":
    s = 3 * np.arange(40) + 5.0
    X, y = make_windows(s, 6)
    model = fit_ar_forecaster(X, y)
    print(round(forecast_step(model, s[:6]), 3))                       # ~ 23.0
    preds = recursive_forecast(model, s[:6], 10)
    print(preds.shape, np.allclose(preds, s[6:16], atol=1e-3))         # (10,) True
    print(round(rmse([1, 2, 3], [1, 2, 5]), 4))                        # 1.1547
    Xs, ys = make_windows(np.arange(10.), 3)
    print(Xs.shape, ys.shape, Xs[0], ys[0])                            # (7, 3) (7,) [0. 1. 2.] 3.0
    rng = np.random.RandomState(7)
    t = np.arange(400)
    series = 0.05 * t + 3 * np.sin(2 * np.pi * t / 20) + rng.randn(400) * 0.3
    X, y = make_windows(series, 20)
    cut = int(len(X) * 0.8)
    m = fit_ar_forecaster(X[:cut], y[:cut])
    print(round(rmse(y[cut:], m.predict(X[cut:])), 3), round(rmse(y[cut:], X[cut:, -1]), 3))   # AR beats persistence
    ramp = 2 * np.arange(60) + 10.0
    Xr, yr = make_windows(ramp, 5)
    mr = fit_ar_forecaster(Xr, yr)
    print(round(forecast_step(mr, ramp[50:55]), 2), np.allclose(recursive_forecast(mr, ramp[:5], 24), ramp[5:29], atol=1e-3))   # 110.0 True
