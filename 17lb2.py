"""Day 17 / Session 1 - Lab Assignment 2: Backprop by Hand - a 2-layer MLP that learns XOR."""
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def forward(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = sigmoid(z2)
    return {"z1": z1, "a1": a1, "z2": z2, "a2": a2}


def bce_loss(y, y_hat, eps=1e-12):
    p = np.clip(y_hat, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def backprop(X, y, params, cache):
    m = X.shape[0]
    dz2 = (cache["a2"] - y) / m                          # sigmoid output + BCE loss
    dW2 = cache["a1"].T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (1.0 - cache["a1"] ** 2)                 # tanh derivative
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def gradient_step(params, grads, lr):
    return {name: params[name] - lr * grads["d" + name] for name in params}


if __name__ == "__main__":
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    rng = np.random.default_rng(0)
    params = {"W1": rng.normal(0, 1, (2, 8)), "b1": np.zeros((1, 8)),
              "W2": rng.normal(0, 1, (8, 1)), "b2": np.zeros((1, 1))}

    # finite-difference check of the analytic gradients
    def loss_with(p):
        return bce_loss(y, forward(X, p)["a2"])

    g = backprop(X, y, params, forward(X, params))
    for name in ("W1", "W2"):
        num = np.zeros_like(params[name])
        for idx in np.ndindex(params[name].shape):
            hi = {k: v.copy() for k, v in params.items()}; hi[name][idx] += 1e-6
            lo = {k: v.copy() for k, v in params.items()}; lo[name][idx] -= 1e-6
            num[idx] = (loss_with(hi) - loss_with(lo)) / 2e-6
        print(name, "max gradient error:", float(np.max(np.abs(num - g["d" + name]))))

    for _ in range(6000):
        cache = forward(X, params)
        params = gradient_step(params, backprop(X, y, params, cache), 0.5)
    pred = (forward(X, params)["a2"] > 0.5).astype(int).ravel()
    print(pred.tolist(), round(bce_loss(y, forward(X, params)["a2"]), 4))
