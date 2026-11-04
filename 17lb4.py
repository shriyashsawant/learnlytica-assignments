"""Day 17 / Session 2 - Lab Assignment 2: The Training Loop - Gradient Descent & the Learning-Rate Regime."""
import numpy as np
from sklearn.datasets import make_blobs


def make_data(seed=0):
    X, y = make_blobs(n_samples=150, centers=3, n_features=2, cluster_std=1.0, random_state=seed)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y


def _softmax(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(X, W, b):
    return X @ W + b


def loss_and_grads(X, y, W, b):
    N = X.shape[0]
    probs = _softmax(forward(X, W, b))
    loss = float(-np.mean(np.log(np.clip(probs[np.arange(N), y], 1e-12, 1.0))))
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), y] = 1.0
    dZ = (probs - onehot) / N
    dW = X.T @ dZ
    db = dZ.sum(axis=0)
    return loss, dW, db


def sgd_step(W, b, dW, db, lr):
    return W - lr * dW, b - lr * db


def train(X, y, lr, epochs, seed=0):
    D, K = X.shape[1], int(np.max(y)) + 1
    W, b = np.zeros((D, K)), np.zeros(K)
    losses = []
    with np.errstate(all="ignore"):
        for _ in range(epochs):
            loss, dW, db = loss_and_grads(X, y, W, b)
            losses.append(loss)                       # record the current loss, then take one step
            W, b = sgd_step(W, b, dW, db, lr)
    return W, b, losses


def lr_effect(X, y, lrs, epochs=200, seed=0):
    out = {}
    for lr in lrs:
        _, _, losses = train(X, y, lr, epochs, seed)
        final = losses[-1]
        out[lr] = "converged" if (np.isfinite(final) and final < losses[0]) else "diverged"
    return out


if __name__ == "__main__":
    X, y = make_data()
    W, b, losses = train(X, y, 0.5, 300)
    print(round(losses[0], 3), "->", round(losses[-1], 3), len(losses))
    print(lr_effect(X, y, [0.01, 0.3, 1000.0]))
    # finite-difference gradient check
    rng = np.random.default_rng(0)
    W0, b0 = rng.normal(size=(2, 3)), rng.normal(size=3)
    _, dW, db = loss_and_grads(X, y, W0, b0)
    num = np.zeros_like(W0)
    for idx in np.ndindex(W0.shape):
        hi, lo = W0.copy(), W0.copy(); hi[idx] += 1e-6; lo[idx] -= 1e-6
        num[idx] = (loss_and_grads(X, y, hi, b0)[0] - loss_and_grads(X, y, lo, b0)[0]) / 2e-6
    print("max gradient error", float(np.abs(num - dW).max()), dW.shape, db.shape)
