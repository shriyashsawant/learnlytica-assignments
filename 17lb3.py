"""Day 17 / Session 2 - Lab Assignment 1: The Output Layer - Numerically-Stable Softmax & Cross-Entropy."""
import numpy as np


def softmax(Z):
    Z = np.asarray(Z, dtype=float)
    shifted = Z - Z.max(axis=1, keepdims=True)      # subtract the row max so exp() can never overflow
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    y = np.asarray(y, dtype=int)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))


def softmax_ce_loss(Z, y):
    return cross_entropy(softmax(Z), y)


def predict(Z):
    return np.argmax(np.asarray(Z, dtype=float), axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(6, 4)) * 3.0
    y = np.array([0, 1, 2, 3, 1, 0])
    print(softmax(Z).sum(axis=1))
    print(softmax_ce_loss(Z, y))
    print(cross_entropy(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]), np.array([0, 1])))
    print(softmax(np.array([[1000., 1000., 1000., 1000.]])))
    print(predict(np.array([[20., 0., 0.], [0., 20., 0.]])))
    from scipy.special import softmax as sp_softmax
    print(np.allclose(softmax(Z), sp_softmax(Z, axis=1)))
