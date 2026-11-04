"""Day 17 / Session 1 - Lab Assignment 1: Neural Net Building Blocks - Activations, Gradients & BCE Loss in NumPy."""
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))      # for z >= 0, exp(-z) is small and safe
    ez = np.exp(z[~pos])                           # for z < 0, exp(z) is small and safe (no overflow)
    out[~pos] = ez / (1.0 + ez)
    return out


def relu(z):
    return np.maximum(0.0, np.asarray(z, dtype=float))


def tanh_act(z):
    return np.tanh(np.asarray(z, dtype=float))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu_grad(z):
    return (np.asarray(z, dtype=float) > 0).astype(float)


def tanh_grad(z):
    return 1.0 - np.tanh(np.asarray(z, dtype=float)) ** 2


def bce_loss(y_true, y_pred, eps=1e-12):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


if __name__ == "__main__":
    z = np.array([-2.0, 0.0, 2.0])
    print(np.round(sigmoid(z), 4))
    print(np.round(sigmoid_grad(z), 4))
    print(relu(np.array([-1.0, 2.0])))
    print(np.round(tanh_grad(0.0), 4))
    print(round(bce_loss([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]), 4))
    print(round(bce_loss([1, 0], [0.0, 1.0]), 4))
    print(sigmoid(np.array([-1000.0, 1000.0])), relu_grad(np.array([-1.0, 0.0, 3.0])))
