"""Day 25 / Session 2 - Lab Assignment 2: Positional Encoding, Layer Norm & the Add-and-Norm Sublayer."""
import numpy as np


def positional_encoding(L, d_model):
    PE = np.zeros((L, d_model))
    pos = np.arange(L)[:, None]
    for i in range(0, d_model, 2):                          # i is the even index 2k
        angle = pos[:, 0] / (10000 ** (i / d_model))        # pos / 10000^(2k/d_model)
        PE[:, i] = np.sin(angle)
        if i + 1 < d_model:
            PE[:, i + 1] = np.cos(angle)                    # the cos of a pair shares the sin's denominator
    return PE


def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)                     # population variance
    xhat = (x - mean) / np.sqrt(var + eps)
    return np.asarray(gamma) * xhat + np.asarray(beta)


def add_and_norm(x, sublayer_out, gamma, beta, eps=1e-5):
    return layer_norm(np.asarray(x, dtype=float) + np.asarray(sublayer_out, dtype=float), gamma, beta, eps)


if __name__ == "__main__":
    rng = np.random.default_rng(250)
    L, d_model = 10, 16
    x = rng.normal(size=(L, d_model)) * 3.0 + 7.0
    sublayer_out = rng.normal(size=(L, d_model))
    gamma = np.ones(d_model); beta = np.zeros(d_model)
    pe = positional_encoding(4, 8)
    print(np.round(pe[0], 3), pe.shape, float(np.abs(positional_encoding(50, 16)).max()) <= 1.0)   # [0. 1. 0. 1. ...] (4, 8) True
    y = layer_norm(x, gamma, beta)
    print(np.round(y.mean(axis=-1), 6)[:3], np.round(y.var(axis=-1), 4)[:3])                       # ~0 ~1
    z = add_and_norm(x, sublayer_out, gamma, beta)
    print(np.allclose(z, layer_norm(x + sublayer_out, gamma, beta)))                               # True
    g, b = rng.normal(size=d_model), rng.normal(size=d_model)
    print(np.allclose(layer_norm(x, g, b), g * layer_norm(x, np.ones(d_model), np.zeros(d_model)) + b))   # True
    print(np.allclose(layer_norm(x * 5 + 3, gamma, beta, eps=1e-12), layer_norm(x, gamma, beta, eps=1e-12), atol=1e-6))   # shift/scale invariant
    P = positional_encoding(6, 8)
    print(np.isclose(P[3, 4], np.sin(3 / 10000 ** (4 / 8))), np.isclose(P[3, 5], np.cos(3 / 10000 ** (4 / 8))))   # True True
