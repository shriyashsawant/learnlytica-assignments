"""Day 25 / Session 2 - Lab Assignment 2: Positional Encoding, Layer Norm & the Add-and-Norm Sublayer."""
import numpy as np

seq_len, d_model = 4, 8
x = np.array([[1., 2., 3., 4., 5., 6., 7., 8.]] * seq_len)     # token vectors
sublayer_out = np.ones((seq_len, d_model)) * 0.5               # a sublayer's output
gamma = np.ones(d_model)                                       # LayerNorm scale
beta = np.zeros(d_model)                                       # LayerNorm shift


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    for k in range(d_model):
        i = k // 2                                             # a sin/cos pair shares the same denominator
        angle = pos[:, 0] / np.power(10000.0, 2 * i / d_model)
        pe[:, k] = np.sin(angle) if k % 2 == 0 else np.cos(angle)      # even index -> sin, odd index -> cos
    return pe


def layer_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)                        # population variance
    xhat = (x - mean) / np.sqrt(var + eps)
    return gamma * xhat + beta


def add_and_norm(x, sublayer_out, gamma, beta, eps=1e-5):
    return layer_norm(np.asarray(x, dtype=float) + np.asarray(sublayer_out, dtype=float), gamma, beta, eps)


if __name__ == "__main__":
    pe = positional_encoding(4, 8)
    print(np.round(pe[0], 3))                                  # [0. 1. 0. 1. 0. 1. 0. 1.]
    print(pe.shape, bool(np.all(np.abs(pe) <= 1)))             # (4, 8) True
    y = layer_norm(x, np.ones(8), np.zeros(8))
    print(np.round(y.mean(axis=-1), 6))                        # ~0
    print(np.round(y.var(axis=-1), 4))                         # ~1
    z = add_and_norm(x, sublayer_out, np.ones(8), np.zeros(8))
    print(np.allclose(z, layer_norm(x + sublayer_out, np.ones(8), np.zeros(8))))      # True
    print(np.round(layer_norm(x, 2 * np.ones(8), np.ones(8)).mean(axis=-1), 4))       # beta shifts the mean to ~1
    print(np.round(pe[1, :4], 4))                              # [sin(1), cos(1), sin(0.1), cos(0.1)] pair denominators
