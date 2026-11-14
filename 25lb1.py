"""Day 25 / Session 1 - Lab Assignment 1: Causal Scaled Dot-Product Self-Attention from Scratch."""
import numpy as np

Q = np.array([[1., 0., 0.], [0., 1., 0.], [1., 1., 0.], [0., 1., 1.]])      # (4, 3)
K = np.array([[1., 0., 1.], [0., 1., 0.], [1., 1., 0.], [0., 0., 1.]])      # (4, 3)
V = np.array([[1., 2.], [0., 1.], [2., 0.], [1., 1.]])                      # (4, 2)


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    shifted = z - np.max(z, axis=axis, keepdims=True)         # subtract the max for stability; -inf stays -inf
    e = np.exp(shifted)                                       # exp(-inf) = 0, so masked entries get weight 0
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    mask = np.zeros((n, n))
    mask[np.triu_indices(n, k=1)] = -np.inf                   # -inf strictly above the diagonal, 0.0 on and below it
    return mask


def attention(Q, K, V, mask=None):
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)                        # row-stochastic
    return weights @ V, weights


if __name__ == "__main__":
    out, w = attention(Q, K, V, causal_mask(4))
    print(out.shape, w.shape)                                 # (4, 2) (4, 4)
    print(np.round(w.sum(axis=1), 6))                         # [1. 1. 1. 1.]
    print(np.round(w[0], 3))                                  # [1. 0. 0. 0.]
    print(np.allclose(np.triu(w, 1), 0))                      # True
    print(causal_mask(3))
    print(np.round(softmax(np.array([[1000.0, 1001.0, -np.inf]])), 4))        # stable, -inf -> 0
