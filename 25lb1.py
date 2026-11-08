"""Day 25 / Session 1 - Lab Assignment 1: Causal Scaled Dot-Product Self-Attention from Scratch."""
import numpy as np


def linear_project(X, W):
    return np.asarray(X, dtype=float) @ np.asarray(W, dtype=float)


def scaled_scores(Q, K):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    return Q @ K.T / np.sqrt(Q.shape[-1])                    # Q K^T / sqrt(d_k)


def causal_mask(n):
    mask = np.zeros((n, n))
    mask[np.triu_indices(n, k=1)] = -np.inf                  # strictly above the diagonal = the future
    return mask


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))      # -inf entries become exp(-inf) = 0
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_self_attention(X, Wq, Wk, Wv):
    Q = linear_project(X, Wq)
    K = linear_project(X, Wk)
    V = linear_project(X, Wv)
    scores = scaled_scores(Q, K) + causal_mask(len(X))
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    X = rng.randn(6, 4); Wq = rng.randn(4, 4); Wk = rng.randn(4, 4); Wv = rng.randn(4, 5)
    out, w = causal_self_attention(X, Wq, Wk, Wv)
    print(out.shape, w.shape)                                # (6, 5) (6, 6)
    print(np.round(w.sum(axis=1), 6))                        # all 1
    print(np.round(w[0], 3), (w > 0).sum(axis=1))            # [1. 0. 0. 0. 0. 0.] [1 2 3 4 5 6]
    print(np.allclose(np.triu(w, 1), 0))                     # True
    print(causal_mask(3))
    from scipy.special import softmax as sp_softmax
    z = rng.randn(4, 5)
    print(np.allclose(softmax(z), sp_softmax(z, axis=-1)), np.allclose(scaled_scores(X, X), X @ X.T / 2.0))   # True True
