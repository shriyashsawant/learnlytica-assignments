"""Day 25 / Session 1 - Lab Assignment 2: Self- vs Cross-Attention & a Future-Leakage Audit."""
import numpy as np


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores + mask                               # additive mask: -inf blocks a position
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def self_attention(X, Wq, Wk, Wv, causal=False):
    X = np.asarray(X, dtype=float)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                         # Q, K and V all come from the SAME sequence
    mask = None
    if causal:
        n = X.shape[0]
        mask = np.zeros((n, n))
        mask[np.triu_indices(n, k=1)] = -np.inf
    return attention(Q, K, V, mask)


def cross_attention(X_dec, X_enc, Wq, Wk, Wv):
    Q = np.asarray(X_dec, dtype=float) @ Wq                  # queries from the decoder
    K = np.asarray(X_enc, dtype=float) @ Wk                  # keys and values from the encoder
    V = np.asarray(X_enc, dtype=float) @ Wv
    return attention(Q, K, V, None)


def future_leakage(weights):
    w = np.asarray(weights, dtype=float)
    if w.shape[0] < 2:
        return 0.0
    return float(np.max(np.triu(w, k=1)))                    # biggest weight on a future position


if __name__ == "__main__":
    rng = np.random.RandomState(3)
    X = rng.randn(6, 4); Wq = rng.randn(4, 4); Wk = rng.randn(4, 4); Wv = rng.randn(4, 4)
    _, w_causal = self_attention(X, Wq, Wk, Wv, causal=True)
    _, w_full = self_attention(X, Wq, Wk, Wv, causal=False)
    print(round(future_leakage(w_causal), 6), future_leakage(w_full) > 0)        # 0.0 True
    print(np.allclose(w_causal.sum(axis=1), 1), np.allclose(w_full.sum(axis=1), 1))
    Xd = rng.randn(3, 4); Xe = rng.randn(7, 4)
    out, wc = cross_attention(Xd, Xe, np.eye(4), np.eye(4), np.eye(4))
    print(wc.shape, out.shape)                                                   # (3, 7) (3, 4)
    print(np.allclose(out, wc @ Xe), np.allclose(wc, softmax(Xd @ Xe.T / 2.0)))  # True True
    Q, K, V = rng.randn(2, 5), rng.randn(4, 5), rng.randn(4, 3)
    o, a = attention(Q, K, V)
    print(o.shape, a.shape, np.allclose(o, softmax(Q @ K.T / np.sqrt(5)) @ V))   # (2, 3) (2, 4) True
