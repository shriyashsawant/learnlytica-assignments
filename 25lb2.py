"""Day 25 / Session 1 - Lab Assignment 2: Self- vs Cross-Attention & a Future-Leakage Audit."""
import numpy as np

X = np.array([[1., 0., 1., 0.],
              [0., 1., 1., 0.],
              [1., 1., 0., 1.]])                  # 3 tokens, d = 4
ctx = np.array([[2., 0., 0., 1.],
                [0., 2., 1., 0.],
                [1., 1., 1., 1.],
                [0., 0., 2., 2.],
                [1., 0., 1., 0.]])                # 5 context tokens, d = 4


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))       # -inf -> 0, every slice sums to 1
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    mask = np.zeros((n, n))
    mask[np.triu_indices(n, k=1)] = -np.inf                   # additive: 0 on/below the diagonal, -inf above it
    return mask


def self_attention(X, mask=None):
    X = np.asarray(X, dtype=float)
    scores = X @ X.T / np.sqrt(X.shape[-1])                   # Q, K and V are all X
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ X, weights                               # weights is (n, n)


def cross_attention(X, context):
    X, context = np.asarray(X, dtype=float), np.asarray(context, dtype=float)
    scores = X @ context.T / np.sqrt(X.shape[-1])             # queries from X, keys/values from the context, no mask
    weights = softmax(scores, axis=-1)                        # (n_q, n_ctx)
    return weights @ context, weights                         # out is (n_q, d)


def audit_future_leakage(weights):
    weights = np.asarray(weights, dtype=float)
    return bool(np.any(np.triu(weights, k=1) > 0))            # True if any token attends strictly ahead


if __name__ == "__main__":
    out, w_full = self_attention(X)
    out, w_causal = self_attention(X, mask=causal_mask(3))
    print(audit_future_leakage(w_causal))                     # False
    print(audit_future_leakage(w_full))                       # True
    out, w = cross_attention(X, ctx)
    print(w.shape, out.shape)                                 # (3, 5) (3, 4)
    print(np.round(w.sum(axis=1), 6), np.round(w_causal.sum(axis=1), 6))      # rows sum to 1
