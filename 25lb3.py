"""Day 25 / Session 2 - Lab Assignment 1: Multi-Head Self-Attention from Scratch."""
import numpy as np


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    dk = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)        # works with or without a leading head axis
    if mask is not None:
        scores = np.where(mask, scores, -1e9)                # False positions get a large negative score
    attn = softmax(scores, axis=-1)
    return attn @ V, attn


def split_heads(x, h):
    x = np.asarray(x, dtype=float)
    L, d_model = x.shape
    d_head = d_model // h
    return x.reshape(L, h, d_head).transpose(1, 0, 2)        # (h, L, d_head): columns k*d_head:(k+1)*d_head -> head k


def combine_heads(x):
    x = np.asarray(x, dtype=float)
    h, L, d_head = x.shape
    return x.transpose(1, 0, 2).reshape(L, h * d_head)


def multi_head_attention(X, Wq, Wk, Wv, Wo, h, mask=None):
    X = np.asarray(X, dtype=float)
    Q = split_heads(X @ Wq, h)
    K = split_heads(X @ Wk, h)
    V = split_heads(X @ Wv, h)
    out, _ = scaled_dot_product_attention(Q, K, V, mask)     # one attention per head, in parallel
    return combine_heads(out) @ Wo


if __name__ == "__main__":
    rng = np.random.default_rng(25)
    L, d_model, h = 5, 8, 2
    X = rng.normal(size=(L, d_model))
    Wq, Wk, Wv, Wo = (rng.normal(size=(d_model, d_model)) for _ in range(4))
    Q = np.zeros((2, 4)); K = np.zeros((5, 4)); V = rng.normal(size=(5, 8))
    out, attn = scaled_dot_product_attention(Q, K, V)
    print(np.round(attn[0], 3), np.allclose(out, V.mean(0)))                    # [0.2 ...] True
    x = rng.normal(size=(6, 12))
    print(split_heads(x, 4).shape, np.allclose(combine_heads(split_heads(x, 4)), x))   # (4, 6, 3) True
    Y = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    print(Y.shape)                                                              # (5, 8)
    mask = np.tril(np.ones((L, L), dtype=bool))
    _, a = scaled_dot_product_attention(rng.normal(size=(L, 4)), rng.normal(size=(L, 4)), rng.normal(size=(L, 4)), mask=mask)
    print(np.round(a[0], 3), np.allclose(a.sum(axis=-1), 1))                    # [1. 0. 0. 0. 0.] True
    Y1 = multi_head_attention(X, Wq, Wk, Wv, Wo, 1)
    o1, _ = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)
    print(np.allclose(Y1, o1 @ Wo))                                             # True (h=1 reduces to plain attention)
    q, k = rng.normal(size=(3, 4)), rng.normal(size=(3, 4))
    print(np.allclose(scaled_dot_product_attention(q, k, np.eye(3))[1], softmax(q @ k.T / 2.0)))   # True
