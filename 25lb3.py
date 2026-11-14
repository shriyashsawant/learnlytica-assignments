"""Day 25 / Session 2 - Lab Assignment 1: Multi-Head Self-Attention from Scratch."""
import numpy as np

L, d_model, h = 3, 4, 2                                   # 3 tokens, model width 4, 2 heads (d_head = 2)
X = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]])
Wq = np.eye(4); Wk = np.eye(4); Wv = np.arange(16.).reshape(4, 4) / 10.; Wo = np.eye(4) * 2.


def _softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))     # numerically stable, over the last axis
    return e / e.sum(axis=-1, keepdims=True)


def split_heads(x, h):
    x = np.asarray(x, dtype=float)
    L, d_model = x.shape
    d_head = d_model // h
    return x.reshape(L, h, d_head).transpose(1, 0, 2)      # (h, L, d_head): columns k*d_head:(k+1)*d_head -> head k


def sdpa(Q, K, V):
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    dk = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(dk)      # works with or without a leading head axis
    attn = _softmax(scores)
    return attn @ V, attn


def multi_head(X, Wq, Wk, Wv, Wo, h):
    X = np.asarray(X, dtype=float)
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                        # project, then split into heads
    out, _ = sdpa(split_heads(Q, h), split_heads(K, h), split_heads(V, h))      # (h, L, d_head)
    concat = out.transpose(1, 0, 2).reshape(X.shape[0], -1)                      # back to (L, d_model)
    return concat @ Wo


if __name__ == "__main__":
    Qz = np.zeros((2, 2)); Kz = np.zeros((4, 2)); Vz = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    out, attn = sdpa(Qz, Kz, Vz)
    print(np.round(attn[0], 3))                             # [0.25 0.25 0.25 0.25]
    print(np.allclose(out, Vz.mean(0)))                     # True
    print(split_heads(X, h).shape)                          # (2, 3, 2)
    Y = multi_head(X, Wq, Wk, Wv, Wo, h)
    print(Y.shape)                                          # (3, 4)
    print(np.allclose(split_heads(X, h)[1], X[:, 2:]), np.round(Y[0], 4))   # head 1 holds columns 2:4
    o3, a3 = sdpa(np.random.rand(2, 3, 2), np.random.rand(2, 3, 2), np.random.rand(2, 3, 2))
    print(o3.shape, a3.shape, np.allclose(a3.sum(-1), 1))   # (2, 3, 2) (2, 3, 3) True
