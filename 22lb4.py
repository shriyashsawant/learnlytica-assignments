"""Day 22 / Session 2 - Lab Assignment 2: Scaled Dot-Product (Luong) Attention & Why sqrt(d_k) Matters."""
import numpy as np


def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def dot_scores(query, keys):
    return np.asarray(keys, dtype=float) @ np.asarray(query, dtype=float)    # e_i = key_i . query, shape (T,)


def scaled_dot_scores(query, keys):
    keys = np.asarray(keys, dtype=float)
    d_k = keys.shape[-1]
    return dot_scores(query, keys) / np.sqrt(d_k)


def attention_weights(scores):
    return stable_softmax(scores)


def context_vector(weights, values):
    return np.asarray(weights) @ np.asarray(values)


def alignment_matrix(Q, K):
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    d_k = K.shape[-1]
    return stable_softmax(Q @ K.T / np.sqrt(d_k), axis=-1)                   # softmax(Q K^T / sqrt(d_k)), row-wise


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    d, T, d_v = 4, 6, 5
    query = rng.normal(size=d)
    keys = rng.normal(size=(T, d))
    values = rng.normal(size=(T, d_v))
    Q = rng.normal(size=(3, d))
    e = dot_scores(query, keys)
    es = scaled_dot_scores(query, keys)
    print(np.allclose(es, e / np.sqrt(4)))                                   # True
    w = attention_weights(es)
    print(round(float(w.sum()), 6), context_vector(w, values).shape)         # 1.0 (5,)
    Aln = alignment_matrix(Q, keys)
    print(Aln.shape, np.allclose(Aln.sum(axis=1), 1))                        # (3, 6) True
    rng2 = np.random.default_rng(3)
    qh = rng2.normal(size=64)
    Kh = rng2.normal(size=(8, 64))
    ent = lambda p: -np.sum(p * np.log(p + 1e-12))
    print(ent(attention_weights(scaled_dot_scores(qh, Kh))) > ent(attention_weights(dot_scores(qh, Kh))))   # True
    E = np.eye(5)                                                            # orthonormal keys
    A = alignment_matrix(E[2:3], E)[0]
    print(int(np.argmax(A)), np.allclose(np.delete(A, 2), np.delete(A, 2)[0]))                              # 2 True
