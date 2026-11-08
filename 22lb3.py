"""Day 22 / Session 2 - Lab Assignment 1: Additive (Bahdanau) Attention from Scratch - Scores, Weights & Context."""
import numpy as np


def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))       # subtract the max so nothing overflows
    return e / np.sum(e, axis=axis, keepdims=True)


def bahdanau_scores(query, keys, W_q, W_k, v_vec):
    # e_i = v . tanh(W_q q + W_k k_i)
    hidden = np.tanh(np.asarray(W_q) @ np.asarray(query) + np.asarray(keys) @ np.asarray(W_k).T)   # (T, d_a)
    return hidden @ np.asarray(v_vec)                                                           # (T,)


def attention_weights(scores):
    return stable_softmax(scores)


def context_vector(weights, values):
    return np.asarray(weights) @ np.asarray(values)           # weighted sum of the value rows, (d_v,)


def alignment_matrix(queries, keys, W_q, W_k, v_vec):
    return np.array([attention_weights(bahdanau_scores(q, keys, W_q, W_k, v_vec)) for q in np.asarray(queries)])


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    d_q, d_k, d_a, d_v, T = 4, 3, 5, 6, 5
    query = rng.normal(size=d_q)
    keys = rng.normal(size=(T, d_k))
    values = rng.normal(size=(T, d_v))
    W_q = rng.normal(size=(d_a, d_q))
    W_k = rng.normal(size=(d_a, d_k))
    v_vec = rng.normal(size=d_a)
    queries = rng.normal(size=(3, d_q))
    e = bahdanau_scores(query, keys, W_q, W_k, v_vec)
    w = attention_weights(e)
    print(e.shape, w.shape, round(float(w.sum()), 6))                         # (5,) (5,) 1.0
    manual = [float(v_vec @ np.tanh(W_q @ query + W_k @ keys[i])) for i in range(T)]
    print(np.allclose(e, manual))                                             # True
    c = context_vector(w, values)
    print(c.shape, np.allclose(c, sum(w[i] * values[i] for i in range(T))))   # (6,) True
    Aln = alignment_matrix(queries, keys, W_q, W_k, v_vec)
    print(Aln.shape, np.allclose(Aln.sum(axis=1), 1))                         # (3, 5) True
    print(attention_weights(bahdanau_scores(query, keys, W_q, W_k, np.zeros(5))))           # [0.2 0.2 0.2 0.2 0.2]
    print(np.isfinite(stable_softmax(np.array([1000.0, 1001.0]))).all(), np.allclose(stable_softmax([1, 2, 3]), stable_softmax([101, 102, 103])))   # True True
