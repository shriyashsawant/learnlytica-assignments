"""Day 22 / Session 2 - Lab Assignment 1: Additive (Bahdanau) Attention from Scratch - Scores, Weights & Context."""
import numpy as np


def softmax(x):
    x = np.asarray(x, dtype=float)
    z = np.exp(x - np.max(x))                       # subtract the max so large logits do not overflow
    return z / z.sum()


def attention_scores(h_dec, H_enc, W1, W2, v):
    h_dec, H_enc = np.asarray(h_dec, dtype=float), np.asarray(H_enc, dtype=float)
    return np.tanh(H_enc @ np.asarray(W1) + h_dec @ np.asarray(W2)) @ np.asarray(v, dtype=float)   # (T,)


def attention(h_dec, H_enc, W1, W2, v):
    weights = softmax(attention_scores(h_dec, H_enc, W1, W2, v))     # alignment weights, sum to 1
    context = weights @ np.asarray(H_enc, dtype=float)               # (d,)
    return weights, context


if __name__ == "__main__":
    H_enc = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    h_dec = np.array([0.5, 0.5])
    v = np.array([1.0, 1.0])
    e = attention_scores(h_dec, H_enc, W1, W2, v)
    print(e.shape)                                                   # (3,)
    w, c = attention(h_dec, H_enc, W1, W2, v)
    print(round(float(w.sum()), 6), c.shape)                         # 1.0 (2,)
    w0, c0 = attention(h_dec, H_enc, W1, W2, np.zeros(2))
    print(w0)                                                        # [0.33333333 0.33333333 0.33333333]
    print(np.allclose(c0, H_enc.mean(axis=0)))                       # True
    print(softmax([1000.0, 1001.0]), np.round(e, 4))                 # no nan on large logits
