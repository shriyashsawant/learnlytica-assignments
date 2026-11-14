"""Day 22 / Session 2 - Lab Assignment 2: Scaled Dot-Product (Luong) Attention & Why sqrt(d_k) Matters."""
import numpy as np

rng = np.random.default_rng(11)
d, T, d_v, Tq = 4, 6, 5, 3                    # key/query dim, source steps, value dim, number of queries
K = rng.normal(size=(T, d))
V = rng.normal(size=(T, d_v))
Q = rng.normal(size=(Tq, d))


def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    z = np.exp(x - np.max(x, axis=axis, keepdims=True))        # subtract the max along the axis for stability
    return z / z.sum(axis=axis, keepdims=True)


def attention_weights(Q, K):
    Q, K = np.asarray(Q, dtype=float), np.asarray(K, dtype=float)
    d_k = K.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d_k), axis=-1)            # (Tq, T), each row sums to 1


def attention(Q, K, V):
    return attention_weights(Q, K) @ np.asarray(V, dtype=float)    # (Tq, d_v)


if __name__ == "__main__":
    W = attention_weights(Q, K)
    print(W.shape, np.allclose(W.sum(axis=1), 1))              # (3, 6) True
    print(attention(Q, K, V).shape)                            # (3, 5)
    rng2 = np.random.default_rng(3)
    qh = rng2.normal(size=(1, 64)); Kh = rng2.normal(size=(8, 64))
    scaled = attention_weights(qh, Kh)
    unscaled = softmax(qh @ Kh.T, axis=-1)
    print(scaled.max() < unscaled.max())                       # True
    # known-answer case: orthonormal keys, query equal to the first key
    eye = np.eye(4)
    print(np.round(attention_weights(eye[:1], eye), 4), np.allclose(attention(eye[:1], eye, eye).sum(), 1))
