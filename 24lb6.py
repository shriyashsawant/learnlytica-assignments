"""Day 24 / Session 3 - Lab Assignment 2: The RNN Sequential Bottleneck & Positional Information Loss."""
import numpy as np


def rnn_sequential_depth(T):
    return int(T)                                            # h_t depends on h_{t-1}: T steps one after another


def attention_sequential_depth(T):
    return 1                                                 # all positions in one parallel matmul


def sequential_speedup(T):
    return rnn_sequential_depth(T) / attention_sequential_depth(T)


def sinusoidal_positional_encoding(T, d):
    PE = np.zeros((T, d))
    for pos in range(T):
        for i in range(0, d, 2):
            angle = pos / (10000 ** (i / d))                 # i is the even index 2k, so this is pos / 10000^(2k/d)
            PE[pos, i] = np.sin(angle)
            if i + 1 < d:
                PE[pos, i + 1] = np.cos(angle)
    return PE


def mean_pool(X):
    return np.asarray(X, dtype=float).mean(axis=0)


def weighted_pool(X, w):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()                                          # normalise the weights to sum to 1
    return (w[:, None] * X).sum(axis=0)


if __name__ == "__main__":
    print(rnn_sequential_depth(64), attention_sequential_depth(64), sequential_speedup(64))     # 64 1 64.0
    PE = sinusoidal_positional_encoding(4, 6)
    print(round(float(PE[0, 0]), 3), round(float(PE[0, 1]), 3))                                  # 0.0 1.0
    print(round(float(PE[1, 0]), 3), round(float(PE[1, 1]), 3))                                  # 0.841 0.54
    print(PE.shape, float(np.abs(PE).max()) <= 1.0)                                              # (4, 6) True
    X = np.array([[1., 2.], [3., 4.], [5., 0.]])
    print(mean_pool(X))                                                                          # [3. 2.]
    rng = np.random.RandomState(11)
    S = rng.randn(7, 5)
    p = rng.permutation(7)
    print(round(float(np.linalg.norm(mean_pool(S) - mean_pool(S[p]))), 8))                       # 0.0 (order lost)
    w = np.arange(1, 8, dtype=float)
    print(np.allclose(weighted_pool(S, np.ones(7)), mean_pool(S)), np.allclose(weighted_pool(S, w), weighted_pool(S[p], w)))   # True False
