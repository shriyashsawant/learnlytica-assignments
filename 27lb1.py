"""Day 27 / Session 1 - Lab Assignment 1: Decoding as Distribution Transforms - Temperature Softmax, Entropy & Sampling."""
import numpy as np


def stable_softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=float) / temperature        # temperature first
    e = np.exp(z - np.max(z))                                # then subtract the max so nothing overflows
    return e / e.sum()


def entropy(probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]                                             # 0 * log 0 is treated as 0
    return float(-np.sum(p * np.log(p)))


def greedy_next_token(logits):
    return int(np.argmax(logits))                            # temperature never changes the argmax


def sample_index(probs, u):
    cumulative = np.cumsum(np.asarray(probs, dtype=float))
    index = int(np.searchsorted(cumulative, u, side="left"))   # smallest i with cumsum[i] >= u
    return min(index, len(cumulative) - 1)


def temperature_entropy(logits, temperatures):
    return np.array([entropy(stable_softmax(logits, T)) for T in temperatures])


if __name__ == "__main__":
    logits = np.array([2.0, 1.0, 0.1, -1.0, 0.5])
    p = stable_softmax(logits, 1.0)
    print(np.round(p, 4), round(float(p.sum()), 6))                         # [0.5585 0.2055 0.0835 0.0278 0.1246] 1.0
    print(round(entropy(p), 4))                                             # 1.217
    print(round(stable_softmax(logits, 2.0).max(), 4), round(stable_softmax(logits, 0.5).max(), 4))   # 0.3719 0.8265
    print(greedy_next_token(logits), np.isfinite(stable_softmax(np.array([1000.0, 999.0, 0.0]))).all())   # 0 True
    print(sample_index(np.array([0.1, 0.2, 0.3, 0.4]), 0.55), sample_index(np.array([0.1, 0.2, 0.3, 0.4]), 0.0))   # 2 0
    print(sample_index(np.array([0.0, 0.3, 0.0, 0.7]), 0.3), sample_index(np.array([0.0, 0.3, 0.0, 0.7]), 0.31))    # 1 3
    print(round(entropy(np.full(5, 0.2)), 4), round(float(np.log(5)), 4), entropy(np.array([0.0, 1.0, 0.0])))      # 1.6094 1.6094 0.0
    h = temperature_entropy(logits, [0.1, 0.5, 1.0, 2.0, 10.0])
    print(np.round(h, 3), bool(np.all(np.diff(h) > 0)))                     # strictly increasing
