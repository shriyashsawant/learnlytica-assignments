"""Day 26 / Session 3 - Lab Assignment 2: Decoding Knobs from Scratch - Temperature, Top-k & Top-p (Nucleus) Filtering."""
import numpy as np


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - np.max(z))                                # stable: subtract the max first
    return e / e.sum()


def apply_temperature(logits, T):
    return softmax(np.asarray(logits, dtype=float) / T)      # T<1 sharpens, T>1 flattens


def _renormalize(kept):
    return kept / kept.sum()


def top_k_filter(probs, k):
    probs = np.asarray(probs, dtype=float)
    k = max(1, int(k))
    order = np.argsort(-probs, kind="stable")                # highest probability first, ties by original index
    kept = np.zeros_like(probs)
    keep_idx = order[:k]
    kept[keep_idx] = probs[keep_idx]
    return _renormalize(kept)


def nucleus_support(probs, p):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    size = int(np.searchsorted(cumulative, p - 1e-12, side="left")) + 1     # smallest prefix with cumulative >= p
    size = min(max(size, 1), len(probs))                                    # always at least the top token
    return np.sort(order[:size])


def top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=float)
    keep_idx = nucleus_support(probs, p)
    kept = np.zeros_like(probs)
    kept[keep_idx] = probs[keep_idx]
    return _renormalize(kept)


if __name__ == "__main__":
    logits = np.array([3.0, 1.0, 0.5, -2.0])
    probs = np.array([0.15, 0.50, 0.05, 0.30])
    print(np.round(softmax(np.array([2.0, 1.0, 0.0])), 4), round(float(softmax(logits).sum()), 6))      # [0.6652 0.2447 0.09] 1.0
    print(np.allclose(apply_temperature(logits, 1.0), softmax(logits)), apply_temperature(logits, 1.0).max() > apply_temperature(logits, 3.0).max())   # True True
    ent = lambda q: -np.sum(q * np.log(q + 1e-12))
    print(ent(apply_temperature(logits, 3.0)) > ent(apply_temperature(logits, 1.0)))                     # True
    print(np.round(top_k_filter(probs, 2), 3))                                                           # [0. 0.625 0. 0.375]
    print(list(nucleus_support(probs, 0.75)))                                                            # [1, 3]
    print(np.round(top_p_filter(probs, 0.75), 3))                                                        # [0. 0.625 0. 0.375]
    print(int(np.argmax(apply_temperature(np.array([2., 1., 0.]), 0.05))), np.round(apply_temperature(np.array([2., 1., 0.]), 0.05), 3))   # 0 [1. 0. 0.]
    print(list(nucleus_support(probs, 1.0)), list(nucleus_support(probs, 0.01)))                          # [0, 1, 2, 3] [1]
    print(np.round(top_k_filter(probs, 10), 3), np.round(top_k_filter(probs, 1), 3))                      # all kept / one-hot at index 1
