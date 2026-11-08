"""Day 27 / Session 1 - Lab Assignment 2: Top-k and Top-p (Nucleus) Filtering - Truncating the Sampling Distribution."""
import numpy as np


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - np.max(z))
    return e / e.sum()


def top_k_mask(probs, k):
    probs = np.asarray(probs, dtype=float)
    k = min(max(int(k), 0), len(probs))                      # clamp k to the vocabulary size
    order = np.argsort(-probs, kind="stable")
    mask = np.zeros(len(probs), dtype=bool)
    mask[order[:k]] = True
    return mask


def top_k_filter(probs, k):
    probs = np.asarray(probs, dtype=float)
    kept = np.where(top_k_mask(probs, k), probs, 0.0)
    return kept / kept.sum()                                 # renormalise; ratios between kept tokens are unchanged


def nucleus_size(probs, p_thresh):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")
    cumulative = np.cumsum(probs[order])
    size = int(np.searchsorted(cumulative, p_thresh - 1e-12, side="left")) + 1    # smallest prefix with cumsum >= p
    return int(min(max(size, 1), len(probs)))                                     # always keep at least one token


def top_p_filter(probs, p_thresh):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")
    keep = order[:nucleus_size(probs, p_thresh)]
    kept = np.zeros_like(probs)
    kept[keep] = probs[keep]
    return kept / kept.sum()


if __name__ == "__main__":
    logits = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
    p = softmax(logits)
    print(np.round(p, 4))                                    # [0.6364 0.2341 0.0861 0.0317 0.0117]
    print(np.round(top_k_filter(p, 2), 4))                   # [0.7311 0.2689 0. 0. 0.]
    print(np.round(top_p_filter(p, 0.9), 4))                 # [0.6652 0.2447 0.09 0. 0.]
    print(nucleus_size(p, 0.9), nucleus_size(p, 0.5), nucleus_size(p, 1.0))               # 3 1 5
    print(top_k_mask(p, 2).tolist())                         # [True, True, False, False, False]
    print(np.allclose(top_k_filter(p, 99), p), np.round(top_p_filter(p, 0.1), 3))          # True one-hot
    sizes = [nucleus_size(p, q) for q in (0.1, 0.5, 0.7, 0.9, 0.99, 1.0)]
    print(sizes, all(a <= b for a, b in zip(sizes, sizes[1:])))                            # non-decreasing
    kept = top_p_filter(p, 0.9)
    print(np.isclose(kept[0] / kept[1], p[0] / p[1]), round(float(kept.sum()), 6))         # True 1.0
