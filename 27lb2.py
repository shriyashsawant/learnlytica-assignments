"""Day 27 / Session 1 - Lab Assignment 2: Top-k and Top-p (Nucleus) Filtering - Truncating the Sampling Distribution."""
import numpy as np

logits = np.array([3.0, 2.0, 1.0, 0.0, -1.0])         # 5-token vocab, already in descending order


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - z.max())
    return e / e.sum()


def top_k_mask(probs, k):
    probs = np.asarray(probs, dtype=float)
    k = int(min(max(k, 0), len(probs)))                       # clamp k to the vocab size
    mask = np.zeros(len(probs), dtype=bool)
    mask[np.argsort(-probs, kind="stable")[:k]] = True        # the k highest-probability tokens
    return mask


def top_k_filter(probs, k):
    probs = np.asarray(probs, dtype=float)
    mask = top_k_mask(probs, max(int(k), 1))                  # always keep at least one token
    out = np.where(mask, probs, 0.0)
    return out / out.sum()                                    # renormalise; ratios between kept tokens are unchanged


def _nucleus_count(probs, p_thresh):
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")                 # highest probability first
    cumulative = np.cumsum(probs[order])
    n = int(np.searchsorted(cumulative, p_thresh - 1e-12) + 1)    # smallest prefix whose mass reaches p_thresh
    return int(min(max(n, 1), len(probs))), order             # always at least one token


def top_p_filter(probs, p_thresh):
    probs = np.asarray(probs, dtype=float)
    n, order = _nucleus_count(probs, p_thresh)
    out = np.zeros_like(probs)
    out[order[:n]] = probs[order[:n]]                         # keep the nucleus, zero the tail
    return out / out.sum()


def nucleus_size(probs, p_thresh):
    return _nucleus_count(probs, p_thresh)[0]                 # how many tokens top-p keeps (int >= 1)


if __name__ == "__main__":
    p = softmax(logits)
    print(np.round(p, 4))                                     # [0.6364 0.2341 0.0861 0.0317 0.0117]
    print(np.round(top_k_filter(p, 2), 4))                    # [0.7311 0.2689 0.     0.     0.    ]
    print(np.round(top_p_filter(p, 0.9), 4))                  # [0.6652 0.2447 0.09   0.     0.    ]
    print(nucleus_size(p, 0.9))                               # 3
    print(nucleus_size(p, 0.5))                               # 1
    print(top_k_mask(p, 2).tolist())                          # [True, True, False, False, False]
    print(np.allclose(top_k_filter(p, 99), p), nucleus_size(p, 1.0))          # True 5
    print([nucleus_size(p, t) for t in (0.1, 0.5, 0.8, 0.9, 0.99, 1.0)])      # non-decreasing
    kept = top_p_filter(p, 0.9)
    print(np.isclose(kept[0] / kept[1], p[0] / p[1]), round(float(kept.sum()), 6))   # ratios preserved, sums to 1
