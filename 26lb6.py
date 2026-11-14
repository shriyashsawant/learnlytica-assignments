"""Day 26 / Session 3 - Lab Assignment 2: Decoding Knobs from Scratch - Temperature, Top-k & Top-p (Nucleus) Filtering."""
import numpy as np

logits = np.array([3.0, 1.0, 0.5, -2.0])              # raw model scores for 4 tokens
probs = np.array([0.15, 0.50, 0.05, 0.30])            # an example next-token distribution


def softmax(logits):
    z = np.asarray(logits, dtype=float)
    e = np.exp(z - z.max())                           # subtract the max for stability
    return e / e.sum()


def apply_temperature(logits, T):
    return softmax(np.asarray(logits, dtype=float) / T)          # T=1 plain softmax; T>1 flattens; T<1 sharpens


def top_k_filter(probs, k):
    probs = np.asarray(probs, dtype=float)
    out = np.zeros_like(probs)
    keep = np.argsort(-probs, kind="stable")[:max(int(k), 1)]    # indices of the k highest probabilities
    out[keep] = probs[keep]
    return out / out.sum()                                       # renormalize


def nucleus_support(probs, p):
    probs = np.asarray(probs, dtype=float)
    if p >= 1.0:
        return np.arange(len(probs))                             # p >= 1 keeps everything
    order = np.argsort(-probs, kind="stable")                    # highest probability first
    cumulative = np.cumsum(probs[order])
    size = int(np.searchsorted(cumulative, p - 1e-12) + 1)       # smallest prefix whose cumulative mass reaches p
    return np.sort(order[:max(size, 1)])                         # always at least the top token; original indices


def top_p_filter(probs, p):
    probs = np.asarray(probs, dtype=float)
    out = np.zeros_like(probs)
    keep = nucleus_support(probs, p)
    out[keep] = probs[keep]
    return out / out.sum()


if __name__ == "__main__":
    print(np.round(softmax(np.array([2.0, 1.0, 0.0])), 4))                         # [0.6652 0.2447 0.09  ]
    print(apply_temperature(logits, 1.0).max() > apply_temperature(logits, 3.0).max())     # True
    print(np.allclose(apply_temperature(logits, 1.0), softmax(logits)))             # True
    print(np.round(top_k_filter(probs, 2), 3))                                      # [0.    0.625 0.    0.375]
    print(list(nucleus_support(probs, 0.75)))                                       # [1, 3]
    print(np.round(top_p_filter(probs, 0.75), 3))                                   # [0.    0.625 0.    0.375]
    print(int(np.argmax(apply_temperature(np.array([2., 1., 0.]), 0.05))))          # 0
    print(np.round(apply_temperature(np.array([2., 1., 0.]), 0.05), 3))             # ~one-hot
    print(list(nucleus_support(probs, 1.0)), list(nucleus_support(probs, 1e-6)))    # [0, 1, 2, 3] [1]
    print(np.round(top_k_filter(probs, 10), 3), np.round(top_k_filter(probs, 1), 3))    # all kept / one-hot at the argmax
