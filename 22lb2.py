"""Day 22 / Session 1 - Lab Assignment 2: Greedy Decoding, the Fixed-Context Bottleneck & Exact-Match Accuracy."""
import numpy as np


def onehot_logits(tokens, V=6):
    # logits whose per-row argmax is exactly `tokens` (a stand-in for a real decoder)
    L = np.full((len(tokens), V), -1.0)
    for t, tok in enumerate(tokens):
        L[t, tok] = 5.0
    return L


def greedy_decode(logits, eos_id=2, max_len=None):
    logits = np.asarray(logits)
    steps = logits.shape[0] if max_len is None else min(max_len, logits.shape[0])
    out = []
    for t in range(steps):
        token = int(np.argmax(logits[t]))
        if token == eos_id:
            break                                   # stop at the first EOS and do not include it
        out.append(token)
    return out


def decode_batch(logits_list, eos_id=2, max_len=None):
    return [greedy_decode(L, eos_id, max_len) for L in logits_list]


def summarize_context(X):
    return np.asarray(X, dtype=float).mean(axis=0)  # one fixed [d] vector, whatever T is


def exact_match(pred, target):
    return 1.0 if list(pred) == list(target) else 0.0


def sequence_accuracy(preds, targets):
    if len(preds) == 0:
        return 0.0
    return float(np.mean([exact_match(p, t) for p, t in zip(preds, targets)]))


if __name__ == "__main__":
    print(greedy_decode(onehot_logits([3, 4, 5, 2, 4])))                      # [3, 4, 5]
    print(greedy_decode(onehot_logits([3, 4, 5, 3])))                         # [3, 4, 5, 3]
    print(greedy_decode(onehot_logits([3, 4, 5, 3]), max_len=2))              # [3, 4]
    X = np.random.default_rng(0).normal(size=(10, 8))
    print(summarize_context(X).shape, np.allclose(summarize_context(X), X.mean(axis=0)))   # (8,) True
    print(summarize_context(X[:3]).shape)                                     # (8,)
    print(sequence_accuracy([[3, 4], [5, 6]], [[3, 4], [5, 9]]), sequence_accuracy([], []))   # 0.5 0.0
    print(decode_batch([onehot_logits([3, 4, 2]), onehot_logits([5, 5, 5]), onehot_logits([4, 2, 3])]))   # [[3, 4], [5, 5, 5], [4]]
    print(np.allclose(summarize_context(X), summarize_context(X[::-1])))      # True (order-blind)
