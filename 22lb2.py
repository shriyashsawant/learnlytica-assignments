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
    steps = len(logits) if max_len is None else min(len(logits), max_len)   # never run past T (or max_len)
    out = []
    for t in range(steps):
        token = int(np.argmax(logits[t]))
        if token == eos_id:                                                 # stop at the first EOS, do not include it
            break
        out.append(token)
    return out


def decode_batch(logits_list, eos_id=2, max_len=None):
    return [greedy_decode(logits, eos_id, max_len) for logits in logits_list]


def exact_match(pred, target):
    return 1.0 if list(pred) == list(target) else 0.0


def sequence_accuracy(preds, targets):
    if len(preds) == 0:
        return 0.0
    return float(np.mean([exact_match(p, t) for p, t in zip(preds, targets)]))


if __name__ == "__main__":
    print(greedy_decode(onehot_logits([3, 4, 5, 2, 4])))                                           # [3, 4, 5]
    print(greedy_decode(onehot_logits([3, 4, 5, 3])))                                              # [3, 4, 5, 3]
    print(decode_batch([onehot_logits([3, 4, 2, 5]), onehot_logits([5, 5, 5])]))                   # [[3, 4], [5, 5, 5]]
    print(sequence_accuracy([[3, 4], [5, 6]], [[3, 4], [5, 9]]))                                   # 0.5
    print(sequence_accuracy([], []), greedy_decode(onehot_logits([3, 4, 5, 3]), max_len=2))        # 0.0 [3, 4]
