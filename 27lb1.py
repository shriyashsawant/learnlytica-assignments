"""Day 27 / Session 1 - Lab Assignment 1: Decoding as Distribution Transforms - Temperature Softmax, Entropy & Sampling."""
import numpy as np

logits = np.array([2.0, 1.0, 0.1, -1.0, 0.5])         # raw scores for a 5-token vocab


def stable_softmax(logits, temperature=1.0):
    z = np.asarray(logits, dtype=float) / temperature  # divide by the temperature first
    z = z - z.max()                                    # then subtract the max so nothing overflows
    e = np.exp(z)
    return e / e.sum()


def entropy(probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]                                       # treat 0 * log(0) as 0
    return float(-np.sum(p * np.log(p)) + 0.0)         # Shannon entropy in nats (+0.0 avoids -0.0)


def greedy_next_token(logits):
    return int(np.argmax(logits))                      # temperature never changes the argmax


def sample_index(probs, u):
    probs = np.asarray(probs, dtype=float)
    cumulative = np.cumsum(probs)
    i = int(np.searchsorted(cumulative, u, side="left"))      # smallest i with cumsum[i] >= u (inverse CDF)
    i = min(i, len(probs) - 1)                                # guard against float round-off at u ~ 1
    while probs[i] == 0 and i < len(probs) - 1:               # a boundary draw never lands on a zero-probability token
        i += 1
    return i


def temperature_entropy(logits, temperatures):
    return np.array([entropy(stable_softmax(logits, T)) for T in temperatures])


if __name__ == "__main__":
    p = stable_softmax(logits, 1.0)
    print(np.round(p, 4))                                                       # [0.5585 0.2055 0.0835 0.0278 0.1246]
    print(round(entropy(p), 4))                                                 # 1.217
    print(round(stable_softmax(logits, 2.0).max(), 4))                          # 0.3719
    print(round(stable_softmax(logits, 0.5).max(), 4))                          # 0.8265
    print(greedy_next_token(logits))                                            # 0
    print(sample_index(np.array([0.1, 0.2, 0.3, 0.4]), 0.55))                   # 2
    big = stable_softmax(np.array([1000.0, 999.0, 0.0]))
    print(np.isnan(big).any(), round(float(big.sum()), 6), int(big.argmax()))   # False 1.0 0
    print(round(entropy(np.full(5, 0.2)), 4), round(float(np.log(5)), 4), entropy([1.0, 0, 0]))   # 1.6094 1.6094 0.0
    te = temperature_entropy(logits, [0.1, 0.5, 1.0, 2.0, 10.0])
    print(np.round(te, 3), bool(np.all(np.diff(te) > 0)))                       # strictly increasing
    print(sample_index(np.array([0.0, 0.5, 0.5]), 0.0), sample_index(np.array([0.5, 0.5]), 0.5), sample_index(np.array([0.5, 0.5]), 0.5001))   # 1 0 1
