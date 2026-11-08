"""Day 25 / Session 3 - Lab Assignment 1: The Masked-LM Objective - 80/10/10 Masking Pipeline & MLM Cross-Entropy."""
import numpy as np


def maskable_positions(tokens, special_ids=()):
    special = set(special_ids)
    return [i for i, t in enumerate(tokens) if t not in special]          # left to right


def choose_mask_positions(tokens, mask_prob=0.15, special_ids=(), seed=0):
    candidates = maskable_positions(tokens, special_ids)
    n = len(candidates)
    if n == 0:
        return []
    k = min(n, max(1, int(round(mask_prob * n))))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(candidates, size=k, replace=False)
    return sorted(int(p) for p in chosen)


def apply_masking(tokens, positions, mask_id, vocab_size, seed=0):
    masked = [int(t) for t in tokens]
    labels = [-100] * len(masked)
    rng = np.random.RandomState(seed)
    for p in sorted(positions):
        labels[p] = int(tokens[p])                                         # the original token is the target
        r = rng.rand()
        if r < 0.8:
            masked[p] = mask_id                                            # 80%: [MASK]
        elif r < 0.9:
            masked[p] = int(rng.randint(0, vocab_size))                    # 10%: a random token
        # else: 10% unchanged
    return masked, labels


def mlm_cross_entropy(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    keep = labels != ignore_index
    if not keep.any():
        return 0.0
    z = logits[keep]
    z = z - z.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))          # stable log-softmax
    return float(-np.mean(log_probs[np.arange(len(z)), labels[keep]]))


if __name__ == "__main__":
    CLS, SEP, MASK = 101, 102, 103
    tokens = [CLS] + list(range(10, 50)) + [SEP]
    print(maskable_positions([CLS, 5, 6, SEP], (CLS, SEP)))                # [1, 2]
    pos = choose_mask_positions(tokens, 0.15, (CLS, SEP), seed=0)
    print(len(pos), pos == sorted(pos), pos == choose_mask_positions(tokens, 0.15, (CLS, SEP), seed=0), 0 not in pos and 41 not in pos)   # 6 True True True
    masked, labels = apply_masking(list(range(10, 30)), [2, 5, 7, 11], MASK, 1000, seed=0)
    print(sum(l != -100 for l in labels), [labels[i] for i in (2, 5, 7, 11)], labels[0])    # 4 [12, 15, 17, 21] -100
    V = 50
    print(round(mlm_cross_entropy(np.zeros((3, V)), np.array([-100, 7, -100])), 4))         # 3.912
    print(mlm_cross_entropy(np.zeros((3, V)), np.array([-100, -100, -100])))                # 0.0
    confident = np.full((2, V), -50.0); confident[0, 3] = 50.0; confident[1, 9] = 50.0
    print(round(mlm_cross_entropy(confident, np.array([3, 9])), 6))                         # ~0
    big = list(range(1000, 11000))
    m, l = apply_masking(big, list(range(10000)), MASK, 20000, seed=1)
    frac_mask = np.mean([a == MASK for a in m]); frac_same = np.mean([a == b for a, b in zip(m, big)])
    print(round(float(frac_mask), 2), round(float(frac_same), 2))                           # ~0.8 ~0.1
