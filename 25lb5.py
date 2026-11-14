"""Day 25 / Session 3 - Lab Assignment 1: The Masked-LM Objective - 80/10/10 Masking Pipeline & MLM Cross-Entropy."""
import numpy as np

tokens = list(range(10, 22))          # a 12-token "sentence"
positions = list(range(10))           # the positions chosen to be masked


def apply_masking(tokens, positions, mask_id, vocab_size, seed=0):
    rng = np.random.RandomState(seed)
    masked = [int(t) for t in tokens]
    labels = [-100] * len(tokens)                       # non-selected positions get label -100
    for p in sorted(positions):                         # iterate in sorted order, as the spec requires
        labels[p] = int(tokens[p])                      # record the original token
        r = rng.rand()
        if r < 0.8:
            masked[p] = int(mask_id)                    # 80%: [MASK]
        elif r < 0.9:
            masked[p] = int(rng.randint(0, vocab_size)) # 10%: a random token
        # else 10%: left unchanged
    return masked, labels


def mlm_loss(logits, labels, ignore_index=-100):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    keep = labels != ignore_index
    if not keep.any():
        return 0.0                                      # nothing is masked
    z = logits[keep]
    z = z - z.max(axis=-1, keepdims=True)               # stable log-softmax
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    picked = log_probs[np.arange(len(z)), labels[keep].astype(int)]
    return float(-picked.mean()) + 0.0                  # mean cross-entropy over masked positions only (+0.0 avoids -0.0)


if __name__ == "__main__":
    masked, labels = apply_masking(list(range(10, 22)), list(range(10)), 103, 1000, seed=0)
    print(sum(l != -100 for l in labels))               # 10
    print(masked, labels)
    V = 50
    print(round(mlm_loss(np.zeros((3, V)), np.array([-100, 7, -100])), 4))      # 3.912
    print(mlm_loss(np.zeros((3, V)), np.full(3, -100)))                         # 0.0
    big = apply_masking(list(range(5000)), list(range(5000)), 103, 1000, seed=1)[0]
    print(round(sum(m == 103 for m in big) / 5000, 2))                          # ~0.8
    conf = np.full((2, V), -20.0); conf[0, 3] = 20.0; conf[1, 9] = 20.0
    print(round(mlm_loss(conf, np.array([3, 9])), 4), mlm_loss(conf, np.array([3, -100])) < 1e-6)   # ~0 ~0
