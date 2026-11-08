"""Day 21 / Session 2 - Lab Assignment 2: Taming the Explosion - Global-Norm Gradient Clipping."""
import numpy as np


def global_norm(grads):
    return float(np.sqrt(sum(float(np.sum(np.asarray(g, dtype=float) ** 2)) for g in grads)))


def did_clip(grads, max_norm):
    return bool(global_norm(grads) > max_norm)


def clip_by_global_norm(grads, max_norm):
    norm = global_norm(grads)
    if norm == 0 or norm <= max_norm:
        return [np.array(g, dtype=float, copy=True) for g in grads]    # nothing to do
    factor = max_norm / norm                                           # ONE shared factor keeps the direction
    return [np.asarray(g, dtype=float) * factor for g in grads]


if __name__ == "__main__":
    grads = [np.array([3.0, 4.0]), np.array([12.0, 0.0])]
    print(round(global_norm(grads), 4))                      # 13.0
    print(did_clip(grads, 6.5), did_clip(grads, 100.0))      # True False
    out = clip_by_global_norm(grads, 6.5)
    print([list(a) for a in out])                            # [[1.5, 2.0], [6.0, 0.0]]
    print(round(global_norm(out), 4))                        # 6.5
    print(clip_by_global_norm(grads, 100.0))                 # unchanged
    print(global_norm([np.array([3.0, 4.0])]))               # 5.0
    print(clip_by_global_norm([np.zeros(3), np.zeros((2, 2))], 1.0))   # zeros, no divide-by-zero
