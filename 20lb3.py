"""Day 20 / Session 2 - Lab Assignment 1: Grad-CAM by Hand - From Feature Maps and Gradients to a Heatmap."""
import numpy as np


def gap_weights(grads):
    return np.asarray(grads, dtype=float).mean(axis=(1, 2))      # one weight per channel


def weighted_sum(fmaps, weights):
    fmaps = np.asarray(fmaps, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.tensordot(weights, fmaps, axes=(0, 0))             # sum_c weights[c] * fmaps[c]


def relu(x):
    return np.maximum(np.asarray(x, dtype=float), 0)


def normalize01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)                                  # constant map: no divide-by-zero
    return (x - lo) / (hi - lo)


def grad_cam(fmaps, grads):
    return normalize01(relu(weighted_sum(fmaps, gap_weights(grads))))


def upsample_nearest(heat, out_h, out_w):
    heat = np.asarray(heat, dtype=float)
    h, w = heat.shape
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = heat[i * h // out_h, j * w // out_w]
    return out


if __name__ == "__main__":
    fmaps = np.array([[[1., 2.], [3., 4.]], [[0., 1.], [1., 0.]]])
    grads = np.array([[[0., 2.], [4., 2.]], [[1., 1.], [1., 5.]]])
    print(gap_weights(grads))                                    # [2. 2.]
    print(weighted_sum(fmaps, np.array([2., 3.])))               # [[2. 7.] [9. 8.]]
    print(grad_cam(fmaps, grads))                                # [[0. 0.667] [1. 1.]]
    print(upsample_nearest(np.array([[0., 1.], [2., 3.]]), 4, 4))
    print(normalize01(np.full((2, 2), 3.0)))                     # zeros
    print(relu([-1, 2]))
