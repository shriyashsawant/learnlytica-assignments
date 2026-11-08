"""Day 19 / Session 1 - Lab Assignment 1: Residual Blocks & Batch Normalization from Scratch."""
import numpy as np


def relu(x):
    return np.maximum(0, np.asarray(x, dtype=float))


def batch_norm(x, gamma, beta, eps=1e-5):
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)                       # one mean per feature (column)
    var = x.var(axis=0)                         # population variance (ddof=0)
    xhat = (x - mean) / np.sqrt(var + eps)
    return np.asarray(gamma) * xhat + np.asarray(beta)


def residual_block(x, W, gamma, beta):
    x = np.asarray(x, dtype=float)
    F = batch_norm(x @ W, gamma, beta)          # the learned residual branch
    return relu(F + x)                          # identity shortcut: add the input straight back


def projection_shortcut(x, Wp):
    return np.asarray(x, dtype=float) @ Wp      # 1x1-style linear projection (N, D_in) -> (N, D_out)


def residual_block_proj(x, W, Wp, gamma, beta):
    x = np.asarray(x, dtype=float)
    F = batch_norm(x @ W, gamma, beta)          # (N, D_out)
    return relu(F + projection_shortcut(x, Wp))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    x = rng.randn(200, 4) * 5.0 + 2.0
    y = batch_norm(x, np.ones(4), np.zeros(4))
    print(np.round(y.mean(0), 6), np.round(y.var(0), 4))     # [0. 0. 0. 0.] [~1. ~1. ~1. ~1.]

    x = rng.randn(20, 5)
    W = rng.randn(5, 5)
    out = residual_block(x, W, np.zeros(5), np.zeros(5))     # F(x)=0 -> relu(x)
    print(np.allclose(out, np.maximum(0, x)))                # True

    print(projection_shortcut(np.ones((2, 3)), np.ones((3, 4))).shape)   # (2, 4)

    # the other autograder checks
    z = batch_norm(rng.randn(500, 3) * 2 + 7, np.full(3, 2.0), np.full(3, 3.0))
    print(np.round(z.mean(0), 3), np.round(z.std(0), 3))     # mean~3, std~2
    xp = rng.randn(10, 4)
    Wp = rng.randn(4, 6)
    o = residual_block_proj(xp, rng.randn(4, 6), Wp, np.zeros(6), np.zeros(6))
    print(o.shape, np.allclose(o, np.maximum(0, xp @ Wp)))   # (10, 6) True
