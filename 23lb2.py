"""Day 23 / Session 1 - Lab Assignment 2: Why Non-Saturating? Generator Gradient Strength vs Discriminator Confidence."""
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def grad_saturating_wrt_logit(fake_logits):
    return -sigmoid(fake_logits)                     # d/ds log(1 - D) = -D


def grad_nonsaturating_wrt_logit(fake_logits):
    return sigmoid(fake_logits) - 1.0                # d/ds (-log D) = D - 1


def grad_magnitude_ratio(fake_logits):
    s = np.asarray(fake_logits, dtype=float)
    with np.errstate(over="ignore"):
        return np.exp(-s)                            # |D - 1| / |D| = (1 - D) / D = exp(-s)


def stronger_gradient(fake_logits):
    return grad_magnitude_ratio(fake_logits) > 1.0   # True exactly where D < 0.5


if __name__ == "__main__":
    fake_logits = np.array([-6.0, -2.0, 0.0, 2.0, 6.0])
    print(np.round(grad_saturating_wrt_logit(fake_logits), 4))        # [-0.0025 -0.1192 -0.5 -0.8808 -0.9975]
    print(np.round(grad_nonsaturating_wrt_logit(fake_logits), 4))     # [-0.9975 -0.8808 -0.5 -0.1192 -0.0025]
    print(np.round(grad_magnitude_ratio(fake_logits), 3))             # [403.429 7.389 1. 0.135 0.002]
    print(stronger_gradient(fake_logits))                             # [ True True False False False]
    gs, gn = grad_saturating_wrt_logit(fake_logits), grad_nonsaturating_wrt_logit(fake_logits)
    print(np.allclose(np.abs(gs) + np.abs(gn), 1), np.allclose(np.abs(gn) / np.abs(gs), grad_magnitude_ratio(fake_logits)))   # True True
