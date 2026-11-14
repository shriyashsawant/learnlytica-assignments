"""Day 23 / Session 1 - Lab Assignment 2: Why Non-Saturating? Generator Gradient Strength vs Discriminator Confidence."""
import numpy as np

d_fake = np.array([0.01, 0.25, 0.5, 0.75, 0.99])      # D(fake): near 0 = confident fake, near 1 = generator fooled D


def grad_saturating(d_fake):
    return -np.asarray(d_fake, dtype=float)             # d/ds log(1 - D) = -D


def grad_nonsaturating(d_fake):
    return np.asarray(d_fake, dtype=float) - 1.0        # d/ds (-log D) = D - 1


def nonsat_is_stronger(d_fake):
    d_fake = np.asarray(d_fake, dtype=float)
    return np.abs(grad_nonsaturating(d_fake)) > np.abs(grad_saturating(d_fake))     # i.e. d_fake < 0.5


if __name__ == "__main__":
    print(np.round(grad_saturating(d_fake), 4))                              # [-0.01 -0.25 -0.5  -0.75 -0.99]
    print(np.round(grad_nonsaturating(d_fake), 4))                           # [-0.99 -0.75 -0.5  -0.25 -0.01]
    print(nonsat_is_stronger(d_fake))                                        # [ True  True False False False]
    print(abs(grad_saturating(np.array([0.01]))[0]))                         # 0.01
    print(abs(grad_nonsaturating(np.array([0.01]))[0]))                      # 0.99
    print(np.allclose(np.abs(grad_saturating(d_fake)) + np.abs(grad_nonsaturating(d_fake)), 1))   # True
