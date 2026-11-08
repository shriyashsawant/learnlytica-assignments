"""Day 27 / Session 2 - Lab Assignment 1: DDPM Forward Process - Beta Schedule, alpha_bar & the Closed-form q_sample."""
import numpy as np


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)


def compute_alpha_bar(betas):
    return np.cumprod(1.0 - np.asarray(betas, dtype=float))     # a cumulative PRODUCT of alpha = 1 - beta


def q_sample(x0, t, alpha_bars, noise):
    ab = np.asarray(alpha_bars, dtype=float)[t]
    return np.sqrt(ab) * np.asarray(x0, dtype=float) + np.sqrt(1.0 - ab) * np.asarray(noise, dtype=float)


def signal_noise_rates(alpha_bars, t):
    ab = float(np.asarray(alpha_bars, dtype=float)[t])
    return float(np.sqrt(ab)), float(np.sqrt(1.0 - ab))


if __name__ == "__main__":
    betas = linear_beta_schedule(1000, 1e-4, 0.02)
    ab = compute_alpha_bar(betas)
    print(betas.shape, betas[0], betas[-1], bool(np.all(np.diff(betas) > 0)))      # (1000,) 0.0001 0.02 True
    print(ab.shape, round(float(ab[0]), 4), round(float(ab[-1]), 6), bool(np.all(np.diff(ab) < 0)))   # (1000,) 0.9999 ~3.9e-05 True
    print(q_sample(np.array([1., 2., 3.]), 0, np.array([0.64]), np.array([1., 1., 1.])))            # [1.4 2.2 3. ]
    print(signal_noise_rates(np.array([0.36]), 0))                                  # (0.6, 0.8)
    print(np.allclose(q_sample(np.array([2., -1.]), 30, ab, np.zeros(2)), np.sqrt(ab[30]) * np.array([2., -1.])))   # True
    rng = np.random.default_rng(0)
    x0, eps = rng.normal(size=(5, 7)), rng.normal(size=(5, 7))
    print(np.allclose(q_sample(x0, 400, ab, eps), np.sqrt(ab[400]) * x0 + np.sqrt(1 - ab[400]) * eps))   # True
    xT = q_sample(rng.normal(size=100000), 999, ab, rng.standard_normal(100000))
    print(round(float(xT.mean()), 2), round(float(xT.var()), 2))                    # ~0 ~1
    s, n = signal_noise_rates(ab, 300)
    print(round(s ** 2 + n ** 2, 10))                                               # 1.0
