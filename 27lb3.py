"""Day 27 / Session 2 - Lab Assignment 1: DDPM Forward Process - Beta Schedule, alpha_bar & the Closed-form q_sample."""
import numpy as np

rng = np.random.default_rng(0)
T = 1000
x0 = rng.normal(size=(8,))                      # a clean sample (any shape works)
noise = rng.standard_normal(size=(8,))          # eps ~ N(0, I), same shape as x0


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)             # (T,), strictly increasing


def compute_alpha_bar(betas):
    return np.cumprod(1.0 - np.asarray(betas, dtype=float))  # cumulative PRODUCT of alpha = 1 - beta


def q_sample(x0, t, alpha_bars, noise):
    ab = float(np.asarray(alpha_bars)[t])
    return np.sqrt(ab) * np.asarray(x0, dtype=float) + np.sqrt(1.0 - ab) * np.asarray(noise, dtype=float)


def signal_noise_rates(alpha_bars, t):
    ab = float(np.asarray(alpha_bars)[t])
    return float(np.sqrt(ab)), float(np.sqrt(1.0 - ab))     # (signal rate, noise rate)


if __name__ == "__main__":
    betas = linear_beta_schedule(1000, 1e-4, 0.02)
    ab = compute_alpha_bar(betas)
    print(ab.shape, round(float(ab[0]), 4))                                           # (1000,) 0.9999
    print(round(float(ab[-1]), 6))                                                    # ~3.9e-05
    print(q_sample(np.array([1., 2., 3.]), 0, np.array([0.64]), np.array([1., 1., 1.])))      # [1.4 2.2 3. ]
    print(signal_noise_rates(np.array([0.36]), 0))                                    # (0.6, 0.8)
    print(np.allclose(q_sample(np.array([2., -1.]), 30, ab, np.zeros(2)), np.sqrt(ab[30]) * np.array([2., -1.])))   # True
    print(bool(np.all(np.diff(betas) > 0)), bool(np.all(np.diff(ab) < 0)), float(betas[0]), float(betas[-1]))     # True True 0.0001 0.02
    xb = np.random.default_rng(1).normal(size=(5, 7)); nb = np.random.default_rng(2).normal(size=(5, 7))
    print(np.allclose(q_sample(xb, 10, ab, nb), np.sqrt(ab[10]) * xb + np.sqrt(1 - ab[10]) * nb))     # True (batch)
    big = np.random.default_rng(3).normal(size=100000) * 3 + 5
    xt = q_sample(big, 999, ab, np.random.default_rng(4).standard_normal(100000))
    print(round(float(xt.mean()), 2), round(float(xt.var()), 2))                      # ~0 ~1
