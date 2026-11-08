"""Day 27 / Session 2 - Lab Assignment 2: DDPM Reverse Step - Per-step SNR, x0-from-noise, Posterior Mean & Variance."""
import numpy as np


def compute_snr(alpha_bars):
    ab = np.asarray(alpha_bars, dtype=float)
    return ab / (1.0 - ab)                                  # signal-to-noise ratio per step


def predict_x0_from_eps(x_t, alpha_bar_t, eps):
    return (np.asarray(x_t, dtype=float) - np.sqrt(1.0 - alpha_bar_t) * np.asarray(eps, dtype=float)) / np.sqrt(alpha_bar_t)


def posterior_mean(x_t, alpha_t, alpha_bar_t, beta_t, eps):
    x_t = np.asarray(x_t, dtype=float)
    eps = np.asarray(eps, dtype=float)
    return (1.0 / np.sqrt(alpha_t)) * (x_t - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps)


def posterior_variance(betas, alpha_bars, t):
    betas = np.asarray(betas, dtype=float)
    alpha_bars = np.asarray(alpha_bars, dtype=float)
    ab_prev = 1.0 if t == 0 else alpha_bars[t - 1]          # convention alpha_bar_{-1} = 1, so beta_tilde_0 = 0
    return float(betas[t] * (1.0 - ab_prev) / (1.0 - alpha_bars[t]))


if __name__ == "__main__":
    betas = np.linspace(1e-4, 0.02, 1000)
    ab = np.cumprod(1.0 - betas)
    print(compute_snr(np.array([0.5, 0.9, 0.8])))                                    # [1. 9. 4.]
    snr = compute_snr(ab)
    print(round(float(snr[0]), 1), round(float(snr[-1]), 5), bool(np.all(np.diff(snr) < 0)))   # >50 <0.01 True
    x0, eps = np.array([1., -2., 0.5]), np.array([0.3, 0.3, 0.3])
    t = 150
    x_t = np.sqrt(ab[t]) * x0 + np.sqrt(1 - ab[t]) * eps
    print(np.round(predict_x0_from_eps(x_t, ab[t], eps), 6))                         # [ 1. -2.  0.5]
    print(posterior_mean(np.array([2., 4.]), 0.81, 0.5, 0.02, np.zeros(2)))          # [2.222.. 4.444..]
    print(np.allclose(posterior_mean(np.array([2., 4.]), 0.81, 0.5, 0.02, np.ones(2)),
                      (np.array([2., 4.]) - 0.02 / np.sqrt(0.5)) / 0.9))             # True
    print(round(posterior_variance(betas, ab, 0), 8), posterior_variance(betas, ab, 500) > 0)   # 0.0 True
    print(np.isclose(posterior_variance(betas, ab, 500), betas[500] * (1 - ab[499]) / (1 - ab[500])))   # True
