"""Day 27 / Session 2 - Lab Assignment 2: DDPM Reverse Step - Per-step SNR, x0-from-noise, Posterior Mean & Variance."""
import numpy as np

rng = np.random.default_rng(2)
T = 1000
betas = np.linspace(1e-4, 0.02, T)
alpha_bars = np.cumprod(1.0 - betas)                 # cumulative product (from lab A1)
alphas = 1.0 - betas

x_t = rng.normal(size=(6,))                          # a noised sample at some step t
eps = rng.normal(size=(6,))                          # the model's PREDICTED noise for x_t


def compute_snr(alpha_bars):
    ab = np.asarray(alpha_bars, dtype=float)
    return ab / (1.0 - ab)                           # SNR_t = alpha_bar_t / (1 - alpha_bar_t)


def predict_x0_from_eps(x_t, alpha_bar_t, eps):
    return (np.asarray(x_t, dtype=float) - np.sqrt(1.0 - alpha_bar_t) * np.asarray(eps, dtype=float)) / np.sqrt(alpha_bar_t)


def posterior_mean(x_t, alpha_t, alpha_bar_t, beta_t, eps):
    return (np.asarray(x_t, dtype=float) - beta_t / np.sqrt(1.0 - alpha_bar_t) * np.asarray(eps, dtype=float)) / np.sqrt(alpha_t)


def posterior_variance(betas, alpha_bars, t):
    betas, alpha_bars = np.asarray(betas, dtype=float), np.asarray(alpha_bars, dtype=float)
    ab_prev = 1.0 if t == 0 else alpha_bars[t - 1]   # convention alpha_bar_{-1} = 1, so the value at t = 0 is exactly 0
    return float(betas[t] * (1.0 - ab_prev) / (1.0 - alpha_bars[t]))


if __name__ == "__main__":
    ab = np.cumprod(1.0 - np.linspace(1e-4, 0.02, 1000))
    print(compute_snr(np.array([0.5, 0.9, 0.8])))                                    # [1. 9. 4.]
    x0_, eps_ = np.array([1., -2., 0.5]), np.array([0.3, 0.3, 0.3])
    t = 150
    xt = np.sqrt(ab[t]) * x0_ + np.sqrt(1 - ab[t]) * eps_
    print(np.round(predict_x0_from_eps(xt, ab[t], eps_), 6))                         # [ 1. -2.  0.5]
    print(posterior_mean(np.array([2., 4.]), 0.81, 0.5, 0.02, np.zeros(2)))          # [2.2222 4.4444]
    b = np.linspace(1e-4, 0.02, 1000)
    print(round(posterior_variance(b, ab, 0), 8))                                    # 0.0
    print(posterior_variance(b, ab, 500) > 0)                                        # True
    snr = compute_snr(ab)
    print(round(float(snr[0]), 1), float(snr[-1]) < 0.01, bool(np.all(np.diff(snr) < 0)))     # >50, True, True
    print(np.allclose(posterior_mean(xt, 0.9, ab[t], 0.1, np.zeros(3)), xt / np.sqrt(0.9)))   # True (eps = 0)
