"""Day 23 / Session 2 - Lab Assignment 2: Advanced VAE Knobs - beta-VAE, Per-dim KL, Posterior Collapse & Annealing."""
import numpy as np


def _kl_per_sample(mu, logvar):
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    return 0.5 * np.sum(np.exp(logvar) + mu ** 2 - 1 - logvar, axis=-1)


def per_dim_kl(mu, logvar):
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    return np.mean(0.5 * (np.exp(logvar) + mu ** 2 - 1 - logvar), axis=0)     # (D,), averaged over the batch


def active_units(mu, logvar, threshold=0.01):
    return int(np.sum(per_dim_kl(mu, logvar) > threshold))                     # strictly greater than the threshold


def beta_vae_loss(recon_log_lik, mu, logvar, beta):
    kl = _kl_per_sample(mu, logvar)
    return float(np.mean(beta * kl - np.asarray(recon_log_lik, dtype=float)))


def kl_annealing_weight(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    return float(min(1.0, max(0.0, step / warmup_steps)))                      # linear warm-up, clamped to [0, 1]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    N = 500
    mu = np.zeros((N, 6)); logvar = np.zeros((N, 6))
    mu[:, 0] = rng.normal(0, 2.0, size=N)
    mu[:, 1] = rng.normal(0, 1.5, size=N)
    logvar[:, 2] = 1.5
    recon_log_lik = rng.normal(size=(N,))
    pk = per_dim_kl(mu, logvar)
    print(np.round(pk, 3), pk.shape)                                           # last three are 0
    print(active_units(mu, logvar))                                            # 3
    rll = recon_log_lik
    print(round(beta_vae_loss(rll, mu, logvar, beta=1.0), 4), round(beta_vae_loss(rll, mu, logvar, beta=4.0), 4))
    print(round(beta_vae_loss(rll, mu, logvar, beta=0.0), 4), round(-float(np.mean(rll)), 4))   # equal
    print(kl_annealing_weight(0, 1000), kl_annealing_weight(500, 1000), kl_annealing_weight(9999, 1000), kl_annealing_weight(-5, 1000))   # 0.0 0.5 1.0 0.0
    print(active_units(np.zeros((10, 4)), np.zeros((10, 4))))                  # 0 (fully collapsed)
