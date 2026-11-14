"""Day 23 / Session 2 - Lab Assignment 2: Advanced VAE Knobs - beta-VAE, Per-dim KL, Posterior Collapse & Annealing."""
import numpy as np

rng = np.random.default_rng(1)
N = 500
mu = np.zeros((N, 6)); logvar = np.zeros((N, 6))
mu[:, 0] = rng.normal(0, 2.0, size=N)             # dim 0: active
mu[:, 1] = rng.normal(0, 1.5, size=N)             # dim 1: active
logvar[:, 2] = 1.5                                # dim 2: active (variance != 1)
# dims 3,4,5 stay mu=0, logvar=0 -> collapsed to the prior (KL = 0)


def per_dim_kl(mu, logvar):
    mu, logvar = np.asarray(mu, dtype=float), np.asarray(logvar, dtype=float)
    kl = 0.5 * (np.exp(logvar) + mu ** 2 - 1.0 - logvar)      # KL contribution of every dimension of every sample
    return kl.mean(axis=0)                                    # averaged over the batch -> shape (D,)


def beta_elbo(recon, kl, beta):
    return recon + beta * kl                                  # beta=1 is the standard VAE, beta=0 drops the KL


if __name__ == "__main__":
    pk = per_dim_kl(mu, logvar)
    print(np.round(pk, 3))                                    # active dims > 0, last three exactly 0
    print(np.round(pk.sum(), 4), np.all(pk >= 0))
    print(beta_elbo(2.0, 3.0, 1.0))                           # 5.0
    print(beta_elbo(2.0, 3.0, 4.0))                           # 14.0
    print(beta_elbo(2.0, 3.0, 0.0))                           # 2.0
    print(pk.shape, pk[:3].min() > pk[3:].max())              # (6,) True (active vs collapsed)
