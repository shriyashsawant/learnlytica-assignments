"""Day 23 / Session 2 - Lab Assignment 1: The VAE Core - Reparameterization Trick, Closed-form KL & ELBO."""
import numpy as np

mu = np.array([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])             # posterior means
logvar = np.array([[0.0, 0.5, -1.0], [1.0, 0.0, 0.2]])          # log-variances (sigma = exp(0.5*logvar))
eps = np.random.default_rng(0).standard_normal(size=mu.shape)   # noise drawn OUTSIDE the network


def reparameterize(mu, logvar, eps):
    mu, logvar, eps = (np.asarray(a, dtype=float) for a in (mu, logvar, eps))
    return mu + np.exp(0.5 * logvar) * eps                       # z = mu + sigma * eps


def kl_divergence(mu, logvar):
    mu, logvar = np.asarray(mu, dtype=float), np.asarray(logvar, dtype=float)
    kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=-1)       # KL(N(mu, sigma^2) || N(0, 1)) per sample
    return kl + 0.0                                                          # turns a stray -0.0 into 0.0


def elbo(recon, kl):
    return np.asarray(recon, dtype=float) - np.asarray(kl, dtype=float)      # per-sample ELBO = recon - kl


if __name__ == "__main__":
    print(kl_divergence(np.zeros((2, 3)), np.zeros((2, 3))))                  # [0. 0.]
    print(kl_divergence(np.ones((2, 3)), np.zeros((2, 3))))                   # [1.5 1.5]
    print(reparameterize(np.array([[0., 0.]]), np.array([[2., -2.]]), np.array([[1., 1.]])))   # [[2.71828183 0.36787944]]
    print(elbo(np.array([-3.0, -3.0]), np.array([1.5, 1.5])))                 # [-4.5 -4.5]
    print(reparameterize(mu, logvar, np.zeros_like(mu)).tolist() == mu.tolist(), kl_divergence(mu, logvar).shape)   # True (2,)
    rg = np.random.default_rng(5); m, lv = rg.normal(size=(4, 3)), rg.normal(size=(4, 3))
    print(np.allclose(kl_divergence(m, lv), 0.5 * np.sum(np.exp(lv) + m ** 2 - 1 - lv, axis=1)))   # True (closed form)
