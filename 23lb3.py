"""Day 23 / Session 2 - Lab Assignment 1: The VAE Core - Reparameterization Trick, Closed-form KL & ELBO."""
import numpy as np


def reparameterize(mu, logvar, eps):
    sigma = np.exp(0.5 * np.asarray(logvar, dtype=float))
    return np.asarray(mu, dtype=float) + sigma * np.asarray(eps, dtype=float)     # z = mu + sigma * eps


def kl_divergence(mu, logvar):
    mu = np.asarray(mu, dtype=float)
    logvar = np.asarray(logvar, dtype=float)
    return -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar), axis=-1)          # per-sample, summed over latent dims


def elbo(recon_log_lik, mu, logvar):
    return np.asarray(recon_log_lik, dtype=float) - kl_divergence(mu, logvar)


def vae_loss(recon_log_lik, mu, logvar):
    return float(np.mean(kl_divergence(mu, logvar) - np.asarray(recon_log_lik, dtype=float)))   # = -mean(ELBO)


if __name__ == "__main__":
    mu = np.ones((2, 3))
    logvar = np.zeros((2, 3))
    print(kl_divergence(mu, logvar))                                                 # [1.5 1.5]
    print(kl_divergence(np.zeros((2, 3)), np.zeros((2, 3))))                         # [0. 0.]
    print(reparameterize(np.array([[0., 0.]]), np.array([[2., -2.]]), np.array([[1., 1.]])))   # [[2.71828183 0.36787944]]
    rll = np.array([-3.0, -3.0])
    print(elbo(rll, mu, logvar))                                                     # [-4.5 -4.5]
    print(round(vae_loss(rll, mu, logvar), 4))                                       # 4.5
    m = np.array([[1.0, -2.0]]); lv = np.array([[0.5, -1.0]])
    print(np.allclose(reparameterize(m, lv, np.zeros((1, 2))), m))                   # True
    rng = np.random.default_rng(0)
    z = reparameterize(np.full((100000, 1), 2.0), np.full((100000, 1), np.log(4.0)), rng.standard_normal((100000, 1)))
    print(round(float(z.mean()), 2), round(float(z.var()), 2))                       # ~2.0 ~4.0
    print((kl_divergence(rng.normal(size=(50, 4)), rng.normal(size=(50, 4))) >= 0).all())   # True
