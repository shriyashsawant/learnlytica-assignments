"""Day 23 / Session 1 - Lab Assignment 1: The GAN Loss Functions - Discriminator, Saturating & Non-Saturating Generator Loss."""
import numpy as np

d_real = np.array([0.9, 0.8, 0.7])      # D's probability that each REAL sample is real
d_fake = np.array([0.2, 0.3, 0.4])      # D's probability that each FAKE sample is real


def d_loss(d_real, d_fake):
    d_real, d_fake = np.asarray(d_real, dtype=float), np.asarray(d_fake, dtype=float)
    return float(-np.mean(np.log(d_real)) - np.mean(np.log(1.0 - d_fake)))     # D wants D(real) -> 1 and D(fake) -> 0


def g_loss_nonsaturating(d_fake):
    d_fake = np.asarray(d_fake, dtype=float)
    return float(-np.mean(np.log(d_fake)))                                    # G wants D(fake) -> 1


def g_loss_saturating(d_fake):
    d_fake = np.asarray(d_fake, dtype=float)
    return float(np.mean(np.log(1.0 - d_fake)))                               # the original minimax term, always <= 0


if __name__ == "__main__":
    print(round(d_loss(d_real, d_fake), 4))                    # 0.5919
    print(round(g_loss_nonsaturating(d_fake), 4))              # 1.2432
    print(round(g_loss_saturating(d_fake), 4))                 # -0.3635
    sure_fake = np.array([1e-4, 1e-4, 1e-4])
    print(round(g_loss_saturating(sure_fake), 4))              # ~ -0.0001
    print(round(g_loss_nonsaturating(sure_fake), 4))           # ~ 9.21
