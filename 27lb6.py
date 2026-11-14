"""Day 27 / Session 3 - Lab Assignment 2: Latent<->Pixel Shape Math & a Deterministic DDIM Step."""
import numpy as np

rng = np.random.RandomState(3)
x_t = rng.randn(4, 8, 8)                   # a (tiny) noisy latent, shaped like SD's (4, H, W)
eps = rng.randn(4, 8, 8)                   # the U-Net's predicted noise for x_t
alpha_t, alpha_prev = 0.7, 0.4             # cumulative signal coeffs (alpha_bar), 0 < .. <= 1


def latent_shape(H, W, C=3, factor=8, latent_channels=4):
    return (latent_channels, H // factor, W // factor)


def pixel_shape(lat_h, lat_w, factor=8, out_channels=3):
    return (out_channels, lat_h * factor, lat_w * factor)


def compression_ratio(H, W, C=3, factor=8, latent_channels=4):
    return float((H * W * C) / (latent_channels * (H // factor) * (W // factor)))


def predict_x0(x_t, eps, alpha_t):
    return (np.asarray(x_t, dtype=float) - np.sqrt(1.0 - alpha_t) * np.asarray(eps, dtype=float)) / np.sqrt(alpha_t)


def ddim_step(x_t, eps, alpha_t, alpha_prev):
    x0 = predict_x0(x_t, eps, alpha_t)
    return np.sqrt(alpha_prev) * x0 + np.sqrt(1.0 - alpha_prev) * np.asarray(eps, dtype=float)     # eta = 0: no random term


if __name__ == "__main__":
    print(latent_shape(512, 512))                                                  # (4, 64, 64)
    print(latent_shape(256, 384))                                                  # (4, 32, 48)
    print(pixel_shape(64, 64))                                                     # (3, 512, 512)
    print(pixel_shape(*latent_shape(512, 512)[1:]) == (3, 512, 512))               # True (round trip)
    print(compression_ratio(512, 512))                                             # 48.0
    a = 0.6
    x0 = np.random.RandomState(1).randn(4, 8, 8)
    e = np.random.RandomState(2).randn(4, 8, 8)
    xt = np.sqrt(a) * x0 + np.sqrt(1 - a) * e
    print(np.allclose(predict_x0(xt, e, a), x0))                                   # True
    print(np.allclose(ddim_step(xt, e, a, a), xt))                                 # True (no-op when alpha_prev == alpha_t)
    print(np.allclose(ddim_step(xt, np.zeros_like(xt), 1.0, 0.25), 0.5 * xt))      # True (alpha_t=1, eps=0 scales by sqrt(alpha_prev))
    out = ddim_step(x_t, eps, alpha_t, alpha_prev)
    ref = np.sqrt(alpha_prev) * (x_t - np.sqrt(1 - alpha_t) * eps) / np.sqrt(alpha_t) + np.sqrt(1 - alpha_prev) * eps
    print(out.shape, np.allclose(out, ref))                                        # (4, 8, 8) True
