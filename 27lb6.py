"""Day 27 / Session 3 - Lab Assignment 2: Latent<->Pixel Shape Math & a Deterministic DDIM Step."""
import numpy as np


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
    return np.sqrt(alpha_prev) * x0 + np.sqrt(1.0 - alpha_prev) * np.asarray(eps, dtype=float)    # eta = 0: no random term


if __name__ == "__main__":
    print(latent_shape(512, 512), latent_shape(256, 384), pixel_shape(64, 64))        # (4, 64, 64) (4, 32, 48) (3, 512, 512)
    print(pixel_shape(*latent_shape(512, 512)[1:]), compression_ratio(512, 512))      # (3, 512, 512) 48.0
    a = 0.6
    x0 = np.random.RandomState(1).randn(4, 8, 8)
    eps = np.random.RandomState(2).randn(4, 8, 8)
    x_t = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
    print(np.allclose(predict_x0(x_t, eps, a), x0))                                   # True
    print(np.allclose(ddim_step(x_t, eps, a, a), x_t))                                # True (no-op)
    xt = np.random.RandomState(3).randn(4, 8, 8)
    print(np.allclose(ddim_step(xt, np.zeros_like(xt), 1.0, 0.4), np.sqrt(0.4) * xt))  # True
    e = np.random.RandomState(4).randn(4, 8, 8)
    manual = np.sqrt(0.4) * ((xt - np.sqrt(0.3) * e) / np.sqrt(0.7)) + np.sqrt(0.6) * e
    print(np.allclose(ddim_step(xt, e, 0.7, 0.4), manual), ddim_step(xt, e, 0.7, 0.4).shape)   # True (4, 8, 8)
