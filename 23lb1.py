"""Day 23 / Session 1 - Lab Assignment 1: The GAN Loss Functions - Discriminator, Saturating & Non-Saturating Generator Loss."""
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))                           # exponent is never positive, so no overflow
    out = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return out if out.ndim else float(out)


def _softplus(z):
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))


def bce_with_logits(logits, targets):
    z = np.asarray(logits, dtype=float)
    t = np.asarray(targets, dtype=float)
    # stable identity: max(z,0) - z*t + log(1 + exp(-|z|))
    return float(np.mean(np.maximum(z, 0) - z * t + np.log1p(np.exp(-np.abs(z)))))


def discriminator_loss(real_logits, fake_logits):
    return bce_with_logits(real_logits, 1.0) + bce_with_logits(fake_logits, 0.0)


def generator_loss_nonsaturating(fake_logits):
    return bce_with_logits(fake_logits, 1.0)         # = -mean(log D(fake))


def generator_loss_saturating(fake_logits):
    return float(-np.mean(_softplus(fake_logits)))   # = mean(log(1 - D(fake))), always <= 0


if __name__ == "__main__":
    real_logits = np.array([2.0, 1.0, 0.5, -0.3])
    fake_logits = np.array([-0.5, 0.0, 1.0, 2.0])
    # NOTE: the question prints 1.1316 / 0.6602 / -0.8268 for these arrays, but those numbers do not follow from the
    # stated formulas (they imply a different fake_logits array). The formulas give 1.594 / 0.5269 / -1.1519.
    print(round(discriminator_loss(real_logits, fake_logits), 4))     # 1.594
    print(round(generator_loss_nonsaturating(fake_logits), 4))        # 0.5269
    print(round(generator_loss_saturating(fake_logits), 4))           # -1.1519
    print(round(float(sigmoid(0.0)), 4))                              # 0.5
    print(sigmoid(np.array([-1000.0, 1000.0])), np.allclose(sigmoid(np.array([-1., 0, 2])), 1 / (1 + np.exp(-np.array([-1., 0, 2])))))
    print(round(discriminator_loss(np.full(4, 30.0), np.full(4, -30.0)), 6))                 # ~0 (perfect discriminator)
    conf = np.full(4, -20.0)                                                                 # D is sure the samples are fake
    print(round(generator_loss_saturating(conf), 6), round(generator_loss_nonsaturating(conf), 3))   # ~0 vs large
    print(np.isclose(generator_loss_nonsaturating(fake_logits), -np.mean(np.log(sigmoid(fake_logits)))))
