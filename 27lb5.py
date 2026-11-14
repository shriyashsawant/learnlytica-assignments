"""Day 27 / Session 3 - Lab Assignment 1: Classifier-Free Guidance - Steering Diffusion Toward the Prompt."""
import numpy as np

rng = np.random.RandomState(0)
eps_uncond = rng.randn(4, 16, 16)          # U-Net output for the empty prompt
eps_cond = rng.randn(4, 16, 16)            # U-Net output for the real prompt


def cfg_combine(eps_uncond, eps_cond, w):
    eps_uncond, eps_cond = np.asarray(eps_uncond, dtype=float), np.asarray(eps_cond, dtype=float)
    return eps_uncond + w * (eps_cond - eps_uncond)           # w=0 unconditional, w=1 conditional, w>1 extrapolates


def conditional_direction(eps_uncond, eps_cond):
    return np.asarray(eps_cond, dtype=float) - np.asarray(eps_uncond, dtype=float)    # the "toward the prompt" vector


def guidance_projection(eps_uncond, eps_cond, w):
    d = conditional_direction(eps_uncond, eps_cond).ravel()
    norm = np.linalg.norm(d)
    if norm == 0:
        return 0.0                                            # the two predictions are identical
    guided = cfg_combine(eps_uncond, eps_cond, w).ravel()
    return float(np.dot(guided, d / norm))                    # projection onto the unit prompt direction


def guidance_alignment(eps_uncond, eps_cond, w):
    d = conditional_direction(eps_uncond, eps_cond).ravel()
    guided = cfg_combine(eps_uncond, eps_cond, w).ravel()
    nd, ng = np.linalg.norm(d), np.linalg.norm(guided)
    if nd == 0 or ng == 0:
        return 0.0
    return float(np.clip(np.dot(guided, d) / (ng * nd), -1.0, 1.0))      # cosine similarity in [-1, 1]


def alignment_curve(eps_uncond, eps_cond, ws):
    return np.array([guidance_alignment(eps_uncond, eps_cond, w) for w in ws])


if __name__ == "__main__":
    u = np.array([1.0, 0.0]); c = np.array([1.0, 1.0])        # direction d = [0, 1]
    print(cfg_combine(u, c, 0.0))                              # [1. 0.]
    print(cfg_combine(u, c, 1.0))                              # [1. 1.]
    print(guidance_projection(u, c, 3.0))                      # 3.0
    print(round(guidance_alignment(u, c, 0.0), 3))             # 0.0
    print(round(guidance_alignment(u, c, 1.0), 3))             # 0.707
    print(np.round(alignment_curve(u, c, [0, 1, 2, 4, 8]), 3)) # [0.    0.707 0.894 0.97  0.992]
    out = cfg_combine(eps_uncond, eps_cond, 7.5)
    print(out.shape, np.allclose(out, eps_uncond + 7.5 * (eps_cond - eps_uncond)))     # (4, 16, 16) True
    print(np.allclose(cfg_combine(eps_uncond, eps_cond, 0), eps_uncond), np.allclose(cfg_combine(eps_uncond, eps_cond, 1), eps_cond))
    print(np.allclose(conditional_direction(eps_uncond, eps_cond), eps_cond - eps_uncond), round(guidance_projection(eps_uncond, eps_cond, 7.5), 4))
    curve = alignment_curve(eps_uncond, eps_cond, [0, 1, 2, 4, 8, 50])
    print(np.round(curve, 3), bool(np.all(np.diff(curve) > 0)))
    print(guidance_projection(u, u, 2.0), guidance_alignment(u, u, 2.0))               # 0.0 0.0
