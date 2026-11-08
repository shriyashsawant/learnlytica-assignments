"""Day 27 / Session 3 - Lab Assignment 1: Classifier-Free Guidance - Steering Diffusion Toward the Prompt."""
import numpy as np


def cfg_combine(eps_uncond, eps_cond, w):
    u = np.asarray(eps_uncond, dtype=float)
    c = np.asarray(eps_cond, dtype=float)
    return u + w * (c - u)                                   # w=0 -> unconditional, w=1 -> conditional, w>1 extrapolates


def conditional_direction(eps_uncond, eps_cond):
    return np.asarray(eps_cond, dtype=float) - np.asarray(eps_uncond, dtype=float)


def guidance_projection(eps_uncond, eps_cond, w):
    d = conditional_direction(eps_uncond, eps_cond).ravel()
    norm = np.linalg.norm(d)
    if norm == 0:
        return 0.0
    guided = cfg_combine(eps_uncond, eps_cond, w).ravel()
    return float(np.dot(guided, d / norm))


def guidance_alignment(eps_uncond, eps_cond, w):
    d = conditional_direction(eps_uncond, eps_cond).ravel()
    g = cfg_combine(eps_uncond, eps_cond, w).ravel()
    nd, ng = np.linalg.norm(d), np.linalg.norm(g)
    if nd == 0 or ng == 0:
        return 0.0
    return float(np.dot(g, d) / (nd * ng))


def alignment_curve(eps_uncond, eps_cond, ws):
    return np.array([guidance_alignment(eps_uncond, eps_cond, w) for w in ws])


if __name__ == "__main__":
    u = np.array([1.0, 0.0]); c = np.array([1.0, 1.0])
    print(cfg_combine(u, c, 0.0), cfg_combine(u, c, 1.0))                  # [1. 0.] [1. 1.]
    print(guidance_projection(u, c, 3.0))                                  # 3.0
    print(round(guidance_alignment(u, c, 0.0), 3), round(guidance_alignment(u, c, 1.0), 3))   # 0.0 0.707
    print(np.round(alignment_curve(u, c, [0, 1, 2, 4, 8]), 3))             # [0. 0.707 0.894 0.97 0.992]
    rng = np.random.RandomState(0)
    eu, ec = rng.randn(4, 16, 16), rng.randn(4, 16, 16)
    print(cfg_combine(eu, ec, 7.5).shape, np.allclose(cfg_combine(eu, ec, 7.5), eu + 7.5 * (ec - eu)))   # (4, 16, 16) True
    print(np.allclose(conditional_direction(eu, ec), ec - eu))             # True
    d = (ec - eu).ravel()
    print(np.isclose(guidance_projection(eu, ec, 2.0), np.dot(cfg_combine(eu, ec, 2.0).ravel(), d / np.linalg.norm(d))))   # True
    print(guidance_projection(eu, eu, 5.0), bool(np.all(np.diff(alignment_curve(eu, ec, [0, 1, 2, 4, 8, 16])) > 0)))      # 0.0 and monotone (once past w=0)
