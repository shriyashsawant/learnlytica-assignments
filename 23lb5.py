"""Day 23 / Session 3 - Lab Assignment 1: Mode Coverage & Nearest-Neighbour Distance for Generative Quality."""
import numpy as np


def pairwise_dist(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]                 # (n, m, d)
    return np.sqrt(np.sum(diff ** 2, axis=-1))           # D[i, j] = ||A[i] - B[j]||_2


def nn_distance(gen, real):
    D = pairwise_dist(gen, real)
    return float(np.mean(D.min(axis=1)))                 # closest real sample for each generated one, averaged


def assign_to_mode(points, modes):
    return np.argmin(pairwise_dist(points, modes), axis=1).astype(int)


def count_covered_modes(gen, modes, radius):
    D = pairwise_dist(modes, gen)                        # (K, n)
    return int(np.sum(D.min(axis=1) <= radius))          # modes with at least one generated sample nearby


def mode_coverage(gen, modes, radius):
    return count_covered_modes(gen, modes, radius) / len(modes)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    modes = np.array([[0., 0.], [5., 0.], [0., 5.], [5., 5.]])
    real = rng.randn(40, 2)
    print(round(nn_distance(real, real), 6))                                       # 0.0
    print(round(nn_distance(real + np.array([0., 0.5]), real), 3))                 # ~0.5 or below (nearest neighbour)
    gen_good = np.vstack([m + 0.2 * rng.randn(30, 2) for m in modes])
    gen_bad = modes[0] + 0.2 * rng.randn(120, 2)
    print(mode_coverage(gen_good, modes, radius=1.0), mode_coverage(gen_bad, modes, radius=1.0))   # 1.0 0.25
    A, B = rng.randn(5, 3), rng.randn(4, 3)
    brute = np.array([[np.linalg.norm(a - b) for b in B] for a in A])
    print(pairwise_dist(A, B).shape, np.allclose(pairwise_dist(A, B), brute))      # (5, 4) True
    print(assign_to_mode(np.array([[0.1, 0.2], [4.8, 5.1], [0.2, 4.9]]), modes))   # [0 3 2]
    print(mode_coverage(gen_good, modes, 0.0001), mode_coverage(gen_good, modes, 100.0))   # 0.0 1.0
