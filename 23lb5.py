"""Day 23 / Session 3 - Lab Assignment 1: Mode Coverage & Nearest-Neighbour Distance for Generative Quality."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

centers = np.array([[0., 0.], [5., 0.], [0., 5.], [5., 5.]])      # 4 ground-truth modes
real, _ = make_blobs(n_samples=400, centers=centers, cluster_std=0.3, random_state=0)
gen, _ = make_blobs(n_samples=400, centers=centers, cluster_std=0.3, random_state=1)    # a healthy generator


def nn_distance(real, gen):
    nn = NearestNeighbors(n_neighbors=1).fit(real)
    dists, _ = nn.kneighbors(gen)
    return float(dists.ravel().mean())                            # mean distance from a generated point to its closest real point


def mode_coverage(real_centers, gen, radius):
    nn = NearestNeighbors(n_neighbors=1).fit(gen)                 # index on the generated points
    dists, _ = nn.kneighbors(np.asarray(real_centers, dtype=float))
    return float((dists.ravel() <= radius).mean())                # fraction of modes with a generated point within radius


if __name__ == "__main__":
    r = np.random.RandomState(0).randn(40, 2)
    print(round(nn_distance(r, r), 6))                                                         # 0.0
    grid = np.array([[i * 3., j * 3.] for i in range(5) for j in range(5)])
    print(round(nn_distance(grid, grid + np.array([0., 0.5])), 3))                             # ~0.5
    rng = np.random.RandomState(0)
    gen_good = np.vstack([c + 0.1 * rng.randn(30, 2) for c in centers])
    gen_bad = centers[0] + 0.1 * rng.randn(120, 2)
    print(mode_coverage(centers, gen_good, radius=1.0))                                        # 1.0
    print(mode_coverage(centers, gen_bad, radius=1.0))                                         # 0.25
    print(mode_coverage(centers, gen_bad, radius=10.0), mode_coverage(centers, gen_bad, radius=0.0))   # 1.0 0.0 (radius sensitivity)
    brute = np.mean([np.min(np.linalg.norm(real - g, axis=1)) for g in gen])
    print(np.isclose(nn_distance(real, gen), brute))                                           # True (matches brute force)
