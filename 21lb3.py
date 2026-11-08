"""Day 21 / Session 2 - Lab Assignment 1: Vanishing/Exploding Gradients - Spectral Radius & the Decay Slope."""
import numpy as np


def spectral_radius(W):
    eigenvalues = np.linalg.eigvals(np.asarray(W, dtype=float))
    return float(max(abs(eigenvalues)))


def propagate_grad_norms(W, g0, k_max):
    W = np.asarray(W, dtype=float)
    g = np.asarray(g0, dtype=float)
    norms = [float(np.linalg.norm(g))]               # norms[0] = ||g0||
    for _ in range(k_max):
        g = W @ g                                    # one more step back through time
        norms.append(float(np.linalg.norm(g)))
    return norms                                     # length k_max + 1


def log_norm_slope(norms):
    k = np.arange(len(norms))
    return float(np.polyfit(k, np.log(norms), 1)[0])  # slope of ln(norm) vs step  ~  ln(rho)


if __name__ == "__main__":
    print(round(spectral_radius(np.diag([0.5, 0.3])), 4))                 # 0.5
    print(round(spectral_radius([[0, 1], [1, 0]]), 4))                    # 1.0
    print(round(spectral_radius(np.diag([0.4, -0.7])), 4))                # 0.7
    norms = propagate_grad_norms([[0.5]], [1.0], 5)
    print([round(x, 4) for x in norms])                                   # [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0312]
    print(round(log_norm_slope(norms), 4))                                # -0.6931
    print(round(log_norm_slope(propagate_grad_norms([[1.2]], [1.0], 20)), 4))   # 0.1823
    print(propagate_grad_norms([[0.5]], [3.0, 4.0][:1], 0)[0], len(propagate_grad_norms(np.eye(2), [3.0, 4.0], 4)))     # 3.0 5
    print(round(log_norm_slope(propagate_grad_norms(np.diag([0.8, 0.1]), [1.0, 1.0], 12)), 3), round(np.log(0.8), 3))   # ~ ln 0.8
