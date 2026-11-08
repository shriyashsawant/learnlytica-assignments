"""Day 19 / Session 1 - Lab Assignment 2: 1x1 Conv Channel Mixer & Skip-Connection Gradient Path."""
import numpy as np


def relu(x):
    return np.maximum(0, np.asarray(x, dtype=float))


def relu_grad(z):
    return (np.asarray(z) > 0).astype(float)    # 1.0 where z > 0, else 0.0


def conv1x1(x, W):
    # mix channels independently at every pixel: out[co,h,w] = sum_ci W[co,ci] * x[ci,h,w]
    return np.einsum('oc,chw->ohw', np.asarray(W, dtype=float), np.asarray(x, dtype=float))


def forward_plain(x, W):
    return relu(conv1x1(x, W))


def forward_residual(x, W):
    x = np.asarray(x, dtype=float)
    return relu(conv1x1(x, W) + x)             # needs C_out == C_in


def block_input_grad(x, W, skip=False):
    """Gradient of L = sum(out) with respect to x."""
    x = np.asarray(x, dtype=float)
    pre = conv1x1(x, W)
    if skip:
        pre = pre + x
    mask = relu_grad(pre)                       # upstream gradient is 1 everywhere
    grad = np.einsum('oc,ohw->chw', np.asarray(W, dtype=float), mask)   # back through the 1x1 conv
    if skip:
        grad = grad + mask                      # the identity Jacobian of the shortcut
    return grad


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    x = rng.randn(3, 4, 4)
    print(conv1x1(x, np.eye(3)).shape)                                   # (3, 4, 4)
    print(np.allclose(conv1x1(x, np.eye(3)), x))                         # True
    W0 = np.zeros((3, 3))
    print(np.allclose(block_input_grad(x, W0, skip=False), 0))           # True
    print(np.allclose(block_input_grad(x, W0, skip=True), (x > 0)))      # True

    W = rng.randn(3, 3) * 0.2
    manual = sum(W[1, c] * x[c, 2, 3] for c in range(3))
    print(np.isclose(conv1x1(x, W)[1, 2, 3], manual))                    # True
    print(np.allclose(forward_residual(x, W), np.maximum(0, conv1x1(x, W) + x)))   # True
    Wtiny = rng.randn(3, 3) * 1e-6
    print(np.linalg.norm(block_input_grad(x, Wtiny, True)) > np.linalg.norm(block_input_grad(x, Wtiny, False)))  # True
