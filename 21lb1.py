"""Day 21 / Session 1 - Lab Assignment 1: The RNN Cell - tanh Recurrence, Unrolling & Parameter Count."""
import numpy as np


def rnn_cell_step(x, h_prev, Wxh, Whh, b):
    # textbook layout: Wxh is (H, D), Whh is (H, H), b is (H,)
    return np.tanh(np.asarray(Wxh) @ np.asarray(x) + np.asarray(Whh) @ np.asarray(h_prev) + np.asarray(b))


def rnn_forward(X, h0, Wxh, Whh, b):
    X = np.asarray(X, dtype=float)
    h = np.asarray(h0, dtype=float)
    states = []
    for t in range(X.shape[0]):
        h = rnn_cell_step(X[t], h, Wxh, Whh, b)      # the previous state is fed back in
        states.append(h)
    return np.array(states)                          # (T, H)


def count_params(D, H):
    return H * D + H * H + H                         # Wxh + Whh + b  ==  H * (D + H + 1)


if __name__ == "__main__":
    D, H = 2, 3
    Wxh = np.array([[0.10, 0.20], [0.00, -0.10], [0.30, 0.10]])
    Whh = np.array([[0.05, 0.00, 0.10], [0.10, 0.05, 0.00], [0.00, 0.10, 0.05]])
    b = np.array([0.00, 0.10, -0.10])
    h0 = np.zeros(3)
    X = np.array([[1.0, 0.5], [-0.5, 1.0]])
    print(np.round(rnn_cell_step(X[0], h0, Wxh, Whh, b), 4))     # [0.1974 0.05   0.2449]
    hs = rnn_forward(X, h0, Wxh, Whh, b)
    print(hs.shape, np.round(hs[-1], 4))                         # (2, 3) [ 0.1823  0.0222 -0.132 ]
    print(count_params(2, 3), count_params(10, 20))              # 18 620
    print(np.allclose(rnn_cell_step(np.zeros(2), np.zeros(3), np.zeros((3, 2)), np.zeros((3, 3)), np.zeros(3)), 0))   # True
    print(np.allclose(rnn_forward(X, np.array([0.5, -0.5, 0.2]), Wxh, Whh, b)[0],
                      rnn_cell_step(X[0], np.array([0.5, -0.5, 0.2]), Wxh, Whh, b)))                                   # True
