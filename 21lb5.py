"""Day 21 / Session 3 - Lab Assignment 1: LSTM Cell Forward Step & Sequence Unroll from Scratch."""
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))                           # never overflows
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def lstm_cell_step(x_t, h_prev, c_prev, Wx, Wh, b):
    H = len(h_prev)
    z = np.asarray(Wx) @ np.asarray(x_t) + np.asarray(Wh) @ np.asarray(h_prev) + np.asarray(b)   # (4H,)
    i = sigmoid(z[0:H])                              # input gate
    f = sigmoid(z[H:2 * H])                          # forget gate
    g = np.tanh(z[2 * H:3 * H])                      # candidate
    o = sigmoid(z[3 * H:4 * H])                      # output gate
    c_t = f * np.asarray(c_prev) + i * g
    h_t = o * np.tanh(c_t)
    return h_t, c_t


def lstm_forward(X, h0, c0, Wx, Wh, b):
    h = np.asarray(h0, dtype=float)
    c = np.asarray(c0, dtype=float)
    states = []
    for x_t in np.asarray(X, dtype=float):
        h, c = lstm_cell_step(x_t, h, c, Wx, Wh, b)
        states.append(h)
    return np.array(states), c                       # H_all (T, H), c_last (H,)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    D, H = 3, 4
    Wx = rng.randn(4 * H, D) * 0.5
    Wh = rng.randn(4 * H, H) * 0.5
    b = rng.randn(4 * H) * 0.3
    X = rng.randn(6, D)
    print(np.round(sigmoid(np.array([0.0, 2.0])), 4), sigmoid(np.array([-1000.0, 1000.0])))   # [0.5 0.8808] [0. 1.]
    H_all, c_last = lstm_forward(X, np.zeros(4), np.zeros(4), Wx, Wh, b)
    print(H_all.shape, c_last.shape)                                                          # (6, 4) (4,)
    b2 = np.zeros(16); b2[0:4] = -30; b2[4:8] = +30
    _, c_t = lstm_cell_step(np.zeros(3), np.zeros(4), np.array([1., 2., 3., 4.]), np.zeros((16, 3)), np.zeros((16, 4)), b2)
    print(np.round(c_t, 4))                                                                   # [1. 2. 3. 4.]
    b3 = np.zeros(16); b3[0:4] = +30; b3[4:8] = -30; b3[8:12] = +30
    _, c_o = lstm_cell_step(np.zeros(3), np.zeros(4), np.array([1., 2., 3., 4.]), np.zeros((16, 3)), np.zeros((16, 4)), b3)
    print(np.round(c_o, 4))                                                                   # ~ [1. 1. 1. 1.] overwritten
    h, c = np.zeros(4), np.zeros(4)
    for t in range(6):
        h, c = lstm_cell_step(X[t], h, c, Wx, Wh, b)
    print(np.allclose(h, H_all[-1]), np.allclose(c, c_last))                                  # True True
