"""Day 21 / Session 3 - Lab Assignment 2: Bidirectional LSTM Wrapper from Scratch."""
import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def lstm_cell_step(x_t, h_prev, c_prev, Wx, Wh, b):
    H = len(h_prev)
    z = np.asarray(Wx) @ np.asarray(x_t) + np.asarray(Wh) @ np.asarray(h_prev) + np.asarray(b)
    i = sigmoid(z[0:H])
    f = sigmoid(z[H:2 * H])
    g = np.tanh(z[2 * H:3 * H])
    o = sigmoid(z[3 * H:4 * H])
    c_t = f * np.asarray(c_prev) + i * g
    h_t = o * np.tanh(c_t)
    return h_t, c_t


def lstm_run(X, Wx, Wh, b):
    H = np.asarray(Wh).shape[1]
    h = np.zeros(H)
    c = np.zeros(H)
    states = []
    for x_t in np.asarray(X, dtype=float):
        h, c = lstm_cell_step(x_t, h, c, Wx, Wh, b)
        states.append(h)
    return np.array(states)                              # (T, H) in input order


def bilstm_forward(X, Wxf, Whf, bf, Wxb, Whb, bb):
    X = np.asarray(X, dtype=float)
    Hf = lstm_run(X, Wxf, Whf, bf)                       # reads left to right
    Hb = lstm_run(X[::-1], Wxb, Whb, bb)[::-1]           # reads right to left, then re-aligned to input order
    return np.concatenate([Hf, Hb], axis=1)              # (T, 2H)


if __name__ == "__main__":
    rng = np.random.RandomState(2)
    D, H = 3, 4
    Wxf = rng.randn(4 * H, D) * 0.5; Whf = rng.randn(4 * H, H) * 0.5; bf = rng.randn(4 * H) * 0.3
    Wxb = rng.randn(4 * H, D) * 0.5; Whb = rng.randn(4 * H, H) * 0.5; bb = rng.randn(4 * H) * 0.3
    X = rng.randn(5, D)
    O = bilstm_forward(X, Wxf, Whf, bf, Wxb, Whb, bb)
    print(O.shape)                                                                       # (5, 8)
    print(np.allclose(O[:, :4], lstm_run(X, Wxf, Whf, bf)))                              # True
    Hb = lstm_run(X[::-1], Wxb, Whb, bb)[::-1]
    print(np.allclose(O[:, 4:], Hb))                                                     # True
    print(np.allclose(O[0, 4:], lstm_run(X[::-1], Wxb, Whb, bb)[-1]))                    # True: t=0 backward state read the whole sequence
    O2 = bilstm_forward(X[::-1], Wxb, Whb, bb, Wxf, Whf, bf)                             # reverse input + swap directions
    print(np.allclose(O2[:, :4], O[:, 4:][::-1]), np.allclose(O2[:, 4:], O[:, :4][::-1]))   # True True
    print(np.round(sigmoid(np.array([0.0, 2.0])), 4))                                    # [0.5 0.8808]
