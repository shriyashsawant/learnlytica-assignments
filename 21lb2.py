"""Day 21 / Session 1 - Lab Assignment 2: Many-to-One RNN Head - Final State to Softmax Class Probabilities."""
import numpy as np


def softmax(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z))                        # subtract the max so large logits cannot overflow
    return e / e.sum()


def rnn_final_state(X, h0, Wxh, Whh, b):
    h = np.asarray(h0, dtype=float)
    for x_t in np.asarray(X, dtype=float):
        h = np.tanh(np.asarray(Wxh) @ x_t + np.asarray(Whh) @ h + np.asarray(b))
    return h                                         # only the last hidden state h_T


def rnn_classify(X, h0, Wxh, Whh, b, Why, by):
    h_T = rnn_final_state(X, h0, Wxh, Whh, b)
    logits = np.asarray(Why) @ h_T + np.asarray(by)
    return softmax(logits)


def predict(probs):
    return int(np.argmax(probs))


def cross_entropy(probs, y_true):
    return float(-np.log(np.asarray(probs)[y_true]))


if __name__ == "__main__":
    Wxh = np.array([[0.10, 0.20], [0.00, -0.10], [0.30, 0.10]])
    Whh = np.array([[0.05, 0.00, 0.10], [0.10, 0.05, 0.00], [0.00, 0.10, 0.05]])
    b = np.array([0.00, 0.10, -0.10])
    Why = np.array([[0.20, -0.10, 0.30], [0.00, 0.40, -0.20], [0.10, 0.10, 0.10]])
    by = np.array([0.00, 0.10, -0.10])
    h0 = np.zeros(3)
    X = np.array([[1.0, 0.5], [-0.5, 1.0]])
    print(np.round(softmax([1.0, 2.0, 3.0]), 4))                 # [0.09   0.2447 0.6652]
    probs = rnn_classify(X, h0, Wxh, Whh, b, Why, by)
    print(np.round(probs, 4), round(float(probs.sum()), 6))      # [0.326  0.3752 0.2987] 1.0
    print(predict(probs), round(cross_entropy(probs, 2), 4))     # 1 1.2082
    print(np.round(softmax([1000.0, 1001.0, 1002.0]), 4), np.allclose(softmax([1, 2, 3]), softmax([101, 102, 103])))   # stable, shift-invariant
