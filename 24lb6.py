"""Day 24 / Session 3 - Lab Assignment 2: The RNN Sequential Bottleneck & Positional Information Loss."""
import numpy as np

seq = np.array([[1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [4.0, 0.0, 0.0]])             # a length-4 sequence of 3-dim token vectors
perm = np.array([3, 2, 1, 0])                 # a reordering (reverse) of the 4 tokens


def encode_mean(seq):
    return np.asarray(seq, dtype=float).mean(axis=0)       # order-invariant: average over the T token rows


def encode_last(seq):
    return np.asarray(seq, dtype=float)[-1]                # the last token's state keeps a trace of order


def contexts_equal(u, v):
    return bool(np.allclose(u, v))


if __name__ == "__main__":
    X = np.array([[1., 2.], [3., 4.], [5., 0.]])
    print(encode_mean(X))                                                          # [3. 2.]
    print(encode_last(X))                                                          # [5. 0.]
    print(contexts_equal(encode_mean(seq), encode_mean(seq[perm])))                # True
    print(contexts_equal(encode_last(seq), encode_last(seq[perm])))                # False
    s1 = np.array([[2., 1.], [0., 5.], [4., 0.]])
    s2 = s1[[2, 1, 0]]
    print(contexts_equal(encode_mean(s1), encode_mean(s2)))                        # True
    print(contexts_equal(encode_last(s1), encode_last(s2)))                        # False
