"""Day 18 / Session 1 - Lab Assignment 1: The Convolution Engine - valid 2D conv, output-size formula, vertical-edge detector."""
import numpy as np


def conv_output_size(W, K, P=0, S=1):
    return (W - K + 2 * P) // S + 1


def conv2d_valid(image, kernel):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    H, W = image.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = image[i:i + kh, j:j + kw]
            out[i, j] = np.sum(region * kernel)        # cross-correlation: the kernel is NOT flipped
    return out


def vertical_edge_kernel():
    return np.array([[-1.0, 0.0, 1.0],
                     [-1.0, 0.0, 1.0],
                     [-1.0, 0.0, 1.0]])


def detect_vertical_edges(image):
    return conv2d_valid(image, vertical_edge_kernel())


if __name__ == "__main__":
    print(conv_output_size(5, 3), conv_output_size(7, 3, 1, 2), conv_output_size(28, 5), conv_output_size(32, 5, 2, 1))
    print(conv2d_valid(np.ones((4, 4)), np.ones((3, 3))))
    edge = np.zeros((5, 6)); edge[:, 3:] = 10.0
    print(detect_vertical_edges(edge))
    print(conv2d_valid(np.ones((5, 7)), np.ones((3, 3))).shape)
    print(detect_vertical_edges(edge.T).max(), detect_vertical_edges(np.ones((5, 5))).max())
    # no-flip check against scipy's correlate
    from scipy.signal import correlate2d
    img = np.arange(25, dtype=float).reshape(5, 5)
    k = np.array([[1., 2., 0.], [0., 1., 3.], [4., 0., 1.]])
    print(np.allclose(conv2d_valid(img, k), correlate2d(img, k, mode="valid")))
