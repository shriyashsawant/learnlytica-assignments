"""Day 18 / Session 1 - Lab Assignment 2: The Feature-Map Pipeline - ReLU, 2x2 max-pooling and a conv->relu->pool forward pass."""
import numpy as np


def conv2d_valid(image, kernel):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    H, W = image.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out


def relu(x):
    return np.maximum(0.0, np.asarray(x, dtype=float))


def pool_output_size(W):
    return W // 2


def maxpool2x2(x):
    x = np.asarray(x, dtype=float)
    H2, W2 = x.shape[0] // 2, x.shape[1] // 2          # a trailing odd row/column is dropped
    out = np.zeros((H2, W2))
    for i in range(H2):
        for j in range(W2):
            out[i, j] = x[2 * i:2 * i + 2, 2 * j:2 * j + 2].max()
    return out


def forward(image, kernel):
    return maxpool2x2(relu(conv2d_valid(image, kernel)))


if __name__ == "__main__":
    print(relu(np.array([-3., -0.2, 0., 2.])))
    x = np.arange(1, 17, dtype=float).reshape(4, 4)
    print(maxpool2x2(x))
    img = np.arange(64, dtype=float).reshape(8, 8)
    print(forward(img, np.ones((3, 3))).shape)
    print(maxpool2x2(np.ones((6, 6))).shape, maxpool2x2(np.ones((5, 7))).shape, [pool_output_size(w) for w in (8, 5, 1)])
    print(maxpool2x2(np.array([[1., 2.], [10., -5.]])), relu(np.array([[-1., 2.]])))
    print(np.array_equal(forward(img, np.ones((3, 3))), maxpool2x2(relu(conv2d_valid(img, np.ones((3, 3)))))))
