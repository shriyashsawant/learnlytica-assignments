"""Day 18 / Session 3 - Lab Assignment 1: Build a CNN Feature Extractor from Scratch (conv -> relu -> maxpool) in NumPy."""
import numpy as np
from sklearn.datasets import load_digits


def load_images():
    d = load_digits()
    return d.images.astype(float), d.target        # (1797, 8, 8), pixels 0..16


def conv2d_valid(img, kernel):
    """VALID 2D cross-correlation (what Conv2D really computes): the kernel is NOT flipped."""
    img = np.asarray(img, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    H, W = img.shape
    kh, kw = kernel.shape
    oh, ow = H - kh + 1, W - kw + 1
    out = np.zeros((oh, ow))
    for i in range(kh):                  # add each kernel weight times the matching shifted copy of the image
        for j in range(kw):
            out += kernel[i, j] * img[i:i + oh, j:j + ow]
    return out


def relu(x):
    return np.maximum(0.0, np.asarray(x, dtype=float))


def max_pool2d(fmap, size=2):
    fmap = np.asarray(fmap, dtype=float)
    H2, W2 = fmap.shape[0] // size, fmap.shape[1] // size
    out = np.zeros((H2, W2))
    for i in range(H2):
        for j in range(W2):
            out[i, j] = fmap[i * size:(i + 1) * size, j * size:(j + 1) * size].max()
    return out


def default_kernels():
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)
    blur = np.ones((3, 3)) / 9.0
    return [sobel_x, sobel_y, laplacian, blur]


def extract_features(X_img, kernels, pool=2):
    features = []
    for img in X_img:
        parts = [max_pool2d(relu(conv2d_valid(img, k)), pool).ravel() for k in kernels]
        features.append(np.concatenate(parts))
    return np.array(features)


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    img = np.arange(9.).reshape(3, 3)
    print(conv2d_valid(img, np.array([[1, 0], [0, -1.]])))
    print(max_pool2d(np.arange(1, 17.).reshape(4, 4), 2))
    X_img, y = load_images()
    F = extract_features(X_img[:5], default_kernels(), pool=2)
    print(F.shape, extract_features(X_img[:20], default_kernels()[:3]).shape)
    full = extract_features(X_img, default_kernels(), pool=2)
    Xtr, Xte, ytr, yte = train_test_split(full, y, test_size=0.25, random_state=0, stratify=y)
    print("test accuracy", round(LogisticRegression(max_iter=5000).fit(Xtr, ytr).score(Xte, yte), 3))
    print(conv2d_valid(np.arange(25.).reshape(5, 5)[:3, :3] * 0 + np.arange(9.).reshape(3, 3), np.arange(9.).reshape(3, 3)))
