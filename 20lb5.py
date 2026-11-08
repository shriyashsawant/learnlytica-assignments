"""Day 20 / Session 3 - Lab Assignment 1: Hand-built Conv-ReLU-Pool Feature Extractor for a Digit Classifier."""
import numpy as np


def conv2d(image, kernel):
    image = np.asarray(image, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    H, W = image.shape
    kh, kw = kernel.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)    # cross-correlation, kernel not flipped
    return out


def relu(x):
    return np.maximum(0, np.asarray(x, dtype=float))


def maxpool2d(x, size=2):
    x = np.asarray(x, dtype=float)
    H, W = x.shape
    oh, ow = H // size, W // size                                    # leftover rows/cols are dropped
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = x[i * size:(i + 1) * size, j * size:(j + 1) * size].max()
    return out


def extract_features(image, kernel):
    return maxpool2d(relu(conv2d(image, kernel)), 2).ravel()


if __name__ == "__main__":
    print(conv2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0], [0, 1]]))                      # [[ 6.  8.] [12. 14.]]
    print(relu([[-1, 2], [3, -4]]))                                                          # [[0. 2.] [3. 0.]]
    print(maxpool2d([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], 2))      # [[ 6.  8.] [14. 16.]]
    print(maxpool2d(np.arange(25).reshape(5, 5), 2).shape)                                   # (2, 2)

    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    f = extract_features(digits.images[0], kernel)
    print(f.shape, f.min() >= 0)                                                             # (9,) True
    bank = [kernel, kernel.T, np.ones((3, 3)) / 9]                                           # a 3-kernel bank
    X = np.array([np.concatenate([extract_features(im, k) for k in bank]) for im in digits.images])
    Xtr, Xte, ytr, yte = train_test_split(X, digits.target, test_size=0.3, random_state=0)
    print(round(LogisticRegression(max_iter=3000).fit(Xtr, ytr).score(Xte, yte), 3))         # well above 0.80
