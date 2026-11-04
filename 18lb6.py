"""Day 18 / Session 3 - Lab Assignment 2: im2col Convolution & the Raw-Pixel vs Conv-Feature Showdown."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_split(seed=0):
    d = load_digits()
    X_img = d.images.astype(float)                      # (1797, 8, 8)
    return train_test_split(X_img, d.target, test_size=0.25, random_state=seed, stratify=d.target)


def im2col(img, k):
    """Each row is one row-major flattened k x k window, visiting output positions row by row."""
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    oh, ow = H - k + 1, W - k + 1
    cols = np.zeros((oh * ow, k * k))
    r = 0
    for i in range(oh):
        for j in range(ow):
            cols[r] = img[i:i + k, j:j + k].ravel()
            r += 1
    return cols


def conv_via_im2col(img, kernel):
    img = np.asarray(img, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    k = kernel.shape[0]
    oh, ow = img.shape[0] - k + 1, img.shape[1] - k + 1
    return (im2col(img, k) @ kernel.reshape(-1)).reshape(oh, ow)      # the whole convolution is ONE matrix multiply


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


def conv_features(X_img, kernels, pool=2):
    rows = []
    for img in X_img:
        parts = [max_pool2d(relu(conv_via_im2col(img, k)), pool).ravel() for k in kernels]
        rows.append(np.concatenate(parts))
    return np.array(rows)


def raw_vs_conv_accuracy(seed=0, kernels=None):
    if kernels is None:
        kernels = default_kernels()
    Xtr, Xte, ytr, yte = load_split(seed)
    raw_tr, raw_te = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    cf_tr, cf_te = conv_features(Xtr, kernels), conv_features(Xte, kernels)

    def fit_score(A, B):
        pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
        return float(pipe.fit(A, ytr).score(B, yte))

    return {
        "raw_acc": fit_score(raw_tr, raw_te),
        "conv_acc": fit_score(cf_tr, cf_te),
        "n_raw_features": int(raw_tr.shape[1]),
        "n_conv_features": int(cf_tr.shape[1]),
    }


if __name__ == "__main__":
    img = np.arange(64.).reshape(8, 8)
    print(im2col(img, 3).shape, im2col(img, 3)[0], im2col(img, 1).shape)
    ker = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1.]])
    print(conv_via_im2col(img, ker).shape)
    from scipy.signal import correlate2d
    print(np.allclose(conv_via_im2col(img, ker), correlate2d(img, ker, mode="valid")))
    Xtr, Xte, ytr, yte = load_split(0)
    print(Xtr.shape, Xte.shape, conv_features(Xtr[:10], default_kernels()[:2]).shape)
    print(raw_vs_conv_accuracy(0))
