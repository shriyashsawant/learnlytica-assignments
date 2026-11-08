"""Day 20 / Session 3 - Lab Assignment 2: Data Augmentation (Shift/Flip) & Evaluation Metrics for a CV Model."""
import numpy as np


def shift_image(img, dy, dx):
    img = np.asarray(img, dtype=float)
    H, W = img.shape
    out = np.zeros_like(img)
    if abs(dy) >= H or abs(dx) >= W:
        return out                                           # everything shifted out of the frame
    src_y = slice(0, H - dy) if dy >= 0 else slice(-dy, H)
    dst_y = slice(dy, H) if dy >= 0 else slice(0, H + dy)
    src_x = slice(0, W - dx) if dx >= 0 else slice(-dx, W)
    dst_x = slice(dx, W) if dx >= 0 else slice(0, W + dx)
    out[dst_y, dst_x] = img[src_y, src_x]
    return out


def flip_image(img):
    return np.asarray(img, dtype=float)[:, ::-1].copy()      # left-right mirror


def augment_dataset(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]              # down, up, right, left
    blocks = [X]
    for dy, dx in shifts:
        blocks.append(np.array([shift_image(im, dy, dx) for im in X]).reshape(X.shape))
    return np.concatenate(blocks, axis=0), np.tile(y, 5)     # every variant keeps its label


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, n_classes):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        M[int(t), int(p)] += 1                               # M[true, predicted]
    return M


def per_class_recall(y_true, y_pred, n_classes):
    M = confusion_matrix(y_true, y_pred, n_classes)
    out = np.zeros(n_classes)
    for c in range(n_classes):
        total = M[c, :].sum()
        out[c] = M[c, c] / total if total > 0 else 0.0
    return out


def macro_f1(y_true, y_pred, n_classes):
    M = confusion_matrix(y_true, y_pred, n_classes)
    scores = []
    for c in range(n_classes):
        tp = M[c, c]
        fp = M[:, c].sum() - tp
        fn = M[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


if __name__ == "__main__":
    print(shift_image([[1, 2], [3, 4]], 0, 1))                       # [[0. 1.] [0. 3.]]
    print(shift_image([[1, 2], [3, 4]], 1, 0))                       # [[0. 0.] [1. 2.]]
    print(flip_image([[1, 2, 3], [4, 5, 6]]))                        # [[3. 2. 1.] [6. 5. 4.]]
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    digits = load_digits()
    images, labels = digits.images, digits.target
    Xa, ya = augment_dataset(images[:2], labels[:2])
    print(Xa.shape, ya.shape, np.array_equal(ya[:2], labels[:2]), np.array_equal(Xa[:2], images[:2]))   # (10, 8, 8) (10,) True True
    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 2, 2]
    print(accuracy(y_true, y_pred), per_class_recall(y_true, y_pred, 3), round(macro_f1(y_true, y_pred, 3), 4))   # 0.8 [1. 0.5 1.] 0.7778
    # robustness: augmented model vs baseline on shifted test digits
    Xtr, Xte, ytr, yte = train_test_split(images, labels, test_size=0.3, random_state=0)
    Xte_shift = np.array([shift_image(im, 1, 1) for im in Xte])
    flat = lambda a: a.reshape(len(a), -1)
    base = LogisticRegression(max_iter=3000).fit(flat(Xtr), ytr)
    Xa, ya = augment_dataset(Xtr, ytr)
    aug = LogisticRegression(max_iter=3000).fit(flat(Xa), ya)
    print(round(base.score(flat(Xte_shift), yte), 3), round(aug.score(flat(Xte_shift), yte), 3), round(aug.score(flat(Xte), yte), 3))
