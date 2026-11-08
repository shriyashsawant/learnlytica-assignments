"""Day 20 / Session 2 - Lab Assignment 2: Saliency Maps and Occlusion Sensitivity - Two Ways to See What a CNN Sees."""
import numpy as np


def saliency_map(grad):
    return np.abs(np.asarray(grad, dtype=float)).max(axis=0)      # |gradient|, then max over channels


def normalize01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def occlude(image, top, left, size, fill=0.0):
    out = np.array(image, dtype=float, copy=True)                 # never mutate the input
    out[top:top + size, left:left + size] = fill
    return out


def occlusion_map(image, score_fn, size, stride, fill=0.0):
    image = np.asarray(image, dtype=float)
    H, W = image.shape
    baseline = score_fn(image)
    rows = list(range(0, H - size + 1, stride))
    cols = list(range(0, W - size + 1, stride))
    grid = np.zeros((len(rows), len(cols)))
    for a, top in enumerate(rows):
        for b, left in enumerate(cols):
            grid[a, b] = baseline - score_fn(occlude(image, top, left, size, fill))   # big drop = important region
    return grid


if __name__ == "__main__":
    grad = np.array([[[1., -9.], [0., 2.]], [[-4., 3.], [7., -1.]]])
    print(saliency_map(grad))                                     # [[4. 9.] [7. 2.]]
    print(normalize01(saliency_map(grad)))

    img = np.zeros((6, 6)); img[2:4, 2:4] = 10.0

    def score_fn(x):
        return float(np.asarray(x)[2:4, 2:4].sum())

    sens = occlusion_map(img, score_fn, size=2, stride=2, fill=0.0)
    print(sens.shape)                                             # (3, 3)
    print(sens)                                                   # 40 at the centre cell only
    patched = occlude(img, 0, 0, 2, fill=5.0)
    print(patched[0:2, 0:2].tolist(), img[0:2, 0:2].tolist())     # [[5,5],[5,5]] [[0,0],[0,0]] (input untouched)
