"""Day 20 / Session 1 - Lab Assignment 1: Segmentation Metrics - IoU, Dice & Soft Dice Loss from Scratch."""
import numpy as np


def binarize(prob, thresh=0.5):
    return (np.asarray(prob) >= thresh).astype(int)


def confusion_counts(pred, true):
    p = np.asarray(pred) > 0
    t = np.asarray(true) > 0
    return {
        "tp": int(np.sum(p & t)),
        "fp": int(np.sum(p & ~t)),
        "fn": int(np.sum(~p & t)),
        "tn": int(np.sum(~p & ~t)),
    }


def iou(pred, true):
    c = confusion_counts(pred, true)
    union = c["tp"] + c["fp"] + c["fn"]
    if union == 0:
        return 1.0                              # both masks empty: perfect agreement
    return c["tp"] / union


def dice_coef(pred, true):
    c = confusion_counts(pred, true)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    if denom == 0:
        return 1.0
    return 2 * c["tp"] / denom


def dice_loss(prob, true, eps=1e-6):
    prob = np.asarray(prob, dtype=float)
    true = np.asarray(true, dtype=float)
    soft = (2 * np.sum(prob * true) + eps) / (np.sum(prob) + np.sum(true) + eps)
    return float(1 - soft)


if __name__ == "__main__":
    true = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    pred = np.array([[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    print(confusion_counts(pred, true))                           # {'tp': 3, 'fp': 1, 'fn': 1, 'tn': 11}
    print(iou(pred, true))                                        # 0.6
    print(dice_coef(pred, true))                                  # 0.75
    print(round(dice_loss(pred.astype(float), true.astype(float)), 4))   # 0.25
    z = np.zeros((5, 5), dtype=int)
    print(iou(z, z), dice_coef(z, z))                             # 1.0 1.0
    print(binarize([[0.2, 0.5], [0.9, 0.49]], 0.5))               # [[0 1] [1 0]]
    a = np.random.RandomState(0).randint(0, 2, (6, 6)); b = np.random.RandomState(1).randint(0, 2, (6, 6))
    print(np.isclose(dice_coef(a, b), 2 * iou(a, b) / (1 + iou(a, b))), round(dice_loss(true.astype(float), true.astype(float)), 6))   # True 0.0
