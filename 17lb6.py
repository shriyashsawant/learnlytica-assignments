"""Day 17 / Session 3 - Lab Assignment 2: Evaluate the Classifier - One-Hot, Confusion Matrix & Per-Class Recall."""
import numpy as np


def one_hot(y, num_classes):
    y = np.asarray(y, dtype=int)
    out = np.zeros((len(y), num_classes), dtype=int)
    out[np.arange(len(y)), y] = 1
    return out


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        cm[t, p] += 1                      # rows = true class, columns = predicted class
    return cm


def overall_accuracy(cm):
    cm = np.asarray(cm)
    total = cm.sum()
    return float(np.trace(cm) / total) if total > 0 else 0.0


def per_class_accuracy(cm):
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1)
    out = np.zeros(len(cm), dtype=float)
    for i, s in enumerate(row_sums):
        out[i] = cm[i, i] / s if s > 0 else 0.0
    return out


def most_confused_pair(cm):
    off = np.array(cm, dtype=int)
    np.fill_diagonal(off, 0)                              # ignore the correct predictions
    t, p = np.unravel_index(np.argmax(off), off.shape)
    return int(t), int(p)


if __name__ == "__main__":
    print(one_hot([0, 2, 1], 3))
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred, 3)
    print(cm)
    print(round(overall_accuracy(cm), 3), np.round(per_class_accuracy(cm), 3), most_confused_pair(cm))
    perfect = confusion_matrix([0, 1, 2], [0, 1, 2], 3)
    print(perfect.tolist(), overall_accuracy(perfect), per_class_accuracy(perfect).tolist())
    print(most_confused_pair(confusion_matrix([2, 2, 2, 0, 1], [3, 3, 2, 0, 1], 4)))
