"""Day 13 / Session 3 - Lab Assignment 1: Evaluation Metrics from Scratch (confusion matrix, PR/F1, ROC-AUC, thresholds)."""
import numpy as np
from scipy.stats import rankdata


def confusion_at(y_true, y_scores, threshold):
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_scores) >= threshold).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (pred == 1))),
        "fp": int(np.sum((y_true == 0) & (pred == 1))),
        "tn": int(np.sum((y_true == 0) & (pred == 0))),
        "fn": int(np.sum((y_true == 1) & (pred == 0))),
    }


def precision_recall_f1(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def roc_auc(y_true, y_scores):
    """AUC = probability a random positive outranks a random negative (Mann-Whitney U / rank formula)."""
    y_true = np.asarray(y_true)
    ranks = rankdata(np.asarray(y_scores))          # ties get the average rank
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.0
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def best_threshold(y_true, y_scores, thresholds):
    best_t, best_f1 = None, -1.0
    for t in thresholds:
        c = confusion_at(y_true, y_scores, t)
        f1 = precision_recall_f1(c["tp"], c["fp"], c["tn"], c["fn"])["f1"]
        if f1 > best_f1:  # strict '>' keeps the first threshold on ties
            best_t, best_f1 = t, f1
    return best_t


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.2, 0.6, 0.4, 0.7, 0.9])
    print(confusion_at(y, s, 0.5))
    c = confusion_at(y, s, 0.5)
    print(precision_recall_f1(c["tp"], c["fp"], c["tn"], c["fn"]))
    print(round(roc_auc(y, s), 3))
    print(best_threshold(y, s, [0.3, 0.5, 0.7]))
    # agreement with scikit-learn on the lab data
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, size=200)
    ss = np.clip(0.5 + 0.3 * (yy * 2 - 1) + rng.normal(0, 0.2, 200), 0, 1)
    print(np.isclose(roc_auc(yy, ss), roc_auc_score(yy, ss)))
    cc = confusion_at(yy, ss, 0.5)
    tn, fp, fn, tp = confusion_matrix(yy, (ss >= 0.5).astype(int)).ravel()
    print(cc == {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}, np.isclose(precision_recall_f1(**cc)["f1"], f1_score(yy, (ss >= 0.5).astype(int))))
    print(precision_recall_f1(0, 0, 5, 3), roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]))
