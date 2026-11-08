"""Day 23 / Session 3 - Lab Assignment 2: Classifier Two-Sample Test & GAN-vs-VAE-vs-Diffusion Comparison."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def two_sample_dataset(real, gen):
    real = np.asarray(real, dtype=float)
    gen = np.asarray(gen, dtype=float)
    X = np.vstack([real, gen])
    y = np.concatenate([np.ones(len(real), dtype=int), np.zeros(len(gen), dtype=int)])   # real = 1, generated = 0
    return X, y


def separability_auc(real, gen, seed=0):
    X, y = two_sample_dataset(real, gen)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def quality_from_auc(auc):
    return float(np.clip(1 - 2 * abs(auc - 0.5), 0.0, 1.0))


def compare_families(prop):
    table = {
        "sampling_speed": "GAN",
        "mode_coverage": "Diffusion",
        "training_stability": "Diffusion",
        "likelihood_estimation": "VAE",
        "latent_interpolation": "VAE",
    }
    return table.get(prop, "unknown")


if __name__ == "__main__":
    rng = np.random.RandomState(7)
    real = rng.randn(400, 2)
    gen_good = rng.randn(400, 2)
    gen_bad = rng.randn(400, 2) + np.array([6.0, 0.0])
    X, y = two_sample_dataset(real, gen_good)
    print(X.shape, y[:2], y[-2:], np.allclose(X, np.vstack([real, gen_good])))    # (800, 2) [1 1] [0 0] True
    a_good, a_bad = separability_auc(real, gen_good, seed=0), separability_auc(real, gen_bad, seed=0)
    print(round(a_good, 2), round(a_bad, 2))                                      # ~0.5 ~1.0
    print(quality_from_auc(0.5), quality_from_auc(0.75), quality_from_auc(1.0), quality_from_auc(0.0))   # 1.0 0.5 0.0 0.0
    print(compare_families("sampling_speed"), compare_families("mode_coverage"), compare_families("training_stability"),
          compare_families("likelihood_estimation"), compare_families("latent_interpolation"), compare_families("nope"))
    print(quality_from_auc(a_good) > quality_from_auc(a_bad) + 0.5)               # True
