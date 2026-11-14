"""Day 23 / Session 3 - Lab Assignment 2: Classifier Two-Sample Test & GAN-vs-VAE-vs-Diffusion Comparison."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

rng = np.random.RandomState(7)
real = rng.randn(400, 2)                                # real data ~ N(0, I)
gen_good = rng.randn(400, 2)                            # same distribution -> hard to tell apart
gen_bad = rng.randn(400, 2) + np.array([6.0, 0.0])      # shifted -> easy to tell apart


def c2st_accuracy(real, gen, seed=0):
    X = np.vstack([real, gen])
    y = np.concatenate([np.ones(len(real)), np.zeros(len(gen))])                 # real = 1, generated = 0
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te)))


def rank_generators(real, gen_dict, seed=0):
    scores = {name: c2st_accuracy(real, g, seed) for name, g in gen_dict.items()}
    return sorted(scores, key=lambda name: abs(scores[name] - 0.5))              # closest to 0.5 = best


if __name__ == "__main__":
    print(round(c2st_accuracy(real, gen_good), 2))                               # ~0.5
    print(round(c2st_accuracy(real, gen_bad), 2))                                # ~1.0
    print(rank_generators(real, {"good": gen_good, "bad": gen_bad}))             # ['good', 'bad']
    print(c2st_accuracy(real, gen_good, seed=3) == c2st_accuracy(real, gen_good, seed=3))   # True (deterministic)
    print(rank_generators(real, {"bad": gen_bad, "good": gen_good}))             # ['good', 'bad'] whatever the dict order
