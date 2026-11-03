"""Day 13 / Session 1 - Assignment: Diagnose and Fix an Overfit Classifier (wine dataset)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# 1. load the data
X, y = load_wine(return_X_y=True)
print(f"[1] X.shape={X.shape}, y.shape={y.shape}, class distribution={np.bincount(y).tolist()}")

# 2. stratified split, class proportions compared
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print(f"[2] train class share {np.round(np.bincount(y_train) / len(y_train), 3).tolist()} | "
      f"test class share {np.round(np.bincount(y_test) / len(y_test), 3).tolist()}  (stratify keeps them the same)")

# 3. unbounded tree: train AND test accuracy
tree = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
train_acc, test_acc = tree.score(X_train, y_train), tree.score(X_test, y_test)
print(f"[3] unbounded tree: train accuracy {train_acc:.4f}, test accuracy {test_acc:.4f} (gap {train_acc - test_acc:.4f}). "
      "Failure mode: overfitting (high variance) - the tree memorises the training rows, so the glowing training score is not honest.")

# 4. bias-variance sweep over max_depth
depths = list(range(1, 13))
tr_acc, te_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_train, y_train)
    tr_acc.append(m.score(X_train, y_train))
    te_acc.append(m.score(X_test, y_test))
plt.figure(figsize=(7, 4))
plt.plot(depths, tr_acc, "o-", label="train accuracy")
plt.plot(depths, te_acc, "o-", label="test accuracy")
plt.axvspan(1, 2, alpha=0.1, color="blue", label="underfitting (too shallow)")
plt.axvspan(6, 12, alpha=0.1, color="red", label="overfitting (train near 1.0, test flat)")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("Bias-variance diagnostic: DecisionTree on wine")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "13a1_depth_sweep.png"), dpi=120)
plt.close()
print(f"[4] depth 1 -> train {tr_acc[0]:.3f} / test {te_acc[0]:.3f} (underfit: both low); "
      f"depth {depths[-1]} -> train {tr_acc[-1]:.3f} / test {te_acc[-1]:.3f} (overfit: train saturates, test does not follow)")

# 5. choose max_depth by 5-fold CV on the TRAINING set only
cv_mean, cv_std = [], []
for d in depths:
    s = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), X_train, y_train, cv=5)
    cv_mean.append(s.mean())
    cv_std.append(s.std())
best_i = int(np.argmax(cv_mean))
best_depth = depths[best_i]
print(f"[5] chosen max_depth={best_depth}: CV mean {cv_mean[best_i]:.4f} +/- {cv_std[best_i]:.4f} "
      f"(picked on training-set CV, not on the single test split)")

# 6. final pipeline: train/test accuracy, classification report, confusion matrix
final = Pipeline([("clf", DecisionTreeClassifier(max_depth=best_depth, random_state=SEED))]).fit(X_train, y_train)
final_train, final_test = final.score(X_train, y_train), final.score(X_test, y_test)
pred = final.predict(X_test)
print(f"\n[6] final pipeline: train accuracy {final_train:.4f}, test accuracy {final_test:.4f}")
print(classification_report(y_test, pred, digits=3))
print("confusion matrix:\n", confusion_matrix(y_test, pred))

# 7. deliberate leakage with a scale-sensitive model (kNN), averaged over 40 random splits so the effect is visible
from sklearn.neighbors import KNeighborsClassifier

leaky_scores, clean_scores = [], []
Xs_full = StandardScaler().fit_transform(X)                                   # leak: the scaler has seen the test rows
for seed in range(40):
    a_tr, a_te, b_tr, b_te = train_test_split(Xs_full, y, test_size=0.25, stratify=y, random_state=seed)
    leaky_scores.append(KNeighborsClassifier().fit(a_tr, b_tr).score(a_te, b_te))
    c_tr, c_te, d_tr, d_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    clean_pipe = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())]).fit(c_tr, d_tr)
    clean_scores.append(clean_pipe.score(c_te, d_te))
leaky, clean_test = float(np.mean(leaky_scores)), float(np.mean(clean_scores))
print(f"\n[7a] kNN over 40 splits: scaler-leak mean accuracy {leaky:.4f} vs leakage-free pipeline {clean_test:.4f} -> difference {leaky - clean_test:+.4f}. "
      "Scaling statistics barely matter on well-behaved wine features, so this difference is within noise; it is not evidence of real inflation or deflation.")

# 7b. a leak that does inflate: pick features using ALL rows' labels, then cross-validate. Pure-noise features isolate the effect.
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

noise = np.random.RandomState(0).randn(len(y), 300)                           # no real signal at all: honest accuracy is about 1/3
leaky_X = SelectKBest(f_classif, k=10).fit_transform(noise, y)                # leak: labels of every row used to choose features
leaky_cv = cross_val_score(LogisticRegression(max_iter=2000), leaky_X, y, cv=5).mean()
clean_cv = cross_val_score(Pipeline([("select", SelectKBest(f_classif, k=10)), ("clf", LogisticRegression(max_iter=2000))]),
                           noise, y, cv=5).mean()
inflation = leaky_cv - clean_cv
print(f"[7b] pure-noise features: leaky CV accuracy {leaky_cv:.4f} vs leakage-free pipeline {clean_cv:.4f} -> inflation {inflation:+.4f}. "
      "Leakage inflates the estimate because the held-out rows' labels already shaped the features; "
      "the leakage-free score is honest (about chance).")

# 8. conclusion (numbers from this run)
print("\nConclusion:")
print(
    f"The original unbounded tree scored {train_acc:.3f} on training but {test_acc:.3f} on test, a gap of {train_acc - test_acc:.3f}, "
    f"so its training score was not trustworthy: it is high-variance overfitting. "
    f"The depth sweep shows depth 1 underfits (train {tr_acc[0]:.3f}, test {te_acc[0]:.3f}) while very deep trees reach train {tr_acc[-1]:.3f} "
    f"without a matching test gain. "
    f"Five-fold CV on the training set picked max_depth={best_depth} (CV mean {cv_mean[best_i]:.3f} +/- {cv_std[best_i]:.3f}), which I trust more "
    f"than one lucky split. "
    f"The final pipeline scored {final_test:.3f} on test with a train-test gap of {final_train - final_test:.3f}. "
    f"The leakage experiment shows the risk: selecting features on the full data inflated a pure-noise model by {inflation:+.3f} ({leaky_cv:.3f} vs {clean_cv:.3f}), "
    f"while scaling on the full data moved a kNN by only {leaky - clean_test:+.4f}, so every preprocessing step must live inside the pipeline. "
    f"Next time I would also repeat the split with several seeds, because with only {len(y_test)} test rows one split is noisy."
)
