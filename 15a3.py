"""Day 15 / Session 3 - Assignment: Diagnose and Cure a Mystery Overfit Model.

Plots are written to PNG files next to this script (matplotlib Agg backend, so it runs headless).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, learning_curve, train_test_split,
                                     validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__))

# 1. reproducible noisy dataset; test set set aside untouched until Task 8
X, y = make_classification(n_samples=2000, n_features=25, n_informative=8, n_redundant=6,
                           flip_y=0.10, class_sep=1.0, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # stratified k-fold keeps the class mix in every fold
print("train", X_train.shape, "test", X_test.shape)

# 2. the model "as handed over": no depth limit  (train accuracy only here - the test set is NOT used)
original = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
orig_train = original.score(X_train, y_train)
orig_cv = np.mean([original.__class__(**original.get_params()).fit(X_train[tr], y_train[tr]).score(X_train[va], y_train[va])
                   for tr, va in cv.split(X_train, y_train)])
print(f"[2] original RF: train acc {orig_train:.4f} vs 5-fold CV acc {orig_cv:.4f} (gap {orig_train - orig_cv:.4f})")
print("    Diagnosis: HIGH VARIANCE (overfit) - near-1.0 training score with a wide train-vs-held-out gap, "
      "not a bias problem (bias would show low scores on BOTH).")

# 3. validation curve over max_depth
depths = np.arange(1, 25)
rf_vc = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
tr_sc, va_sc = validation_curve(rf_vc, X_train, y_train, param_name="max_depth", param_range=depths, cv=cv, n_jobs=-1)
tr_m, va_m = tr_sc.mean(axis=1), va_sc.mean(axis=1)
best_depth = int(depths[va_m.argmax()])
plt.figure(figsize=(7, 4))
plt.plot(depths, tr_m, "o-", label="train")
plt.plot(depths, va_m, "o-", label="CV validation")
plt.axvline(best_depth, color="k", ls="--", label=f"CV peak (max_depth={best_depth})")
plt.axvspan(depths[0], max(best_depth - 2, 1), alpha=0.1, color="blue", label="underfit region")
plt.axvspan(best_depth + 4, depths[-1], alpha=0.1, color="red", label="overfit region")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("Validation curve (RandomForest)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "15a3_validation_curve.png"), dpi=120)
plt.close()
print(f"[3] validation curve: CV peak {va_m.max():.4f} at max_depth={best_depth}; "
      f"shallow trees underfit (depth 1 CV {va_m[0]:.4f}); deep trees overfit "
      f"(train {tr_m[-1]:.4f} vs CV {va_m[-1]:.4f}).")

# 4. learning curve for the overfit model (fixed model, growing training size)
sizes, lc_tr, lc_va = learning_curve(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
                                     X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 6), cv=cv, n_jobs=-1)
lc_tr_m, lc_va_m = lc_tr.mean(axis=1), lc_va.mean(axis=1)
plt.figure(figsize=(7, 4))
plt.plot(sizes, lc_tr_m, "o-", label="train")
plt.plot(sizes, lc_va_m, "o-", label="CV validation")
plt.fill_between(sizes, lc_tr_m, lc_va_m, alpha=0.1, label="train-CV gap")
plt.xlabel("training examples")
plt.ylabel("accuracy")
plt.title("Learning curve (unrestricted RandomForest)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "15a3_learning_curve.png"), dpi=120)
plt.close()
gap_first, gap_last = lc_tr_m[0] - lc_va_m[0], lc_tr_m[-1] - lc_va_m[-1]
shrinking = gap_last < gap_first - 0.01
print(f"[4] learning curve: train-CV gap {gap_first:.4f} at {sizes[0]} samples -> {gap_last:.4f} at {sizes[-1]} samples "
      f"(CV score {lc_va_m[-2]:.4f} -> {lc_va_m[-1]:.4f}). The gap is "
      f"{'still shrinking' if shrinking else 'barely shrinking'} at full data, so collecting more data is "
      f"{'a useful but slow' if shrinking else 'a weak'} lever compared with regularising the model.")

# 5. cure the model two different ways, tuned with CV on the training data only
# (a) regularise the forest (depth + leaf size + max_features). Selection uses the one-standard-error rule on the
#     CV results: the most regularised setting whose CV score is within 1 SE of the best one (still CV-only).
rf_search = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
                         {"max_depth": [6, 10, 15, None], "min_samples_leaf": [1, 3, 5, 10, 20],
                          "max_features": ["sqrt", 0.5]}, cv=cv, n_jobs=-1).fit(X_train, y_train)
res = pd.DataFrame(rf_search.cv_results_)
best_i = res["mean_test_score"].idxmax()
threshold = res.loc[best_i, "mean_test_score"] - res.loc[best_i, "std_test_score"] / np.sqrt(cv.get_n_splits())
ok = res[res["mean_test_score"] >= threshold].copy()
ok["complexity"] = ok["param_max_depth"].map(lambda d: 30 if d is None else d) / ok["param_min_samples_leaf"].astype(float)
pick = ok.sort_values(["complexity", "mean_test_score"], ascending=[True, False]).iloc[0]
cure_a_params = {k: pick["param_" + k] for k in ("max_depth", "min_samples_leaf", "max_features")}
cure_a_params = {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else (v.item() if hasattr(v, 'item') else v))
                 for k, v in cure_a_params.items()}
cure_a_model = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, **cure_a_params).fit(X_train, y_train)
cure_a_cv = pick["mean_test_score"]
# (b) L2-regularised logistic regression with its own tuned strength, features standardised
cure_b = GridSearchCV(Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=5000))]),
                      {"lr__C": [0.001, 0.01, 0.1, 1, 10]}, cv=cv, n_jobs=-1).fit(X_train, y_train)
print(f"[5] cure (a) RF params {cure_a_params} CV {cure_a_cv:.4f} (best raw CV {res.loc[best_i, 'mean_test_score']:.4f}); "
      f"cure (b) LogReg params {cure_b.best_params_} CV {cure_b.best_score_:.4f}")
print("    CV scheme: stratified 5-fold with a fixed seed on the TRAIN split only - it keeps the class balance in every "
      "fold and the test set is never consulted while tuning.")

# 6/7. summary table; the test set is evaluated here, once, for the finished models
models = {"original RF (no limit)": original, "cured #1 RF (depth/leaf)": cure_a_model,
          "cured #2 LogReg (L2, tuned C)": cure_b.best_estimator_}
table = pd.DataFrame([{"model": n, "train_acc": m.score(X_train, y_train), "test_acc": m.score(X_test, y_test)}
                      for n, m in models.items()])
table["gap"] = table["train_acc"] - table["test_acc"]
print("\n[6] summary\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# 8. final choice = the cure with the better CV score; report its honest test accuracy
final_name = "cured #1 RF (depth/leaf)" if cure_a_cv >= cure_b.best_score_ else "cured #2 LogReg (L2, tuned C)"
t = table.set_index("model")
o, f = t.loc["original RF (no limit)"], t.loc[final_name]
print(f"\n[8] final model: {final_name}  honest test accuracy = {f['test_acc']:.4f}")
print("\nConclusion:")
print(
    f"The handed-over forest scored {o['train_acc']:.4f} on training but only {o['test_acc']:.4f} on test, a gap of "
    f"{o['gap']:.4f}, and the validation curve showed CV accuracy peaking at max_depth={best_depth} while deeper trees "
    f"kept memorising the training set, so the diagnosis is high variance rather than bias. "
    f"The learning curve's train-CV gap went from {gap_first:.4f} to {gap_last:.4f} as data grew, so more data alone "
    f"would help only slowly. "
    f"Regularising with the one-standard-error CV settings {cure_a_params} gave test accuracy "
    f"{t.loc['cured #1 RF (depth/leaf)', 'test_acc']:.4f} with a gap of {t.loc['cured #1 RF (depth/leaf)', 'gap']:.4f}, "
    f"and the L2 logistic regression (C={cure_b.best_params_['lr__C']}) reached "
    f"{t.loc['cured #2 LogReg (L2, tuned C)', 'test_acc']:.4f} with a gap of {t.loc['cured #2 LogReg (L2, tuned C)', 'gap']:.4f}. "
    f"I chose {final_name} by CV score and its honest test accuracy is {f['test_acc']:.4f}, "
    f"closing the train-test gap from {o['gap']:.4f} to {f['gap']:.4f}. "
    f"Caveat: with flip_y=0.10 about a tenth of the labels were randomly reassigned, which is irreducible error and sets "
    f"a ceiling on accuracy of roughly 0.95, so no model can honestly reach 100% here."
)
