"""Day 16 / Session 2 - Assignment: Tune and Blend an Ensemble on a Real-ish Dataset."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__))

# 1. data + stratified 60/20/20 train/validation/test split
X, y = make_classification(n_samples=4000, n_features=20, n_informative=10, n_redundant=4, flip_y=0.05,
                           class_sep=0.9, random_state=SEED)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
print("train", X_tr.shape, "val", X_val.shape, "test", X_te.shape)

test_acc, val_acc = {}, {}


def record(name, model):
    test_acc[name] = accuracy_score(y_te, model.predict(X_te))
    val_acc[name] = accuracy_score(y_val, model.predict(X_val))


# 2. baseline tree
tree = DecisionTreeClassifier(random_state=SEED).fit(X_tr, y_tr)
record("tree", tree)
tree_train = tree.score(X_tr, y_tr)
print(f"\n[2] tree: train {tree_train:.4f} vs test {test_acc['tree']:.4f} (gap {tree_train - test_acc['tree']:.4f}) "
      "-> a near-1.0 train score with a big gap means HIGH VARIANCE, not high bias.")

# 3. bagging / random forest (attacks variance) and gradient boosting (attacks bias)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
record("random forest (bagging)", rf)
gb = GradientBoostingClassifier(n_estimators=300, random_state=SEED).fit(X_tr, y_tr)
record("gradient boosting (untuned)", gb)
print(f"[3] random forest test {test_acc['random forest (bagging)']:.4f} - averaging many deep decorrelated trees cuts VARIANCE; "
      f"gradient boosting test {test_acc['gradient boosting (untuned)']:.4f} - adding shallow trees that fix residuals cuts BIAS "
      "(but can overfit if run too long).")

# 4. early stopping with staged_predict on the validation set
stage_val = [accuracy_score(y_val, p) for p in gb.staged_predict(X_val)]
best_n = int(np.argmax(stage_val) + 1)
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(stage_val) + 1), stage_val)
plt.axvline(best_n, color="r", ls="--", label=f"best n_estimators = {best_n}")
plt.xlabel("number of trees")
plt.ylabel("validation accuracy")
plt.title("Early stopping with staged_predict")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "16a2_early_stopping.png"), dpi=120)
plt.close()
gb_es = GradientBoostingClassifier(n_estimators=best_n, random_state=SEED).fit(X_tr, y_tr)
print(f"\n[4] validation accuracy peaks at {max(stage_val):.4f} with {best_n} trees (vs {stage_val[-1]:.4f} at 300); "
      f"refit at {best_n} -> test accuracy {gb_es.score(X_te, y_te):.4f}")

# 5. hyperparameter sweep, judged on VALIDATION only
rows = []
for lr in (0.05, 0.1, 0.2):
    for depth in (2, 3, 4):
        m = GradientBoostingClassifier(n_estimators=best_n, learning_rate=lr, max_depth=depth, random_state=SEED).fit(X_tr, y_tr)
        rows.append({"learning_rate": lr, "max_depth": depth, "val_acc": accuracy_score(y_val, m.predict(X_val))})
sweep = pd.DataFrame(rows)
print("\n[5] boosting sweep (validation accuracy):\n", sweep.pivot(index="learning_rate", columns="max_depth", values="val_acc").round(4).to_string())
bc = sweep.loc[sweep["val_acc"].idxmax()]
gb_tuned = GradientBoostingClassifier(n_estimators=best_n, learning_rate=float(bc["learning_rate"]),
                                      max_depth=int(bc["max_depth"]), random_state=SEED).fit(X_tr, y_tr)
record("tuned boosting", gb_tuned)
print(f"    best on validation: learning_rate={bc['learning_rate']}, max_depth={int(bc['max_depth'])} "
      f"(val {bc['val_acc']:.4f}) -> test {test_acc['tuned boosting']:.4f}")

# 6. soft voting and stacking over diverse base models (all expose predict_proba)
bases = [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
         ("rf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
         ("gb", GradientBoostingClassifier(n_estimators=best_n, random_state=SEED)),
         ("nb", GaussianNB())]
voting = VotingClassifier(bases, voting="soft").fit(X_tr, y_tr)
stacking = StackingClassifier(bases, final_estimator=LogisticRegression(max_iter=2000), cv=5).fit(X_tr, y_tr)
record("voting (soft)", voting)
record("stacking", stacking)
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
record("bagging", bag)
print(f"\n[6] soft voting test {test_acc['voting (soft)']:.4f}; stacking test {test_acc['stacking']:.4f}")

# 7. final bar chart + recommendation (model shipped = best VALIDATION accuracy among the finished candidates)
chart = {k: test_acc[k] for k in ("tree", "bagging", "tuned boosting", "voting (soft)", "stacking")}
plt.figure(figsize=(7, 4))
plt.bar(list(chart), list(chart.values()))
plt.ylim(min(chart.values()) - 0.05, 1.0)
plt.ylabel("test accuracy")
plt.xticks(rotation=20)
plt.title("Test accuracy by model")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "16a2_test_accuracy.png"), dpi=120)
plt.close()
ship = max(chart, key=lambda k: val_acc[k])
print("\nRecommendation:")
print(
    f"I would ship the {ship} model (best validation accuracy {val_acc[ship]:.4f}, test accuracy {test_acc[ship]:.4f}). "
    f"The single decision tree scored {tree_train:.4f} on training but {test_acc['tree']:.4f} on test, so it is a high-variance "
    f"model; bagging lifted test accuracy to {test_acc['bagging']:.4f} by averaging that variance away. "
    f"Gradient boosting attacks bias instead: tuned with early stopping at {best_n} trees "
    f"(learning_rate={bc['learning_rate']}, max_depth={int(bc['max_depth'])}) it reached {test_acc['tuned boosting']:.4f}, and stopping "
    f"early mattered because validation accuracy peaked at {max(stage_val):.4f} rather than at the 300-tree maximum ({stage_val[-1]:.4f}). "
    f"Soft voting ({test_acc['voting (soft)']:.4f}) and stacking ({test_acc['stacking']:.4f}) blend diverse models to cut variance, "
    f"but a blend is only as good as its weakest members: voting weights them equally, while stacking learns the weights "
    f"from out-of-fold predictions. "
    f"The gaps between the top models are small relative to sampling noise on a {len(y_te)}-row test set, so the choice "
    f"rests on validation evidence and overfitting risk: the shipped model's meta-learner or tree count was fixed on "
    f"validation or out-of-fold data, never on the test set."
)
