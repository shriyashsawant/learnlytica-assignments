"""Day 13 / Session 3 - Assignment: Evaluate, Persist, and Serve a Classifier (imbalanced churn-style data).

Writes churn_model.joblib, churn_model.meta.json, app.py (FastAPI stub) and 13a3_roc.png next to this script / notebook.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# 1. imbalanced data, stratified split, class balance
X, y = make_classification(n_samples=5000, n_features=15, n_informative=6, n_redundant=3, weights=[0.9, 0.1],
                           flip_y=0.02, class_sep=0.9, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
print(f"[1] class balance: train positive rate {y_train.mean():.3f}, test positive rate {y_test.mean():.3f}")

# 2. pipeline + test accuracy
pipe = Pipeline([("scale", StandardScaler()),
                 ("clf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1))])
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
pred = pipe.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"[2] test accuracy {acc:.4f}  (a model that always says 'no' would score {1 - y_test.mean():.4f})")

# 3. confusion matrix and by-hand precision / recall / F1
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
prec_hand = tp / (tp + fp) if tp + fp else 0.0
rec_hand = tp / (tp + fn)
f1_hand = 2 * prec_hand * rec_hand / (prec_hand + rec_hand) if prec_hand + rec_hand else 0.0
print(f"\n[3] confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"    by hand: precision {prec_hand:.4f}, recall {rec_hand:.4f}, F1 {f1_hand:.4f}")
print(f"    sklearn: precision {precision_score(y_test, pred):.4f}, recall {recall_score(y_test, pred):.4f}")
assert np.isclose(prec_hand, precision_score(y_test, pred)) and np.isclose(rec_hand, recall_score(y_test, pred))
print(classification_report(y_test, pred, digits=3))

# 4. ROC-AUC from probabilities + ROC curve with the random baseline
auc = roc_auc_score(y_test, proba)
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"model (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random baseline")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("ROC curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "13a3_roc.png"), dpi=120)
plt.close()
print(f"[4] ROC-AUC {auc:.4f}: if I pick one churner and one non-churner at random, the model ranks the churner higher "
      f"{auc:.1%} of the time.")

# 5. threshold sweep + cost-based operating point. The threshold is CHOSEN on out-of-fold train predictions (not on test).
cost_fn, cost_fp = 10.0, 1.0       # losing a customer we failed to flag is assumed 10x worse than a wasted retention offer
oof = cross_val_predict(pipe, X_train, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
costs = []
for t in thresholds:
    p = (oof >= t).astype(int)
    costs.append(cost_fn * np.sum((y_train == 1) & (p == 0)) + cost_fp * np.sum((y_train == 0) & (p == 1)))
chosen = thresholds[int(np.argmin(costs))]
print(f"\n[5] threshold sweep on the TEST set (precision vs recall); operating threshold chosen from train out-of-fold cost = {chosen}")
print("    threshold  precision  recall")
for t in thresholds:
    p = (proba >= t).astype(int)
    print(f"      {t:>4}      {precision_score(y_test, p, zero_division=0):.3f}     {recall_score(y_test, p):.3f}")
final_pred = (proba >= chosen).astype(int)
final_prec, final_rec = precision_score(y_test, final_pred, zero_division=0), recall_score(y_test, final_pred)
print(f"    recommended threshold {chosen}: precision {final_prec:.3f}, recall {final_rec:.3f}. A false negative (a churner we miss) costs "
      f"about {cost_fn / cost_fp:.0f}x a false positive (an unnecessary retention offer), so we accept lower precision to raise recall.")

# 6. persist the WHOLE pipeline + a manifest
artifact = os.path.join(OUT, "churn_model.joblib")
joblib.dump(pipe, artifact)
meta = {
    "version": "1.0.0",
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "scikit_learn_version": sklearn.__version__,
    "n_features": int(X_train.shape[1]),
    "chosen_threshold": chosen,
    "metrics_at_threshold": {"precision": round(float(final_prec), 4), "recall": round(float(final_rec), 4),
                             "roc_auc": round(float(auc), 4)},
}
with open(os.path.join(OUT, "churn_model.meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("\n[6] saved churn_model.joblib and churn_model.meta.json:", json.dumps(meta))

# 7. fresh load: predictions must be identical
reloaded = joblib.load(artifact)
same = bool(np.array_equal(reloaded.predict(X_test), pipe.predict(X_test))
            and np.allclose(reloaded.predict_proba(X_test), pipe.predict_proba(X_test)))
print(f"[7] reloaded model gives identical predictions and probabilities: {same}")
assert same

# 8. FastAPI stub (written as a file; no need to run uvicorn)
app_code = '''import json

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Loaded ONCE when the process starts, not on every request: unpickling is slow and the same object can serve every call.
model = joblib.load("churn_model.joblib")
threshold = json.load(open("churn_model.meta.json"))["chosen_threshold"]

app = FastAPI(title="Churn model service")


class Features(BaseModel):
    values: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: Features):
    probability = float(model.predict_proba(np.array([features.values]))[0, 1])
    return {"churn_probability": probability, "churn": probability >= threshold}
'''
with open(os.path.join(OUT, "app.py"), "w") as f:
    f.write(app_code)
print("[8] wrote app.py (FastAPI /health and /predict). The model is loaded at startup, not per request, because loading "
      "reads and unpickles the file (slow, and repeated for every call it would add latency and disk I/O), while the loaded "
      "pipeline is read-only and safe to reuse across requests.")

# 9. evaluation summary
print("\nEvaluation summary:")
print(
    f"The data is imbalanced ({y_test.mean():.1%} positives), so accuracy ({acc:.3f}) is misleading: always answering 'no' would score "
    f"{1 - y_test.mean():.3f}. "
    f"At the default 0.5 threshold the model has precision {precision_score(y_test, pred):.3f} and recall {recall_score(y_test, pred):.3f}, "
    f"with ROC-AUC {auc:.3f}. "
    f"I would put recall at the chosen operating point in the SLA (recall {final_rec:.3f} at threshold {chosen}), with a precision floor, "
    f"because missing a churner is far more costly than a wasted offer. "
    f"The threshold {chosen} was picked from cost on out-of-fold training predictions, not from the test set. "
    f"The artifact carries a concrete risk: joblib/pickle executes code on load, so only load files this team produced, and a different "
    f"scikit-learn version (trained on {sklearn.__version__}) can break or silently change the model, which is why the manifest records the version."
)
