"""Day 16 / Session 3 - Assignment: Improve a Credit-Default Classifier End to End (imbalanced data)."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

# 1. imbalanced data (~7% positive), stratified split with a fixed seed, positive rate in each split
X, y = make_classification(n_samples=10000, n_features=18, n_informative=5, n_redundant=3, n_repeated=0,
                           weights=[0.93, 0.07], flip_y=0.01, class_sep=0.8, random_state=SEED)
cols = [f"f{i}" for i in range(X.shape[1])]
seg = (X[:, 0] > np.median(X[:, 0])).astype(int)       # a real-feature slice carried through the split
X_tr, X_te, y_tr, y_te, seg_tr, seg_te = train_test_split(X, y, seg, test_size=0.25, stratify=y, random_state=SEED)
print(f"[1] positive rate: all {y.mean():.4f} | train {y_tr.mean():.4f} | test {y_te.mean():.4f}  (test frozen from here on)")


def scores(y_true, pred, proba):
    return {"accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred), "F1": f1_score(y_true, pred), "PR-AUC": average_precision_score(y_true, proba)}


# 2. plain baseline
baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
p_base = baseline.predict_proba(X_te)[:, 1]
pred_base = (p_base >= 0.5).astype(int)
sc_base = scores(y_te, pred_base, p_base)
print(f"\n[2] baseline: accuracy {sc_base['accuracy']:.4f} but recall only {sc_base['recall']:.4f} "
      f"(precision {sc_base['precision']:.4f}, F1 {sc_base['F1']:.4f}, PR-AUC {sc_base['PR-AUC']:.4f}).")
print(f"    Accuracy paradox: always predicting 'no default' already scores {1 - y_te.mean():.4f} accuracy, "
      "so a high accuracy says nothing about catching defaulters.")

# 3. error analysis
cm = confusion_matrix(y_te, pred_base)
print("\n[3] confusion matrix (rows=true, cols=pred):\n", cm)
for name, mask in (("segment f0 > median", seg_te == 1), ("segment f0 <= median", seg_te == 0)):
    print(f"    recall on slice '{name}': {recall_score(y_te[mask], pred_base[mask]):.4f}  (n_pos={int(y_te[mask].sum())})")
fn = np.where((y_te == 1) & (pred_base == 0))[0]
fn_df = pd.DataFrame(X_te[fn][:, :5], columns=cols[:5]).assign(p_default=p_base[fn]).head(5)
print("    first 5 false negatives (first 5 features + predicted default probability):\n", fn_df.round(3).to_string())
neg_mean = p_base[y_te == 0].mean()
print(f"    Pattern: the missed defaulters have a mean predicted probability of {p_base[fn].mean():.3f} versus {neg_mean:.3f} for true "
      f"non-defaulters, so they are {'ranked higher' if p_base[fn].mean() > neg_mean else 'not ranked higher'} but still land far below the 0.5 cut-off, "
      "which is too strict for a 7% class.")

# 4. two imbalance levers, compared on the same frozen test set
weighted = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(X_tr, y_tr)
p_w = weighted.predict_proba(X_te)[:, 1]
sc_w = scores(y_te, (p_w >= 0.5).astype(int), p_w)

# threshold tuned on the PR curve of OUT-OF-FOLD training probabilities (never the test set)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                        X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_tr, oof)
f1c = 2 * prec * rec / (prec + rec + 1e-9)
best_thr = float(thr[np.argmax(f1c[:-1])])
sc_t = scores(y_te, (p_w >= best_thr).astype(int), p_w)
print(f"\n[4] lever 1 class_weight='balanced' at 0.5: recall {sc_w['recall']:.4f}, precision {sc_w['precision']:.4f}, F1 {sc_w['F1']:.4f}")
print(f"    lever 2 threshold tuning (F1-optimal on the PR curve of out-of-fold train probabilities = {best_thr:.3f}): "
      f"recall {sc_t['recall']:.4f}, precision {sc_t['precision']:.4f}, F1 {sc_t['F1']:.4f}")
print("    (SMOTE stretch: if used it must sit INSIDE an imblearn Pipeline so it is re-fit on each training fold only; "
      "resampling before the split would copy synthetic neighbours of test rows into training and inflate scores.)")

# 5. feature selection with RFECV (scoring F1), refit on the reduced set
rfecv = RFECV(LogisticRegression(max_iter=2000, class_weight="balanced"), step=1, cv=cv, scoring="f1")
Xs_tr = StandardScaler().fit(X_tr)
rfecv.fit(Xs_tr.transform(X_tr), y_tr)
kept = [c for c, k in zip(cols, rfecv.support_) if k]
dropped = [c for c, k in zip(cols, rfecv.support_) if not k]
print(f"\n[5] RFECV kept {rfecv.n_features_} features: {kept}\n    dropped: {dropped}")
sel = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
sel.fit(X_tr[:, rfecv.support_], y_tr)
oof_s = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                          X_tr[:, rfecv.support_], y_tr, cv=cv, method="predict_proba")[:, 1]
ps, rs, ts = precision_recall_curve(y_tr, oof_s)
thr_s = float(ts[np.argmax((2 * ps * rs / (ps + rs + 1e-9))[:-1])])
p_s = sel.predict_proba(X_te[:, rfecv.support_])[:, 1]
sc_s = scores(y_te, (p_s >= thr_s).astype(int), p_s)

# 6. scorecard
card = pd.DataFrame([sc_base, sc_w, sc_t, sc_s],
                    index=["baseline", "imbalance-handled (class_weight)", "threshold-tuned", "feature-selected"])
print("\n[6] before/after scorecard on the frozen test set\n" + card.round(4).to_string())

# 7. diagnosis (numbers pulled from the scorecard)
gain = card["F1"].iloc[-1] - card["F1"].iloc[0]
best_stage = card["F1"].idxmax()
print("\nDiagnosis:")
print(
    f"The baseline looked healthy at {sc_base['accuracy']:.3f} accuracy yet found only {sc_base['recall']:.3f} of the defaulters "
    f"(F1 {sc_base['F1']:.3f}), which is the accuracy paradox on a {y.mean():.0%} positive class. "
    f"class_weight='balanced' moved recall to {sc_w['recall']:.3f} at the cost of precision ({sc_w['precision']:.3f}), and tuning the threshold "
    f"on the training PR curve (to {best_thr:.2f}) gave F1 {sc_t['F1']:.3f}. "
    f"RFECV kept {rfecv.n_features_} of {len(cols)} features and the reduced model scored F1 {sc_s['F1']:.3f}, so selection "
    f"{'helped slightly' if sc_s['F1'] > sc_t['F1'] else 'did not add F1'} by removing noise columns. "
    f"Overall F1 changed by {gain:+.3f} from baseline to final and the best stage was '{best_stage}'. "
    f"PR-AUC went from {card['PR-AUC'].iloc[0]:.3f} to {card['PR-AUC'].iloc[-1]:.3f} across stages ("
    f"{'it fell' if card['PR-AUC'].iloc[-1] < card['PR-AUC'].iloc[0] else 'it rose'}), which shows the class weights and threshold "
    f"re-balanced the precision/recall trade-off rather than improving how well the model RANKS defaulters; the ranking is the bottleneck. "
    f"Next step: try a non-linear model such as gradient boosting with tuned depth, because the linear boundary caps PR-AUC near {card['PR-AUC'].max():.3f}."
)
