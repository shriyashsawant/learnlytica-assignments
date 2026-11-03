"""Day 15 / Session 1 - Assignment: Justify a Model Choice with Evidence (lending scenario)."""
import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42

# 1. data, fixed random_state + stratified split
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# 2. five model families (scaler wraps the distance/margin models)
models = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    "SVM": make_pipeline(StandardScaler(), SVC(random_state=SEED)),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
}
traits = {
    "LogisticRegression": "linear, coefficients explain every decision, very cheap",
    "RandomForest": "non-linear, feature importances, moderate cost",
    "GradientBoosting": "most accurate trees, sequential and slower, harder to explain",
    "SVM": "margin model, needs scaling, weak interpretability",
    "kNN": "distance model, needs scaling, slow predict, no model to explain",
}

# 3. 5-fold CV (scaling happens per fold inside the pipeline) + fit time on the full training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0
    rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std(),
                 "fit_seconds": fit_seconds, "traits": traits[name]})

# 4. single sorted DataFrame
results = pd.DataFrame(rows).sort_values("cv_mean", ascending=False).reset_index(drop=True)
print(results.to_string(float_format=lambda v: f"{v:.4f}"))

# 5. scaling ablation: SVM and kNN with and without a scaler
ablation = []
for name, est in (("SVM", SVC(random_state=SEED)), ("kNN", KNeighborsClassifier())):
    scaled = cross_val_score(models[name], X_train, y_train, cv=cv).mean()
    raw = cross_val_score(est, X_train, y_train, cv=cv).mean()
    ablation.append({"model": name, "scaled": scaled, "unscaled": raw, "drop": scaled - raw})
ablation = pd.DataFrame(ablation)
print("\nScaling ablation:\n", ablation.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# 6. interpretability evidence
names = np.array(data.feature_names)
rf = models["RandomForest"]
rf_top = [str(n) for n in names[np.argsort(rf.feature_importances_)[::-1][:3]]]
lr_coef = models["LogisticRegression"][-1].coef_[0]
lr_top = [str(n) for n in names[np.argsort(np.abs(lr_coef))[::-1][:3]]]
print("\nRandomForest top-3 importances:", rf_top)
print("LogisticRegression top-3 |coef| (scaled features):", lr_top)

# 7. recommendation (numbers pulled from the results above)
by = results.set_index("model")
pick = "LogisticRegression"
best = results.iloc[0]
text = (
    f"Recommendation: deploy {pick}. "
    f"Its 5-fold CV accuracy is {by.loc[pick, 'cv_mean']:.4f} (+/- {by.loc[pick, 'cv_std']:.4f}), against {best['cv_mean']:.4f} "
    f"for the top model ({best['model']}), so the accuracy gap is {best['cv_mean'] - by.loc[pick, 'cv_mean']:+.4f}, "
    f"which is {'within' if best['cv_mean'] - by.loc[pick, 'cv_mean'] <= by.loc[pick, 'cv_std'] else 'beyond'} one fold std of the chosen model. "
    f"For interpretability, every decision can be explained by its coefficients; the top drivers are {', '.join(lr_top)}, "
    f"and the forest independently points at {', '.join(rf_top)}. "
    f"For nightly CPU retraining it fits in {by.loc[pick, 'fit_seconds']:.4f}s versus {by.loc['GradientBoosting', 'fit_seconds']:.4f}s for gradient boosting. "
    f"Scaling matters: removing it costs SVM {ablation.loc[0, 'drop']:.4f} and kNN {ablation.loc[1, 'drop']:.4f} accuracy, "
    f"so the pipeline keeps the StandardScaler inside every fold. "
    f"Even if a tree ensemble edged ahead on accuracy it would lose on the compliance constraint, so the linear model is the pick."
)
print("\n" + text)

# held-out check for the chosen model (touched once, at the end)
print(f"\nHeld-out test accuracy of {pick}: {models[pick].score(X_test, y_test):.4f}")
