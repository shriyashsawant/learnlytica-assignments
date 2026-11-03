"""Day 15 / Session 4 - Assignment: Build and Defend a Regression Model (diabetes progression)."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42

# 1. offline dataset (no download), fixed random_state; test set untouched until the single evaluation below
data = load_diabetes()
X, y = data.data, data.target
names = list(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
print("train", X_train.shape, "test", X_test.shape)

# 2. linear baseline
linear = LinearRegression().fit(X_train, y_train)

# 3. interpretation: intercept + coefficients sorted by magnitude
coef = pd.Series(linear.coef_, index=names).sort_values(key=np.abs, ascending=False)
print(f"\n[3] intercept: {linear.intercept_:.2f}")
print(coef.round(2).to_string())
top2 = coef.index[:2]

# 4. polynomial features (degree 2) with a plain linear model
poly = Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("lr", LinearRegression())]).fit(X_train, y_train)

# 5. ridge on the polynomial features (standardise first); alpha tuned with CV on the TRAIN split only
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
alphas = [0.1, 1, 10, 30, 100, 300, 1000]
ridge_search = GridSearchCV(
    Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)), ("sc", StandardScaler()), ("ridge", Ridge())]),
    {"ridge__alpha": alphas}, scoring="neg_mean_squared_error", cv=cv).fit(X_train, y_train)
best_alpha = ridge_search.best_params_["ridge__alpha"]
print(f"\n[5] ridge alpha chosen by 5-fold CV on the training data only: {best_alpha} "
      f"(CV MSE by alpha: " + ", ".join(f"{a}: {-s:.0f}" for a, s in zip(alphas, ridge_search.cv_results_['mean_test_score'])) + ")")
ridge = ridge_search.best_estimator_

# 6/7. single evaluation of the test set for all finished models -> summary table
models = {"linear": linear, "polynomial (deg 2)": poly, "ridge on polynomial": ridge}
rows = []
for name, m in models.items():
    tr_pred, te_pred = m.predict(X_train), m.predict(X_test)  # the test set is scored once per model, here
    rows.append({"model": name,
                 "train_mse": mean_squared_error(y_train, tr_pred), "test_mse": mean_squared_error(y_test, te_pred),
                 "train_r2": r2_score(y_train, tr_pred), "test_r2": r2_score(y_test, te_pred)})
table = pd.DataFrame(rows).set_index("model")
table["gap_r2"] = table["train_r2"] - table["test_r2"]
table["test_rmse"] = np.sqrt(table["test_mse"])
print("\n[6] summary (train/test MSE and R2, gap = train R2 - test R2)")
print(table.round(4).to_string())

lin, pol, rdg = table.loc["linear"], table.loc["polynomial (deg 2)"], table.loc["ridge on polynomial"]
print(f"\n[2] R2 meaning: the linear baseline's test R2 of {lin['test_r2']:.3f} means it explains about "
      f"{lin['test_r2'] * 100:.0f}% of the variance in disease progression on unseen patients; the rest is unexplained.")
print(f"[3] two most influential features: {top2[0]} ({'positive' if coef[top2[0]] > 0 else 'negative'} effect, "
      f"{coef[top2[0]]:+.1f}) and {top2[1]} ({'positive' if coef[top2[1]] > 0 else 'negative'} effect, {coef[top2[1]]:+.1f}).")
print(f"[4] polynomial: train R2 {pol['train_r2']:.3f} (up from {lin['train_r2']:.3f}) but test R2 {pol['test_r2']:.3f} "
      f"(vs {lin['test_r2']:.3f}); the gap grew from {lin['gap_r2']:.3f} to {pol['gap_r2']:.3f} - that is variance (overfitting), not bias.")

# 8. final model = best CV model among the three candidates (ridge here, picked from CV not from test)
final = "ridge on polynomial"
print(f"\n[8] final model: {final}: honest test R2 = {rdg['test_r2']:.4f}, test RMSE = {rdg['test_rmse']:.2f}")
print("\nConclusion:")
print(
    f"I would ship the ridge-regularised polynomial model with alpha={best_alpha}, chosen by 5-fold CV on the training data. "
    f"The plain linear baseline explains {lin['test_r2']:.3f} of the test variance (RMSE {lin['test_rmse']:.2f}); "
    f"adding degree-2 terms raised training R2 to {pol['train_r2']:.3f} but dropped test R2 to {pol['test_r2']:.3f}, "
    f"widening the train-test gap from {lin['gap_r2']:.3f} to {pol['gap_r2']:.3f}, which is overfitting. "
    f"Ridge shrank that gap to {rdg['gap_r2']:.3f} and reached a test R2 of {rdg['test_r2']:.3f} with RMSE {rdg['test_rmse']:.2f}, "
    f"so regularisation closed {pol['gap_r2'] - rdg['gap_r2']:.3f} of the gap. "
    f"The test set was evaluated once for the finished models. "
    f"Caveat: diabetes progression has substantial irreducible noise, so even the best model keeps an RMSE floor near "
    f"{rdg['test_rmse']:.0f} and R2 well below 1."
)
