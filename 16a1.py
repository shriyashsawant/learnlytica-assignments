"""Day 16 / Session 1 - Assignment: Build and Evaluate a Feature Pipeline with PCA.

Dataset choice: a synthetic TaskFlow-style churn table generated in code (no network, fully reproducible, and we can
plant a skewed numeric column, a genuinely ordered column and a high-cardinality categorical on purpose).
"""
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

warnings.filterwarnings("ignore", category=FutureWarning)  # TargetEncoder.random_state deprecation on sklearn>=1.9
SEED = 42
OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(SEED)

# 1. synthetic churn data: >=3 numeric, >=2 categorical, binary target
n = 4000
signup = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_charges": rng.normal(70, 25, n).clip(15, 150),
    "usage_hours": rng.lognormal(mean=2.5, sigma=1.0, size=n),          # heavily right-skewed
    "support_calls": rng.poisson(1.5, n),
    "plan": rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.3, 0.2]),
    "region": rng.choice(["north", "south", "east", "west"], n),
    "contract": rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.2]),  # genuinely ordered
    "zip_code": ["z%03d" % z for z in rng.integers(0, 600, n)],           # high cardinality
    "signup_date": signup,
})
zip_effect = {z: rng.normal(0, 0.6) for z in df["zip_code"].unique()}
logit = (-1.0 - 0.03 * df["tenure_months"] + 0.012 * df["monthly_charges"] + 0.5 * np.log1p(df["usage_hours"]) * 0.2
         + 0.25 * df["support_calls"] + df["contract"].map({"month-to-month": 0.9, "one-year": 0.0, "two-year": -0.8})
         + df["zip_code"].map(zip_effect) * 0.5)
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
print(f"[1] synthetic churn data {df.shape}, churn rate {y.mean():.3f}")

# 2. short EDA
print("\n[2] dtypes:\n", df.dtypes.to_string())
cats = ["plan", "region", "contract", "zip_code"]
nums = ["tenure_months", "monthly_charges", "usage_hours", "support_calls"]
print("categorical cardinality:", {c: int(df[c].nunique()) for c in cats})
skew = df[nums].skew().round(2)
print("numeric skew:\n", skew.to_string())
print(f"-> skewed numeric: usage_hours (skew {skew['usage_hours']}); high-cardinality categorical: zip_code ({df['zip_code'].nunique()} levels)")

# 4. derived features (computed row-wise, so no leakage)
df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure_months"] + 1)      # hypothesis: new, expensive customers churn
df["signup_month"] = df["signup_date"].dt.month                                    # hypothesis: seasonal promotions drive churn
df["calls_x_usage"] = df["support_calls"] * np.log1p(df["usage_hours"])            # hypothesis: heavy users with problems leave
df["tenure_bin"] = pd.cut(df["tenure_months"], [0, 12, 36, 100], labels=["new", "mid", "loyal"]).astype(str)
derived_num = ["charges_per_tenure", "calls_x_usage", "signup_month"]
print("\n[4] derived features: charges_per_tenure (ratio), signup_month (date-derived), calls_x_usage (interaction), tenure_bin (binned)")
df = df.drop(columns=["signup_date"])

# 3. one ColumnTransformer with several transformer types, all fit inside the pipeline / CV folds
log_cols = ["usage_hours"]
scale_cols = ["tenure_months", "monthly_charges", "support_calls"] + derived_num
onehot_cols = ["plan", "region", "tenure_bin"]
preprocess = ColumnTransformer([
    ("log_scale", Pipeline([("log", FunctionTransformer(np.log1p)), ("sc", StandardScaler())]), log_cols),
    ("scale", StandardScaler(), scale_cols),
    ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_cols),
    ("ordinal", OrdinalEncoder(categories=[["month-to-month", "one-year", "two-year"]]), ["contract"]),
    ("target_enc", TargetEncoder(target_type="binary", random_state=SEED), ["zip_code"]),   # out-of-fold inside fit_transform
])

# 5. leakage demonstration: naive full-data mean encoding vs TargetEncoder
naive = df["zip_code"].map(pd.Series(y, index=df.index).groupby(df["zip_code"]).mean())
proper = TargetEncoder(target_type="binary", random_state=SEED).fit_transform(df[["zip_code"]], y).ravel()
corr_naive, corr_proper = np.corrcoef(naive, y)[0, 1], np.corrcoef(proper, y)[0, 1]
print(f"\n[5] correlation with target: naive full-data mean encoding {corr_naive:.3f}  vs  TargetEncoder (cross-fitted) {corr_proper:.3f}")
print("    Explanation: the naive encoder computes each zip's mean of y using the very rows it then encodes, so every row's own "
      "label leaks into its feature - with ~7 rows per zip the correlation is inflated and will not hold on new data. "
      "TargetEncoder fits its category means on other folds (cross fitting) plus shrinkage, so a row never sees its own "
      "label and the correlation is an honest, lower figure.")

# 6. Pipeline + classifier, cross-validated (preprocessing is inside the pipeline -> refit per fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
full_pipe = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=5000))])
cv_full = cross_val_score(full_pipe, df, y, cv=cv, scoring="roc_auc")
print(f"\n[6] CV ROC-AUC, full engineered features: {cv_full.mean():.4f} +/- {cv_full.std():.4f} "
      "(scaler, encoders and model are all refit inside every fold - nothing is fit before the split)")

# 7. PCA on the standardised engineered matrix (exploration done on the training split only)
X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.25, stratify=y, random_state=SEED)
explore = Pipeline([("prep", preprocess), ("sc", StandardScaler()), ("pca", PCA(random_state=SEED))]).fit(X_tr, y_tr)
cum = np.cumsum(explore.named_steps["pca"].explained_variance_ratio_)
k90, k95 = int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.95) + 1)
plt.figure(figsize=(7, 4))
plt.plot(np.arange(1, len(cum) + 1), cum, "o-")
plt.axhline(0.90, color="r", ls="--", label=f"90% -> {k90} components")
plt.axhline(0.95, color="g", ls="--", label=f"95% -> {k95} components")
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.title("PCA on the standardised engineered matrix")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "16a1_pca_cumulative_variance.png"), dpi=120)
plt.close()
print(f"\n[7] of {len(cum)} engineered columns: {k90} components keep 90% of the variance and {k95} keep 95%")

# 8. full features vs PCA-reduced features, both cross-validated with PCA inside the pipeline
pca_pipe = Pipeline([("prep", preprocess), ("sc", StandardScaler()), ("pca", PCA(n_components=k90, random_state=SEED)),
                     ("clf", LogisticRegression(max_iter=5000))])
cv_pca = cross_val_score(pca_pipe, df, y, cv=cv, scoring="roc_auc")
delta = cv_pca.mean() - cv_full.mean()
verdict = "helped" if delta > 0.005 else "hurt" if delta < -0.005 else "was neutral"
print(f"[8] CV ROC-AUC full engineered: {cv_full.mean():.4f} | PCA (k={k90}): {cv_pca.mean():.4f} | difference {delta:+.4f}")
print(f"Conclusion: PCA {verdict} here. The engineered matrix has {len(cum)} columns but the classifier is linear and already "
      f"regularised, so compressing to {k90} components "
      + ("discarded little useful signal and gave a smaller, faster model" if verdict != "hurt" else
         "threw away low-variance directions that still carried churn signal (variance is not the same as predictive value)")
      + ".")
