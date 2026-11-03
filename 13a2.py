"""Day 13 / Session 2 - Assignment: A Reusable, Leak-Free Pipeline on a New Dataset (Breast Cancer)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
rng = np.random.RandomState(SEED)

# 1. load as a DataFrame
df = load_breast_cancer(as_frame=True).frame
print(f"[1] shape {df.shape}; target balance {df['target'].value_counts(normalize=True).round(3).to_dict()}")
df.info()

# 2. engineer a categorical column and inject ~5% missing values into two numeric columns
df["radius_size"] = pd.qcut(df["mean radius"], q=3, labels=["small", "medium", "large"]).astype(str)
for col in ("mean texture", "mean area"):
    df.loc[rng.choice(len(df), int(0.05 * len(df)), replace=False), col] = np.nan
print(f"\n[2] added categorical 'radius_size'; missing values now: { {c: int(df[c].isna().sum()) for c in ('mean texture', 'mean area')} }")

X = df.drop(columns="target")
y = df["target"]
numeric_cols = [c for c in X.columns if c != "radius_size"]

# 3. stratified 70/15/15 split BEFORE any transformer is fitted
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
print("\n[3] positive-class share  train {:.3f} | val {:.3f} | test {:.3f}  (sizes {}/{}/{})".format(
    y_train.mean(), y_val.mean(), y_test.mean(), len(y_train), len(y_val), len(y_test)))

# 4. ColumnTransformer (numeric: impute + scale, categorical: impute + one-hot) inside ONE Pipeline
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["radius_size"]),
])
pipe = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=5000))])

# 5. cross-validate on the training split
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"\n[5] 5-fold CV ROC-AUC on train: mean {auc.mean():.4f} +/- {auc.std():.4f}")

# 6. grid search: preprocessor parameter (imputer strategy) + two model parameters
grid = {"pre__num__imputer__strategy": ["mean", "median"],
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"]}
search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
print(f"[6] best_params_ {search.best_params_}  best_score_ {search.best_score_:.4f}")
val_auc = roc_auc_score(y_val, search.best_estimator_.predict_proba(X_val)[:, 1])
print(f"    validation ROC-AUC of the best estimator: {val_auc:.4f}")

# 7. the test set is scored exactly once, here
best = search.best_estimator_
test_proba = best.predict_proba(X_test)[:, 1]
test_pred = best.predict(X_test)
test_acc, test_auc = accuracy_score(y_test, test_pred), roc_auc_score(y_test, test_proba)
print(f"\n[7] TEST (scored once): accuracy {test_acc:.4f}, ROC-AUC {test_auc:.4f}")
print("confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, test_pred))

# 8. leakage demonstration: choose features with ALL labels before CV vs inside the Pipeline.
#    Pure-noise features are used on purpose so the true accuracy is about chance and the inflation is easy to see.
noise = np.random.RandomState(0).randn(len(y), 300)
leaky_X = SelectKBest(f_classif, k=10).fit_transform(noise, y)                       # leak: labels of held-out rows used
leaky_score = cross_val_score(LogisticRegression(max_iter=2000), leaky_X, y, cv=cv).mean()
clean_score = cross_val_score(Pipeline([("select", SelectKBest(f_classif, k=10)),
                                        ("clf", LogisticRegression(max_iter=2000))]), noise, y, cv=cv).mean()
print(f"\n[8] leaky CV accuracy {leaky_score:.4f} vs in-Pipeline CV accuracy {clean_score:.4f} (gap {leaky_score - clean_score:+.4f})")
print("    Explanation: in the leaky version the feature selector looked at every row's label, including the rows later used "
      "as validation folds, so it picked the noise columns that happened to correlate with those labels and the CV score "
      "looks far better than it really is. Inside the Pipeline the selector is re-fit on each training fold only, so the "
      "held-out fold stays unseen and the score falls back to roughly chance, which is the honest answer for noise.")

# 9. conclusion with specific numbers
print("\nConclusion:")
print(
    f"The pipeline imputes, scales and one-hot encodes inside one Pipeline, so nothing is fitted on validation or test rows. "
    f"Five-fold CV on the training split gave ROC-AUC {auc.mean():.3f} +/- {auc.std():.3f}, and the grid search chose "
    f"{search.best_params_} with best_score_ {search.best_score_:.3f}. "
    f"The test set was scored once: accuracy {test_acc:.3f} and ROC-AUC {test_auc:.3f}, close to the CV estimate, so the evaluation looks honest. "
    f"The leakage demo showed a gap of {leaky_score - clean_score:+.3f} ({leaky_score:.3f} vs {clean_score:.3f}) on pure noise, which is why every fitted step stays in the pipeline. "
    f"Before deploying I would check calibration and repeat the split with other seeds, because the test set has only {len(y_test)} rows."
)
