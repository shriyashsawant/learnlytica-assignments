"""Day 13 / Session 2 - Lab Assignment 1: End-to-End Pipeline for Mixed-Type Data (ColumnTransformer)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_df(n=600, seed=0):
    rng = np.random.RandomState(seed)
    z = rng.randn(n)
    age = 40 + 10 * z + rng.randn(n) * 3
    balance = 5000 + 2000 * z + rng.randn(n) * 500
    tenure = 24 + 6 * z + rng.randn(n) * 4
    plan = np.where(z > 0.3, "premium", np.where(z < -0.3, "basic", "standard"))
    region = rng.choice(["north", "south", "east", "west"], size=n)
    y = (z + rng.randn(n) * 0.4 > 0).astype(int)
    df = pd.DataFrame({"age": age, "balance": balance, "tenure": tenure, "plan": plan, "region": region})
    return df, y


NUMERIC = ["age", "balance", "tenure"]
CATEGORICAL = ["plan", "region"]


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), list(numeric_cols)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols)),
    ])


def build_pipeline(numeric_cols, categorical_cols):
    return Pipeline([
        ("pre", build_preprocessor(numeric_cols, categorical_cols)),
        ("clf", LogisticRegression(max_iter=5000)),
    ])


def n_features_out(preprocessor, df):
    out = preprocessor.fit_transform(df)
    return int(out.shape[1])


def fit_evaluate(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    return round(float(pipe.score(X_test, y_test)), 3)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df, y = make_df()
    prep = build_preprocessor(NUMERIC, CATEGORICAL)
    print(n_features_out(prep, df))
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.2, stratify=y, random_state=0)
    print(fit_evaluate(build_pipeline(NUMERIC, CATEGORICAL), Xtr, ytr, Xte, yte))
    # unseen category at inference must not crash
    unseen = Xte.copy()
    unseen.iloc[0, unseen.columns.get_loc("plan")] = "enterprise"
    print(build_pipeline(NUMERIC, CATEGORICAL).fit(Xtr, ytr).predict(unseen)[:3])
