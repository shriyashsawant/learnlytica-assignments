"""Day 16 / Session 1 - Lab Assignment 1: Feature Engineering by Hand + a Leakage-Free Pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def standardize_fit(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # zero-variance columns must never divide by zero
    return mean, std


def standardize_transform(X, mean, std):
    return (np.asarray(X, dtype=float) - mean) / std  # uses the passed-in (train) statistics


def one_hot(labels, categories):
    index = {c: i for i, c in enumerate(categories)}
    out = np.zeros((len(labels), len(categories)))
    for r, lab in enumerate(labels):
        j = index.get(lab)
        if j is not None:  # an unknown label leaves an all-zero row
            out[r, j] = 1.0
    return out


def build_pipeline(numeric_cols, categorical_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), list(numeric_cols)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols)),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({"age": rng.normal(40, 10, n), "income": rng.normal(50000, 15000, n),
                       "city": rng.choice(["NY", "LA", "SF"], n)})
    city_bonus = df["city"].map({"SF": 1.0, "LA": 0.0, "NY": -1.0}).values
    score = (df["income"].values - 50000.0) / 15000.0 + city_bonus
    y = (score > 0).astype(int)

    X = np.array([[1., 100., 5.], [3., 300., 5.], [5., 500., 5.], [7., 700., 5.]])
    m, s = standardize_fit(X)
    print(s)
    Z = standardize_transform(X, m, s)
    print(Z[:, :2].mean(0), Z[:, :2].std(0))
    print(one_hot(["a", "b", "c", "a"], ["a", "b", "c"]))
    print(one_hot(["z"], ["a", "b"]))
    pipe = build_pipeline(["age", "income"], ["city"]).fit(df, y)
    print(pipe.named_steps["pre"].transform(df).shape[1], pipe.score(df, y))
