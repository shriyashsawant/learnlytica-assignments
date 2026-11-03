"""Day 13 / Session 1 - Lab Assignment 2: Train/Test Discipline & Cross-Validation."""
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def stratified_split(X, y, test_size=0.2, seed=0):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])


def kfold_score(pipe, X, y, k=5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    return round(float(cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()), 3)


def honest_test_score(pipe, X, y):
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    pipe.fit(X_train, y_train)
    return round(float(pipe.score(X_test, y_test)), 3)


if __name__ == "__main__":
    d = load_wine()
    X, y = d.data, d.target
    print(kfold_score(build_pipeline(), X, y, 5))
    print(honest_test_score(build_pipeline(), X, y))
