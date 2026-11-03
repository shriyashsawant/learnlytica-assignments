"""Day 15 / Session 2 - Lab Assignment 2: Tune a Model and Prove the Lift over Baseline."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

GRID = {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]}


def make_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("svc", SVC())])


def baseline_score(X, y, cv):
    return round(float(cross_val_score(make_pipeline(), X, y, cv=cv, scoring="accuracy").mean()), 4)


def build_search(cv):
    return GridSearchCV(make_pipeline(), GRID, scoring="accuracy", cv=cv)


def tune(X, y, cv):
    search = build_search(cv).fit(X, y)
    return {"best_params": search.best_params_, "best_score": round(float(search.best_score_), 4)}


def improvement(X, y, cv):
    return round(tune(X, y, cv)["best_score"] - baseline_score(X, y, cv), 4)


if __name__ == "__main__":
    d = load_breast_cancer()
    X, y = d.data, d.target
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print(baseline_score(X, y, cv))
    r = tune(X, y, cv)
    print(r["best_params"])
    print(r["best_score"])
    print(improvement(X, y, cv))
