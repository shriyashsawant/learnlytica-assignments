"""Day 13 / Session 2 - Lab Assignment 2: Hyperparameter Tuning & Leakage-Free CV (GridSearchCV)."""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])


def param_grid():
    return {"clf__C": [0.01, 0.1, 1.0, 10.0]}


def grid_search(X, y, k=5):
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    return GridSearchCV(build_pipeline(), param_grid(), cv=cv, scoring="accuracy").fit(X, y)


def best_c(gs):
    return float(gs.best_params_["clf__C"])


def test_accuracy(gs, X_test, y_test):
    return round(float(gs.best_estimator_.score(X_test, y_test)), 3)


if __name__ == "__main__":
    X, y = make_classification(n_samples=500, n_features=20, n_informative=8, n_redundant=4, class_sep=1.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    gs = grid_search(X_train, y_train, 5)
    print(best_c(gs))
    print(round(gs.best_score_, 3))
    print(test_accuracy(gs, X_test, y_test))
