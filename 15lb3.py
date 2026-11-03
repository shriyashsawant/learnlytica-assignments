"""Day 15 / Session 2 - Lab Assignment 1: A Fair Multi-Model Comparison Harness."""
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold, cross_val_score


def make_cv(k=5, seed=0):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)


def score_model(model, X, y, cv):
    return round(float(cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()), 4)


def leaderboard(models, X, y, cv):
    rows = [(name, score_model(model, X, y, cv)) for name, model in models.items()]
    return sorted(rows, key=lambda r: r[1], reverse=True)


def best_model_name(models, X, y, cv):
    return leaderboard(models, X, y, cv)[0][0]


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    d = load_wine()
    X, y = d.data, d.target
    models = {
        "logreg": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
        "tree": DecisionTreeClassifier(random_state=0),
        "nb": GaussianNB(),
    }
    cv = make_cv(5, 0)
    print(leaderboard(models, X, y, cv))
    print(best_model_name(models, X, y, cv))
