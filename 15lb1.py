import time
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone


def load_data():
    d = load_wine()
    return d.data, d.target


def model_zoo():
    return {
        "logreg": LogisticRegression(max_iter=5000),
        "tree": DecisionTreeClassifier(random_state=0),
        "forest": RandomForestClassifier(n_estimators=100, random_state=0),
        "nb": GaussianNB(),
        "knn": KNeighborsClassifier(),
    }


def benchmark(models, X, y, seed=0):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    rows = []
    for name, est in models.items():
        est = clone(est)
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        acc = float((est.predict(X_te) == y_te).mean())
        rows.append({"name": name, "accuracy": round(acc, 4), "fit_time": float(fit_time)})
    return rows


def leaderboard(results):
    return sorted(results, key=lambda r: (-r["accuracy"], r["fit_time"]))


if __name__ == "__main__":
    X, y = load_data()
    rows = benchmark(model_zoo(), X, y, 0)
    for r in leaderboard(rows):
        print(r["name"], r["accuracy"], round(r["fit_time"], 4))
