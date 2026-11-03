"""Day 15 / Session 1 - Lab Assignment 2: Scaling Ablation - which models need feature scaling?"""
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_data():
    d = load_wine()
    return d.data, d.target


def scaled_vs_unscaled(estimator, X, y, seed=0):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    raw = clone(estimator).fit(X_tr, y_tr)
    pipe = make_pipeline(StandardScaler(), clone(estimator)).fit(X_tr, y_tr)  # scaler fit on train only
    return {
        "unscaled": round(float(raw.score(X_te, y_te)), 4),
        "scaled": round(float(pipe.score(X_te, y_te)), 4),
    }


def needs_scaling(estimator, X, y, seed=0, threshold=0.02):
    r = scaled_vs_unscaled(estimator, X, y, seed)
    return bool(r["scaled"] - r["unscaled"] > threshold)


def ablation(X, y, seed=0):
    zoo = {
        "knn": KNeighborsClassifier(5),
        "svm": SVC(),
        "tree": DecisionTreeClassifier(random_state=0),
        "forest": RandomForestClassifier(n_estimators=100, random_state=0),
    }
    out = {}
    for name, est in zoo.items():
        r = scaled_vs_unscaled(est, X, y, seed)
        gain = round(r["scaled"] - r["unscaled"], 4)
        out[name] = {"unscaled": r["unscaled"], "scaled": r["scaled"], "gain": gain, "needs_scaling": gain > 0.02}
    return out


if __name__ == "__main__":
    X, y = make_data()
    tbl = ablation(X, y, 0)
    for name, r in tbl.items():
        print(name, r["unscaled"], r["scaled"], r["gain"], r["needs_scaling"])
