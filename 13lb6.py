"""Day 13 / Session 3 - Lab Assignment 2: Model Persistence & Serving - Save, Reload & Serve with joblib."""
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_save(path, seed=0):
    d = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(d.data, d.target, test_size=0.2, stratify=d.target, random_state=seed)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])
    model.fit(X_train, y_train)
    joblib.dump(model, path)  # the WHOLE pipeline, so the scaler travels with the model
    return model


def load_model(path):
    return joblib.load(path)


def predict_proba_positive(model, X):
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def roundtrip_consistent(model, path, X):
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    return bool(np.array_equal(reloaded.predict(X), model.predict(X)))


if __name__ == "__main__":
    import os
    import tempfile

    d = load_breast_cancer()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.2, stratify=d.target, random_state=0)
    folder = tempfile.mkdtemp()
    m = train_and_save(os.path.join(folder, "model.joblib"), 0)
    srv = load_model(os.path.join(folder, "model.joblib"))
    print((srv.predict(Xte) == m.predict(Xte)).all())
    print(predict_proba_positive(srv, Xte[:3]).round(3))
    print(roundtrip_consistent(m, os.path.join(folder, "rt.joblib"), Xte))
    print(type(srv).__name__, list(srv.named_steps), round(float(srv.score(Xte, yte)), 3))
    print(np.allclose(srv.named_steps["clf"].coef_, m.named_steps["clf"].coef_))
