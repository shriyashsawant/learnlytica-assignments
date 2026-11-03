"""Day 13 / Session 1 - Lab Assignment 1: Build & Evaluate Your First Classifier."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def load_split(random_state=42):
    d = load_breast_cancer()
    return train_test_split(d.data, d.target, test_size=0.2, stratify=d.target, random_state=random_state)


def train_model(X_train, y_train):
    return LogisticRegression(max_iter=5000).fit(X_train, y_train)


def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred)), 3),
        "recall": round(float(recall_score(y_test, pred)), 3),
        "f1": round(float(f1_score(y_test, pred)), 3),
    }


def predict_one(model, x):
    return int(model.predict(np.asarray(x).reshape(1, -1))[0])


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = load_split(42)
    m = train_model(Xtr, ytr)
    print(evaluate(m, Xte, yte))
    print(predict_one(m, Xte[0]))
    print(confusion_counts(np.array([1, 1, 1, 0, 0]), np.array([1, 0, 1, 1, 0])))
