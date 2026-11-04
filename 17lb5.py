"""Day 17 / Session 3 - Lab Assignment 1: Deploy the MLP Inference Path - relu + softmax forward pass in numpy."""
import numpy as np


def relu(Z):
    return np.maximum(0.0, np.asarray(Z, dtype=float))


def softmax(Z):
    Z = np.asarray(Z, dtype=float)
    e = np.exp(Z - Z.max(axis=1, keepdims=True))     # subtract the row max for numerical stability
    return e / e.sum(axis=1, keepdims=True)


def mlp_forward(X, params):
    A = np.asarray(X, dtype=float)
    last = len(params) - 1
    for i, (W, b) in enumerate(params):
        Z = A @ W + b
        A = softmax(Z) if i == last else relu(Z)     # relu on hidden layers, softmax on the output layer
    return A


def predict(X, params):
    return np.argmax(mlp_forward(X, params), axis=1).astype(int)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.neural_network import MLPClassifier

    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    clf = MLPClassifier(hidden_layer_sizes=(30,), activation="relu", max_iter=300, random_state=1).fit(X, y)
    params = list(zip(clf.coefs_, clf.intercepts_))
    probs = mlp_forward(X, params)
    print(probs.shape, np.allclose(probs.sum(axis=1), 1.0))
    preds = predict(X, params)
    print(preds[:8], round(accuracy(y, preds), 3), round(clf.score(X, y), 3))
    print(np.array_equal(preds, clf.predict(X)), np.allclose(probs, clf.predict_proba(X)))
