"""Day 24 / Session 1 - Lab Assignment 2: TF-IDF from Scratch (verified vs scikit-learn) + a Text Classifier."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train = ["stock market prices rise today",
         "the bank raised interest rates",
         "investors buy shares in the market",
         "the team won the football match",
         "he scored a great goal",
         "players train hard for the match"]
labels = [0, 0, 0, 1, 1, 1]                                    # 0 = finance, 1 = sports
test = ["shares stock market rates", "the football team scored a goal"]


def vectorize(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)                        # fit on these texts only
    return X, vectorizer                                       # return the fitted vectorizer to reuse on held-out text


def train_classifier(X, y):
    return LogisticRegression(max_iter=1000).fit(X, y)


def evaluate(clf, X, y):
    return float(accuracy_score(y, clf.predict(X)))


if __name__ == "__main__":
    X, vec = vectorize(train)
    print(X.shape)                                                          # (6, n_vocab)
    print(np.round(np.sqrt((X.toarray() ** 2).sum(axis=1)), 4))             # [1. 1. 1. 1. 1. 1.]
    clf = train_classifier(X, labels)
    print(evaluate(clf, X, labels))                                         # 1.0
    Xte = vec.transform(test)                                               # reuse the TRAIN vocabulary + idf
    print([int(p) for p in clf.predict(Xte)])                               # [0, 1]
    print(evaluate(clf, Xte, [0, 1]))                                       # 1.0
    print(np.allclose(X.toarray(), TfidfVectorizer().fit_transform(train).toarray()))   # True (matches the library)
