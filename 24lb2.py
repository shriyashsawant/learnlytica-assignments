"""Day 24 / Session 1 - Lab Assignment 2: TF-IDF from Scratch (verified vs scikit-learn) + a Text Classifier."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def compute_idf(counts):
    counts = np.asarray(counts, dtype=float)
    n = counts.shape[0]
    df = (counts > 0).sum(axis=0)                            # number of documents containing each term
    return np.log((1 + n) / (1 + df)) + 1                    # scikit-learn's smoothed idf


def l2_normalize(mat):
    mat = np.asarray(mat, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)                  # zero rows stay zero, no nan
    return mat / safe


def tfidf_transform(counts, idf):
    return l2_normalize(np.asarray(counts, dtype=float) * np.asarray(idf, dtype=float))


def tfidf_fit_transform(counts):
    idf = compute_idf(counts)
    return tfidf_transform(counts, idf)


def classify(train_counts, train_labels, test_counts):
    idf = compute_idf(train_counts)                          # fitted on the TRAIN counts only
    Xtr = tfidf_transform(train_counts, idf)
    Xte = tfidf_transform(test_counts, idf)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, train_labels)
    return model.predict(Xte)


if __name__ == "__main__":
    counts = np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]])
    print(np.round(compute_idf(counts), 4))                          # [1.6931 1.     1.6931]
    T = tfidf_fit_transform(counts)
    print(np.round(T[0], 4))                                         # [0.861  0.5085 0.    ]
    print(np.round(np.sqrt((T ** 2).sum(axis=1)), 4))                # [1. 1. 1.]
    print(l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]])))          # [[0.6 0.8] [0. 0.]]

    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    corpus = ["stock market prices rise today", "the bank raised interest rates",
              "investors buy shares in the market", "the team won the football match",
              "he scored a great goal", "players train hard for the match"]
    C = CountVectorizer().fit_transform(corpus).toarray()
    ref = TfidfVectorizer().fit_transform(corpus).toarray()
    print(np.allclose(tfidf_fit_transform(C), ref, atol=1e-8))       # True

    cv = CountVectorizer().fit(corpus[:2] + corpus[3:5])
    Ctr = cv.transform(corpus[:2] + corpus[3:5]).toarray()
    Cte = cv.transform(["market prices rise", "football match goal"]).toarray()
    print(classify(Ctr, [0, 0, 1, 1], Cte))                          # [0 1]
    tv = TfidfVectorizer().fit(corpus[:2] + corpus[3:5])
    print(np.allclose(tfidf_transform(Cte, compute_idf(Ctr)), tv.transform(["market prices rise", "football match goal"]).toarray()))   # True
