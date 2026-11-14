"""Day 24 / Session 3 - Lab Assignment 1: Embedding Mean-Pool Classifier vs Bag-of-Words."""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(3)
dim = 12
c0 = rng.randn(dim); c1 = -c0                                    # two opposite "meaning" directions
emb = {}
for i in range(10): emb[f"a{i}"] = c0 + 0.15 * rng.randn(dim)    # class-0 training words
for i in range(10): emb[f"b{i}"] = c1 + 0.15 * rng.randn(dim)    # class-1 training words
docs = ["a1 a4 a2", "b3 b0", "a7 a1", "b9 b2 b5"]                # space-separated tokens
y = np.array([0, 1, 0, 1])


def mean_pool(text, emb, dim):
    vectors = [np.asarray(emb[t], dtype=float) for t in text.split() if t in emb]    # skip tokens that are not in emb
    if not vectors:
        return np.zeros(dim)                                                         # nothing known (or empty text)
    return np.mean(vectors, axis=0)


def embed_matrix(texts, emb, dim):
    return np.array([mean_pool(t, emb, dim) for t in texts]).reshape(len(texts), dim)


def bow_features(train_texts, test_texts):
    vectorizer = CountVectorizer().fit(train_texts)                                  # ONE vectorizer, fitted on train only
    return vectorizer.transform(train_texts).toarray(), vectorizer.transform(test_texts).toarray()


def classify_accuracy(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(accuracy_score(yte, clf.predict(Xte)))


if __name__ == "__main__":
    e = {"p": [2., 4.], "q": [4., 0.]}
    print(mean_pool("p q", e, 2))                                    # [3. 2.]
    print(mean_pool("p zzz", e, 2))                                  # [2. 4.]
    print(mean_pool("zzz", e, 2), mean_pool("", e, 2))               # [0. 0.] [0. 0.]
    Xtr, Xte = bow_features(["apple apple banana", "cherry"], ["zzz qqq"])
    print(Xtr)                                                       # [[2 1 0] [0 0 1]]
    print(Xte)                                                       # [[0 0 0]]
    a = np.array([[0, 0], [0, 0.1], [5, 5], [5, 4.9]]); b = np.array([0, 0, 1, 1])
    print(classify_accuracy(a, b, np.array([[0.1, 0.1], [4.9, 5.1]]), np.array([0, 1])))   # 1.0
    # embeddings generalise to unseen synonyms while bag-of-words cannot
    emb2 = dict(emb)
    for i in range(10, 14): emb2[f"a{i}"] = c0 + 0.15 * rng.randn(dim); emb2[f"b{i}"] = c1 + 0.15 * rng.randn(dim)
    train_t = [f"a{i} a{(i + 1) % 10}" for i in range(10)] + [f"b{i} b{(i + 1) % 10}" for i in range(10)]
    train_y = np.array([0] * 10 + [1] * 10)
    test_t = ["a10 a11", "b12 b13", "a12 a13", "b10 b11"]; test_y = np.array([0, 1, 0, 1])
    emb_acc = classify_accuracy(embed_matrix(train_t, emb2, dim), train_y, embed_matrix(test_t, emb2, dim), test_y)
    Btr, Bte = bow_features(train_t, test_t)
    print(emb_acc, classify_accuracy(Btr, train_y, Bte, test_y))      # 1.0 vs chance (0.5)
