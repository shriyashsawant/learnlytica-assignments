"""Day 24 / Session 3 - Lab Assignment 1: Embedding Mean-Pool Classifier vs Bag-of-Words."""
import numpy as np
from sklearn.linear_model import LogisticRegression


def mean_pool(tokens, emb, dim):
    vectors = [np.asarray(emb[t], dtype=float) for t in tokens if t in emb]    # skip unknown tokens
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


def embed_matrix(docs, emb, dim):
    return np.array([mean_pool(doc, emb, dim) for doc in docs]).reshape(len(docs), dim)


def build_vocab(docs):
    tokens = sorted({t for doc in docs for t in doc})
    return {t: i for i, t in enumerate(tokens)}               # indices follow sorted token order


def bow_matrix(docs, vocab):
    M = np.zeros((len(docs), len(vocab)))
    for i, doc in enumerate(docs):
        for t in doc:
            if t in vocab:                                   # out-of-vocabulary tokens are ignored
                M[i, vocab[t]] += 1
    return M


def classify_accuracy(Xtr, ytr, Xte, yte):
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)
    return float(model.score(Xte, yte))


if __name__ == "__main__":
    emb = {"p": [2., 4.], "q": [4., 0.], "r": [0., 0.]}
    print(mean_pool(["p", "q"], emb, 2), mean_pool(["p", "zzz"], emb, 2), mean_pool(["zzz"], emb, 2), mean_pool([], emb, 2))   # [3. 2.] [2. 4.] [0. 0.] [0. 0.]
    docs = [["apple", "apple", "banana"], ["cherry"]]
    v = build_vocab(docs)
    print(v)                                                  # {'apple': 0, 'banana': 1, 'cherry': 2}
    print(bow_matrix(docs, v))                                # [[2. 1. 0.] [0. 0. 1.]]
    print(embed_matrix([["p"], ["q", "p"]], emb, 2))
    Xtr = np.array([[0, 0], [0, 0.1], [5, 5], [5, 4.9]]); ytr = np.array([0, 0, 1, 1])
    Xte = np.array([[0.1, 0.1], [4.9, 5.1]]); yte = np.array([0, 1])
    print(classify_accuracy(Xtr, ytr, Xte, yte))              # 1.0

    # the headline comparison: embeddings generalise to unseen synonyms, bag-of-words cannot
    rng = np.random.RandomState(3)
    dim = 12
    c0 = rng.randn(dim); c1 = -c0
    e = {}
    for i in range(10):
        e[f"a{i}"] = c0 + 0.15 * rng.randn(dim)
    for i in range(10):
        e[f"b{i}"] = c1 + 0.15 * rng.randn(dim)
    train_docs = [["a1", "a4", "a2"], ["b3", "b0"], ["a7", "a1"], ["b9", "b2", "b5"]]
    ytrain = np.array([0, 1, 0, 1])
    test_docs = [["a0", "a3"], ["b4", "b6"], ["a5", "a8", "a9"], ["b7", "b8"]]       # words never seen in training
    ytest = np.array([0, 1, 0, 1])
    emb_acc = classify_accuracy(embed_matrix(train_docs, e, dim), ytrain, embed_matrix(test_docs, e, dim), ytest)
    vb = build_vocab(train_docs)
    bow_acc = classify_accuracy(bow_matrix(train_docs, vb), ytrain, bow_matrix(test_docs, vb), ytest)
    print(emb_acc, bow_acc)                                   # 1.0 vs ~0.5 (chance)
    print(bow_matrix([["zzz"]], vb).sum(), np.linalg.norm(mean_pool(["a1"], e, dim)) > 0)         # 0.0 True
