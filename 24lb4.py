"""Day 24 / Session 2 - Lab Assignment 2: A Classical NLP Pipeline - Tokenize, Mean-Pool Embeddings & PCA-2D."""
import re

import numpy as np


def tokenize(sentence):
    return re.findall(r"[a-z0-9]+", str(sentence).lower())


def embed_document(sentence, embeddings, vocab):
    embeddings = np.asarray(embeddings, dtype=float)
    index = {w: i for i, w in enumerate(vocab)}
    vectors = [embeddings[index[t]] for t in tokenize(sentence) if t in index]    # skip out-of-vocabulary tokens
    if not vectors:
        return np.zeros(embeddings.shape[1])
    return np.mean(vectors, axis=0)


def document_similarity(s1, s2, embeddings, vocab):
    a = embed_document(s1, embeddings, vocab)
    b = embed_document(s2, embeddings, vocab)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pca_2d(X):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)                                  # centre the data
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)        # rows of Vt = principal directions, by variance
    return Xc @ Vt[:2].T


if __name__ == "__main__":
    vocab = ["good", "great", "bad", "movie", "film", "popcorn"]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.0]], dtype=float)
    print(tokenize("Good, GREAT movie!!"), tokenize("...!?"), tokenize(""))      # ['good', 'great', 'movie'] [] []
    print(embed_document("good movie", embeddings, vocab))                        # [0.5 0.5 0. ]
    print(embed_document("good movie zzz qqq", embeddings, vocab))                # same, OOV ignored
    print(embed_document("zzz qqq", embeddings, vocab))                           # [0. 0. 0.]
    print(round(document_similarity("good movie", "great film", embeddings, vocab), 2),
          round(document_similarity("good movie", "bad movie", embeddings, vocab), 2))   # ~0.95 ~0.0
    Z = pca_2d(np.random.RandomState(0).randn(50, 5))
    print(Z.shape, np.allclose(Z.mean(0), 0), Z[:, 0].var() >= Z[:, 1].var())     # (50, 2) True True
    rng = np.random.RandomState(1)
    base = rng.randn(30, 2) * [3.0, 1.0]
    X = base @ rng.randn(2, 6)                                                    # rank-2 data in 6-D
    Z2 = pca_2d(X)
    d = lambda A: np.linalg.norm(A[:, None] - A[None], axis=-1)
    print(np.allclose(d(X), d(Z2)))                                               # True (distances preserved)
