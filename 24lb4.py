"""Day 24 / Session 2 - Lab Assignment 2: A Classical NLP Pipeline - Tokenize, Mean-Pool Embeddings & PCA-2D."""
import re

import numpy as np
from sklearn.decomposition import PCA

vocab = ["good", "great", "bad", "movie", "film", "popcorn"]
embeddings = np.array([
    [1.0, 0.0, 0.0],      # good
    [0.9, 0.1, 0.0],      # great
    [-1.0, 0.0, 0.0],     # bad
    [0.0, 1.0, 0.0],      # movie
    [0.0, 0.9, 0.1],      # film
    [0.0, 0.0, 1.0],      # popcorn
], dtype=float)


def tokenize(sentence):
    return re.findall(r"[a-z0-9]+", sentence.lower())


def embed_document(sentence, embeddings, vocab):
    embeddings = np.asarray(embeddings, dtype=float)
    index = {w: i for i, w in enumerate(vocab)}
    vectors = [embeddings[index[t]] for t in tokenize(sentence) if t in index]      # skip out-of-vocabulary tokens
    if not vectors:
        return np.zeros(embeddings.shape[1])                                         # nothing known: a zero vector
    return np.mean(vectors, axis=0)                                                  # mean-pool the word vectors


def pca_2d(X):
    return PCA(n_components=2).fit_transform(np.asarray(X, dtype=float))             # (n_docs, 2); sklearn centres the data


def explained_variance_2d(X):
    pca = PCA(n_components=2).fit(np.asarray(X, dtype=float))
    return float(np.sum(pca.explained_variance_ratio_))                              # share of variance the 2-D plot keeps


if __name__ == "__main__":
    print(tokenize("Good, GREAT movie!!"))                                          # ['good', 'great', 'movie']
    print(embed_document("good movie", embeddings, vocab))                          # [0.5 0.5 0. ]
    print(embed_document("zzz qqq", embeddings, vocab))                             # [0. 0. 0.]
    docs = ["good movie", "great film", "bad movie", "popcorn", "good great"]
    X = np.vstack([embed_document(s, embeddings, vocab) for s in docs])
    Z = pca_2d(X)
    print(Z.shape, np.allclose(Z.mean(axis=0), 0))                                  # (5, 2) True
    print(round(explained_variance_2d(X), 3), tokenize("?!"))                       # a value in [0, 1], []
    flat = np.column_stack([np.arange(6.0), 2 * np.arange(6.0), np.zeros(6)])       # intrinsically 1-D data
    print(round(explained_variance_2d(flat), 3))                                    # ~1.0
