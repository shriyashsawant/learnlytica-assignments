"""Day 24 / Session 2 - Lab Assignment 1: Cosine Similarity, Nearest Neighbours & Word Analogies from Scratch."""
import numpy as np


def l2_normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.copy()                                      # a zero vector is returned unchanged
    return v / norm


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _top_k(target, embeddings, vocab, k, exclude):
    embeddings = np.asarray(embeddings, dtype=float)
    scored = []
    for i, word in enumerate(vocab):
        if word in exclude:
            continue
        scored.append((-cosine_similarity(target, embeddings[i]), i, word))
    scored.sort()                                            # most similar first, ties broken by vocab order
    return [word for _, _, word in scored[:k]]


def nearest_neighbours(word, embeddings, vocab, k):
    target = np.asarray(embeddings, dtype=float)[list(vocab).index(word)]
    return _top_k(target, embeddings, vocab, k, exclude={word})


def analogy(a, b, c, embeddings, vocab, k=1):
    E = np.asarray(embeddings, dtype=float)
    idx = {w: i for i, w in enumerate(vocab)}
    target = E[idx[b]] - E[idx[a]] + E[idx[c]]               # "a is to b as c is to ?"
    return _top_k(target, E, vocab, k, exclude={a, b, c})


if __name__ == "__main__":
    vocab = ["king", "queen", "man", "woman", "prince", "princess", "dog", "cat", "apple"]
    embeddings = np.array([
        [1.0, 0.9, 0.0], [1.0, -0.9, 0.0], [0.2, 1.0, 0.0], [0.2, -1.0, 0.0], [0.8, 0.8, 0.1],
        [0.7, -0.7, 0.2], [0.0, 0.1, 1.0], [0.0, -0.1, 1.0], [-1.0, 0.0, 0.2]], dtype=float)
    print(round(cosine_similarity([1, 0, 0], [2, 0, 0]), 3), round(cosine_similarity([1, 0, 0], [0, 1, 0]), 3), round(cosine_similarity([1, 0], [-3, 0]), 3))   # 1.0 0.0 -1.0
    print(l2_normalize(np.array([3., 4.])), l2_normalize(np.zeros(3)))        # [0.6 0.8] [0. 0. 0.]
    print(nearest_neighbours("king", embeddings, vocab, 2))                   # ['prince', 'man']
    print(analogy("man", "king", "woman", embeddings, vocab))                 # ['queen']
    print(analogy("king", "queen", "prince", embeddings, vocab))              # ['princess']
    print(cosine_similarity(np.zeros(3), [1, 2, 3]), len(nearest_neighbours("dog", embeddings, vocab, 4)), "dog" in nearest_neighbours("dog", embeddings, vocab, 8))
