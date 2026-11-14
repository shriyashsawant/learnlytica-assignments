"""Day 24 / Session 2 - Lab Assignment 1: Cosine Similarity, Nearest Neighbours & Word Analogies from Scratch."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

vocab = ["king", "queen", "man", "woman", "prince", "princess", "dog", "cat", "apple"]
embeddings = np.array([
    [1.0, 0.9, 0.0],      # king
    [1.0, -0.9, 0.0],     # queen
    [0.2, 1.0, 0.0],      # man
    [0.2, -1.0, 0.0],     # woman
    [0.8, 0.8, 0.1],      # prince
    [0.7, -0.7, 0.2],     # princess
    [0.0, 0.1, 1.0],      # dog
    [0.0, -0.1, 1.0],     # cat
    [-1.0, 0.0, 0.2],     # apple
], dtype=float)


def cos_sim(a, b):
    a = np.asarray(a, dtype=float).reshape(1, -1)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    return float(cosine_similarity(a, b)[0, 0])


def nearest_neighbours(word, embeddings, vocab, k):
    embeddings = np.asarray(embeddings, dtype=float)
    i = list(vocab).index(word)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(vocab)), metric="cosine").fit(embeddings)
    _, idx = nn.kneighbors(embeddings[i].reshape(1, -1))
    return [vocab[j] for j in idx[0] if j != i][:k]             # most similar first, never the word itself


def analogy(a, b, c, embeddings, vocab, k=1):
    embeddings = np.asarray(embeddings, dtype=float)
    vocab = list(vocab)
    target = embeddings[vocab.index(b)] - embeddings[vocab.index(a)] + embeddings[vocab.index(c)]
    sims = cosine_similarity(target.reshape(1, -1), embeddings)[0]
    out = []
    for j in np.argsort(-sims):                                  # highest cosine first
        if vocab[j] not in (a, b, c):                            # exclude the three input words
            out.append(vocab[j])
        if len(out) == k:
            break
    return out


if __name__ == "__main__":
    print(round(cos_sim([1, 0, 0], [2, 0, 0]), 3))                   # 1.0
    print(round(cos_sim([1, 0, 0], [0, 1, 0]), 3))                   # 0.0
    print(round(cos_sim([1, 0, 0], [-3, 0, 0]), 3))                  # -1.0
    print(nearest_neighbours("king", embeddings, vocab, 2))          # ['prince', 'man']
    print(nearest_neighbours("dog", embeddings, vocab, 3))           # ['cat', ...]
    print(analogy("man", "king", "woman", embeddings, vocab))        # ['queen']
    print(analogy("king", "queen", "prince", embeddings, vocab))     # ['princess']
