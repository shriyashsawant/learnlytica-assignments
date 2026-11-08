"""Day 24 / Session 1 - Lab Assignment 1: Tokenization, Normalization & Bag-of-Words from Scratch."""
import re

import numpy as np


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())      # lowercase, keep runs of letters/digits only


def remove_stopwords(tokens, stopwords):
    stop = set(stopwords)
    return [t for t in tokens if t not in stop]              # keeps order and repeats


def preprocess(text, stopwords):
    return remove_stopwords(tokenize(text), stopwords)


def build_vocabulary(docs, stopwords):
    vocab = set()
    for doc in docs:
        vocab.update(preprocess(doc, stopwords))
    return sorted(vocab)


def bag_of_words(docs, vocab, stopwords):
    index = {word: j for j, word in enumerate(vocab)}
    matrix = np.zeros((len(docs), len(vocab)), dtype=int)
    for i, doc in enumerate(docs):
        for token in preprocess(doc, stopwords):
            if token in index:                               # out-of-vocabulary tokens contribute nothing
                matrix[i, index[token]] += 1
    return matrix


if __name__ == "__main__":
    docs = ["the cat sat on the mat", "the dog sat", "cat and dog play"]
    stopwords = {"the", "on", "and"}
    print(tokenize("Hello, World! It's 2024."))                       # ['hello', 'world', 'it', 's', '2024']
    print(tokenize(""), tokenize("...!?"))                            # [] []
    print(preprocess("The DOG, the dog!", {"the"}))                   # ['dog', 'dog']
    print(remove_stopwords(["a", "b", "a", "c"], {"b"}))              # ['a', 'a', 'c']
    print(build_vocabulary(docs, stopwords))                          # ['cat', 'dog', 'mat', 'play', 'sat']
    B = bag_of_words(docs, ['cat', 'dog', 'mat', 'play', 'sat'], stopwords)
    print(B)                                                          # [[1 0 1 0 1] [0 1 0 0 1] [1 1 0 1 0]]
    print(bag_of_words(["dog dog dog cat"], ["cat", "dog"], set()))   # [[1 3]] true counts
    print(np.array_equal(bag_of_words(["sat cat"], ["cat", "sat"], set()), bag_of_words(["cat sat"], ["cat", "sat"], set())))   # True (order ignored)
