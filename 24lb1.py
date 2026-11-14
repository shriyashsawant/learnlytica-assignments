"""Day 24 / Session 1 - Lab Assignment 1: Tokenization, Normalization & Bag-of-Words from Scratch."""
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

docs = ["the cat sat on the mat", "the dog sat", "cat and dog play"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())          # lowercase, keep runs of letters/digits, drop punctuation


def _vectorizer():
    # our own tokenizer already lowercases, so switch the library's lowercasing and token pattern off
    return CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)


def vocabulary(docs):
    vectorizer = _vectorizer().fit(docs)
    return list(vectorizer.get_feature_names_out())        # CountVectorizer sorts the vocabulary


def bag_of_words(docs):
    vectorizer = _vectorizer()
    X = vectorizer.fit_transform(docs).toarray().astype(int)       # true counts, shape (n_docs, len(vocab))
    return X, list(vectorizer.get_feature_names_out())


if __name__ == "__main__":
    print(tokenize("Hello, World! It's 2024."))            # ['hello', 'world', 'it', 's', '2024']
    print(tokenize("!!! ???"))                             # []
    print(vocabulary(docs))                                # ['and', 'cat', 'dog', 'mat', 'on', 'play', 'sat', 'the']
    X, vocab = bag_of_words(docs)
    print(vocab)
    print(X)
    print(X[0, vocab.index("the")], X[1, vocab.index("the")], X.shape)       # 2 1 (3, 8)
    print(X.sum(axis=1).tolist() == [len(tokenize(d)) for d in docs])        # True: counts match the hand-written tokenizer
