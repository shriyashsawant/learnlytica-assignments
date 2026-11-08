"""Day 26 / Session 1 - Lab Assignment 1: Extractive Summarization by Word-Frequency Scoring + ROUGE-1."""
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def split_sentences(text):
    parts = re.split(r"[.!?]+", str(text))
    return [p.strip() for p in parts if p.strip()]


def word_frequencies(text):
    counts = Counter(tokenize(text))
    if not counts:
        return {}
    top = max(counts.values())
    return {word: count / top for word, count in counts.items()}     # the most frequent word maps to 1.0


def score_sentences(sentences, freqs):
    return [float(sum(freqs.get(token, 0.0) for token in tokenize(s))) for s in sentences]


def extractive_summary(text, k):
    sentences = split_sentences(text)
    if k <= 0 or not sentences:
        return []
    scores = score_sentences(sentences, word_frequencies(text))
    ranked = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))     # ties -> the earlier sentence
    chosen = sorted(ranked[:k])                                               # back to the original order
    return [sentences[i] for i in chosen]


def rouge1(candidate, reference):
    cand = Counter(tokenize(candidate))
    ref = Counter(tokenize(reference))
    n_cand, n_ref = sum(cand.values()), sum(ref.values())
    if n_cand == 0 or n_ref == 0:
        return 0.0
    overlap = sum(min(count, ref[token]) for token, count in cand.items())
    precision, recall = overlap / n_cand, overlap / n_ref
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    doc = ("Solar power is a clean energy source. Solar energy comes from the sun. "
           "Cats are fluffy animals. The weather today is mild and pleasant.")
    print(tokenize("Hi, 2024!"), split_sentences(doc)[:2])
    print(word_frequencies("the cat the cat the dog"))                    # {'the': 1.0, 'cat': 0.666..., 'dog': 0.333...}
    print(word_frequencies(""))                                           # {}
    print(extractive_summary(doc, 2))                                     # the two solar sentences, original order
    print(extractive_summary(doc, 99) == split_sentences(doc), extractive_summary(doc, 0))
    print(round(rouge1("the cat sat", "the cat ran"), 4), round(rouge1("the cat sat on the mat", "the cat sat on the mat"), 4))   # 0.6667 1.0
    print(rouge1("dog", "cat"), rouge1("", "cat"), round(rouge1("the the the", "the cat"), 4))                                    # 0.0 0.0 0.4
    print(score_sentences(["solar energy", "cats"], {"solar": 1.0, "energy": 0.5}))                                               # [1.5, 0.0]
