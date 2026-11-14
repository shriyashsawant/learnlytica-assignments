"""Day 26 / Session 1 - Lab Assignment 1: Extractive Summarization by Word-Frequency Scoring + ROUGE-1."""
import re
from collections import Counter

doc = ("Solar power is a clean energy source. "
       "Solar energy comes from the sun. "
       "Cats are fluffy animals. "
       "The weather today is mild and pleasant.")
reference_summary = "Solar power is clean energy from the sun."


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())                 # lowercase, maximal runs of letters/digits


def split_sentences(text):
    parts = re.split(r"[.!?]+", text)                             # split on runs of . ! ?
    return [p.strip() for p in parts if p.strip()]                # stripped, non-empty, no terminal punctuation


def word_frequencies(text):
    counts = Counter(tokenize(text))
    if not counts:
        return {}
    peak = max(counts.values())
    return {word: c / peak for word, c in counts.items()}         # the most frequent word maps to 1.0


def score_sentences(sentences, freqs):
    return [float(sum(freqs.get(tok, 0.0) for tok in tokenize(s))) for s in sentences]    # absent tokens count 0


def extractive_summary(text, k):
    sentences = split_sentences(text)
    if k <= 0:
        return []
    if k >= len(sentences):
        return sentences
    scores = score_sentences(sentences, word_frequencies(text))
    best = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))[:k]     # ties go to the earlier sentence
    return [sentences[i] for i in sorted(best)]                                  # keep the original order


def rouge1(candidate, reference):
    cand, ref = Counter(tokenize(candidate)), Counter(tokenize(reference))
    n_cand, n_ref = sum(cand.values()), sum(ref.values())
    if n_cand == 0 or n_ref == 0:
        return 0.0
    overlap = sum(min(cand[t], ref[t]) for t in cand)             # clipped unigram overlap
    p, r = overlap / n_cand, overlap / n_ref
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


if __name__ == "__main__":
    print(tokenize("Hi, 2024!"))                                   # ['hi', '2024']
    print(split_sentences(doc)[:2])
    print(word_frequencies("the cat the cat the dog"))             # {'the': 1.0, 'cat': 0.666..., 'dog': 0.333...}
    print(extractive_summary(doc, 2))                              # ['Solar power is a clean energy source', 'Solar energy comes from the sun']
    print(extractive_summary(doc, 0), len(extractive_summary(doc, 10)))     # [] 4
    print(round(rouge1("the cat sat", "the cat ran"), 4))          # 0.6667
    print(round(rouge1("the cat sat on the mat", "the cat sat on the mat"), 4))     # 1.0
    print(rouge1("a b", "c d"), rouge1("", "x"), round(rouge1("the the the", "the cat"), 4))   # 0.0 0.0 0.4
    print(round(rouge1(extractive_summary(doc, 2)[0], reference_summary), 4))
