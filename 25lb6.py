"""Day 25 / Session 3 - Lab Assignment 2: BERT Inputs & Pooling - WordPiece Tokenizer + [CLS]/Mean Pooling."""
import numpy as np

vocab = {"play", "##ing", "the", "cat", "##s", "un", "##aff", "##able"}
embeddings = np.arange(12, dtype=float).reshape(4, 3)       # (seq_len=4, hidden=3)
mask = np.array([1, 1, 0, 1])                               # row 2 is padding


def wordpiece_tokenize(word, vocab, unk_token="[UNK]"):
    pieces = []
    start = 0
    while start < len(word):
        end = len(word)
        current = None
        while start < end:                                  # longest match first, shrink the end until it is in vocab
            piece = word[start:end]
            if start > 0:
                piece = "##" + piece                        # every piece after the first carries the ## prefix
            if piece in vocab:
                current = piece
                break
            end -= 1
        if current is None:
            return [unk_token]                              # nothing matched here: the whole word is unknown
        pieces.append(current)
        start = end                                         # continue from where the piece ended
    return pieces


def tokenize_text(text, vocab, unk_token="[UNK]"):
    out = []
    for word in text.split():
        out.extend(wordpiece_tokenize(word, vocab, unk_token))
    return out


def cls_pool(embeddings):
    return np.asarray(embeddings, dtype=float)[0]           # the [CLS] vector at position 0


def mean_pool(embeddings, mask):
    embeddings = np.asarray(embeddings, dtype=float)
    mask = np.asarray(mask)
    real = mask == 1
    if not real.any():
        return np.zeros(embeddings.shape[1])                # no real token at all
    return embeddings[real].mean(axis=0)                    # average over the non-padding rows only


if __name__ == "__main__":
    v = {"play", "##ing", "the", "cat", "##s"}
    print(wordpiece_tokenize("playing", v))                 # ['play', '##ing']
    print(wordpiece_tokenize("zzz", v))                     # ['[UNK]']
    print(tokenize_text("the cats", v))                     # ['the', 'cat', '##s']
    E = np.arange(12, dtype=float).reshape(4, 3)
    print(cls_pool(E))                                      # [0. 1. 2.]
    print(mean_pool(E, np.array([1, 1, 0, 1])))             # [3. 4. 5.]
    print(mean_pool(E, np.array([0, 0, 0, 0])))             # [0. 0. 0.]
    print(wordpiece_tokenize("unaffable", vocab), wordpiece_tokenize("playz", v))   # ['un', '##aff', '##able'] ['[UNK]']
