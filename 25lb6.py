"""Day 25 / Session 3 - Lab Assignment 2: BERT Inputs & Pooling - WordPiece Tokenizer + [CLS]/Mean Pooling."""
import numpy as np


def wordpiece_tokenize(word, vocab, unk_token="[UNK]", max_chars=100):
    if len(word) > max_chars:
        return [unk_token]
    pieces = []
    start = 0
    while start < len(word):
        end = len(word)
        current = None
        while start < end:                                   # shrink the end until the substring is in the vocab
            piece = word[start:end]
            if start > 0:
                piece = "##" + piece                         # continuation pieces carry the ## prefix
            if piece in vocab:
                current = piece
                break
            end -= 1
        if current is None:
            return [unk_token]                               # nothing matched: the whole word is unknown
        pieces.append(current)
        start = end
    return pieces


def tokenize_text(text, vocab, unk_token="[UNK]"):
    tokens = []
    for word in str(text).split():
        tokens.extend(wordpiece_tokenize(word, vocab, unk_token))
    return tokens


def cls_pool(hidden_states):
    return np.asarray(hidden_states, dtype=float)[0]         # the [CLS] vector sits at position 0


def masked_mean_pool(hidden_states, attention_mask):
    H = np.asarray(hidden_states, dtype=float)
    keep = np.asarray(attention_mask) == 1
    if not keep.any():
        return np.zeros(H.shape[1])
    return H[keep].mean(axis=0)                              # padding rows are ignored


if __name__ == "__main__":
    vocab = {"play", "##ing", "the", "cat", "##s", "un", "##aff", "##able"}
    print(wordpiece_tokenize("playing", vocab), wordpiece_tokenize("zzz", vocab))          # ['play', '##ing'] ['[UNK]']
    print(tokenize_text("the cats", vocab), tokenize_text("the cats playing", vocab))      # ['the', 'cat', '##s'] [... 'play', '##ing']
    print(tokenize_text("unaffable qqq", vocab))                                           # ['un', '##aff', '##able', '[UNK]']
    print(wordpiece_tokenize("playing", vocab | {"p", "pl"}))                              # longest match still 'play'
    H = np.arange(12, dtype=float).reshape(4, 3)
    print(cls_pool(H), masked_mean_pool(H, np.array([1, 1, 0, 0])))                        # [0. 1. 2.] [1.5 2.5 3.5]
    print(np.allclose(masked_mean_pool(H, np.ones(4)), H.mean(axis=0)), masked_mean_pool(H, np.zeros(4)))   # True [0. 0. 0.]
