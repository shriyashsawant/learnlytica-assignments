"""Day 26 / Session 2 - Lab Assignment 2: Context-Window Budget Manager & Multilingual Token Cost."""
import math

conversation = [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Tell me a very long detailed story now"},
    {"role": "assistant", "content": "Once upon a time"},
    {"role": "user", "content": "continue please"},
]
sentences = {
    "english": "I do not speak much",
    "spanish": "no hablo mucho",
    "japanese": "watashi wa amari hanasanai",
}


def count_tokens_ws(text):
    return len(text.split())                                    # whitespace words; runs of spaces collapse, "" gives 0


def count_tokens_char(text, chars_per_token=4):
    return int(math.ceil(len(text) / chars_per_token))          # real tokenizers average ~4 characters per token


def fit_to_budget(messages, max_tokens, token_count_fn):
    # NOTE: the question's printed example (budget 10 -> ['system', 'assistant', 'user']) does not follow from its own
    # rule: those three messages already total 11 words. This follows the written rule (total <= max_tokens).
    kept = [dict(m) for m in messages]

    def total():
        return sum(token_count_fn(m["content"]) for m in kept)

    while total() > max_tokens:
        oldest = next((i for i, m in enumerate(kept) if m["role"] != "system"), None)
        if oldest is None:
            break                                               # only system messages left: always keep them
        del kept[oldest]                                        # drop the oldest non-system message
    return kept


def multilingual_cost(sentences, token_count_fn):
    return {language: token_count_fn(text) for language, text in sentences.items()}


if __name__ == "__main__":
    print(count_tokens_ws("the quick brown fox"))               # 4
    print(count_tokens_char("the quick brown fox"))             # 5
    kept = fit_to_budget(conversation, 10, count_tokens_ws)
    print([m["role"] for m in kept], sum(count_tokens_ws(m["content"]) for m in kept))   # ['system', 'user'] 7
    print(multilingual_cost(sentences, count_tokens_ws))        # {'english': 5, 'spanish': 3, 'japanese': 4}
    print([m["role"] for m in fit_to_budget(conversation, 1, count_tokens_ws)])           # ['system'] (system always kept)
    print(fit_to_budget(conversation, 100, count_tokens_ws) == conversation)             # True (already fits)
    print(count_tokens_ws("  a   b  "), count_tokens_ws(""), count_tokens_char(""), count_tokens_char("abcde", 2))   # 2 0 0 3
    no_sys = [{"role": "user", "content": "one two three"}, {"role": "assistant", "content": "four five"},
              {"role": "user", "content": "six"}]
    print([m["content"] for m in fit_to_budget(no_sys, 3, count_tokens_ws)])            # ['four five', 'six']
