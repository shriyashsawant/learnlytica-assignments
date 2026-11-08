"""Day 26 / Session 2 - Lab Assignment 2: Context-Window Budget Manager & Multilingual Token Cost."""
import math


def count_tokens_ws(text):
    return len(str(text).split())                            # runs of spaces collapse; "" -> 0


def count_tokens_char(text, chars_per_token=4):
    return int(math.ceil(len(str(text)) / chars_per_token))


def fit_to_budget(messages, max_tokens, token_count_fn):
    # NOTE: the question's printed example (budget 10 -> ['system','assistant','user']) keeps 11 tokens, which breaks the
    # question's own rule "total <= max_tokens"; this follows the written rule (system tokens count, system is never dropped).
    kept = list(messages)

    def total():
        return sum(token_count_fn(m["content"]) for m in kept)

    while total() > max_tokens:
        oldest = next((i for i, m in enumerate(kept) if m["role"] != "system"), None)
        if oldest is None:
            break                                            # only system messages left: keep them even if over budget
        del kept[oldest]                                     # drop the oldest non-system message
    return kept


def multilingual_cost(sentences, token_count_fn):
    return {language: token_count_fn(text) for language, text in sentences.items()}


if __name__ == "__main__":
    conversation = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Tell me a very long detailed story now"},
        {"role": "assistant", "content": "Once upon a time"},
        {"role": "user", "content": "continue please"},
    ]
    print(count_tokens_ws("the quick brown fox"), count_tokens_char("the quick brown fox"))     # 4 5
    print(count_tokens_ws("  a   b  "), count_tokens_ws(""))                                    # 2 0
    kept = fit_to_budget(conversation, 10, count_tokens_ws)
    print([m["role"] for m in kept], sum(count_tokens_ws(m["content"]) for m in kept))          # ['system', 'user'] 7
    print([m["role"] for m in fit_to_budget(conversation, 12, count_tokens_ws)])                # ['system', 'assistant', 'user']
    print([m["role"] for m in fit_to_budget(conversation, 2, count_tokens_ws)])                 # ['system'] (kept even though over budget)
    print(fit_to_budget(conversation, 100, count_tokens_ws) == conversation)                    # True (already fits)
    print(multilingual_cost({"english": "I do not speak much", "spanish": "no hablo mucho",
                             "japanese": "watashi wa amari hanasanai"}, count_tokens_ws))       # {'english': 5, 'spanish': 3, 'japanese': 4}
    no_sys = [{"role": "user", "content": "a b c"}, {"role": "assistant", "content": "d e"}, {"role": "user", "content": "f"}]
    print([m["content"] for m in fit_to_budget(no_sys, 3, count_tokens_ws)])                    # ['d e', 'f']
