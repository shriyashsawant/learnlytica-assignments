"""Day 26 / Session 1 - Lab Assignment 2: Lexicon Sentiment Scoring & Few-Shot Prompt Formatting."""
import re

NEGATORS = {"not", "no", "never"}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def sentiment_score(text, pos_words, neg_words):
    tokens = tokenize(text)
    total = 0
    for i, token in enumerate(tokens):
        value = 1 if token in pos_words else (-1 if token in neg_words else 0)
        if value and i > 0 and tokens[i - 1] in NEGATORS:
            value = -value                                   # "not good" -> -1, "not bad" -> +1
        total += value
    return int(total)


def classify_sentiment(text, pos_words, neg_words):
    score = sentiment_score(text, pos_words, neg_words)
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def evaluate(texts, labels, pos_words, neg_words):
    if len(texts) == 0:
        return 0.0
    right = sum(classify_sentiment(t, pos_words, neg_words) == y for t, y in zip(texts, labels))
    return right / len(texts)


def build_few_shot_prompt(examples, query, instruction="Classify the sentiment of the review as positive or negative."):
    blocks = [instruction]
    for text, label in examples:
        blocks.append(f"Review: {text}\nSentiment: {label}")        # each label stays with its own review
    blocks.append(f"Review: {query}\nSentiment:")                    # dangling: the model completes it
    return "\n\n".join(blocks)


if __name__ == "__main__":
    POS = {"good", "great", "love", "excellent", "happy"}
    NEG = {"bad", "terrible", "hate", "awful", "sad"}
    print(sentiment_score("good good bad", POS, NEG))                # 1
    print(classify_sentiment("not good", POS, NEG), classify_sentiment("not bad", POS, NEG), classify_sentiment("the movie exists", POS, NEG))   # negative positive neutral
    print(classify_sentiment("GOOD, bad!", POS, NEG), classify_sentiment("good bad", POS, NEG))                                                 # neutral neutral (tie)
    print(round(evaluate(["good", "bad", "okay"], ["positive", "negative", "positive"], POS, NEG), 4), evaluate([], [], POS, NEG))               # 0.6667 0.0
    texts = ["i love it", "so happy", "just awful", "very sad", "the movie exists"]
    labels = ["positive", "positive", "negative", "negative", "neutral"]
    print(evaluate(texts, labels, POS, NEG))                                                                                                     # 1.0
    pr = build_few_shot_prompt([("loved it", "positive"), ("hated it", "negative")], "it was fine")
    print(pr)
    print(pr.count("Review:"), pr.rstrip().endswith("Sentiment:"))                                                                               # 3 True
    print(repr(build_few_shot_prompt([], "x")))                                                                                                  # zero-shot works
