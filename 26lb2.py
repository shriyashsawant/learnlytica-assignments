"""Day 26 / Session 1 - Lab Assignment 2: Lexicon Sentiment Scoring & Few-Shot Prompt Formatting."""
import re

POS = {"good", "great", "love", "excellent", "happy"}
NEG = {"bad", "terrible", "hate", "awful", "sad"}
NEGATORS = {"not", "no", "never"}
texts = ["i love it", "so happy", "just awful", "very sad", "the movie exists"]
labels = ["positive", "positive", "negative", "negative", "neutral"]


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def sentiment_score(text, pos_words, neg_words):
    tokens = tokenize(text)
    total = 0
    for i, tok in enumerate(tokens):
        value = 1 if tok in pos_words else -1 if tok in neg_words else 0
        if value and i > 0 and tokens[i - 1] in NEGATORS:        # an immediately preceding negator flips the value
            value = -value
        total += value
    return int(total)


def classify_sentiment(text, pos_words, neg_words):
    score = sentiment_score(text, pos_words, neg_words)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def evaluate(texts, labels, pos_words, neg_words):
    if len(texts) == 0:
        return 0.0
    right = sum(classify_sentiment(t, pos_words, neg_words) == y for t, y in zip(texts, labels))
    return float(right / len(texts))


def build_few_shot_prompt(examples, query, instruction="Classify the sentiment of the review as positive or negative."):
    blocks = [f"Review: {text}\nSentiment: {label}" for text, label in examples]
    blocks.append(f"Review: {query}\nSentiment:")                  # dangling label: the model completes it
    return instruction + "\n\n" + "\n\n".join(blocks)


if __name__ == "__main__":
    print(sentiment_score("good good bad", POS, NEG))               # 1
    print(classify_sentiment("not good", POS, NEG))                 # negative
    print(classify_sentiment("not bad", POS, NEG), sentiment_score("Not BAD!", POS, NEG))   # positive 1
    print(classify_sentiment("the movie exists", POS, NEG))         # neutral
    print(classify_sentiment("good bad", POS, NEG))                 # neutral (tie)
    print(round(evaluate(["good", "bad", "okay"], ["positive", "negative", "positive"], POS, NEG), 4))   # 0.6667
    print(evaluate(texts, labels, POS, NEG), evaluate([], [], POS, NEG))      # 1.0 0.0
    pr = build_few_shot_prompt([("loved it", "positive")], "it was fine")
    print(pr.count("Review:"), pr.rstrip().endswith("Sentiment:"))  # 2 True
    print(build_few_shot_prompt([("loved it", "positive"), ("hated it", "negative")], "it was fine"))
    print(repr(build_few_shot_prompt([], "x")))                     # zero-shot works
