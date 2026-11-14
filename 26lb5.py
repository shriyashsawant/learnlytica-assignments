"""Day 26 / Session 3 - Lab Assignment 1: LLM API Integration - Token-Cost Estimator & Exponential-Backoff Retry."""
import time

import numpy as np

# Vendor rate cards - prices are per 1,000 tokens (the standard billing unit)
MODELS = [
    {"name": "big", "in_rate": 0.003, "out_rate": 0.015},
    {"name": "small", "in_rate": 0.001, "out_rate": 0.002},
    {"name": "huge", "in_rate": 0.010, "out_rate": 0.030},
]

TRANSIENT = {429, 500, 502, 503}


# A fake endpoint that returns HTTP-like statuses from a scripted list
def make_endpoint(statuses, body="OK"):
    state = {"n": 0}

    def call():
        i = state["n"]
        state["n"] += 1
        s = statuses[i] if i < len(statuses) else statuses[-1]
        return {"status": s, "body": body}
    call.state = state
    return call


def estimate_cost(input_tokens, output_tokens, in_rate_per_1k, out_rate_per_1k):
    return input_tokens / 1000 * in_rate_per_1k + output_tokens / 1000 * out_rate_per_1k


def estimate_batch_cost(requests, in_rate_per_1k, out_rate_per_1k):
    return sum(estimate_cost(r["input_tokens"], r["output_tokens"], in_rate_per_1k, out_rate_per_1k) for r in requests)


def cheapest_model(models, input_tokens, output_tokens):
    best_name, best_cost = None, None
    for m in models:
        cost = estimate_cost(input_tokens, output_tokens, m["in_rate"], m["out_rate"])
        if best_cost is None or cost < best_cost:               # strict <, so the first model wins ties
            best_name, best_cost = m["name"], cost
    return best_name


def backoff_delay(attempt, base=1.0, factor=2.0, cap=60.0):
    return min(cap, base * factor ** attempt)                   # attempt is 0-indexed


def call_with_retries(fn, max_attempts=5, base=1.0, factor=2.0, cap=60.0, sleep=None):
    sleep = sleep if sleep is not None else time.sleep
    for attempt in range(max_attempts):
        response = fn()
        status = response["status"]
        if status == 200:
            return response["body"]
        if status not in TRANSIENT:
            raise ValueError(f"non-retryable status {status}")  # permanent error: fail immediately, no retry
        if attempt < max_attempts - 1:
            sleep(backoff_delay(attempt, base, factor, cap))    # wait between attempts, not after the last one
    raise RuntimeError(f"gave up after {max_attempts} transient failures")


if __name__ == "__main__":
    print(round(estimate_cost(1000, 500, 0.003, 0.015), 4))                    # 0.0105
    print(round(estimate_cost(2500, 0, 0.002, 0.02), 4))                       # 0.005
    print(round(estimate_batch_cost([{"input_tokens": 1000, "output_tokens": 1000},
                                     {"input_tokens": 500, "output_tokens": 0}], 0.001, 0.002), 4))   # 0.0035
    print(cheapest_model(MODELS, 1000, 1000))                                  # small
    print([backoff_delay(a) for a in range(4)], backoff_delay(10))             # [1.0, 2.0, 4.0, 8.0] 60.0
    delays = []
    ep = make_endpoint([503, 500, 200], body="done")
    print(call_with_retries(ep, sleep=delays.append), delays)                  # done [1.0, 2.0]
    print(ep.state["n"])                                                       # 3
    d1 = []; ok = make_endpoint([200])
    print(call_with_retries(ok, sleep=d1.append), ok.state["n"], d1)           # OK 1 []
    bad = make_endpoint([400])
    try:
        call_with_retries(bad, sleep=lambda s: None)
    except ValueError as e:
        print("ValueError", bad.state["n"])                                    # ValueError 1
    d2 = []; down = make_endpoint([503])
    try:
        call_with_retries(down, max_attempts=4, sleep=d2.append)
    except RuntimeError as e:
        print("RuntimeError", down.state["n"], len(d2))                        # RuntimeError 4 3
