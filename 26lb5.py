"""Day 26 / Session 3 - Lab Assignment 1: LLM API Integration - Token-Cost Estimator & Exponential-Backoff Retry."""
import time

TRANSIENT = {429, 500, 502, 503}


def estimate_cost(input_tokens, output_tokens, in_rate_per_1k, out_rate_per_1k):
    return input_tokens / 1000 * in_rate_per_1k + output_tokens / 1000 * out_rate_per_1k


def estimate_batch_cost(requests, in_rate_per_1k, out_rate_per_1k):
    return sum(estimate_cost(r["input_tokens"], r["output_tokens"], in_rate_per_1k, out_rate_per_1k) for r in requests)


def cheapest_model(models, input_tokens, output_tokens):
    best_name = None
    best_cost = None
    for m in models:
        cost = estimate_cost(input_tokens, output_tokens, m["in_rate"], m["out_rate"])
        if best_cost is None or cost < best_cost:            # strict <, so the first model wins ties
            best_name, best_cost = m["name"], cost
    return best_name


def backoff_delay(attempt, base=1.0, factor=2.0, cap=60.0):
    return min(cap, base * factor ** attempt)


def call_with_retries(fn, max_attempts=5, base=1.0, factor=2.0, cap=60.0, sleep=None):
    if sleep is None:
        sleep = time.sleep
    for attempt in range(max_attempts):
        response = fn()
        status = response["status"]
        if status == 200:
            return response["body"]
        if status not in TRANSIENT:
            raise ValueError(f"non-retryable status {status}")          # e.g. 400 / 401 / 404: retrying cannot help
        if attempt < max_attempts - 1:
            sleep(backoff_delay(attempt, base, factor, cap))             # wait, then try again
    raise RuntimeError(f"gave up after {max_attempts} transient failures")


if __name__ == "__main__":
    MODELS = [{"name": "big", "in_rate": 0.003, "out_rate": 0.015},
              {"name": "small", "in_rate": 0.001, "out_rate": 0.002},
              {"name": "huge", "in_rate": 0.010, "out_rate": 0.030}]

    def make_endpoint(statuses, body="OK"):
        state = {"n": 0}

        def call():
            i = state["n"]
            state["n"] += 1
            s = statuses[i] if i < len(statuses) else statuses[-1]
            return {"status": s, "body": body}
        call.state = state
        return call

    print(round(estimate_cost(1000, 500, 0.003, 0.015), 4), round(estimate_cost(2500, 0, 0.002, 0.02), 4))     # 0.0105 0.005
    print(round(estimate_batch_cost([{"input_tokens": 1000, "output_tokens": 1000}, {"input_tokens": 500, "output_tokens": 0}], 0.001, 0.002), 4))   # 0.0035
    print(cheapest_model(MODELS, 1000, 1000), [backoff_delay(a) for a in range(4)], backoff_delay(10))          # small [1.0, 2.0, 4.0, 8.0] 60.0
    delays = []
    ep = make_endpoint([503, 500, 200], body="done")
    print(call_with_retries(ep, sleep=delays.append), delays, ep.state["n"])                                   # done [1.0, 2.0] 3
    d2, ep2 = [], make_endpoint([200])
    print(call_with_retries(ep2, sleep=d2.append), d2, ep2.state["n"])                                         # OK [] 1
    try:
        call_with_retries(make_endpoint([400]), sleep=lambda s: None)
    except ValueError as e:
        print("ValueError:", e)
    d3, ep3 = [], make_endpoint([503])
    try:
        call_with_retries(ep3, max_attempts=4, sleep=d3.append)
    except RuntimeError as e:
        print("RuntimeError:", e, ep3.state["n"], len(d3))                                                     # 4 calls, 3 sleeps
