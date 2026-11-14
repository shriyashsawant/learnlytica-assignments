"""Day 26 / Session 2 - Lab Assignment 1: Chat Message Format & Prompt Rendering for Conversational LLMs."""

VALID_ROLES = ("system", "user", "assistant")

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hola, ¿cómo estás?"},
    {"role": "assistant", "content": "¡Muy bien, gracias!"},
    {"role": "user", "content": "Great, switch to English please."},
]


def make_message(role, content):
    if role not in VALID_ROLES:
        raise ValueError(f"unknown role: {role!r}")
    return {"role": role, "content": content}


def render_prompt(messages, add_generation_prompt=False):
    text = "\n".join(f"<|{m['role']}|>\n{m['content']}<|end|>" for m in messages)
    if add_generation_prompt:
        text += "\n<|assistant|>"                         # the cue that tells the model to start replying
    return text


def validate_conversation(messages):
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return False
        if m["role"] not in VALID_ROLES or not isinstance(m["content"], str):
            return False
    roles = [m["role"] for m in messages]
    if roles.count("system") > 1 or ("system" in roles and roles[0] != "system"):
        return False                                       # at most one system message, and it must come first
    rest = roles[1:] if roles[0] == "system" else roles
    expected = "user"                                      # a chat starts with the user, then strictly alternates
    for role in rest:
        if role != expected:
            return False
        expected = "assistant" if expected == "user" else "user"
    return True


def merge_system(messages):
    systems = [m["content"] for m in messages if m["role"] == "system"]
    others = [dict(m) for m in messages if m["role"] != "system"]
    if not systems:
        return others
    return [{"role": "system", "content": "\n".join(systems)}] + others     # one system message at the front


if __name__ == "__main__":
    msgs = [make_message("system", "You are helpful."), make_message("user", "Hi")]
    print(render_prompt(msgs))
    print(render_prompt([make_message("user", "Hi")], add_generation_prompt=True))
    print(validate_conversation(msgs))                                     # True
    print(validate_conversation([make_message("assistant", "hi")]))        # False
    print(validate_conversation(conversation))                             # True
    print(validate_conversation([make_message("user", "a"), make_message("system", "s")]))     # False (system not first)
    print(validate_conversation([make_message("system", "a"), make_message("system", "b"), make_message("user", "u")]))   # False
    print(make_message("user", "hello") == {"role": "user", "content": "hello"})
    try:
        make_message("robot", "x")
    except ValueError as e:
        print("ValueError:", e)
    mixed = [make_message("system", "A"), make_message("user", "u"), make_message("system", "B"), make_message("assistant", "a")]
    print(merge_system(mixed))
    print(render_prompt(conversation).count("<|end|>"))                    # 4 (one per message)
