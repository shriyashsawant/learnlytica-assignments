"""Day 26 / Session 2 - Lab Assignment 1: Chat Message Format & Prompt Rendering for Conversational LLMs."""

VALID_ROLES = ("system", "user", "assistant")


def make_message(role, content):
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role: {role!r}")
    return {"role": role, "content": content}


def render_prompt(messages, add_generation_prompt=False):
    text = "\n".join(f"<|{m['role']}|>\n{m['content']}<|end|>" for m in messages)
    if add_generation_prompt:
        text += "\n<|assistant|>"                            # the cue that tells the model to start replying
    return text


def validate_conversation(messages):
    if not isinstance(messages, list) or len(messages) == 0:
        return False
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in VALID_ROLES or not isinstance(m.get("content"), str):
            return False
    systems = [i for i, m in enumerate(messages) if m["role"] == "system"]
    if len(systems) > 1 or (systems and systems[0] != 0):
        return False                                         # at most one system message, and it must be first
    turns = messages[1:] if systems else messages
    expected = "user"                                        # a chat starts with the user and alternates
    for m in turns:
        if m["role"] != expected:
            return False
        expected = "assistant" if expected == "user" else "user"
    return True


def merge_system(messages):
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    others = [m for m in messages if m["role"] != "system"]
    if not system_parts:
        return list(others)
    return [make_message("system", "\n".join(system_parts))] + others


if __name__ == "__main__":
    msgs = [make_message("system", "You are helpful."), make_message("user", "Hi")]
    print(render_prompt(msgs))
    print(render_prompt([make_message("user", "Hi")], add_generation_prompt=True))
    print(validate_conversation(msgs), validate_conversation([make_message("assistant", "hi")]))        # True False
    conv = [make_message("system", "s"), make_message("user", "u"), make_message("assistant", "a"), make_message("user", "u2")]
    print(validate_conversation(conv), validate_conversation([make_message("user", "u"), make_message("system", "s")]),
          validate_conversation([make_message("system", "a"), make_message("system", "b"), make_message("user", "u")]))   # True False False
    print(render_prompt(conv).count("<|end|>"), make_message("user", "hello"))                           # 4 {...}
    try:
        make_message("robot", "x")
    except ValueError as e:
        print("ValueError:", e)
    print(merge_system([make_message("system", "A"), make_message("user", "u"), make_message("system", "B")]))
