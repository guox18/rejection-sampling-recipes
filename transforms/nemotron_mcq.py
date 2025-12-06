"""
Transform function for Nemotron Post-Training Dataset MCQ format.

Converts the original format (with assistant response) to pipeline format
(user message only + metadata with ground truth answer).
"""

import re


def transform(item: dict) -> dict | None:
    """
    Transform Nemotron MCQ item to pipeline format.

    Input format:
    {
        "uuid": "...",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "question..."},
            {"role": "assistant", "content": "explanation... \\boxed{A}"}
        ],
        ...
    }

    Output format:
    {
        "id": "...",
        "messages": [{"role": "user", "content": "question..."}],
        "metadata": {"answer": "A", "category": "stem", ...}
    }
    """
    # Extract answer from assistant response
    messages = item.get("messages", [])
    assistant_msg = None
    user_messages = []

    for msg in messages:
        if msg["role"] == "assistant":
            assistant_msg = msg["content"]
        elif msg["role"] == "user":
            user_messages.append(msg)
        elif msg["role"] == "system" and msg["content"]:
            # Include non-empty system messages
            user_messages.insert(0, msg)

    if not assistant_msg or not user_messages:
        return None

    # Extract answer from \boxed{}
    answer = extract_boxed_answer(assistant_msg)
    if not answer:
        return None

    return {
        "id": item.get("uuid", ""),
        "messages": user_messages,
        "metadata": {
            "answer": answer,
            "category": item.get("category", ""),
            "generator": item.get("generator", ""),
            "original_response": assistant_msg,  # Keep for reference
        },
    }


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{} format."""
    # Match \boxed{X} where X is typically a single letter for MCQ
    patterns = [
        r"\\boxed\{([A-Z])\}",  # \boxed{A}
        r"\\boxed\{\\text\{([A-Z])\}\}",  # \boxed{\text{A}}
        r"\\boxed\{\\textbf\{([A-Z])\}\}",  # \boxed{\textbf{A}}
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]  # Return last match (final answer)

    return None
