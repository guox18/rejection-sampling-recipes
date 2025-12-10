"""
Response processor: split thinking and final response.

Supports multiple formats:
- <think>...</think> format (R1, QwQ, etc.)
- <|channel|>analysis<|message|> format (GPT-OSS, etc.)
"""


def split_response(raw_response: str) -> tuple[str, str]:
    """
    Split raw response into thinking and final response.

    Args:
        raw_response: Model's raw output

    Returns:
        Tuple of (thinking, response):
        - thinking: The thinking/analysis process (empty string if none)
        - response: The final response for judging
    """
    thinking = ""
    response = raw_response.strip()

    # Format 1: <think>...</think> (R1, QwQ, etc.)
    if "<think>" in raw_response:
        if "</think>" in raw_response:
            parts = raw_response.split("</think>", 1)
            # Extract thinking content (remove <think> tag)
            thinking_part = parts[0]
            if "<think>" in thinking_part:
                thinking = thinking_part.split("<think>", 1)[-1].strip()
            else:
                thinking = thinking_part.strip()
            response = parts[-1].strip()
        else:
            # Truncated - thinking incomplete, no final response
            thinking = raw_response.replace("<think>", "").strip()
            response = ""

    # Format 2: GPT-OSS <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>
    elif "<|channel|>analysis<|message|>" in raw_response:
        separator = "<|end|><|start|>assistant<|channel|>final<|message|>"
        if separator in raw_response:
            parts = raw_response.split(separator, 1)
            # Extract thinking (remove channel tags)
            thinking_part = parts[0]
            if "<|channel|>analysis<|message|>" in thinking_part:
                thinking = thinking_part.split("<|channel|>analysis<|message|>", 1)[-1].strip()
            else:
                thinking = thinking_part.strip()
            response = parts[-1].strip()
        else:
            # Truncated - no final response
            thinking = raw_response.replace("<|channel|>analysis<|message|>", "").strip()
            response = ""

    # Fallback: check for </think> without <think> (some chat templates)
    elif "</think>" in raw_response:
        parts = raw_response.split("</think>", 1)
        thinking = parts[0].strip()
        response = parts[-1].strip()

    return thinking, response


def clip_thinking(raw_response: str) -> str:
    """
    Remove thinking process, return only final response.

    This is a convenience function that returns just the response part.

    Args:
        raw_response: Model's raw output

    Returns:
        Final response (for judging)
    """
    _, response = split_response(raw_response)
    return response


def has_final_answer(raw_response: str) -> bool:
    """
    Check if response has a final answer section (after thinking).

    This is NOT for truncation detection! Truncation should be determined by:
    - API: finish_reason == "stop" (not "length")
    - vLLM: eos_token present at end

    This function only checks if the response structure has a final answer,
    which may be empty if the model was cut off mid-thinking.

    Args:
        raw_response: Model's raw output

    Returns:
        True if response has a final answer section
    """
    _, response = split_response(raw_response)
    return bool(response.strip())
