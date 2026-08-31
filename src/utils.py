"""
utils.py
--------
Small, dependable helpers: safely turning the model's raw text output into
a Python dict (never letting malformed JSON crash the app) and a couple of
formatting helpers used by app.py.
"""

import json
import re


def safe_json_parse(raw_text: str):
    """
    Strip accidental ```json fences / stray text around the JSON object,
    then try to parse it. Returns None (never raises) if parsing fails so
    the caller can show a friendly fallback instead of crashing.
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # Remove ```json ... ``` or ``` ... ``` fences if the model added them.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # If there's extra prose around the JSON, grab the outermost {...}.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


REQUIRED_KEYS = [
    "summary", "possible_conditions", "urgency_level",
    "recommended_next_steps", "questions_for_doctor", "warning_signs",
]


def validate_schema(data) -> bool:
    """True only if every key from the required JSON schema is present."""
    return isinstance(data, dict) and all(k in data for k in REQUIRED_KEYS)


def format_symptom_list(selected, free_text: str) -> str:
    """Combine multiselect choices with any free-text symptoms into one string."""
    items = list(selected)
    if free_text and free_text.strip():
        items.append(free_text.strip())
    return ", ".join(items) if items else "None reported"
