"""JSON repair and extraction for LLM structured output.

LLMs frequently produce malformed JSON: markdown code fences, trailing commas,
single-quoted strings, unquoted keys, or extra text around the JSON object.

This module implements a **deterministic repair chain** — a sequence of
progressively more aggressive fixes, each tried in order.  No LLM re-calls,
no regex-only hacks that break on edge cases.
"""

from __future__ import annotations

import json
import re


def parse_structured(raw: str) -> dict | None:
    """Attempt to extract and parse a JSON object from *raw* LLM output.

    The repair chain, in order:

    1. Direct ``json.loads`` on the stripped string.
    2. Strip `` ```json `` / `` ``` `` code fences, then ``json.loads``.
    3. Apply lightweight fix-ups (trailing commas, single quotes, unquoted
       keys) and retry ``json.loads``.
    4. Regex-extract the first ``{ ... }`` block (greedy-brace-aware) and
       ``json.loads`` it.
    5. Give up — return ``None``.

    Even after a successful parse, the result is only returned if it is a
    ``dict`` that contains an ``"action"`` key.  This is a **fast pre-check**;
    full :class:`Action` validation happens in :meth:`Action.from_dict`.

    Returns
    -------
    dict | None
        The parsed JSON dict if successful **and** containing ``"action"``;
        otherwise ``None``.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # Step 1 — direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # Step 2 — strip code fences
    result = _try_parse(_strip_code_fences(text))
    if result is not None:
        return result

    # Step 3 — lightweight fix-ups
    result = _try_parse(_light_fix(text))
    if result is not None:
        return result

    # Step 4 — regex-extract first { ... } block
    extracted = _extract_first_json_block(text)
    if extracted:
        result = _try_parse(extracted)
        if result is not None:
            return result
        # also try fix-up + extract combo
        fixed = _light_fix(extracted)
        result = _try_parse(fixed)
        if result is not None:
            return result

    # Step 5 — give up
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches ```json ... ``` or ``` ... ```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove a single outer pair of ```json / ``` markers, if present."""
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


def _try_parse(text: str) -> dict | None:
    """``json.loads`` with a safety net — returns a dict or None."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action" in obj:
            return obj
        return None
    except (json.JSONDecodeError, ValueError):
        return None


# -- lightweight fix-ups ------------------------------------------------------

_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_SINGLE_QUOTED_STR_RE = re.compile(r"'((?:[^'\\]|\\.)*?)'")
_UNQUOTED_KEY_RE = re.compile(r'([{,]\s*)([a-zA-Z_]\w*)(\s*:)')


def _light_fix(text: str) -> str:
    """Apply a series of conservative JSON fix-ups.

    These fix the most common LLM JSON mistakes without introducing
    aggressive heuristics that might corrupt valid-but-unusual JSON.
    """
    # Remove markdown code fences first (they break all other steps)
    text = _strip_code_fences(text)

    # 1. Trailing commas before } or ]
    text = _TRAILING_COMMA_RE.sub(r"\1", text)

    # 2. Single-quoted strings → double-quoted (naive but effective for
    #    typical LLM output without complex escape sequences)
    text = _SINGLE_QUOTED_STR_RE.sub(r'"\1"', text)

    # 3. Unquoted keys: { key: ... } → { "key": ... }
    text = _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', text)

    return text


# -- brace extraction ---------------------------------------------------------


def _extract_first_json_block(text: str) -> str | None:
    """Extract the first ``{ ... }`` block from *text* using brace counting.

    This handles JSON objects nested inside surrounding explanatory text,
    and correctly skips strings that contain literal braces.
    """
    # Find the first opening brace
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\" and in_string:
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None  # unbalanced braces
