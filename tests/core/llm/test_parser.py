"""Tests for core.llm.parser — the 5-layer JSON repair and extraction chain."""

from __future__ import annotations

import pytest

from three_kingdoms_ai_agent.core.llm.parser import (
    _extract_first_json_block,
    _light_fix,
    _strip_code_fences,
    parse_structured,
)


class TestParseStructured:
    """Top-level parse_structured() — full repair chain."""

    # -- happy path -----------------------------------------------------------

    def test_plain_json(self):
        result = parse_structured('{"action": "exit"}')
        assert result == {"action": "exit"}

    def test_json_with_whitespace(self):
        result = parse_structured('  \n  {"action": "switch", "target": "x"}  ')
        assert result == {"action": "switch", "target": "x"}

    # -- code fences ----------------------------------------------------------

    def test_json_with_fences(self):
        result = parse_structured('```json\n{"action": "exit"}\n```')
        assert result == {"action": "exit"}

    def test_fences_no_json_label(self):
        result = parse_structured('```\n{"action": "exit"}\n```')
        assert result == {"action": "exit"}

    def test_fences_extra_whitespace(self):
        result = parse_structured('```json  \n{"action": "exit"}\n```')
        assert result == {"action": "exit"}

    # -- trailing commas ------------------------------------------------------

    def test_trailing_comma_in_object(self):
        result = parse_structured('{"action": "exit",}')
        assert result == {"action": "exit"}

    def test_trailing_comma_after_target(self):
        result = parse_structured('{"action": "switch", "target": "r",}')
        assert result == {"action": "switch", "target": "r"}

    # -- single quotes --------------------------------------------------------

    def test_single_quoted_value(self):
        result = parse_structured("{'action': 'exit'}")
        assert result == {"action": "exit"}

    def test_single_quoted_keys_and_values(self):
        result = parse_structured("{'action': 'switch', 'target': 'media'}")
        assert result == {"action": "switch", "target": "media"}

    # -- unquoted keys --------------------------------------------------------

    def test_unquoted_key(self):
        result = parse_structured("{action: 'exit'}")
        assert result == {"action": "exit"}

    def test_multiple_unquoted_keys(self):
        result = parse_structured("{action: 'switch', target: 'chat'}")
        assert result == {"action": "switch", "target": "chat"}

    # -- extraction from surrounding text -------------------------------------

    def test_json_buried_in_text(self):
        result = parse_structured(
            '好的，我理解了\n{"action": "exit"}\n以上是我的回复。'
        )
        assert result == {"action": "exit"}

    def test_multiple_json_objects_extracts_first(self):
        result = parse_structured(
            '{"action": "exit"}\nsome text\n{"action": "switch", "target": "x"}'
        )
        assert result == {"action": "exit"}

    # -- edge cases -----------------------------------------------------------

    def test_empty_string(self):
        assert parse_structured("") is None

    def test_whitespace_only(self):
        assert parse_structured("   \n  ") is None

    def test_no_action_key(self):
        """Even valid JSON without 'action' key returns None."""
        assert parse_structured('{"foo": "bar"}') is None

    def test_array_wrapped_dict_extracted(self):
        """When an action dict is inside a JSON array, we extract it.
        LLMs sometimes wrap the object in [...] — being robust is better
        than rejecting valid action data.
        """
        result = parse_structured('[{"action": "exit"}]')
        assert result == {"action": "exit"}

    def test_primitive_is_not_a_dict(self):
        """JSON primitives are not valid action containers."""
        assert parse_structured('"action"') is None
        assert parse_structured("42") is None

    def test_malformed_unrecoverable(self):
        assert parse_structured("this is not json at all") is None

    def test_broken_beyond_repair(self):
        """Truly broken JSON that the fix-ups can't help."""
        assert parse_structured('{"action": "exit" broken') is None

    # -- real-world LLM output patterns ---------------------------------------

    def test_llm_with_explanatory_prefix(self):
        result = parse_structured(
            '根据你的要求，我将输出以下JSON：\n{"action": "tool", "target": "player", "params": {"song": "关羽之歌"}}'
        )
        assert result == {
            "action": "tool",
            "target": "player",
            "params": {"song": "关羽之歌"},
        }

    def test_chinese_characters_in_json(self):
        """JSON with Chinese values should parse correctly."""
        result = parse_structured('{"action": "switch", "target": "吃什么"}')
        assert result == {"action": "switch", "target": "吃什么"}


# ---------------------------------------------------------------------------
# Low-level helpers (white-box)
# ---------------------------------------------------------------------------


class TestStripCodeFences:
    def test_json_label(self):
        assert _strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'

    def test_no_label(self):
        assert _strip_code_fences('```\n{"a":1}\n```') == '{"a":1}'

    def test_no_fences(self):
        text = '{"a":1}'
        assert _strip_code_fences(text) is text  # same object ref


class TestLightFix:
    def test_trailing_comma_object(self):
        assert _light_fix('{"a": 1,}') == '{"a": 1}'

    def test_trailing_comma_array(self):
        assert _light_fix('{"a": [1,2,]}') == '{"a": [1,2]}'

    def test_single_quotes_basic(self):
        fixed = _light_fix("{'a': 'hello'}")
        assert '"' in fixed
        assert "'" not in fixed

    def test_unquoted_keys(self):
        fixed = _light_fix("{key: 'value'}")
        assert '"key"' in fixed


class TestExtractFirstJsonBlock:
    def test_simple(self):
        assert _extract_first_json_block('{"a": 1}') == '{"a": 1}'

    def test_nested(self):
        assert _extract_first_json_block('{"a": {"b": 2}}') == '{"a": {"b": 2}}'

    def test_with_surrounding_text(self):
        block = _extract_first_json_block('hello {"a": 1} world')
        assert block == '{"a": 1}'

    def test_brace_in_string(self):
        block = _extract_first_json_block('{"a": "{not a brace}"}')
        assert block == '{"a": "{not a brace}"}'

    def test_no_brace(self):
        assert _extract_first_json_block("no braces here") is None

    def test_unbalanced(self):
        assert _extract_first_json_block('{"a": 1') is None
