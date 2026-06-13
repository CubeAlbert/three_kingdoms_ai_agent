"""Tests for core.llm.action — ActionType, Action, and from_dict validation."""

from __future__ import annotations

import pytest

from three_kingdoms_ai_agent.core.llm.action import Action, ActionType


class TestActionTypeEnum:
    def test_has_three_members(self):
        assert len(ActionType) == 3
        assert ActionType.SWITCH.value == "switch"
        assert ActionType.EXIT.value == "exit"
        assert ActionType.TOOL.value == "tool"

    def test_switch_is_not_exit(self):
        assert ActionType.SWITCH != ActionType.EXIT


class TestActionFromDictSwitch:
    def test_valid_switch(self):
        action = Action.from_dict({"action": "switch", "target": "recipe"})
        assert action is not None
        assert action.type == ActionType.SWITCH
        assert action.target == "recipe"
        assert action.params is None

    def test_switch_strips_target_whitespace(self):
        action = Action.from_dict({"action": "switch", "target": "  recipe  "})
        assert action is not None
        assert action.target == "recipe"

    def test_switch_missing_target_returns_none(self):
        assert Action.from_dict({"action": "switch"}) is None

    def test_switch_empty_target_returns_none(self):
        assert Action.from_dict({"action": "switch", "target": ""}) is None

    def test_switch_whitespace_only_target_returns_none(self):
        assert Action.from_dict({"action": "switch", "target": "   "}) is None

    def test_switch_non_string_target_returns_none(self):
        assert Action.from_dict({"action": "switch", "target": 42}) is None
        assert Action.from_dict({"action": "switch", "target": None}) is None


class TestActionFromDictExit:
    def test_valid_exit_no_target(self):
        action = Action.from_dict({"action": "exit"})
        assert action is not None
        assert action.type == ActionType.EXIT
        assert action.target is None

    def test_valid_exit_target_none(self):
        action = Action.from_dict({"action": "exit", "target": None})
        assert action is not None
        assert action.type == ActionType.EXIT

    def test_valid_exit_target_empty_string(self):
        action = Action.from_dict({"action": "exit", "target": ""})
        assert action is not None
        assert action.type == ActionType.EXIT

    def test_exit_with_meaningful_target_returns_none(self):
        assert Action.from_dict({"action": "exit", "target": "something"}) is None


class TestActionFromDictTool:
    def test_valid_tool(self):
        action = Action.from_dict({"action": "tool", "target": "player"})
        assert action is not None
        assert action.type == ActionType.TOOL
        assert action.target == "player"
        assert action.params is None

    def test_tool_with_params(self):
        action = Action.from_dict(
            {"action": "tool", "target": "player", "params": {"url": "xxx"}}
        )
        assert action is not None
        assert action.params == {"url": "xxx"}

    def test_tool_params_not_dict_stored_as_none(self):
        action = Action.from_dict(
            {"action": "tool", "target": "player", "params": "not-dict"}
        )
        assert action is not None
        assert action.params is None

    def test_tool_missing_target_returns_none(self):
        assert Action.from_dict({"action": "tool"}) is None

    def test_tool_empty_target_returns_none(self):
        assert Action.from_dict({"action": "tool", "target": ""}) is None


class TestActionFromDictInvalid:
    def test_none_input(self):
        assert Action.from_dict(None) is None

    def test_non_dict_input(self):
        assert Action.from_dict("action: exit") is None
        assert Action.from_dict(42) is None
        assert Action.from_dict([]) is None

    def test_empty_dict(self):
        assert Action.from_dict({}) is None

    def test_no_action_key(self):
        assert Action.from_dict({"target": "recipe"}) is None

    def test_unknown_action_value(self):
        assert Action.from_dict({"action": "dance"}) is None
        assert Action.from_dict({"action": "SWITCH"}) is None  # case-sensitive

    def test_extra_keys_ignored(self):
        """Unknown keys do not invalidate the action."""
        action = Action.from_dict(
            {"action": "exit", "extra": "ignored", "foo": "bar"}
        )
        assert action is not None
        assert action.type == ActionType.EXIT


class TestActionDataclass:
    def test_manual_construction(self):
        action = Action(type=ActionType.TOOL, target="player", params={"k": "v"})
        assert action.type == ActionType.TOOL
        assert action.target == "player"
        assert action.params == {"k": "v"}

    def test_defaults(self):
        action = Action(type=ActionType.EXIT)
        assert action.target is None
        assert action.params is None
