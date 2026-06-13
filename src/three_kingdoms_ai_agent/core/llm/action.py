"""Action type definitions and validation for structured LLM output.

An *Action* represents a control signal embedded in the LLM's JSON response.
It is validated deterministically — no LLM is involved in parsing or validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    """Well-known action types that the orchestrator / sub-agents understand."""

    SWITCH = "switch"  # route to another agent (target = agent_id)
    EXIT = "exit"  # leave the current agent / end the conversation
    TOOL = "tool"  # invoke an external tool (target = tool_name)


@dataclass
class Action:
    """A validated, structured control signal extracted from the LLM response.

    Create via :meth:`from_dict` — never construct directly unless you are
    certain the values are valid.
    """

    type: ActionType
    target: str | None = None
    params: dict | None = field(default=None, repr=False)

    # -- validation -----------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict | None) -> Action | None:
        """Strictly validate a raw parsed dict into an :class:`Action`.

        Returns ``None`` for **any** invalid input — no exceptions are raised.
        The caller simply treats a ``None`` result as a non-structured response.

        Validation rules (all must pass):

        1. *d* must be a non-empty dict containing the key ``"action"``.
        2. ``d["action"]`` must be one of ``"switch"`` / ``"exit"`` / ``"tool"``.
        3. ``"switch"`` → ``target`` must be a non-empty string.
        4. ``"exit"``   → ``target`` must be absent, ``None``, or an empty string.
        5. ``"tool"``   → ``target`` must be a non-empty string; ``params`` is
           optional and stored as-is if present.
        """
        if not isinstance(d, dict) or "action" not in d:
            return None

        raw_action = d.get("action")
        if raw_action not in _ACTION_VALUE_MAP:
            return None

        action_type = _ACTION_VALUE_MAP[raw_action]
        target = d.get("target")

        # --- switch ---------------------------------------------------------------
        if action_type == ActionType.SWITCH:
            if not isinstance(target, str) or not target.strip():
                return None
            return cls(type=ActionType.SWITCH, target=target.strip())

        # --- exit -----------------------------------------------------------------
        if action_type == ActionType.EXIT:
            if target is not None and target != "":
                return None  # exit must NOT have a meaningful target
            return cls(type=ActionType.EXIT)

        # --- tool -----------------------------------------------------------------
        if action_type == ActionType.TOOL:
            if not isinstance(target, str) or not target.strip():
                return None
            params = d.get("params")
            return cls(
                type=ActionType.TOOL,
                target=target.strip(),
                params=params if isinstance(params, dict) else None,
            )

        return None  # unreachable; kept for safety


# -- lookup helpers -----------------------------------------------------------

_ACTION_VALUE_MAP: dict[str, ActionType] = {
    "switch": ActionType.SWITCH,
    "exit": ActionType.EXIT,
    "tool": ActionType.TOOL,
}
