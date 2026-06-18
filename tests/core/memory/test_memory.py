"""Tests for core.memory — MemoryManager ABC, WindowMemory, and LongTermMemory."""

from __future__ import annotations

import pytest

from three_kingdoms_ai_agent.core.memory import (
    LongTermMemory,
    MemoryManager,
    WindowMemory,
)


# =========================================================================
# MemoryManager ABC
# =========================================================================


class TestMemoryManagerABC:
    """Verify that MemoryManager cannot be instantiated directly."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            MemoryManager()  # type: ignore[abstract]


# =========================================================================
# WindowMemory
# =========================================================================


class TestWindowMemoryInit:
    def test_default_window_size_is_10(self):
        wm = WindowMemory()
        assert wm._window_size == 10
        assert len(wm) == 0

    def test_custom_window_size(self):
        wm = WindowMemory(window_size=5)
        assert wm._window_size == 5
        assert len(wm) == 0

    def test_window_size_zero_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            WindowMemory(window_size=0)

    def test_window_size_negative_raises(self):
        with pytest.raises(ValueError, match=">= 1"):
            WindowMemory(window_size=-1)

    def test_is_memory_manager_subclass(self):
        assert issubclass(WindowMemory, MemoryManager)


class TestWindowMemoryAdd:
    def test_add_single_turn(self):
        wm = WindowMemory()
        wm.add("user", "hello")
        assert len(wm) == 1
        ctx = wm.get_context()
        assert ctx == [{"role": "user", "content": "hello"}]

    def test_add_multiple_turns_chronological(self):
        wm = WindowMemory()
        wm.add("user", "Q1")
        wm.add("assistant", "A1")
        wm.add("user", "Q2")
        assert len(wm) == 3
        ctx = wm.get_context()
        assert ctx == [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]


class TestWindowMemoryGetContext:
    def test_empty_returns_empty_list(self):
        wm = WindowMemory()
        assert wm.get_context() == []

    def test_get_context_without_limit_returns_all(self):
        wm = WindowMemory()
        for i in range(5):
            wm.add("user", f"msg{i}")
        assert len(wm.get_context()) == 5

    def test_get_context_with_limit_returns_last_n(self):
        wm = WindowMemory()
        for i in range(10):
            wm.add("user", f"msg{i}")
        ctx = wm.get_context(limit=3)
        assert ctx == [
            {"role": "user", "content": "msg7"},
            {"role": "user", "content": "msg8"},
            {"role": "user", "content": "msg9"},
        ]

    def test_get_context_limit_larger_than_stored(self):
        wm = WindowMemory()
        wm.add("user", "only")
        ctx = wm.get_context(limit=100)
        assert ctx == [{"role": "user", "content": "only"}]

    def test_get_context_limit_none_same_as_all(self):
        wm = WindowMemory()
        wm.add("user", "hello")
        assert wm.get_context(limit=None) == wm.get_context()

    def test_get_context_returns_chat_compatible_dicts(self):
        """Each entry must be a plain dict with 'role' and 'content' keys — the
        format LLMClient.chat() expects."""
        wm = WindowMemory()
        wm.add("system", "you are a helpful assistant")
        wm.add("user", "hi")
        ctx = wm.get_context()
        for entry in ctx:
            assert isinstance(entry, dict)
            assert set(entry.keys()) == {"role", "content"}
            assert isinstance(entry["role"], str)
            assert isinstance(entry["content"], str)


class TestWindowMemoryEviction:
    def test_evicts_oldest_when_full(self):
        wm = WindowMemory(window_size=3)
        wm.add("user", "1")
        wm.add("assistant", "2")
        wm.add("user", "3")  # window full
        assert len(wm) == 3

        wm.add("assistant", "4")  # should evict "1"
        assert len(wm) == 3
        ctx = wm.get_context()
        assert ctx == [
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
        ]

    def test_window_size_one(self):
        wm = WindowMemory(window_size=1)
        wm.add("user", "first")
        assert len(wm) == 1
        wm.add("user", "second")
        assert len(wm) == 1
        assert wm.get_context() == [{"role": "user", "content": "second"}]


class TestWindowMemoryLongTerm:
    def test_store_long_term_is_noop(self):
        wm = WindowMemory()
        wm.store_long_term("key", {"a": 1})  # should not raise

    def test_recall_long_term_returns_none(self):
        wm = WindowMemory()
        assert wm.recall_long_term("any_key") is None


class TestWindowMemoryClear:
    def test_clear_drops_all_turns(self):
        wm = WindowMemory()
        wm.add("user", "A")
        wm.add("assistant", "B")
        assert len(wm) == 2
        wm.clear()
        assert len(wm) == 0
        assert wm.get_context() == []

    def test_clear_then_add(self):
        wm = WindowMemory()
        wm.add("user", "old")
        wm.clear()
        wm.add("user", "new")
        assert wm.get_context() == [{"role": "user", "content": "new"}]


class TestWindowMemoryRepr:
    def test_repr_shows_window_size_and_count(self):
        wm = WindowMemory(window_size=8)
        wm.add("user", "hi")
        r = repr(wm)
        assert "WindowMemory" in r
        assert "8" in r
        assert "1" in r  # turn count


# =========================================================================
# LongTermMemory
# =========================================================================


class TestLongTermMemory:
    def test_is_memory_manager_subclass(self):
        assert issubclass(LongTermMemory, MemoryManager)

    def test_add_is_noop(self):
        ltm = LongTermMemory()
        ltm.add("user", "anything")  # should not raise

    def test_get_context_returns_empty_list(self):
        ltm = LongTermMemory()
        assert ltm.get_context() == []
        assert ltm.get_context(limit=5) == []

    def test_store_long_term_is_noop(self):
        ltm = LongTermMemory()
        ltm.store_long_term("k", {"v": 1})  # should not raise

    def test_recall_long_term_returns_none(self):
        ltm = LongTermMemory()
        assert ltm.recall_long_term("missing") is None

    def test_clear_is_noop(self):
        ltm = LongTermMemory()
        ltm.clear()  # should not raise

    def test_repr(self):
        ltm = LongTermMemory()
        assert repr(ltm) == "LongTermMemory()"


# =========================================================================
# Integration: polymorphic usage via ABC
# =========================================================================


class TestPolymorphicUsage:
    """Both WindowMemory and LongTermMemory should be usable through the
    MemoryManager ABC interface."""

    @pytest.mark.parametrize("backend", [WindowMemory(), LongTermMemory()])
    def test_add_does_not_raise(self, backend: MemoryManager):
        backend.add("user", "test")

    @pytest.mark.parametrize("backend", [WindowMemory(), LongTermMemory()])
    def test_get_context_returns_list_of_dict(self, backend: MemoryManager):
        ctx = backend.get_context()
        assert isinstance(ctx, list)

    @pytest.mark.parametrize("backend", [WindowMemory(), LongTermMemory()])
    def test_store_long_term_does_not_raise(self, backend: MemoryManager):
        backend.store_long_term("k", None)

    @pytest.mark.parametrize("backend", [WindowMemory(), LongTermMemory()])
    def test_recall_long_term_returns_none(self, backend: MemoryManager):
        result = backend.recall_long_term("any")
        # WindowMemory/LongTermMemory both return None in MVP
        assert result is None

    @pytest.mark.parametrize("backend", [WindowMemory(), LongTermMemory()])
    def test_clear_does_not_raise(self, backend: MemoryManager):
        backend.clear()
