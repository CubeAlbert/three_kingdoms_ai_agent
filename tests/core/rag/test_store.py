"""Tests for :mod:`core.rag.store`."""

import pytest

from three_kingdoms_ai_agent.core.rag.store import Match, SqliteVecStore


class TestSqliteVecStore:
    """Test sqlite-vec backed vector store."""

    @pytest.fixture
    def store(self):
        """Return a fresh in-memory store for each test."""
        return SqliteVecStore(":memory:")

    # -- empty state -----------------------------------------------------------

    def test_count_empty(self, store):
        """count() should return 0 for an unpopulated store."""
        assert store.count() == 0

    def test_search_empty(self, store):
        """search() should return an empty list when no vectors exist."""
        results = store.search([0.1, 0.2, 0.3])
        assert results == []

    def test_add_then_count(self, store):
        """count() should reflect inserted vectors."""
        store.add(0, [0.1, 0.2, 0.3], {"agent_id": "a", "sub_type": "s", "text": "t"})
        assert store.count() == 1

        store.add(1, [0.4, 0.5, 0.6], {"agent_id": "b", "sub_type": "s2", "text": "t2"})
        assert store.count() == 2

    # -- search ----------------------------------------------------------------

    def test_search_finds_nearest(self, store):
        """search() should return the closest match first."""
        store.add(0, [1.0, 0.0, 0.0], {"agent_id": "x", "sub_type": "a", "text": "far"})
        store.add(1, [0.1, 0.0, 0.0], {"agent_id": "y", "sub_type": "b", "text": "near"})

        results = store.search([0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

        # "near" (rowid=1) should be closer to [0,0,0] than "far" (rowid=0)
        assert results[0].rowid == 1
        assert results[0].text == "near"
        assert results[0].agent_id == "y"
        assert results[0].sub_type == "b"
        assert results[0].distance < results[1].distance

    def test_search_respects_top_k(self, store):
        """search() should return at most top_k results."""
        for i in range(5):
            store.add(i, [float(i)] * 3, {"agent_id": "a", "sub_type": "s", "text": f"t{i}"})

        results = store.search([0.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_search_similarity_property(self, store):
        """Match.similarity should convert distance to [0, 1] correctly."""
        store.add(0, [0.1, 0.0], {"agent_id": "a", "sub_type": "s", "text": "t"})

        results = store.search([0.1, 0.0], top_k=1)
        # Same vector → distance ~0 → similarity ~1.0
        assert results[0].similarity > 0.99

    # -- replace semantics -----------------------------------------------------

    def test_add_replace_same_rowid(self, store):
        """Adding with the same rowid should replace both vector and metadata."""
        store.add(0, [1.0, 0.0], {"agent_id": "first", "sub_type": "s", "text": "old"})
        store.add(0, [0.0, 1.0], {"agent_id": "second", "sub_type": "t", "text": "new"})

        assert store.count() == 1  # still one row

        # The stored metadata should now be "second"
        results = store.search([0.0, 1.0], top_k=1)
        assert results[0].agent_id == "second"
        assert results[0].text == "new"

    # -- clear -----------------------------------------------------------------

    def test_clear_then_reuse(self, store):
        """After clear(), store should be empty and accept new data."""
        store.add(0, [1.0, 0.0], {"agent_id": "a", "sub_type": "s", "text": "t"})
        assert store.count() == 1

        store.clear()
        assert store.count() == 0
        assert store.search([0.0, 1.0]) == []

        # Re-add after clear
        store.add(0, [0.5, 0.5], {"agent_id": "fresh", "sub_type": "x", "text": "y"})
        assert store.count() == 1
        results = store.search([0.5, 0.5], top_k=1)
        assert results[0].agent_id == "fresh"

    # -- lazy table creation ---------------------------------------------------

    def test_lazy_table_creation_dimension(self, store):
        """The vector table should be created on first add() with the correct
        dimension."""
        # Use an unusual dimension to verify it's captured
        store.add(0, [0.0] * 42, {"agent_id": "a", "sub_type": "s", "text": "t"})
        assert store.count() == 1
        # Searching should work
        results = store.search([0.0] * 42, top_k=1)
        assert len(results) == 1
