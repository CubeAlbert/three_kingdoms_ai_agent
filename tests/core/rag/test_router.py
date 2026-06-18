"""Tests for :mod:`core.rag.router`."""

from unittest.mock import MagicMock

import pytest

from three_kingdoms_ai_agent.core.rag.router import RouteResult, Router


class TestRouteResult:
    """Test the RouteResult dataclass."""

    def test_fields(self):
        result = RouteResult(
            agent_id="recipe_agent",
            sub_type="喝什么",
            meme_text="当浮一大白",
            similarity=0.92,
        )
        assert result.agent_id == "recipe_agent"
        assert result.sub_type == "喝什么"
        assert result.meme_text == "当浮一大白"
        assert result.similarity == 0.92


class TestRouterRoute:
    """Test runtime routing logic."""

    @pytest.fixture
    def mock_embedder(self):
        """An Embedder that returns a fixed vector."""
        emb = MagicMock()
        emb.embed.return_value = [0.1, 0.2, 0.3]
        emb.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        return emb

    @pytest.fixture
    def mock_store(self):
        """A vector store with controllable search results."""
        store = MagicMock()
        store.count.return_value = 1  # not empty
        return store

    def test_route_hit_above_threshold(self, mock_embedder, mock_store):
        """When the top match exceeds threshold, a RouteResult is returned."""
        from three_kingdoms_ai_agent.core.rag.store import Match

        mock_store.search.return_value = [
            Match(
                rowid=0,
                agent_id="recipe_agent",
                sub_type="喝什么",
                text="当浮一大白",
                distance=0.05,
            )
        ]

        router = Router(mock_embedder, mock_store, threshold=0.75)
        result = router.route("来喝一杯")

        assert result is not None
        assert result.agent_id == "recipe_agent"
        assert result.sub_type == "喝什么"
        assert result.meme_text == "当浮一大白"
        assert result.similarity == 0.95  # 1.0 - 0.05

    def test_route_miss_below_threshold(self, mock_embedder, mock_store):
        """When the top match is below threshold, None is returned."""
        from three_kingdoms_ai_agent.core.rag.store import Match

        mock_store.search.return_value = [
            Match(
                rowid=0,
                agent_id="chat_agent",
                sub_type="废话文学",
                text="老夫回答你之前...",
                distance=0.6,  # similarity = 0.4 < threshold 0.75
            )
        ]

        router = Router(mock_embedder, mock_store, threshold=0.75)
        result = router.route("今天天气不错")

        assert result is None

    def test_route_empty_store(self, mock_embedder, mock_store):
        """When the store is empty, route() returns None."""
        mock_store.count.return_value = 0

        router = Router(mock_embedder, mock_store, threshold=0.75)
        result = router.route("当浮一大白")

        assert result is None
        mock_store.search.assert_not_called()

    def test_route_picks_first_above_threshold(self, mock_embedder, mock_store):
        """When multiple matches exist, the first one above threshold wins."""
        from three_kingdoms_ai_agent.core.rag.store import Match

        mock_store.search.return_value = [
            Match(rowid=0, agent_id="a", sub_type="s1", text="t1", distance=0.1),  # sim=0.90
            Match(rowid=1, agent_id="b", sub_type="s2", text="t2", distance=0.2),  # sim=0.80
        ]

        router = Router(mock_embedder, mock_store, threshold=0.75)
        result = router.route("test")

        assert result.agent_id == "a"  # first above threshold

    def test_route_skips_match_below_then_picks_next(self, mock_embedder, mock_store):
        """A sub-threshold match is skipped; the next viable match is used."""
        from three_kingdoms_ai_agent.core.rag.store import Match

        mock_store.search.return_value = [
            Match(rowid=0, agent_id="bad", sub_type="x", text="noisy", distance=0.8),  # sim=0.20
            Match(rowid=1, agent_id="good", sub_type="y", text="clean", distance=0.1),  # sim=0.90
        ]

        router = Router(mock_embedder, mock_store, threshold=0.75)
        result = router.route("test")

        assert result is not None
        assert result.agent_id == "good"


class TestRouterLoadMemes:
    """Test the _load_memes static method."""

    def test_load_valid_yaml(self, tmp_path):
        """_load_memes should parse a valid flat memes YAML."""
        yaml_path = tmp_path / "memes.yaml"
        yaml_path.write_text("""
memes:
  - text: "台词1"
    agent_id: agent_a
    sub_type: type_x
  - text: "台词2"
    agent_id: agent_b
    sub_type: type_y
""", encoding="utf-8")

        memes = Router._load_memes(str(yaml_path))
        assert len(memes) == 2
        assert memes[0] == {"text": "台词1", "agent_id": "agent_a", "sub_type": "type_x"}
        assert memes[1] == {"text": "台词2", "agent_id": "agent_b", "sub_type": "type_y"}

    def test_load_missing_file(self):
        """_load_memes should return empty list for a non-existent file."""
        memes = Router._load_memes("nonexistent/file.yaml")
        assert memes == []

    def test_load_skips_incomplete_entries(self, tmp_path):
        """_load_memes should skip dicts missing required keys."""
        yaml_path = tmp_path / "memes.yaml"
        yaml_path.write_text("""
memes:
  - text: "ok"
    agent_id: a
    sub_type: b
  - text: "no agent_id"
    sub_type: c
  - not_a_dict
""", encoding="utf-8")

        memes = Router._load_memes(str(yaml_path))
        assert len(memes) == 1
        assert memes[0]["text"] == "ok"
