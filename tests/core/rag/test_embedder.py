"""Tests for :mod:`core.rag.embedder`."""

from unittest.mock import MagicMock

from three_kingdoms_ai_agent.core.rag.embedder import Embedder


class TestEmbedder:
    """Test the thin Embedder wrapper."""

    def test_embed_passthrough(self):
        """embed() should delegate to client.embed() and return its result."""
        mock_client = MagicMock()
        mock_client.embed.return_value = [0.1, 0.2, 0.3]

        embedder = Embedder(mock_client)
        result = embedder.embed("你好")

        mock_client.embed.assert_called_once_with("你好")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_passthrough(self):
        """embed_batch() should delegate to client.embed_batch() and return
        its result."""
        mock_client = MagicMock()
        mock_client.embed_batch.return_value = [[0.1], [0.2], [0.3]]

        embedder = Embedder(mock_client)
        result = embedder.embed_batch(["a", "b", "c"])

        mock_client.embed_batch.assert_called_once_with(["a", "b", "c"])
        assert result == [[0.1], [0.2], [0.3]]

    def test_embed_batch_empty(self):
        """embed_batch() with empty list should return empty list without
        calling the client."""
        mock_client = MagicMock()

        embedder = Embedder(mock_client)
        result = embedder.embed_batch([])

        assert result == []
        mock_client.embed_batch.assert_not_called()
