"""Router — meme matching via semantic search.

At startup, loads ``data/memes.yaml``, embeds every meme phrase, and stores
them in a :class:`VectorStore`.  At runtime, embeds the user message and
does a deterministic similarity search — no LLM is involved in routing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from .embedder import Embedder
from .store import SqliteVecStore, VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RouteResult
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """The result of a successful meme match.

    Attributes
    ----------
    agent_id : str
        The target sub-agent identifier (e.g. ``"recipe_agent"``).
    sub_type : str
        The sub-type for prompt selection (e.g. ``"喝什么"``).
    meme_text : str
        The matched meme trigger text (for display / context).
    similarity : float
        Cosine similarity [0, 1] between the user message and the matched meme.
    """

    agent_id: str
    sub_type: str
    meme_text: str
    similarity: float


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Deterministic meme router.

    Startup flow::

        router = Router.from_config(llm_client, settings, "data/memes.yaml")
        # → loads YAML, embeds all memes, populates the vector store

    Runtime flow::

        hit = router.route("当浮一大白")
        if hit:
            print(hit.agent_id)  # "recipe_agent"
            print(hit.sub_type)  # "喝什么"
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        threshold: float = 0.75,
        top_k: int = 3,
    ) -> None:
        """Create a Router from pre-built components.

        Prefer :meth:`from_config` for the common bootstrapping path.
        """
        self._embedder = embedder
        self._store = store
        self._threshold = threshold
        self._top_k = top_k

    # -- factory ---------------------------------------------------------------

    @classmethod
    def from_config(cls, llm_client, settings, memes_path: str = "data/memes.yaml") -> Router:
        """Build a Router from the config-driven bootstrapping path.

        1. Load *memes_path* (YAML flat list)
        2. Create a :class:`SqliteVecStore` at *settings.rag.db_path*
        3. If the store is empty, embed all memes and populate it
        4. Return a ready-to-use :class:`Router`

        Parameters
        ----------
        llm_client : LLMClient
            The configured LLM client (used for embedding).
        settings : Settings
            Application settings (``rag.similarity_threshold``, ``rag.top_k``,
            ``rag.db_path``).
        memes_path : str
            Path to the YAML meme corpus.

        Returns
        -------
        Router
            Ready to route.
        """
        embedder = Embedder(llm_client)
        store = SqliteVecStore(settings.rag.db_path)

        if store.count() == 0:
            memes = cls._load_memes(memes_path)
            if not memes:
                logger.warning("No memes found in %s — router will always miss.", memes_path)
            else:
                cls._populate_store(
                    embedder, store, memes, batch_size=settings.rag.embed_batch_size
                )
                logger.info("Populated store with %d meme vectors.", len(memes))

        return cls(
            embedder=embedder,
            store=store,
            threshold=settings.rag.similarity_threshold,
            top_k=settings.rag.top_k,
        )

    # -- runtime routing -------------------------------------------------------

    def route(self, user_text: str) -> RouteResult | None:
        """Try to match *user_text* against the meme corpus.

        Parameters
        ----------
        user_text : str
            The raw user input.

        Returns
        -------
        RouteResult or None
            ``None`` when no meme exceeds the similarity threshold.
        """
        if self._store.count() == 0:
            return None

        query_vec = self._embedder.embed(user_text)
        matches = self._store.search(query_vec, top_k=self._top_k)

        for match in matches:
            if match.similarity >= self._threshold:
                logger.debug(
                    "Meme hit: %r → agent=%s sub_type=%s sim=%.3f",
                    match.text,
                    match.agent_id,
                    match.sub_type,
                    match.similarity,
                )
                return RouteResult(
                    agent_id=match.agent_id,
                    sub_type=match.sub_type,
                    meme_text=match.text,
                    similarity=match.similarity,
                )

        return None

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _load_memes(path: str) -> list[dict]:
        """Load the flat meme list from a YAML file.

        Returns
        -------
        list[dict]
            Each dict has ``text``, ``agent_id``, ``sub_type``.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("Meme corpus not found: %s", file_path)
            return []

        with open(file_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        memes = data.get("memes", [])
        # Basic validation — skip malformed entries
        valid = []
        for i, entry in enumerate(memes):
            if not isinstance(entry, dict):
                logger.warning("Skipping non-dict meme entry %d: %r", i, entry)
                continue
            if not all(k in entry for k in ("text", "agent_id", "sub_type")):
                logger.warning("Skipping incomplete meme entry %d: %r", i, entry)
                continue
            valid.append(entry)
        return valid

    @staticmethod
    def _populate_store(
        embedder: Embedder,
        store: VectorStore,
        memes: list[dict],
        batch_size: int = 10,
    ) -> None:
        """Embed all memes in batches and write them into *store*.

        Chunked to respect provider batch-size limits (e.g. some services cap
        at 10 inputs per embedding request).
        """
        for batch_start in range(0, len(memes), batch_size):
            batch = memes[batch_start:batch_start + batch_size]
            texts = [m["text"] for m in batch]
            vectors = embedder.embed_batch(texts)

            for offset, (meme, vector) in enumerate(zip(batch, vectors)):
                idx = batch_start + offset
                store.add(
                    rowid=idx,
                    vector=vector,
                    metadata={
                        "agent_id": meme["agent_id"],
                        "sub_type": meme["sub_type"],
                        "text": meme["text"],
                    },
                )
