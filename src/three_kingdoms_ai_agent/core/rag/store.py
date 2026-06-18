"""Vector store — ABC and sqlite-vec implementation.

Uses `sqlite-vec <https://github.com/asg017/sqlite-vec>`_ (zero-dependency
SQLite extension) for vector storage and cosine-distance search.  Metadata
(agent_id, sub_type, meme text) lives in a companion table joined via rowid.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------


@dataclass
class Match:
    """A single search result from the vector store."""

    rowid: int
    agent_id: str
    sub_type: str
    text: str
    distance: float  # cosine distance: 0 = identical, 2 = opposite

    @property
    def similarity(self) -> float:
        """Convert cosine distance to similarity [0, 1]."""
        return 1.0 - self.distance


# ---------------------------------------------------------------------------
# VectorStore ABC
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Abstract interface for a vector store.

    Subclasses are responsible for persistence, indexing, and search.
    """

    @abstractmethod
    def add(self, rowid: int, vector: list[float], metadata: dict) -> None:
        """Insert one vector with its metadata.

        Parameters
        ----------
        rowid : int
            Unique integer id for this entry.
        vector : list[float]
            The embedding vector.
        metadata : dict
            Must contain ``agent_id``, ``sub_type``, and ``text``.
        """
        ...

    @abstractmethod
    def search(self, vector: list[float], top_k: int = 3) -> list[Match]:
        """Find the *top_k* nearest neighbours to *vector*.

        Parameters
        ----------
        vector : list[float]
            Query embedding.
        top_k : int
            Max results to return.

        Returns
        -------
        list[Match]
            Sorted by ascending distance (closest first).
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored vectors."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored vectors and metadata (idempotent)."""
        ...


# ---------------------------------------------------------------------------
# SqliteVecStore
# ---------------------------------------------------------------------------


class SqliteVecStore(VectorStore):
    """sqlite-vec backed vector store.

    Data is stored in a single ``.db`` file with two tables:

    * ``meme_vectors`` — ``vec0`` virtual table holding embeddings
    * ``meme_metadata`` — companion table with ``agent_id``, ``sub_type``, ``text``

    The two tables are joined on ``rowid``.

    Usage::

        store = SqliteVecStore("data/memes.db")
        store.add(0, vec, {"agent_id": "recipe", "sub_type": "喝什么", "text": "..."})
        matches = store.search(query_vec, top_k=3)
    """

    _TABLE_VEC = "meme_vectors"
    _TABLE_META = "meme_metadata"

    def __init__(self, db_path: str | Path) -> None:
        """Open (or create) the store at *db_path*.

        The path is created with parent directories if needed.  The vector
        table is created lazily on the first :meth:`add` call so we know
        the embedding dimension.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_connection()
        self._tables_created = False

    # -- VectorStore interface -------------------------------------------------

    def add(self, rowid: int, vector: list[float], metadata: dict) -> None:
        """Insert one vector and its metadata.

        Deletes any existing row with the same *rowid* first (sqlite-vec
        ``vec0`` tables don't support ``INSERT OR REPLACE``).

        On first call the vector table is created with the dimension of
        *vector*.
        """
        if not self._tables_created:
            self._create_tables(len(vector))

        # vec0 virtual tables don't support INSERT OR REPLACE — we must
        # DELETE first for idempotent upsert behaviour.
        self._conn.execute(
            f"DELETE FROM {self._TABLE_VEC} WHERE rowid = ?", (rowid,)
        )
        self._conn.execute(
            f"DELETE FROM {self._TABLE_META} WHERE rowid = ?", (rowid,)
        )

        vec_json = json.dumps(vector)
        self._conn.execute(
            f"INSERT INTO {self._TABLE_VEC}(rowid, embedding) VALUES (?, ?)",
            (rowid, vec_json),
        )
        self._conn.execute(
            f"INSERT INTO {self._TABLE_META}"
            f"(rowid, agent_id, sub_type, text) VALUES (?, ?, ?, ?)",
            (
                rowid,
                metadata.get("agent_id", ""),
                metadata.get("sub_type", ""),
                metadata.get("text", ""),
            ),
        )
        self._conn.commit()

    def search(self, vector: list[float], top_k: int = 3) -> list[Match]:
        """Find the *top_k* nearest neighbours by cosine distance.

        Returns an empty list when the store hasn't been populated yet.
        """
        if not self._tables_created:
            return []

        query_json = json.dumps(vector)
        rows = self._conn.execute(
            f"SELECT v.rowid, m.agent_id, m.sub_type, m.text, v.distance "
            f"FROM {self._TABLE_VEC} v "
            f"JOIN {self._TABLE_META} m ON v.rowid = m.rowid "
            f"WHERE v.embedding MATCH ? AND k = ? "
            f"ORDER BY v.distance",
            (query_json, top_k),
        ).fetchall()
        return [
            Match(rowid=r[0], agent_id=r[1], sub_type=r[2], text=r[3], distance=r[4])
            for r in rows
        ]

    def count(self) -> int:
        """Return the total number of stored vectors.

        Returns 0 when the store hasn't been populated yet.
        """
        if not self._tables_created:
            return 0
        row = self._conn.execute(
            f"SELECT COUNT(*) FROM {self._TABLE_VEC}"
        ).fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        """Drop both tables and reset state.

        Tables will be re-created lazily on the next :meth:`add` call.
        """
        self._conn.execute(f"DROP TABLE IF EXISTS {self._TABLE_VEC}")
        self._conn.execute(f"DROP TABLE IF EXISTS {self._TABLE_META}")
        self._conn.commit()
        self._tables_created = False

    # -- internals -------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        """Open a SQLite connection and load the vec0 extension.

        Table creation is deferred to the first :meth:`add` call.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.commit()
        return conn

    def _create_tables(self, dim: int) -> None:
        """Create the vector table (with embedding *dim*) and metadata table."""
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._TABLE_VEC} "
            f"USING vec0(embedding float[{dim}])"
        )
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self._TABLE_META} ("
            f"  rowid INTEGER PRIMARY KEY,"
            f"  agent_id TEXT NOT NULL,"
            f"  sub_type TEXT NOT NULL,"
            f"  text TEXT NOT NULL"
            f")"
        )
        self._conn.commit()
        self._tables_created = True
