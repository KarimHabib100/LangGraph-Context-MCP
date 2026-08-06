"""Default storage backend: SQLite with the ``sqlite-vec`` extension.

Zero configuration by design (DEC-002) — the index is a single file at
``<repo_root>/.langgraph-context/index.db``, created on demand, with no server to run.

Schema (DEC-011):
  ``meta``          key/value; holds the vector width the vec0 table was created with
  ``repositories``  one row per indexed repo (prd.md's Repository model + model/dimension)
  ``graphs``        one row per graph, structure stored as a JSON document
  ``chunks_vec``    a ``vec0`` virtual table: the embedding plus the metadata a hit needs

Cosine is declared explicitly on the vector column — vec0's default metric is L2, and RISK-004
forbids assuming the two backends' defaults agree. Combined with the unit-length vectors
``normalize_vector`` produces, this backend and pgvector rank identically.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import sqlite_vec

from ..parser.graph_model import GraphDef
from .base import (
    BACKEND_SQLITE,
    EmbeddingChunk,
    RepositoryInfo,
    SearchResult,
    VectorStore,
    graph_from_dict,
    normalize_vector,
)

logger = logging.getLogger(__name__)

# Where the default index lives, relative to the repository being indexed — never relative to
# this tool's install directory.
INDEX_DIR_NAME = ".langgraph-context"
INDEX_FILE_NAME = "index.db"

_DIMENSION_META_KEY = "vector_dimension"


def default_index_path(repo_root: Path) -> Path:
    """``<repo_root>/.langgraph-context/index.db`` — the zero-config index location."""
    return repo_root.resolve() / INDEX_DIR_NAME / INDEX_FILE_NAME


class SqliteStore(VectorStore):
    """``VectorStore`` backed by SQLite + ``sqlite-vec``."""

    def __init__(self, db_path: Path, dimension: int | None = None) -> None:
        """Open (creating if needed) the index at ``db_path``.

        ``dimension`` lets the caller create the vector table up front; when omitted it is
        created on the first write or search, whose vectors carry the width.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self._db_path))
        self._connection.row_factory = sqlite3.Row
        self._load_extension()
        self._create_base_tables()
        self._vector_dimension: int | None = self._read_stored_dimension()
        if dimension is not None:
            self._ensure_vector_table(dimension)

    # ----------------------------------------------------------------------------------------
    # VectorStore contract
    # ----------------------------------------------------------------------------------------
    def upsert_chunks(self, chunks: list[EmbeddingChunk]) -> None:
        if not chunks:
            return
        self._ensure_vector_table(chunks[0].dimension)
        for chunk in chunks:
            if chunk.dimension != self._vector_dimension:
                raise ValueError(
                    f"Chunk {chunk.id} has dimension {chunk.dimension}, but this index stores "
                    f"{self._vector_dimension}-dimensional vectors."
                )

        # vec0 rejects INSERT OR REPLACE, so an upsert is an explicit delete followed by an
        # insert. Both run in one transaction, so a failure cannot leave a chunk missing.
        with self._connection:
            self._connection.executemany(
                "DELETE FROM chunks_vec WHERE chunk_id = ?",
                [(chunk.id,) for chunk in chunks],
            )
            self._connection.executemany(
                """
                INSERT INTO chunks_vec(
                    chunk_id, embedding, repo_id, graph_id, node_id, node_name, file_path,
                    line_start, line_end, docstring, chunk_text, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        sqlite_vec.serialize_float32(normalize_vector(list(chunk.vector))),
                        chunk.repo_id,
                        chunk.graph_id,
                        chunk.node_id,
                        chunk.node_name,
                        chunk.file_path,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.docstring,
                        chunk.text,
                        chunk.embedding_model,
                    )
                    for chunk in chunks
                ],
            )

    def search(
        self, query_vector: list[float], top_k: int, filters: dict
    ) -> list[SearchResult]:
        self.validate_search_args(top_k, filters)
        if self._vector_dimension is None:
            return []  # nothing indexed yet — an empty result, not an error
        if len(query_vector) != self._vector_dimension:
            raise ValueError(
                f"Query vector has dimension {len(query_vector)}, but this index stores "
                f"{self._vector_dimension}-dimensional vectors. Re-index with the same "
                f"embedding model."
            )

        where = ["embedding MATCH ?", "k = ?"]
        params: list[object] = [
            sqlite_vec.serialize_float32(normalize_vector(query_vector)),
            top_k,
        ]
        for key in ("repo_id", "graph_id"):
            if key in filters:
                where.append(f"{key} = ?")
                params.append(filters[key])

        rows = self._connection.execute(
            f"""
            SELECT node_id, node_name, graph_id, file_path, line_start, docstring, distance
            FROM chunks_vec
            WHERE {" AND ".join(where)}
            ORDER BY distance
            """,
            params,
        ).fetchall()

        return [
            SearchResult(
                node_id=row["node_id"],
                node_name=row["node_name"],
                graph_id=row["graph_id"],
                file_path=row["file_path"],
                line_start=row["line_start"],
                # vec0 returns cosine *distance*; the interface promises similarity.
                score=1.0 - float(row["distance"]),
                docstring=row["docstring"],
            )
            for row in rows
        ]

    def get_graph(self, graph_id: str) -> GraphDef | None:
        row = self._connection.execute(
            "SELECT graph_json FROM graphs WHERE graph_id = ?", (graph_id,)
        ).fetchone()
        if row is None:
            return None
        return graph_from_dict(json.loads(row["graph_json"]))

    def delete_repo(self, repo_id: str) -> None:
        with self._connection:
            if self._vector_dimension is not None:
                self._connection.execute(
                    "DELETE FROM chunks_vec WHERE repo_id = ?", (repo_id,)
                )
            self._connection.execute("DELETE FROM graphs WHERE repo_id = ?", (repo_id,))
            self._connection.execute(
                "DELETE FROM repositories WHERE repo_id = ?", (repo_id,)
            )

    def upsert_graphs(
        self,
        repo_id: str,
        root_path: str,
        graphs: list[GraphDef],
        embedding_model: str,
        dimension: int,
        indexed_at: datetime,
    ) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO repositories(
                    repo_id, root_path, last_indexed_at, backend_type, embedding_model, dimension
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id) DO UPDATE SET
                    root_path = excluded.root_path,
                    last_indexed_at = excluded.last_indexed_at,
                    backend_type = excluded.backend_type,
                    embedding_model = excluded.embedding_model,
                    dimension = excluded.dimension
                """,
                (
                    repo_id,
                    root_path,
                    indexed_at.isoformat(),
                    BACKEND_SQLITE,
                    embedding_model,
                    dimension,
                ),
            )
            self._connection.executemany(
                """
                INSERT INTO graphs(graph_id, repo_id, graph_json) VALUES (?, ?, ?)
                ON CONFLICT(graph_id) DO UPDATE SET
                    repo_id = excluded.repo_id,
                    graph_json = excluded.graph_json
                """,
                [
                    (graph.id, repo_id, json.dumps(graph.to_dict()))
                    for graph in graphs
                ],
            )

    def list_graphs(self, repo_id: str) -> list[GraphDef]:
        rows = self._connection.execute(
            "SELECT graph_json FROM graphs WHERE repo_id = ? ORDER BY graph_id", (repo_id,)
        ).fetchall()
        return [graph_from_dict(json.loads(row["graph_json"])) for row in rows]

    def get_repository(self, repo_id: str) -> RepositoryInfo | None:
        row = self._connection.execute(
            "SELECT * FROM repositories WHERE repo_id = ?", (repo_id,)
        ).fetchone()
        if row is None:
            return None
        return RepositoryInfo(
            repo_id=row["repo_id"],
            root_path=row["root_path"],
            last_indexed_at=(
                datetime.fromisoformat(row["last_indexed_at"])
                if row["last_indexed_at"]
                else None
            ),
            backend_type=row["backend_type"],
            embedding_model=row["embedding_model"],
            dimension=row["dimension"],
        )

    def close(self) -> None:
        self._connection.close()

    # ----------------------------------------------------------------------------------------
    # Schema management
    # ----------------------------------------------------------------------------------------
    def _load_extension(self) -> None:
        try:
            self._connection.enable_load_extension(True)
            sqlite_vec.load(self._connection)
        finally:
            self._connection.enable_load_extension(False)

    def _create_base_tables(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS repositories(
                    repo_id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    last_indexed_at TEXT,
                    backend_type TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    dimension INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS graphs(
                    graph_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    graph_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS graphs_repo_id ON graphs(repo_id)"
            )

    def _read_stored_dimension(self) -> int | None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = ?", (_DIMENSION_META_KEY,)
        ).fetchone()
        return int(row["value"]) if row is not None else None

    def _ensure_vector_table(self, dimension: int) -> None:
        """Create the vec0 table at ``dimension``, or verify the existing one matches it.

        A mismatch means the index was built with a different embedding model. Failing loudly is
        the point: silently searching across incompatible vectors returns confident nonsense.
        """
        if self._vector_dimension is not None:
            if self._vector_dimension != dimension:
                raise ValueError(
                    f"This index stores {self._vector_dimension}-dimensional vectors but the "
                    f"current embedding model produces {dimension}. Delete "
                    f"{self._db_path} and re-index, or restore the original model."
                )
            return

        with self._connection:
            self._connection.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding float[{int(dimension)}] distance_metric=cosine,
                    repo_id TEXT,
                    graph_id TEXT,
                    +node_id TEXT,
                    +node_name TEXT,
                    +file_path TEXT,
                    +line_start INTEGER,
                    +line_end INTEGER,
                    +docstring TEXT,
                    +chunk_text TEXT,
                    +embedding_model TEXT
                )
                """
            )
            self._connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (_DIMENSION_META_KEY, str(int(dimension))),
            )
        self._vector_dimension = int(dimension)
