"""Opt-in storage backend: PostgreSQL with the ``pgvector`` extension and an HNSW index.

Activated only when ``DATABASE_URL`` is set (DEC-002); the SQLite backend remains the
zero-config default. This backend is for users who already run Postgres — it is never required
to try the tool.

Schema mirrors the SQLite backend exactly (DEC-011), with table names prefixed
``langgraph_context_`` so the index can live in an existing database without colliding with the
user's own tables. Everything — extension, tables, and the HNSW index — is created on first
connection if absent; a v1 tool this small does not ask users to run migrations.

The HNSW index is declared ``vector_cosine_ops`` and every vector is unit-normalized before it
is written or queried, so this backend's approximate search and SQLite's brute-force scan rank
the same way rather than accidentally differing (RISK-004).

This module is imported lazily by ``factory.py``: ``psycopg`` and ``pgvector`` ship in the
optional ``[pgvector]`` extra, so a default SQLite install must never import it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from ..parser.graph_model import GraphDef
from .base import (
    BACKEND_PGVECTOR,
    EmbeddingChunk,
    RepositoryInfo,
    SearchResult,
    VectorStore,
    graph_from_dict,
    normalize_vector,
)

logger = logging.getLogger(__name__)

# Prefixed so this tool's tables are unmistakable inside a shared database.
REPOSITORIES_TABLE = "langgraph_context_repositories"
GRAPHS_TABLE = "langgraph_context_graphs"
CHUNKS_TABLE = "langgraph_context_chunks"
META_TABLE = "langgraph_context_meta"
HNSW_INDEX = "langgraph_context_chunks_embedding_hnsw"

# Seconds to wait for a TCP connection before failing. Without this, an unreachable host makes
# the client hang indefinitely instead of reporting a clear error (Phase 3 scenario 3.13).
CONNECT_TIMEOUT_SECONDS = 10

_DIMENSION_META_KEY = "vector_dimension"


class PgvectorStore(VectorStore):
    """``VectorStore`` backed by PostgreSQL + ``pgvector``."""

    def __init__(self, database_url: str, dimension: int | None = None) -> None:
        """Connect to ``database_url`` and ensure the schema exists.

        Raises ``psycopg.OperationalError`` with the driver's own message if the server is
        unreachable — a clear, immediate failure rather than a hang or a half-working store.
        """
        self._connection = psycopg.connect(
            database_url, connect_timeout=CONNECT_TIMEOUT_SECONDS, autocommit=False
        )
        self._ensure_extension()
        register_vector(self._connection)
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

        with self._connection.cursor() as cursor:
            cursor.executemany(
                f"""
                INSERT INTO {CHUNKS_TABLE}(
                    chunk_id, node_id, repo_id, graph_id, node_name, file_path, line_start,
                    line_end, docstring, chunk_text, embedding_model, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    node_id = EXCLUDED.node_id,
                    repo_id = EXCLUDED.repo_id,
                    graph_id = EXCLUDED.graph_id,
                    node_name = EXCLUDED.node_name,
                    file_path = EXCLUDED.file_path,
                    line_start = EXCLUDED.line_start,
                    line_end = EXCLUDED.line_end,
                    docstring = EXCLUDED.docstring,
                    chunk_text = EXCLUDED.chunk_text,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding
                """,
                [
                    (
                        chunk.id,
                        chunk.node_id,
                        chunk.repo_id,
                        chunk.graph_id,
                        chunk.node_name,
                        chunk.file_path,
                        chunk.line_start,
                        chunk.line_end,
                        chunk.docstring,
                        chunk.text,
                        chunk.embedding_model,
                        Vector(normalize_vector(list(chunk.vector))),
                    )
                    for chunk in chunks
                ],
            )
        self._connection.commit()

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

        vector = Vector(normalize_vector(query_vector))
        conditions = []
        params: list[object] = [vector]
        for key in ("repo_id", "graph_id"):
            if key in filters:
                conditions.append(f"{key} = %s")
                params.append(filters[key])
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([vector, top_k])

        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT node_id, node_name, graph_id, file_path, line_start, docstring,
                       1 - (embedding <=> %s) AS score
                FROM {CHUNKS_TABLE}
                {where}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                params,
            )
            rows = cursor.fetchall()

        return [
            SearchResult(
                node_id=row[0],
                node_name=row[1],
                graph_id=row[2],
                file_path=row[3],
                line_start=row[4],
                docstring=row[5],
                score=float(row[6]),
            )
            for row in rows
        ]

    def get_graph(self, graph_id: str) -> GraphDef | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT graph_json FROM {GRAPHS_TABLE} WHERE graph_id = %s", (graph_id,)
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return graph_from_dict(_as_dict(row[0]))

    def delete_repo(self, repo_id: str) -> None:
        with self._connection.cursor() as cursor:
            if self._vector_dimension is not None:
                cursor.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE repo_id = %s", (repo_id,))
            cursor.execute(f"DELETE FROM {GRAPHS_TABLE} WHERE repo_id = %s", (repo_id,))
            cursor.execute(f"DELETE FROM {REPOSITORIES_TABLE} WHERE repo_id = %s", (repo_id,))
        self._connection.commit()

    def upsert_graphs(
        self,
        repo_id: str,
        root_path: str,
        graphs: list[GraphDef],
        embedding_model: str,
        dimension: int,
        indexed_at: datetime,
    ) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {REPOSITORIES_TABLE}(
                    repo_id, root_path, last_indexed_at, backend_type, embedding_model, dimension
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (repo_id) DO UPDATE SET
                    root_path = EXCLUDED.root_path,
                    last_indexed_at = EXCLUDED.last_indexed_at,
                    backend_type = EXCLUDED.backend_type,
                    embedding_model = EXCLUDED.embedding_model,
                    dimension = EXCLUDED.dimension
                """,
                (
                    repo_id,
                    root_path,
                    indexed_at,
                    BACKEND_PGVECTOR,
                    embedding_model,
                    dimension,
                ),
            )
            cursor.executemany(
                f"""
                INSERT INTO {GRAPHS_TABLE}(graph_id, repo_id, graph_json) VALUES (%s, %s, %s)
                ON CONFLICT (graph_id) DO UPDATE SET
                    repo_id = EXCLUDED.repo_id,
                    graph_json = EXCLUDED.graph_json
                """,
                [
                    (graph.id, repo_id, json.dumps(graph.to_dict()))
                    for graph in graphs
                ],
            )
        self._connection.commit()

    def list_graphs(self, repo_id: str) -> list[GraphDef]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT graph_json FROM {GRAPHS_TABLE} WHERE repo_id = %s ORDER BY graph_id",
                (repo_id,),
            )
            rows = cursor.fetchall()
        return [graph_from_dict(_as_dict(row[0])) for row in rows]

    def get_repository(self, repo_id: str) -> RepositoryInfo | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT repo_id, root_path, last_indexed_at, backend_type, embedding_model,
                       dimension
                FROM {REPOSITORIES_TABLE} WHERE repo_id = %s
                """,
                (repo_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return RepositoryInfo(
            repo_id=row[0],
            root_path=row[1],
            last_indexed_at=row[2],
            backend_type=row[3],
            embedding_model=row[4],
            dimension=row[5],
        )

    def close(self) -> None:
        if not self._connection.closed:
            self._connection.close()

    # ----------------------------------------------------------------------------------------
    # Schema management
    # ----------------------------------------------------------------------------------------
    def _ensure_extension(self) -> None:
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._connection.commit()
        except psycopg.errors.InsufficientPrivilege as exc:
            self._connection.rollback()
            raise psycopg.errors.InsufficientPrivilege(
                "The 'vector' extension is not installed and this role may not create it. "
                "Ask a superuser to run: CREATE EXTENSION vector;"
            ) from exc

    def _create_base_tables(self) -> None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {META_TABLE}(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {REPOSITORIES_TABLE}(
                    repo_id TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    last_indexed_at TIMESTAMPTZ,
                    backend_type TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    dimension INTEGER NOT NULL
                )
                """
            )
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {GRAPHS_TABLE}(
                    graph_id TEXT PRIMARY KEY,
                    repo_id TEXT NOT NULL,
                    graph_json JSONB NOT NULL
                )
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {GRAPHS_TABLE}_repo_id ON {GRAPHS_TABLE}(repo_id)"
            )
        self._connection.commit()

    def _read_stored_dimension(self) -> int | None:
        with self._connection.cursor() as cursor:
            cursor.execute(
                f"SELECT value FROM {META_TABLE} WHERE key = %s", (_DIMENSION_META_KEY,)
            )
            row = cursor.fetchone()
        return int(row[0]) if row is not None else None

    def _ensure_vector_table(self, dimension: int) -> None:
        """Create the chunks table and its HNSW index at ``dimension``, or verify a match.

        A mismatch means the index was built with a different embedding model; failing loudly
        beats searching across incompatible vectors and returning confident nonsense.
        """
        if self._vector_dimension is not None:
            if self._vector_dimension != dimension:
                raise ValueError(
                    f"This index stores {self._vector_dimension}-dimensional vectors but the "
                    f"current embedding model produces {dimension}. Drop the "
                    f"{CHUNKS_TABLE} table and re-index, or restore the original model."
                )
            return

        with self._connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE}(
                    chunk_id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    graph_id TEXT NOT NULL,
                    node_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    docstring TEXT,
                    chunk_text TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding vector({int(dimension)}) NOT NULL
                )
                """
            )
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {CHUNKS_TABLE}_repo_id ON {CHUNKS_TABLE}(repo_id)"
            )
            # Cosine explicitly — pgvector's default operator class is L2 (RISK-004).
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {HNSW_INDEX}
                ON {CHUNKS_TABLE} USING hnsw (embedding vector_cosine_ops)
                """
            )
            cursor.execute(
                f"""
                INSERT INTO {META_TABLE}(key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (_DIMENSION_META_KEY, str(int(dimension))),
            )
        self._connection.commit()
        self._vector_dimension = int(dimension)


def _as_dict(value: object) -> dict:
    """A ``JSONB`` column comes back already decoded; tolerate a raw string too."""
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise TypeError(f"Unexpected graph_json payload type: {type(value)!r}")
