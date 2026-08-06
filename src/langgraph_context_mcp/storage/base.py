"""Backend-agnostic vector storage interface and the value objects that cross it.

Two backends implement ``VectorStore``: ``sqlite_store.SqliteStore`` (the zero-config default)
and ``pgvector_store.PgvectorStore`` (opt-in via ``DATABASE_URL``). Per DEC-002 they must behave
identically, which task 2.8's shared contract test enforces — so any behaviour that both must
share lives here, not duplicated in each backend.

The physical schema behind these methods is fixed by DEC-011:
  - graph structure is persisted as one JSON document per ``GraphDef``, keyed by ``graph_id``
  - chunk rows denormalize the node metadata a search result needs, so one query answers a search
  - every vector is L2-normalized here (``normalize_vector``) and both backends are explicitly
    configured for cosine distance, so their rankings agree by construction (RISK-004)

No module outside this package may talk to ``sqlite3`` or ``psycopg`` directly (claude.md).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Self

from ..parser.graph_model import (
    ConditionalRoute,
    EdgeDef,
    GraphDef,
    NodeDef,
    ToolBinding,
)

# Backend identifiers, persisted with the repository row and reported by the index/status output.
BACKEND_SQLITE = "sqlite"
BACKEND_PGVECTOR = "pgvector"

# The filter keys ``search()`` understands. Anything else is a caller error, not a silent no-op:
# a filter that is quietly ignored returns confident, wrong results.
SUPPORTED_SEARCH_FILTERS = frozenset({"repo_id", "graph_id"})


def make_repo_id(repo_root: Path) -> str:
    """Stable repository ID: the resolved absolute path, POSIX-style.

    Must match the ``repo_id`` the parser stamps onto every ``GraphDef`` (see ``ast_walker``),
    so chunks, graphs, and the repository row all key on the same value.
    """
    return repo_root.resolve().as_posix()


def make_chunk_id(node_id: str) -> str:
    """Stable chunk ID. One chunk per node (DEC-005), so this is a pure function of the node."""
    return f"{node_id}::chunk"


def normalize_vector(vector: list[float]) -> list[float]:
    """Return ``vector`` scaled to unit length.

    Applied on every write and every query by both backends. On unit vectors cosine distance is
    a pure function of the dot product, which is what makes ``sqlite-vec``'s brute-force scan and
    ``pgvector``'s HNSW index rank identically instead of coincidentally (RISK-004). A
    zero-length vector has no direction to preserve and is returned unchanged.
    """
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        return list(vector)
    return [component / magnitude for component in vector]


def graph_from_dict(payload: dict) -> GraphDef:
    """Rebuild a ``GraphDef`` from the document written by ``GraphDef.to_dict()`` (DEC-011).

    Lives here rather than on the dataclass because persistence is the storage layer's concern
    and both backends must rehydrate identically. Round-trips exactly: the dataclasses are frozen
    and ``to_dict()`` emits every field.
    """
    return GraphDef(
        id=payload["id"],
        repo_id=payload["repo_id"],
        file_path=payload["file_path"],
        variable_name=payload["variable_name"],
        entry_point=payload["entry_point"],
        nodes=tuple(NodeDef(**node) for node in payload["nodes"]),
        edges=tuple(EdgeDef(**edge) for edge in payload["edges"]),
        conditional_routes=tuple(
            ConditionalRoute(**route) for route in payload["conditional_routes"]
        ),
        tool_bindings=tuple(ToolBinding(**binding) for binding in payload["tool_bindings"]),
    )


@dataclass(frozen=True)
class EmbeddingChunk:
    """One embedded node — prd.md's EmbeddingChunk plus denormalized retrieval metadata.

    The first five fields are prd.md's model. The rest are carried on the row so ``search()`` can
    answer ``semantic_search_nodes``'s full output contract without a second lookup (DEC-011).
    ``vector`` is stored normalized.
    """

    id: str
    node_id: str
    vector: tuple[float, ...]
    embedding_model: str
    dimension: int
    repo_id: str
    graph_id: str
    node_name: str
    file_path: str
    line_start: int
    line_end: int
    docstring: str | None
    text: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "vector": list(self.vector),
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "repo_id": self.repo_id,
            "graph_id": self.graph_id,
            "node_name": self.node_name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "docstring": self.docstring,
            "text": self.text,
        }


@dataclass(frozen=True)
class SearchResult:
    """One semantic search hit. ``score`` is cosine similarity: higher is closer, max 1.0."""

    node_id: str
    node_name: str
    graph_id: str
    file_path: str
    line_start: int
    score: float
    docstring: str | None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "graph_id": self.graph_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "score": self.score,
            "docstring": self.docstring,
        }


@dataclass(frozen=True)
class RepositoryInfo:
    """prd.md's Repository model, plus what the index was built with.

    ``embedding_model`` and ``dimension`` are persisted so a later query with a different model
    is detected instead of silently producing meaningless rankings.
    """

    repo_id: str
    root_path: str
    last_indexed_at: datetime | None
    backend_type: str
    embedding_model: str
    dimension: int

    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "root_path": self.root_path,
            "last_indexed_at": (
                self.last_indexed_at.isoformat() if self.last_indexed_at else None
            ),
            "backend_type": self.backend_type,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
        }


class VectorStore(ABC):
    """Persistence for graph structure and node embeddings.

    Implementations own a connection and must be closed (or used as a context manager). They are
    injected, never reached for as a singleton (claude.md).
    """

    # ----------------------------------------------------------------------------------------
    # The contract from task 2.3
    # ----------------------------------------------------------------------------------------
    @abstractmethod
    def upsert_chunks(self, chunks: list[EmbeddingChunk]) -> None:
        """Insert or replace chunks, keyed on ``chunk.id``. Re-running with the same input is a
        no-op in effect, never a duplicate row."""

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int, filters: dict
    ) -> list[SearchResult]:
        """Return the ``top_k`` nearest chunks by cosine similarity, best first.

        ``filters`` accepts the keys in ``SUPPORTED_SEARCH_FILTERS``; any other key raises
        ``ValueError`` rather than being ignored.
        """

    @abstractmethod
    def get_graph(self, graph_id: str) -> GraphDef | None:
        """Rehydrate one stored graph, or ``None`` if that ``graph_id`` is not indexed."""

    @abstractmethod
    def delete_repo(self, repo_id: str) -> None:
        """Remove a repository's row plus all of its graphs and chunks. Missing repo = no-op."""

    # ----------------------------------------------------------------------------------------
    # Supporting members (DEC-011): each exists because a prd.md contract cannot be served
    # without it, not as new scope.
    # ----------------------------------------------------------------------------------------
    @abstractmethod
    def upsert_graphs(
        self,
        repo_id: str,
        root_path: str,
        graphs: list[GraphDef],
        embedding_model: str,
        dimension: int,
        indexed_at: datetime,
    ) -> None:
        """Persist a repository row and its graph documents. Without this ``get_graph`` could
        never return anything."""

    @abstractmethod
    def list_graphs(self, repo_id: str) -> list[GraphDef]:
        """Every stored graph for a repository. ``get_graph_summary`` enumerates; ``get_graph``
        alone cannot."""

    @abstractmethod
    def get_repository(self, repo_id: str) -> RepositoryInfo | None:
        """The stored repository row (prd.md's Repository model), or ``None`` if never indexed."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection. Safe to call more than once."""

    # ----------------------------------------------------------------------------------------
    # Concrete helpers — defined once here so the two backends cannot drift
    # ----------------------------------------------------------------------------------------
    def is_indexed(self, repo_id: str) -> bool:
        """Whether this repository has been indexed. Drives prd.md's ``not_indexed`` error."""
        return self.get_repository(repo_id) is not None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @staticmethod
    def validate_search_args(top_k: int, filters: dict) -> None:
        """Shared argument validation, so both backends reject the same inputs identically."""
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        unknown = set(filters) - SUPPORTED_SEARCH_FILTERS
        if unknown:
            raise ValueError(
                f"Unsupported search filter(s): {sorted(unknown)}. "
                f"Supported: {sorted(SUPPORTED_SEARCH_FILTERS)}"
            )
