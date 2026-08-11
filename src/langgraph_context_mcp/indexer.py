"""Ties the pipeline together: parse -> chunk -> embed -> store.

This is the composition root of the engine. It is also the only place that decides *what* text
represents a node: one chunk per ``NodeDef`` (DEC-005), built from the node's own line span
(DEC-012). Both the CLI and the MCP tools call into here rather than re-implementing indexing.

Nothing is cached in a module-level global. The store and the embedding provider are injected —
callers that pass neither get the configured defaults, and tests pass fakes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .embeddings.base import EmbeddingProvider
from .embeddings.nomic_provider import NomicEmbeddingProvider
from .parser.graph_model import RESOLUTION_PARTIAL, GraphDef, NodeDef
from .parser.repo_scanner import scan_repository
from .storage.base import (
    EmbeddingChunk,
    VectorStore,
    make_chunk_id,
    make_repo_id,
)
from .storage.factory import get_vector_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexResult:
    """Outcome of one indexing run — the exact shape prd.md's ``index_repo`` tool returns."""

    graphs_found: int
    nodes_indexed: int
    edges_indexed: int
    partial_nodes: int
    backend: str
    duration_ms: int

    def to_dict(self) -> dict:
        return {
            "graphs_found": self.graphs_found,
            "nodes_indexed": self.nodes_indexed,
            "edges_indexed": self.edges_indexed,
            "partial_nodes": self.partial_nodes,
            "backend": self.backend,
            "duration_ms": self.duration_ms,
        }


def index_repository(
    repo_root: Path,
    store: VectorStore | None = None,
    embedder: EmbeddingProvider | None = None,
) -> IndexResult:
    """Scan ``repo_root``, embed every graph node, and persist the result.

    Passing ``store`` or ``embedder`` overrides the configured defaults (``DATABASE_URL`` decides
    the backend; embeddings are local). When this function creates the store itself it also
    closes it; an injected store belongs to the caller and is left open.

    Indexing is idempotent: the repository's existing rows are removed first, so re-running
    produces neither duplicates nor stale rows for nodes that no longer exist (DEC-011). A
    repository with no LangGraph usage is a valid outcome — it is recorded as indexed with zero
    graphs, not an error.

    Raises ``FileNotFoundError`` if the path does not exist and ``NotADirectoryError`` if it is
    not a directory; both map to prd.md's ``index_repo`` error cases in the MCP layer.
    """
    started = time.perf_counter()
    repo_root = Path(repo_root).resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Path does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {repo_root}")

    graphs = scan_repository(repo_root)
    repo_id = make_repo_id(repo_root)
    embedder = embedder or NomicEmbeddingProvider()

    owns_store = store is None
    store = store or get_vector_store(repo_root, dimension=embedder.dimension)
    try:
        chunks = _build_chunks(graphs, repo_root, embedder)
        store.delete_repo(repo_id)
        store.upsert_graphs(
            repo_id=repo_id,
            root_path=repo_root.as_posix(),
            graphs=graphs,
            embedding_model=embedder.model_name,
            dimension=embedder.dimension,
            indexed_at=datetime.now(UTC),
        )
        store.upsert_chunks(chunks)
        repository = store.get_repository(repo_id)
        backend = repository.backend_type if repository else "unknown"
    finally:
        if owns_store:
            store.close()

    return IndexResult(
        graphs_found=len(graphs),
        nodes_indexed=sum(len(graph.nodes) for graph in graphs),
        edges_indexed=sum(len(graph.edges) for graph in graphs),
        partial_nodes=sum(
            1
            for graph in graphs
            for node in graph.nodes
            if node.resolution == RESOLUTION_PARTIAL
        ),
        backend=backend,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


# --------------------------------------------------------------------------------------------
# Chunking (DEC-005 / DEC-012)
# --------------------------------------------------------------------------------------------
def build_chunk_text(node: NodeDef, repo_root: Path, source_cache: dict | None = None) -> str:
    """Return the text embedded for ``node``: its decorators, signature, docstring, and body.

    Read as the source lines ``[line_start, line_end]``, which is the whole of DEC-005's chunk:
    since DEC-013 the parser's ``line_start`` is the node's first decorator, and the span already
    contains the docstring (it is the function's first statement), so all three components are
    present exactly once with nothing concatenated twice.

    Falls back to the docstring, then to the node name, if the file cannot be read; every node
    always yields non-empty text.
    """
    lines = _source_lines(node.source_file, repo_root, source_cache)
    if lines:
        start = max(node.line_start - 1, 0)
        end = min(node.line_end, len(lines))
        text = "\n".join(line.rstrip() for line in lines[start:end]).strip()
        if text:
            return text

    if node.docstring:
        return node.docstring
    return node.name


def _build_chunks(
    graphs: list[GraphDef], repo_root: Path, embedder: EmbeddingProvider
) -> list[EmbeddingChunk]:
    """Build one chunk per node and embed them all in a single batched call."""
    source_cache: dict[str, list[str] | None] = {}
    nodes: list[tuple[GraphDef, NodeDef]] = [
        (graph, node) for graph in graphs for node in graph.nodes
    ]
    if not nodes:
        return []

    texts = [build_chunk_text(node, repo_root, source_cache) for _graph, node in nodes]
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding provider returned {len(vectors)} vectors for {len(texts)} chunks."
        )

    return [
        EmbeddingChunk(
            id=make_chunk_id(node.id),
            node_id=node.id,
            vector=tuple(vector),
            embedding_model=embedder.model_name,
            dimension=len(vector),
            repo_id=graph.repo_id,
            graph_id=graph.id,
            node_name=node.name,
            file_path=node.source_file,
            line_start=node.line_start,
            line_end=node.line_end,
            docstring=node.docstring,
            text=text,
        )
        for (graph, node), text, vector in zip(nodes, texts, vectors)
    ]


def _source_lines(
    source_file: str, repo_root: Path, source_cache: dict | None
) -> list[str] | None:
    """Read a node's source file once per index run. ``source_file`` is repo-relative when the
    parser could make it so, absolute otherwise."""
    if source_cache is not None and source_file in source_cache:
        return source_cache[source_file]

    candidate = repo_root / source_file
    if not candidate.exists():
        candidate = Path(source_file)

    lines: list[str] | None
    try:
        lines = candidate.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s for chunking (%s)", source_file, exc)
        lines = None

    if source_cache is not None:
        source_cache[source_file] = lines
    return lines
