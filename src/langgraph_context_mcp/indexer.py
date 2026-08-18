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
from .parser.repo_scanner import resolve_repo_root, scan_repository
from .storage.base import (
    EmbeddingChunk,
    VectorStore,
    make_chunk_id,
    make_repo_id,
)
from .storage.factory import get_vector_store

logger = logging.getLogger(__name__)

# Cap on the padded volume of one embedding call, as `len(batch) * longest_member_chars`
# (DEC-016). `fastembed` pads every member of a batch to its longest member and attention cost
# scales with `batch_size * max_seq_len**2`, so embedding a whole repo at once is both slower than
# batching it (2.3x on real code) and able to exhaust memory outright — a 273-chunk corpus with one
# 5,521-char node attempted an 18.9 GB allocation and crashed.
#
# 4000 is the measured minimax choice across three chunk-length regimes: within 2% of the best
# strategy on long/variable, short/uniform, and mixed corpora, where no fixed batch size does
# better than 8.8% worst-case. Characters are a deliberate proxy for tokens — this is a bound, not
# a target, so an approximation is safe and costs no tokenizer pass. Calibrated for
# `nomic-embed-text-v1.5` on CPU; re-measure if DEC-003's default model changes.
EMBED_BATCH_CHAR_BUDGET = 4000


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

    Raises ``ValueError`` for an empty path or one containing ``..``, ``FileNotFoundError`` if the
    path does not exist, and ``NotADirectoryError`` if it is not a directory. The latter two map to
    prd.md's ``index_repo`` error cases in the MCP layer; the first is a third case Phase 4 must
    name (DEC-014).
    """
    started = time.perf_counter()
    # Validated here as well as inside scan_repository (DEC-014): failing at the entry point means
    # no store and no embedding model are ever constructed for a path we were never going to scan.
    repo_root = resolve_repo_root(repo_root)
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


def plan_embedding_batches(texts: list[str], budget: int | None = None) -> list[list[int]]:
    """Group ``texts`` into length-sorted batches bounded by a padded-size budget (DEC-016).

    Returns lists of indices into ``texts``. Sorting by length keeps each batch's members
    near-uniform, so little is wasted on padding; the budget caps
    ``len(batch) * longest_member``, which is the padded volume the model materialises and the
    quantity that overflows memory when everything is embedded at once.

    A text larger than the whole budget forms a batch of one. It is still embedded in full —
    this function never drops or truncates a chunk (DEC-005).

    ``budget`` defaults to ``EMBED_BATCH_CHAR_BUDGET``, resolved on each call rather than bound as
    a default argument, so the module constant stays the single source of truth.
    """
    effective_budget = EMBED_BATCH_CHAR_BUDGET if budget is None else budget
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0

    for index in order:
        candidate_longest = max(longest, len(texts[index]))
        if current and (len(current) + 1) * candidate_longest > effective_budget:
            batches.append(current)
            current, longest = [index], len(texts[index])
        else:
            current.append(index)
            longest = candidate_longest

    if current:
        batches.append(current)
    return batches


def _embed_in_batches(texts: list[str], embedder: EmbeddingProvider) -> list[list[float]]:
    """Embed ``texts`` in bounded batches, returning vectors in the caller's original order.

    Batching is invisible to the caller: grouping changes only how the work is issued, never the
    result. The model masks padded positions, so a chunk's vector does not depend on which batch
    it landed in — asserted exactly, not within a tolerance, by the tests for DEC-016.
    """
    batches = plan_embedding_batches(texts)
    logger.debug(
        "Embedding %d chunks in %d batch(es) (budget %d chars)",
        len(texts),
        len(batches),
        EMBED_BATCH_CHAR_BUDGET,
    )

    vectors: list[list[float] | None] = [None] * len(texts)
    for batch in batches:
        embedded = embedder.embed([texts[index] for index in batch])
        if len(embedded) != len(batch):
            raise RuntimeError(
                f"Embedding provider returned {len(embedded)} vectors for {len(batch)} chunks."
            )
        for index, vector in zip(batch, embedded):
            vectors[index] = vector

    missing = [index for index, vector in enumerate(vectors) if vector is None]
    if missing:
        raise RuntimeError(f"No embedding produced for chunk index(es) {missing}.")
    return [vector for vector in vectors if vector is not None]


def _build_chunks(
    graphs: list[GraphDef], repo_root: Path, embedder: EmbeddingProvider
) -> list[EmbeddingChunk]:
    """Build one chunk per node and embed them in bounded batches (DEC-016)."""
    source_cache: dict[str, list[str] | None] = {}
    nodes: list[tuple[GraphDef, NodeDef]] = [
        (graph, node) for graph in graphs for node in graph.nodes
    ]
    if not nodes:
        return []

    texts = [build_chunk_text(node, repo_root, source_cache) for _graph, node in nodes]
    vectors = _embed_in_batches(texts, embedder)

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
        # utf-8-sig strips a leading UTF-8 BOM if present (DEC-022), same reasoning as
        # ast_walker._parse_file: a no-op when absent, so this never changes non-BOM output.
        lines = candidate.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read %s for chunking (%s)", source_file, exc)
        lines = None

    if source_cache is not None:
        source_cache[source_file] = lines
    return lines
