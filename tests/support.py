"""Shared test helpers: the embedding double and the graph/chunk builders.

Kept out of ``conftest.py`` so that both ``conftest.py`` (which defines fixtures) and
``vector_store_contract.py`` (which defines the shared backend contract) can import them without
importing each other.
"""

from __future__ import annotations

import zlib
from datetime import UTC, datetime
from pathlib import Path

from langgraph_context_mcp.embeddings.base import EmbeddingProvider
from langgraph_context_mcp.parser.graph_model import (
    CONDITION_VALUE_KNOWN,
    EDGE_CONDITIONAL,
    EDGE_NORMAL,
    RESOLUTION_FULL,
    TOOL_RESOLUTION_NOT_APPLICABLE,
    ConditionalRoute,
    EdgeDef,
    GraphDef,
    NodeDef,
    make_edge_id,
    make_graph_id,
    make_node_id,
    make_route_id,
)
from langgraph_context_mcp.storage.base import (
    EmbeddingChunk,
    VectorStore,
    make_chunk_id,
)

# Every store fixture and hand-built vector in the suite uses this width. The pgvector backend
# fixes its column width the first time the table is created, so the whole suite must agree.
TEST_DIMENSION = 16

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_graphs"

FAKE_MODEL_NAME = "fake-hashing-embedder"


class FakeEmbeddingProvider(EmbeddingProvider):
    """A deterministic bag-of-words hashing embedder — no model, no download, no network.

    Similar text lands on similar vectors, which is enough to assert ranking behaviour. Uses
    ``crc32`` rather than ``hash()`` because Python randomizes string hashing per process, and a
    test that changes its answer between runs is worse than no test.
    """

    def __init__(self, dimension: int = TEST_DIMENSION) -> None:
        self._dimension = dimension
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return FAKE_MODEL_NAME

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in text.lower().replace("_", " ").replace(".", " ").split():
            vector[zlib.crc32(token.encode("utf-8")) % self._dimension] += 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector


def build_graph(repo_id: str, token: str, node_names: list[str]) -> GraphDef:
    """A small but complete ``GraphDef``: every node, one normal edge, and one conditional edge
    with its mirroring route, so round-trip tests cover every child collection."""
    file_path = f"{token}/agent.py"
    graph_id = make_graph_id(file_path, "graph")

    nodes = tuple(
        NodeDef(
            id=make_node_id(graph_id, name),
            graph_id=graph_id,
            name=name,
            source_file=file_path,
            line_start=10 + index,
            line_end=20 + index,
            docstring=f"Docstring for {name}.",
            function_body_hash=f"hash-{name}",
            resolution=RESOLUTION_FULL,
            tool_resolution=TOOL_RESOLUTION_NOT_APPLICABLE,
        )
        for index, name in enumerate(node_names)
    )

    edges: list[EdgeDef] = []
    routes: list[ConditionalRoute] = []
    if len(node_names) >= 2:
        first, second = node_names[0], node_names[1]
        normal_id = make_edge_id(graph_id, first, second, EDGE_NORMAL)
        edges.append(
            EdgeDef(
                id=normal_id,
                graph_id=graph_id,
                source_node_id=make_node_id(graph_id, first),
                target_node_id=make_node_id(graph_id, second),
                type=EDGE_NORMAL,
                condition_function_name=None,
            )
        )
        conditional_id = make_edge_id(graph_id, second, first, EDGE_CONDITIONAL, "retry")
        edges.append(
            EdgeDef(
                id=conditional_id,
                graph_id=graph_id,
                source_node_id=make_node_id(graph_id, second),
                target_node_id=make_node_id(graph_id, first),
                type=EDGE_CONDITIONAL,
                condition_function_name="should_retry",
            )
        )
        routes.append(
            ConditionalRoute(
                id=make_route_id(conditional_id),
                edge_id=conditional_id,
                condition_value="retry",
                target_node_id=make_node_id(graph_id, first),
                value_resolution=CONDITION_VALUE_KNOWN,
            )
        )

    return GraphDef(
        id=graph_id,
        repo_id=repo_id,
        file_path=file_path,
        variable_name="graph",
        entry_point=node_names[0] if node_names else None,
        nodes=nodes,
        edges=tuple(edges),
        conditional_routes=tuple(routes),
        tool_bindings=(),
    )


def one_hot(index: int, dimension: int = TEST_DIMENSION) -> list[float]:
    """A unit vector along one axis — two of these are orthogonal, so cosine similarity is 0."""
    vector = [0.0] * dimension
    vector[index % dimension] = 1.0
    return vector


def build_chunk(
    graph: GraphDef,
    node: NodeDef,
    vector: list[float],
    text: str = "chunk text",
    embedding_model: str = FAKE_MODEL_NAME,
) -> EmbeddingChunk:
    return EmbeddingChunk(
        id=make_chunk_id(node.id),
        node_id=node.id,
        vector=tuple(vector),
        embedding_model=embedding_model,
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


def store_graph(
    store: VectorStore, graph: GraphDef, dimension: int = TEST_DIMENSION
) -> None:
    """Persist one graph plus its repository row, the way the indexer does."""
    store.upsert_graphs(
        repo_id=graph.repo_id,
        root_path=f"/tmp/{graph.repo_id}",
        graphs=[graph],
        embedding_model=FAKE_MODEL_NAME,
        dimension=dimension,
        indexed_at=datetime.now(UTC),
    )


def configured_backend() -> str:
    """The backend name the factory will actually choose, per DEC-002's rule.

    Tests that assert on the reported backend must respect the ambient ``DATABASE_URL`` rather
    than hardcoding ``"sqlite"``. CI sets that variable job-wide, so a hardcoded expectation
    fails there while passing on a developer machine — which is exactly how QA-5-04 went
    unnoticed from Phase 4 until Phase 5.
    """
    import os

    return "pgvector" if (os.environ.get("DATABASE_URL") or "").strip() else "sqlite"


def requires_sqlite_backend() -> None:
    """Skip a test that is inherently SQLite-specific when Postgres is configured.

    The same clean-skip convention ``conftest.pgvector_store`` uses in the other direction, so
    backend-specific tests never silently assume the default.
    """
    import os

    import pytest

    if (os.environ.get("DATABASE_URL") or "").strip():
        pytest.skip("DATABASE_URL is set — this test inspects the on-disk SQLite index file")
