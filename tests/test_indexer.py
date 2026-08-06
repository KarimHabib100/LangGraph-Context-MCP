"""End-to-end pipeline tests: parse -> chunk -> embed -> store.

Most tests use ``FakeEmbeddingProvider`` because they are about pipeline wiring, not embedding
quality. The final test is the exception and the reason this phase exists: it runs the real
local ``nomic-embed-text-v1.5`` model and asserts that a natural-language query finds the right
node — Phase 2's semantic-search exit criterion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from support import TEST_DIMENSION, FakeEmbeddingProvider

from langgraph_context_mcp.embeddings.nomic_provider import (
    DEFAULT_MODEL_DIMENSION,
    DEFAULT_MODEL_NAME,
    NomicEmbeddingProvider,
)
from langgraph_context_mcp.indexer import build_chunk_text, index_repository
from langgraph_context_mcp.storage.base import make_repo_id
from langgraph_context_mcp.storage.sqlite_store import SqliteStore, default_index_path


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------
def test_indexes_the_fixture_repo_end_to_end(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """The fixture's 4 nodes and 5 edges (3 normal + 2 conditional branches) are all indexed."""
    result = index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)

    assert result.graphs_found == 1
    assert result.nodes_indexed == 4
    assert result.edges_indexed == 5
    assert result.partial_nodes == 0
    assert result.backend == "sqlite"
    assert result.duration_ms >= 0
    assert set(result.to_dict()) == {
        "graphs_found",
        "nodes_indexed",
        "edges_indexed",
        "partial_nodes",
        "backend",
        "duration_ms",
    }


def test_indexed_nodes_are_searchable(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)
    repo_id = make_repo_id(fixture_repo)

    results = sqlite_store.search(
        fake_embedder.embed(["fetch the requested data"])[0],
        top_k=4,
        filters={"repo_id": repo_id},
    )

    assert len(results) == 4
    assert {result.node_name for result in results} == {
        "check_auth_token",
        "fetch_data",
        "format_response",
        "handle_error",
    }


def test_graph_structure_is_persisted(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """The stored graph is the parser's output, unchanged — Phase 4 reads structure from here."""
    index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)
    repo_id = make_repo_id(fixture_repo)

    graphs = sqlite_store.list_graphs(repo_id)

    assert len(graphs) == 1
    graph = graphs[0]
    assert graph.variable_name == "graph"
    assert graph.entry_point == "check_auth_token"
    assert {node.name for node in graph.nodes} == {
        "check_auth_token",
        "fetch_data",
        "format_response",
        "handle_error",
    }
    assert {route.condition_value for route in graph.conditional_routes} == {
        "authorized",
        "unauthorized",
    }
    assert sqlite_store.get_graph(graph.id) == graph


def test_embedding_is_one_batched_call(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """All chunks go through the provider in a single call, per task 2.7."""
    index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)

    assert len(fake_embedder.embed_calls) == 1
    assert len(fake_embedder.embed_calls[0]) == 4


def test_reindexing_is_idempotent(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """Indexing twice leaves one row per node and one row per graph (Phase 3 scenario 3.9)."""
    first = index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)
    second = index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)
    repo_id = make_repo_id(fixture_repo)

    assert first.to_dict()["nodes_indexed"] == second.to_dict()["nodes_indexed"]
    assert len(sqlite_store.list_graphs(repo_id)) == 1
    results = sqlite_store.search(
        fake_embedder.embed(["anything"])[0], top_k=50, filters={"repo_id": repo_id}
    )
    assert len(results) == 4


def test_reindexing_drops_nodes_that_no_longer_exist(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """Stale rows are removed, not stranded — a plain upsert would leave the deleted node."""
    index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)
    repo_id = make_repo_id(fixture_repo)

    graph_file = fixture_repo / "simple_graph.py"
    source = graph_file.read_text(encoding="utf-8")
    graph_file.write_text(
        source.replace('graph.add_node("handle_error", handle_error)\n', ""),
        encoding="utf-8",
    )
    index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)

    results = sqlite_store.search(
        fake_embedder.embed(["anything"])[0], top_k=50, filters={"repo_id": repo_id}
    )
    assert "handle_error" not in {result.node_name for result in results}
    assert len(results) == 3


def test_zero_config_run_creates_the_index_under_the_repo(
    fixture_repo: Path, fake_embedder: FakeEmbeddingProvider, monkeypatch
) -> None:
    """With no DATABASE_URL, indexing writes ``.langgraph-context/index.db`` and nothing else."""
    monkeypatch.delenv("DATABASE_URL", raising=False)

    result = index_repository(fixture_repo, embedder=fake_embedder)

    assert result.backend == "sqlite"
    assert default_index_path(fixture_repo).exists()

    store = SqliteStore(default_index_path(fixture_repo))
    try:
        assert store.is_indexed(make_repo_id(fixture_repo)) is True
    finally:
        store.close()


def test_repo_with_no_langgraph_usage_is_not_an_error(
    tmp_path: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """An empty result is a valid outcome: recorded as indexed with zero graphs, per prd.md."""
    (tmp_path / "plain.py").write_text("def hello():\n    return 1\n", encoding="utf-8")

    result = index_repository(tmp_path, store=sqlite_store, embedder=fake_embedder)

    assert result.graphs_found == 0
    assert result.nodes_indexed == 0
    assert sqlite_store.is_indexed(make_repo_id(tmp_path)) is True
    assert sqlite_store.list_graphs(make_repo_id(tmp_path)) == []


def test_syntax_error_file_does_not_stop_the_scan(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """One broken file must not cost the rest of the repository (RISK-001)."""
    (fixture_repo / "broken.py").write_text("def oops(:\n", encoding="utf-8")

    result = index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)

    assert result.graphs_found == 1
    assert result.nodes_indexed == 4


def test_missing_path_raises_file_not_found(
    tmp_path: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    with pytest.raises(FileNotFoundError):
        index_repository(tmp_path / "nope", store=sqlite_store, embedder=fake_embedder)


def test_file_path_raises_not_a_directory(
    tmp_path: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    target = tmp_path / "a_file.py"
    target.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        index_repository(target, store=sqlite_store, embedder=fake_embedder)


# --------------------------------------------------------------------------------------------
# Chunk construction (DEC-005 / DEC-012)
# --------------------------------------------------------------------------------------------
def test_chunk_text_is_docstring_decorators_and_body(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)
    graph = sqlite_store.list_graphs(make_repo_id(fixture_repo))[0]
    node = next(node for node in graph.nodes if node.name == "check_auth_token")

    text = build_chunk_text(node, fixture_repo)

    assert text.startswith("def check_auth_token(")
    assert "Validate the caller's auth token" in text  # the docstring
    assert 'token.startswith("valid-")' in text  # the body
    assert "def fetch_data" not in text  # and nothing from the neighbouring node


def test_chunk_text_includes_decorators(tmp_path: Path) -> None:
    """DEC-005 names decorators explicitly; DEC-012 captures them by extending the span up."""
    source = tmp_path / "decorated.py"
    source.write_text(
        "from langgraph.graph import StateGraph\n"
        "\n"
        "def trace(func):\n"
        "    return func\n"
        "\n"
        "@trace\n"
        "@trace\n"
        "def decorated_node(state):\n"
        '    """Decorated node docstring."""\n'
        "    return state\n"
        "\n"
        "graph = StateGraph(dict)\n"
        'graph.add_node("decorated_node", decorated_node)\n',
        encoding="utf-8",
    )

    from langgraph_context_mcp.parser.repo_scanner import scan_repository

    graph = scan_repository(tmp_path)[0]
    node = next(node for node in graph.nodes if node.name == "decorated_node")
    text = build_chunk_text(node, tmp_path)

    assert text.count("@trace") == 2
    assert "Decorated node docstring." in text
    assert "def trace(func):" not in text  # the decorator's own definition is not swept in


def test_chunk_text_falls_back_when_the_file_is_gone(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """Every node must produce non-empty text, even if its source has since disappeared."""
    index_repository(fixture_repo, store=sqlite_store, embedder=fake_embedder)
    graph = sqlite_store.list_graphs(make_repo_id(fixture_repo))[0]
    node = next(node for node in graph.nodes if node.name == "fetch_data")

    text = build_chunk_text(node, Path(fixture_repo / "does-not-exist"))

    assert text == node.docstring


# --------------------------------------------------------------------------------------------
# Embedding provider
# --------------------------------------------------------------------------------------------
def test_importing_the_package_does_not_load_fastembed() -> None:
    """Import must stay fast and side-effect-free — no ONNX session, no model download."""
    for module in [name for name in sys.modules if name.startswith("fastembed")]:
        del sys.modules[module]

    import langgraph_context_mcp  # noqa: F401
    from langgraph_context_mcp.embeddings import nomic_provider  # noqa: F401

    assert not any(name.startswith("fastembed") for name in sys.modules)


def test_provider_reports_metadata_without_loading_the_model() -> None:
    provider = NomicEmbeddingProvider()

    assert provider.model_name == DEFAULT_MODEL_NAME
    assert provider.dimension == DEFAULT_MODEL_DIMENSION
    assert provider._model is None, "the model must load on first embed(), not before"


def test_model_name_can_be_overridden_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CONTEXT_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-Q")

    assert NomicEmbeddingProvider().model_name == "nomic-ai/nomic-embed-text-v1.5-Q"


def test_empty_batch_needs_no_model() -> None:
    provider = NomicEmbeddingProvider()

    assert provider.embed([]) == []
    assert provider._model is None


# --------------------------------------------------------------------------------------------
# Phase 2 exit criterion — the real local model
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def nomic_provider() -> NomicEmbeddingProvider:
    """The real provider, loaded once per session (the first use downloads the model)."""
    return NomicEmbeddingProvider()


def test_semantic_query_finds_the_authentication_node(
    fixture_repo: Path, tmp_path: Path, nomic_provider: NomicEmbeddingProvider
) -> None:
    """Phase 2's exit criterion: "authentication" must surface ``check_auth_token``.

    Runs the genuine local ONNX model — no API key, no network call after the one-time model
    download (DEC-003 / COR-002).
    """
    store = SqliteStore(tmp_path / "real" / "index.db", dimension=nomic_provider.dimension)
    try:
        index_repository(fixture_repo, store=store, embedder=nomic_provider)
        repo_id = make_repo_id(fixture_repo)

        def search(query: str) -> list[str]:
            return [
                result.node_name
                for result in store.search(
                    nomic_provider.embed_query(query), top_k=3, filters={"repo_id": repo_id}
                )
            ]

        # tasks.md's literal criterion — the bare topic word ranks the node first.
        assert search("authentication")[0] == "check_auth_token"
        # prd.md's phrasing. Only top-3 is asserted: "handles" pulls `handle_error` up by name
        # alone, so the margin here is genuinely thin (RISK-003).
        assert "check_auth_token" in search("which node handles authentication")
        assert "check_auth_token" in search("validate the caller's token")
    finally:
        store.close()


def test_real_provider_dimension_matches_its_vectors(
    nomic_provider: NomicEmbeddingProvider,
) -> None:
    """The declared dimension must match reality, or every stored vector is mis-sized."""
    vectors = nomic_provider.embed(["a short node body"])

    assert len(vectors) == 1
    assert len(vectors[0]) == nomic_provider.dimension == DEFAULT_MODEL_DIMENSION
    assert TEST_DIMENSION != DEFAULT_MODEL_DIMENSION  # the fake and the real model differ
