"""The shared ``VectorStore`` contract — defined once, executed against every backend.

DEC-002 promises that the SQLite default and the pgvector opt-in behave identically, and
RISK-004 warns that they could silently diverge. Two independently written test files would be
exactly how that divergence goes unnoticed, so the assertions live here and
``test_sqlite_store.py`` / ``test_pgvector_store.py`` each parametrize over ``CONTRACT_TESTS``
with their own store fixture. One definition, two executions.

This module is deliberately not named ``test_*``: pytest must not collect it directly, because a
contract function needs a store and there is no such thing as a backend-less run.

Each function takes ``(store, repo_id)`` and confines every write to that ``repo_id``, so the
pgvector backend's shared tables stay isolated per test.
"""

from __future__ import annotations

import pytest
from support import TEST_DIMENSION, build_chunk, build_graph, one_hot, store_graph


def contract_search_returns_nearest_first(store, repo_id: str) -> None:
    """Results come back ranked by cosine similarity, best first."""
    graph = build_graph(repo_id, "rank", ["alpha", "beta", "gamma"])
    store_graph(store, graph)
    store.upsert_chunks(
        [
            build_chunk(graph, graph.nodes[0], one_hot(0)),
            build_chunk(graph, graph.nodes[1], one_hot(1)),
            build_chunk(graph, graph.nodes[2], one_hot(2)),
        ]
    )

    results = store.search(one_hot(1), top_k=3, filters={"repo_id": repo_id})

    # alpha and gamma are both orthogonal to the query, so only beta's position is defined.
    assert results[0].node_name == "beta"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert {result.node_name for result in results[1:]} == {"alpha", "gamma"}


def contract_score_is_cosine_similarity(store, repo_id: str) -> None:
    """Identical vectors score 1.0 and orthogonal vectors score 0.0 — on every backend.

    This is RISK-004's parity assertion: both backends are configured for cosine over
    unit-normalized vectors, so these numbers are backend-independent, not approximately similar.
    """
    graph = build_graph(repo_id, "cosine", ["same", "orthogonal"])
    store_graph(store, graph)
    store.upsert_chunks(
        [
            build_chunk(graph, graph.nodes[0], one_hot(0)),
            build_chunk(graph, graph.nodes[1], one_hot(1)),
        ]
    )

    results = store.search(one_hot(0), top_k=2, filters={"repo_id": repo_id})
    by_name = {result.node_name: result.score for result in results}

    assert by_name["same"] == pytest.approx(1.0, abs=1e-5)
    assert by_name["orthogonal"] == pytest.approx(0.0, abs=1e-5)


def contract_unnormalized_vectors_are_normalized_on_write(store, repo_id: str) -> None:
    """Vector magnitude must not affect ranking — only direction does."""
    graph = build_graph(repo_id, "magnitude", ["scaled", "other"])
    store_graph(store, graph)
    long_vector = [value * 42.0 for value in one_hot(3)]
    store.upsert_chunks(
        [
            build_chunk(graph, graph.nodes[0], long_vector),
            build_chunk(graph, graph.nodes[1], one_hot(4)),
        ]
    )

    results = store.search(one_hot(3), top_k=2, filters={"repo_id": repo_id})

    assert results[0].node_name == "scaled"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def contract_search_respects_top_k(store, repo_id: str) -> None:
    graph = build_graph(repo_id, "topk", ["one", "two", "three", "four"])
    store_graph(store, graph)
    store.upsert_chunks(
        [
            build_chunk(graph, node, one_hot(index))
            for index, node in enumerate(graph.nodes)
        ]
    )

    assert len(store.search(one_hot(0), top_k=2, filters={"repo_id": repo_id})) == 2
    assert len(store.search(one_hot(0), top_k=4, filters={"repo_id": repo_id})) == 4


def contract_search_filters_by_repo(store, repo_id: str) -> None:
    """A repo filter must exclude every other repository's chunks."""
    mine = build_graph(repo_id, "mine", ["mine_node"])
    theirs = build_graph(f"{repo_id}-other", "theirs", ["their_node"])
    store_graph(store, mine)
    store_graph(store, theirs)
    store.upsert_chunks(
        [
            build_chunk(mine, mine.nodes[0], one_hot(5)),
            build_chunk(theirs, theirs.nodes[0], one_hot(5)),
        ]
    )
    try:
        results = store.search(one_hot(5), top_k=10, filters={"repo_id": repo_id})
        assert [result.node_name for result in results] == ["mine_node"]
    finally:
        store.delete_repo(f"{repo_id}-other")


def contract_search_filters_by_graph(store, repo_id: str) -> None:
    first = build_graph(repo_id, "graph_a", ["a_node"])
    second = build_graph(repo_id, "graph_b", ["b_node"])
    store_graph(store, first)
    store_graph(store, second)
    store.upsert_chunks(
        [
            build_chunk(first, first.nodes[0], one_hot(6)),
            build_chunk(second, second.nodes[0], one_hot(6)),
        ]
    )

    results = store.search(one_hot(6), top_k=10, filters={"graph_id": second.id})

    assert [result.node_name for result in results] == ["b_node"]


def contract_search_unknown_repo_is_empty(store, repo_id: str) -> None:
    """Querying a repository that was never indexed returns nothing — not an error, not a
    borrowed result from another repository."""
    assert store.search(one_hot(0), top_k=5, filters={"repo_id": repo_id}) == []


def contract_search_rejects_unknown_filter(store, repo_id: str) -> None:
    """An unrecognized filter must fail loudly; silently ignoring it returns confident, wrong
    results."""
    with pytest.raises(ValueError, match="Unsupported search filter"):
        store.search(one_hot(0), top_k=5, filters={"node_name": "anything"})


def contract_search_rejects_invalid_top_k(store, repo_id: str) -> None:
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        store.search(one_hot(0), top_k=0, filters={"repo_id": repo_id})


def contract_upsert_is_idempotent(store, repo_id: str) -> None:
    """Re-indexing the same repository twice must not duplicate rows (Phase 3 scenario 3.9)."""
    graph = build_graph(repo_id, "idempotent", ["only_node"])
    store_graph(store, graph)
    chunks = [build_chunk(graph, graph.nodes[0], one_hot(7))]

    store.upsert_chunks(chunks)
    store.upsert_chunks(chunks)
    store_graph(store, graph)

    results = store.search(one_hot(7), top_k=10, filters={"repo_id": repo_id})
    assert len(results) == 1
    assert len(store.list_graphs(repo_id)) == 1


def contract_upsert_updates_existing_chunk(store, repo_id: str) -> None:
    """A re-index with changed content replaces the old vector rather than adding a second."""
    graph = build_graph(repo_id, "update", ["moving_node"])
    store_graph(store, graph)
    store.upsert_chunks([build_chunk(graph, graph.nodes[0], one_hot(8), text="before")])
    store.upsert_chunks([build_chunk(graph, graph.nodes[0], one_hot(9), text="after")])

    old_position = store.search(one_hot(8), top_k=10, filters={"repo_id": repo_id})
    new_position = store.search(one_hot(9), top_k=10, filters={"repo_id": repo_id})

    assert len(new_position) == 1
    assert new_position[0].score == pytest.approx(1.0, abs=1e-5)
    assert old_position[0].score == pytest.approx(0.0, abs=1e-5)


def contract_search_result_carries_node_metadata(store, repo_id: str) -> None:
    """A hit must answer prd.md's ``semantic_search_nodes`` output without a second lookup."""
    graph = build_graph(repo_id, "metadata", ["described_node"])
    store_graph(store, graph)
    store.upsert_chunks([build_chunk(graph, graph.nodes[0], one_hot(10))])

    result = store.search(one_hot(10), top_k=1, filters={"repo_id": repo_id})[0]

    assert result.node_name == "described_node"
    assert result.node_id == graph.nodes[0].id
    assert result.graph_id == graph.id
    assert result.file_path == graph.nodes[0].source_file
    assert result.line_start == graph.nodes[0].line_start
    assert result.docstring == "Docstring for described_node."
    assert set(result.to_dict()) >= {
        "node_name",
        "file_path",
        "line_start",
        "score",
        "docstring",
    }


def contract_get_graph_round_trips(store, repo_id: str) -> None:
    """A stored graph must come back byte-for-byte equal, children included."""
    graph = build_graph(repo_id, "roundtrip", ["first", "second"])
    store_graph(store, graph)

    loaded = store.get_graph(graph.id)

    assert loaded is not None
    assert loaded == graph
    assert loaded.to_dict() == graph.to_dict()
    assert len(loaded.edges) == 2
    assert len(loaded.conditional_routes) == 1


def contract_get_graph_unknown_id_is_none(store, repo_id: str) -> None:
    assert store.get_graph(f"{repo_id}::does-not-exist") is None


def contract_list_graphs_scoped_to_repo(store, repo_id: str) -> None:
    first = build_graph(repo_id, "list_a", ["node_a"])
    second = build_graph(repo_id, "list_b", ["node_b"])
    elsewhere = build_graph(f"{repo_id}-other", "list_c", ["node_c"])
    store_graph(store, first)
    store_graph(store, second)
    store_graph(store, elsewhere)
    try:
        listed = store.list_graphs(repo_id)
        assert {graph.id for graph in listed} == {first.id, second.id}
    finally:
        store.delete_repo(f"{repo_id}-other")


def contract_repository_metadata_is_persisted(store, repo_id: str) -> None:
    """prd.md's Repository model round-trips, including which model built the index."""
    assert store.is_indexed(repo_id) is False
    assert store.get_repository(repo_id) is None

    graph = build_graph(repo_id, "repo_meta", ["node"])
    store_graph(store, graph)

    repository = store.get_repository(repo_id)
    assert repository is not None
    assert repository.repo_id == repo_id
    assert repository.embedding_model == "fake-hashing-embedder"
    assert repository.dimension == TEST_DIMENSION
    assert repository.backend_type in {"sqlite", "pgvector"}
    assert repository.last_indexed_at is not None
    assert store.is_indexed(repo_id) is True


def contract_delete_repo_removes_everything(store, repo_id: str) -> None:
    graph = build_graph(repo_id, "deleted", ["doomed_node"])
    store_graph(store, graph)
    store.upsert_chunks([build_chunk(graph, graph.nodes[0], one_hot(11))])

    store.delete_repo(repo_id)

    assert store.get_repository(repo_id) is None
    assert store.is_indexed(repo_id) is False
    assert store.list_graphs(repo_id) == []
    assert store.get_graph(graph.id) is None
    assert store.search(one_hot(11), top_k=10, filters={"repo_id": repo_id}) == []


def contract_delete_unknown_repo_is_a_noop(store, repo_id: str) -> None:
    store.delete_repo(f"{repo_id}-never-existed")


def contract_dimension_mismatch_is_rejected(store, repo_id: str) -> None:
    """A vector of the wrong width must be refused, never coerced or silently stored."""
    graph = build_graph(repo_id, "mismatch", ["node"])
    store_graph(store, graph)
    wrong_width = build_chunk(graph, graph.nodes[0], [1.0] * (TEST_DIMENSION + 1))

    with pytest.raises(ValueError, match="dimension"):
        store.upsert_chunks([wrong_width])

    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0] * (TEST_DIMENSION + 1), top_k=1, filters={"repo_id": repo_id})


def contract_empty_upsert_is_a_noop(store, repo_id: str) -> None:
    store.upsert_chunks([])


# Executed by both backend test modules. Adding a function here adds it to both.
CONTRACT_TESTS = [
    contract_search_returns_nearest_first,
    contract_score_is_cosine_similarity,
    contract_unnormalized_vectors_are_normalized_on_write,
    contract_search_respects_top_k,
    contract_search_filters_by_repo,
    contract_search_filters_by_graph,
    contract_search_unknown_repo_is_empty,
    contract_search_rejects_unknown_filter,
    contract_search_rejects_invalid_top_k,
    contract_upsert_is_idempotent,
    contract_upsert_updates_existing_chunk,
    contract_search_result_carries_node_metadata,
    contract_get_graph_round_trips,
    contract_get_graph_unknown_id_is_none,
    contract_list_graphs_scoped_to_repo,
    contract_repository_metadata_is_persisted,
    contract_delete_repo_removes_everything,
    contract_delete_unknown_repo_is_a_noop,
    contract_dimension_mismatch_is_rejected,
    contract_empty_upsert_is_a_noop,
]
