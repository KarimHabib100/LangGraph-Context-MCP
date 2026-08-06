"""pgvector backend: the same shared ``VectorStore`` contract, plus pgvector-only behaviour.

The contract assertions are imported from ``vector_store_contract.py`` — the identical list that
``test_sqlite_store.py`` runs — so both backends are held to one definition of correct rather
than two that could silently diverge (DEC-002 / RISK-004 / task 2.8).

Every test here skips cleanly when ``DATABASE_URL`` is unset, which is the normal state on a
developer machine with no Postgres. CI provides a Postgres service container with the ``vector``
extension available, so these run for real there.
"""

from __future__ import annotations

import os

import pytest
from support import TEST_DIMENSION, build_chunk, build_graph, one_hot, store_graph
from vector_store_contract import CONTRACT_TESTS

from langgraph_context_mcp.storage.factory import get_vector_store
from langgraph_context_mcp.storage.pgvector_store import (
    CHUNKS_TABLE,
    HNSW_INDEX,
    PgvectorStore,
)


@pytest.mark.parametrize("contract_test", CONTRACT_TESTS, ids=lambda test: test.__name__)
def test_vector_store_contract(contract_test, pgvector_store, repo_id: str) -> None:
    """Every shared contract behaviour, run against the pgvector backend."""
    contract_test(pgvector_store, repo_id)


def test_hnsw_index_is_created_with_cosine_ops(pgvector_store) -> None:
    """RISK-004's mitigation, asserted against the live schema rather than assumed.

    pgvector's default operator class is L2. If this index were built without
    ``vector_cosine_ops``, ranking would quietly diverge from the SQLite backend.
    """
    with pgvector_store._connection.cursor() as cursor:
        cursor.execute("SELECT indexdef FROM pg_indexes WHERE indexname = %s", (HNSW_INDEX,))
        row = cursor.fetchone()

    assert row is not None, f"{HNSW_INDEX} was not created on first connection"
    definition = row[0].lower()
    assert "using hnsw" in definition
    assert "vector_cosine_ops" in definition


def test_schema_is_created_on_first_connection(pgvector_store) -> None:
    """No manual migration step: the tables exist as soon as a store is constructed."""
    with pgvector_store._connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (CHUNKS_TABLE,))
        assert cursor.fetchone()[0] is not None


def test_connecting_twice_is_idempotent(pgvector_store, repo_id: str) -> None:
    """A second store against the same database must not fail on already-existing objects."""
    second = PgvectorStore(os.environ["DATABASE_URL"], dimension=TEST_DIMENSION)
    try:
        graph = build_graph(repo_id, "second_connection", ["shared_node"])
        store_graph(second, graph)
        second.upsert_chunks([build_chunk(graph, graph.nodes[0], one_hot(2))])
        assert pgvector_store.get_graph(graph.id) == graph
    finally:
        second.close()


def test_unreachable_host_fails_fast_with_a_clear_error() -> None:
    """An unreachable Postgres must raise promptly, never hang (Phase 3 scenario 3.13)."""
    import psycopg

    unreachable = "postgresql://user:pass@127.0.0.1:1/does_not_exist"
    with pytest.raises(psycopg.OperationalError):
        PgvectorStore(unreachable)


def test_factory_selects_pgvector_when_database_url_is_set(
    pgvector_store, tmp_path, repo_id: str
) -> None:
    """DEC-002's rule end to end: the environment variable is the entire decision."""
    store = get_vector_store(tmp_path, dimension=TEST_DIMENSION)
    try:
        assert isinstance(store, PgvectorStore)
        graph = build_graph(repo_id, "via_factory", ["node"])
        store_graph(store, graph)
        assert store.get_repository(repo_id).backend_type == "pgvector"
    finally:
        store.close()
