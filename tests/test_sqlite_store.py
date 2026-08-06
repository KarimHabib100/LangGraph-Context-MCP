"""SQLite backend: the shared ``VectorStore`` contract plus SQLite-only behaviour.

The contract assertions themselves live in ``vector_store_contract.py`` and are shared with
``test_pgvector_store.py`` — one definition, executed against both backends, so the two cannot
drift apart (DEC-002 / RISK-004 / task 2.8). Only behaviour genuinely specific to this backend
(the on-disk index location, extension loading, persistence across connections) is written here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from support import TEST_DIMENSION, build_chunk, build_graph, one_hot, store_graph
from vector_store_contract import CONTRACT_TESTS

from langgraph_context_mcp.storage.sqlite_store import (
    INDEX_DIR_NAME,
    INDEX_FILE_NAME,
    SqliteStore,
    default_index_path,
)


@pytest.mark.parametrize("contract_test", CONTRACT_TESTS, ids=lambda test: test.__name__)
def test_vector_store_contract(contract_test, sqlite_store, repo_id: str) -> None:
    """Every shared contract behaviour, run against the SQLite backend."""
    contract_test(sqlite_store, repo_id)


def test_default_index_path_is_under_the_indexed_repo(tmp_path: Path) -> None:
    """The index belongs to the repo being indexed, never to this tool's install directory."""
    assert default_index_path(tmp_path) == tmp_path.resolve() / INDEX_DIR_NAME / INDEX_FILE_NAME


def test_index_directory_is_created_on_demand(tmp_path: Path) -> None:
    db_path = tmp_path / ".langgraph-context" / "index.db"
    assert not db_path.parent.exists()

    store = SqliteStore(db_path, dimension=TEST_DIMENSION)
    try:
        assert db_path.parent.is_dir()
        assert db_path.exists()
    finally:
        store.close()


def test_data_survives_reopening_the_file(tmp_path: Path, repo_id: str) -> None:
    """A written index is readable by a fresh connection — the point of persisting at all."""
    db_path = tmp_path / ".langgraph-context" / "index.db"
    graph = build_graph(repo_id, "persisted", ["kept_node"])

    first = SqliteStore(db_path, dimension=TEST_DIMENSION)
    try:
        store_graph(first, graph)
        first.upsert_chunks([build_chunk(graph, graph.nodes[0], one_hot(1))])
    finally:
        first.close()

    second = SqliteStore(db_path)
    try:
        assert second.is_indexed(repo_id) is True
        assert second.get_graph(graph.id) == graph
        results = second.search(one_hot(1), top_k=5, filters={"repo_id": repo_id})
        assert [result.node_name for result in results] == ["kept_node"]
    finally:
        second.close()


def test_reopening_recovers_the_stored_dimension(tmp_path: Path) -> None:
    """A store opened without an explicit dimension adopts the one already on disk."""
    db_path = tmp_path / "index.db"
    SqliteStore(db_path, dimension=TEST_DIMENSION).close()

    reopened = SqliteStore(db_path)
    try:
        with pytest.raises(ValueError, match="dimension"):
            reopened.search([1.0] * (TEST_DIMENSION + 1), top_k=1, filters={})
    finally:
        reopened.close()


def test_search_before_anything_is_indexed_is_empty(tmp_path: Path) -> None:
    """A brand-new index has no vector table yet; searching it is empty, not an error."""
    store = SqliteStore(tmp_path / "index.db")
    try:
        assert store.search([0.0] * TEST_DIMENSION, top_k=5, filters={}) == []
    finally:
        store.close()


def test_close_is_safe_and_context_manager_closes(tmp_path: Path) -> None:
    with SqliteStore(tmp_path / "index.db", dimension=TEST_DIMENSION) as store:
        assert store.is_indexed("/nothing") is False
    # Closing twice must not raise — cleanup paths call it defensively.
    store.close()


def test_corrupt_index_file_fails_loudly(tmp_path: Path) -> None:
    """A damaged index must raise, never quietly return wrong or empty results."""
    import sqlite3

    db_path = tmp_path / "index.db"
    db_path.write_bytes(b"this is definitely not a sqlite database" * 10)

    with pytest.raises(sqlite3.DatabaseError):
        SqliteStore(db_path, dimension=TEST_DIMENSION)
