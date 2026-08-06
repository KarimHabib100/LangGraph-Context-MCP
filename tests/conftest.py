"""Fixtures shared by the Phase 2 storage, embedding, and indexer tests.

The ``VectorStore`` contract is exercised against both backends. ``pgvector`` skips cleanly when
``DATABASE_URL`` is unset (a local developer with no Postgres) and runs for real in CI, where a
Postgres service container provides one.

The pgvector backend writes into the user's own database, so its tables persist across tests.
Every test therefore namespaces its data under a unique ``repo_id`` and filters by it, and
teardown deletes exactly what the test created — nothing is ever dropped.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from support import FIXTURE_REPO, TEST_DIMENSION, FakeEmbeddingProvider

from langgraph_context_mcp.storage.pgvector_store import PgvectorStore
from langgraph_context_mcp.storage.sqlite_store import SqliteStore


@pytest.fixture
def fake_embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def repo_id() -> str:
    """A repo ID unique to one test, so tests sharing pgvector tables cannot collide."""
    return f"/test-repo/{uuid4().hex}"


@pytest.fixture
def sqlite_store(tmp_path: Path):
    store = SqliteStore(tmp_path / ".langgraph-context" / "index.db", dimension=TEST_DIMENSION)
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def pgvector_store(repo_id: str):
    """A ``PgvectorStore`` against ``DATABASE_URL``, or a clean skip when it is unset."""
    database_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not database_url:
        pytest.skip("DATABASE_URL is not set — the pgvector backend is exercised in CI")

    store = PgvectorStore(database_url, dimension=TEST_DIMENSION)
    try:
        yield store
    finally:
        try:
            store.delete_repo(repo_id)
        finally:
            store.close()


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    """A writable copy of ``tests/fixtures/sample_graphs``.

    Copied rather than used in place so an index written to the default
    ``.langgraph-context/index.db`` location lands in ``tmp_path``, never in the source tree.
    """
    destination = tmp_path / "sample_repo"
    shutil.copytree(FIXTURE_REPO, destination)
    return destination
