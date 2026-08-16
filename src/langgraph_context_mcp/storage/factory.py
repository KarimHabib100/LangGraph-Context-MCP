"""Chooses the storage backend for a repository.

The rule is the whole of DEC-002: ``DATABASE_URL`` set means PostgreSQL + ``pgvector``; unset
means SQLite at ``<repo_root>/.langgraph-context/index.db``. There is no configuration file and
no flag — the presence of one environment variable is the entire decision.

Every call returns a *new* store. The connection is injected into whatever needs it and closed by
its owner; nothing here caches a store in a module-level global (claude.md).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import VectorStore
from .sqlite_store import SqliteStore, default_index_path

logger = logging.getLogger(__name__)

DATABASE_URL_ENV_VAR = "DATABASE_URL"


def configured_database_url() -> str:
    """The trimmed ``DATABASE_URL``, or an empty string when unset — the whole of DEC-002's rule."""
    return (os.environ.get(DATABASE_URL_ENV_VAR) or "").strip()


def open_existing_store(repo_root: Path, dimension: int | None = None) -> VectorStore | None:
    """Return a store for *reading*, or ``None`` when this repository demonstrably has no index.

    Read-only callers use this instead of ``get_vector_store``: ``SqliteStore`` creates its
    directory and database file on construction, so answering "not indexed" through the normal
    factory would write an empty index into the user's repository as the side effect of a question
    (DEC-018). Creating the index is ``index_repository``'s job, and only its job.

    The pgvector backend always returns a store: whether a repository row exists is a question only
    the server can answer, and connecting is how you ask it.
    """
    if configured_database_url():
        return get_vector_store(repo_root, dimension=dimension)

    index_path = default_index_path(repo_root)
    if not index_path.exists():
        logger.debug("No sqlite index at %s — repository is not indexed", index_path)
        return None
    return SqliteStore(index_path, dimension=dimension)


def get_vector_store(repo_root: Path, dimension: int | None = None) -> VectorStore:
    """Return the store for ``repo_root``: ``PgvectorStore`` if ``DATABASE_URL`` is set, else
    ``SqliteStore``.

    ``dimension`` creates the vector table up front when the caller already knows the embedding
    width; when omitted, the table is created on the first write or search instead. The caller
    owns the returned store and must close it (or use it as a context manager).

    Constructing a ``SqliteStore`` creates the index file if it is missing, which is correct for an
    indexing run and wrong for a read-only query — those use ``open_existing_store``.
    """
    database_url = configured_database_url()
    if database_url:
        # Imported here, not at module scope: psycopg and pgvector live in the optional
        # [pgvector] extra, and a default SQLite install must not need them present.
        from .pgvector_store import PgvectorStore

        logger.info("Using pgvector backend (DATABASE_URL is set)")
        return PgvectorStore(database_url, dimension=dimension)

    index_path = default_index_path(repo_root)
    logger.info("Using sqlite backend at %s", index_path)
    return SqliteStore(index_path, dimension=dimension)
