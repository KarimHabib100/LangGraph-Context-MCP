"""Repo-path validation at the point file I/O begins (DEC-014).

These cover the two defects Phase 3 QA reproduced against a real build — QA-3-07 (an empty path
silently indexing the current working directory) and QA-3-08 (``..`` escaping the named
directory) — plus the symlink case neither of those rules can see.

The guard lives in ``repo_scanner`` because that is where the filesystem walk happens; the
indexer calls it again at its entry point, and both layers are asserted here so a future refactor
cannot quietly drop one of them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from support import FakeEmbeddingProvider

from langgraph_context_mcp.indexer import index_repository
from langgraph_context_mcp.parser.repo_scanner import (
    _is_within,
    resolve_repo_root,
    scan_repository,
)

# Every input that must be rejected as "no directory was actually named". Each of these resolves
# to the current working directory (or a near-miss of it) rather than to anything the caller meant.
EMPTY_PATHS = ["", "   ", "\t", "\n", "\t\n  "]

# Every shape of `..` traversal. The resolved form of each is an ordinary absolute path with no
# evidence left, which is why the check runs on the raw argument.
TRAVERSAL_PATHS = ["..", "../..", "../../..", "repo/../..", "./../sibling", "a/b/../../../c"]


# --------------------------------------------------------------------------------------------
# resolve_repo_root — the guard itself
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw", EMPTY_PATHS)
def test_empty_path_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        resolve_repo_root(raw)


@pytest.mark.parametrize("raw", TRAVERSAL_PATHS)
def test_traversal_path_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match=r"may not contain '\.\.'"):
        resolve_repo_root(raw)


def test_none_is_rejected() -> None:
    with pytest.raises(ValueError, match="required"):
        resolve_repo_root(None)


def test_valid_path_resolves_to_absolute(tmp_path: Path) -> None:
    resolved = resolve_repo_root(tmp_path)
    assert resolved.is_absolute()
    assert resolved == tmp_path.resolve()


def test_dot_is_accepted_and_is_the_documented_invocation(tmp_path: Path) -> None:
    """prd.md's documented flow is ``index .`` — a single dot must survive validation."""
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        assert resolve_repo_root(".") == tmp_path.resolve()
    finally:
        os.chdir(previous)


# --------------------------------------------------------------------------------------------
# QA-3-07 — an empty path must not silently mean "the current working directory"
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw", EMPTY_PATHS)
def test_scan_repository_rejects_empty_path(raw: str) -> None:
    with pytest.raises(ValueError):
        scan_repository(raw)


@pytest.mark.parametrize("raw", EMPTY_PATHS)
def test_index_repository_rejects_empty_path(
    raw: str, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    with pytest.raises(ValueError):
        index_repository(raw, store=sqlite_store, embedder=fake_embedder)


def test_empty_path_does_not_index_the_working_directory(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """QA-3-07, exactly as reproduced: cwd is a real repo, and `""` must still refuse.

    Before the fix this returned a successful ``IndexResult`` describing the working directory —
    a directory the caller never named — and wrote an index into it.
    """
    previous = Path.cwd()
    os.chdir(fixture_repo)
    try:
        with pytest.raises(ValueError):
            index_repository("", store=sqlite_store, embedder=fake_embedder)
        assert not (fixture_repo / ".langgraph-context").exists()
    finally:
        os.chdir(previous)


# --------------------------------------------------------------------------------------------
# QA-3-08 — `..` must not escape the named directory
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw", TRAVERSAL_PATHS)
def test_scan_repository_rejects_traversal(raw: str) -> None:
    with pytest.raises(ValueError):
        scan_repository(raw)


def test_index_repository_rejects_traversal_out_of_the_named_repo(
    fixture_repo: Path, sqlite_store, fake_embedder: FakeEmbeddingProvider
) -> None:
    """QA-3-08, exactly as reproduced: ``repo / ".." / ".." / ".."``.

    Before the fix this walked out of the fixture and indexed unrelated trees, reporting success.
    """
    escaping = fixture_repo / ".." / ".." / ".."
    with pytest.raises(ValueError, match=r"may not contain '\.\.'"):
        index_repository(escaping, store=sqlite_store, embedder=fake_embedder)


def test_traversal_is_rejected_even_when_it_stays_inside(fixture_repo: Path) -> None:
    """``repo/subdir/..`` lands back inside the repo, but is still refused.

    Documented in DEC-014 as a deliberate over-rejection: the rule is claude.md's, read literally.
    """
    with pytest.raises(ValueError):
        scan_repository(fixture_repo / "subdir" / "..")


# --------------------------------------------------------------------------------------------
# Rule 3 — containment, the case the first two rules cannot see
# --------------------------------------------------------------------------------------------
def test_symlinked_file_pointing_outside_the_root_is_skipped(
    fixture_repo: Path, tmp_path: Path
) -> None:
    """A legitimate path argument, but a file inside it links out of the tree."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret_graph.py"
    secret.write_text(
        "from langgraph.graph import StateGraph\n"
        "def leaked(state):\n"
        "    return state\n"
        "g = StateGraph(dict)\n"
        'g.add_node("leaked", leaked)\n',
        encoding="utf-8",
    )

    link = fixture_repo / "linked_graph.py"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted in this environment")

    graphs = scan_repository(fixture_repo)

    assert all("leaked" not in [node.name for node in graph.nodes] for graph in graphs)
    assert all(Path(graph.file_path).name != "linked_graph.py" for graph in graphs)


def test_containment_predicate_accepts_inside_and_rejects_outside(tmp_path: Path) -> None:
    """Rule 3's predicate, tested directly.

    The symlink test above is the integration form, but symlink creation needs elevation on
    Windows and skips there — so the predicate itself is asserted unconditionally rather than
    letting a security control go unexercised on a developer machine.
    """
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    inside = root / "pkg" / "mod.py"
    inside.write_text("x = 1\n", encoding="utf-8")
    outside = tmp_path / "elsewhere.py"
    outside.write_text("x = 1\n", encoding="utf-8")

    assert _is_within(inside, root.resolve()) is True
    assert _is_within(root / "missing.py", root.resolve()) is True
    assert _is_within(outside, root.resolve()) is False
    assert _is_within(tmp_path, root.resolve()) is False


def test_normal_scan_still_works(fixture_repo: Path) -> None:
    """The guard must not disturb the ordinary path — the fixture still parses in full."""
    graphs = scan_repository(fixture_repo)

    assert len(graphs) == 1
    assert len(graphs[0].nodes) == 5
    assert {node.name for node in graphs[0].nodes} == {
        "fetch_data",
        "check_auth_token",
        "enrich_data",
        "handle_error",
        "format_response",
    }
