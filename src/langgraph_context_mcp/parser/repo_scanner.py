"""Walk a repository, parse every Python file, and aggregate the graphs found.

Respects ``.gitignore`` with a minimal, dependency-free pattern matcher (the common cases:
directory names, ``*.ext`` globs, anchored and unanchored patterns). We deliberately do NOT
pull in ``pathspec`` unless a real case proves the manual matcher insufficient — see the Phase 1
prompt's guidance and DEC-006's neighbourhood. If ``pathspec`` ever becomes necessary, that is
a new DEC entry, not a silent addition.

A file with a syntax error (or an unreadable/non-UTF-8 file) is skipped with a warning; the
scan of the rest of the repository continues. This function never raises on a bad file.
"""

from __future__ import annotations

import logging
import os
from fnmatch import fnmatch
from pathlib import Path

from .ast_walker import find_graph_definitions
from .graph_model import GraphDef
from .resolver import resolve_cross_file_references

logger = logging.getLogger(__name__)

# Directories that are always pruned regardless of .gitignore — build/tooling/VCS noise plus
# this tool's own index directory. Keeps the scan fast and avoids indexing dependencies.
_DEFAULT_IGNORE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".langgraph-context",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "build",
        "dist",
        ".eggs",
        ".idea",
        ".vscode",
    }
)


def scan_repository(repo_root: Path) -> list[GraphDef]:
    """Scan ``repo_root`` for LangGraph definitions and return every resolved ``GraphDef``.

    Walks all ``.py`` files under ``repo_root`` (honouring ``.gitignore`` and the default
    ignore set), parses each with the AST walker, runs the bounded-depth cross-file resolver on
    every graph, and aggregates the results. A repo with no LangGraph usage returns an empty
    list — that is a valid, non-error outcome.
    """
    repo_root = repo_root.resolve()
    matcher = _GitignoreMatcher.load(repo_root)

    graphs: list[GraphDef] = []
    for py_file in _iter_python_files(repo_root, matcher):
        try:
            file_graphs = find_graph_definitions(py_file, repo_root)
        except Exception as exc:  # noqa: BLE001 — last-resort guard so one file can't abort the scan
            logger.warning("Skipping %s: unexpected parse failure (%s)", py_file, exc)
            continue
        for graph in file_graphs:
            try:
                graphs.append(resolve_cross_file_references(graph, repo_root))
            except Exception as exc:  # noqa: BLE001 — keep the partial graph if resolution errors
                logger.warning(
                    "Cross-file resolution failed for %s, keeping partial result (%s)",
                    graph.id,
                    exc,
                )
                graphs.append(graph)
    return graphs


def _iter_python_files(repo_root: Path, matcher: _GitignoreMatcher):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        current = Path(dirpath)
        # Prune ignored directories in place so os.walk does not descend into them.
        kept = []
        for dirname in dirnames:
            if dirname in _DEFAULT_IGNORE_DIRS or dirname.endswith(".egg-info"):
                continue
            rel = _rel_posix(current / dirname, repo_root)
            if matcher.is_ignored(rel, is_dir=True):
                continue
            kept.append(dirname)
        dirnames[:] = kept

        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            file_path = current / filename
            rel = _rel_posix(file_path, repo_root)
            if matcher.is_ignored(rel, is_dir=False):
                continue
            yield file_path


def _rel_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.name


class _GitignoreMatcher:
    """A minimal ``.gitignore`` matcher — enough for the common patterns, no dependency.

    Supported: blank lines and ``#`` comments (ignored), trailing-slash directory patterns,
    leading-slash anchored patterns, ``*``/``?`` globs, and unanchored patterns that match by
    basename anywhere in the tree. Negation (``!``) is intentionally not supported and such
    lines are ignored — a conservative choice that never wrongly excludes a file.
    """

    def __init__(self, patterns: list[tuple[str, bool, bool]]) -> None:
        # each pattern: (glob, dir_only, anchored)
        self._patterns = patterns

    @classmethod
    def load(cls, repo_root: Path) -> _GitignoreMatcher:
        patterns: list[tuple[str, bool, bool]] = []
        gitignore = repo_root / ".gitignore"
        try:
            lines = gitignore.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return cls(patterns)

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith(("#", "!")):
                continue
            dir_only = line.endswith("/")
            if dir_only:
                line = line[:-1]
            anchored = line.startswith("/")
            if anchored:
                line = line[1:]
            if line:
                patterns.append((line, dir_only, anchored))
        return cls(patterns)

    def is_ignored(self, rel_path: str, is_dir: bool) -> bool:
        name = rel_path.rsplit("/", 1)[-1]
        for glob, dir_only, anchored in self._patterns:
            if dir_only and not is_dir:
                continue
            if anchored:
                if fnmatch(rel_path, glob) or fnmatch(rel_path, f"{glob}/*"):
                    return True
                continue
            if fnmatch(name, glob):
                return True
            if "/" in glob and fnmatch(rel_path, glob):
                return True
        return False
