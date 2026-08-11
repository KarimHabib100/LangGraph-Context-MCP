"""Walk a repository, parse every Python file, and aggregate the graphs found.

Respects ``.gitignore`` with a minimal, dependency-free pattern matcher (the common cases:
directory names, ``*.ext`` globs, anchored and unanchored patterns). We deliberately do NOT
pull in ``pathspec`` unless a real case proves the manual matcher insufficient — see the Phase 1
prompt's guidance and DEC-006's neighbourhood. If ``pathspec`` ever becomes necessary, that is
a new DEC entry, not a silent addition.

A file with a syntax error (or an unreadable/non-UTF-8 file) is skipped with a warning; the
scan of the rest of the repository continues. This function never raises on a bad file.

This module is also where repo-path validation lives (DEC-014), because this is the point at
which file I/O actually begins. ``resolve_repo_root`` rejects empty input and ``..`` traversal,
and the walk additionally refuses to hand back any file that resolves outside the root — see
claude.md's KEY ARCHITECTURAL RULES and RISK-SEC-001.
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


def resolve_repo_root(repo_root: Path | str) -> Path:
    """Validate a caller-supplied repository path and return it resolved and absolute.

    This is the guard claude.md's KEY ARCHITECTURAL RULES require and RISK-SEC-001 names, applied
    here because this module is where file I/O begins (DEC-014). Two rejections, both
    ``ValueError``:

    - **Empty or whitespace-only input.** ``Path("").resolve()`` silently means "the current
      working directory", so a caller that named no directory would otherwise get an unrelated one
      indexed and reported as a success (QA-3-07).
    - **A ``..`` component in the argument.** Checked on the raw input, because resolving is what
      destroys the evidence — ``<repo>/../../..`` resolves to a perfectly ordinary-looking absolute
      path with nothing left to detect (QA-3-08).

    Existence and directory-ness are deliberately *not* checked here: those belong to the caller
    that wants to distinguish them (``index_repository`` raises ``FileNotFoundError`` /
    ``NotADirectoryError``), and ``scan_repository`` keeps its existing empty-result behaviour for
    a path that is not there.
    """
    if repo_root is None:
        raise ValueError("Repository path is required, got None.")

    raw = os.fspath(repo_root)
    if not raw.strip():
        raise ValueError(
            "Repository path is empty or whitespace-only. Pass an explicit directory — an "
            "empty path would silently resolve to the current working directory."
        )

    candidate = Path(raw)
    if any(part == ".." for part in candidate.parts):
        raise ValueError(
            f"Repository path may not contain '..' path traversal: {raw!r}. "
            f"Pass an absolute path, or run from inside the repository and use '.'."
        )

    return candidate.resolve()


def scan_repository(repo_root: Path) -> list[GraphDef]:
    """Scan ``repo_root`` for LangGraph definitions and return every resolved ``GraphDef``.

    Walks all ``.py`` files under ``repo_root`` (honouring ``.gitignore`` and the default
    ignore set), parses each with the AST walker, runs the bounded-depth cross-file resolver on
    every graph, and aggregates the results. A repo with no LangGraph usage returns an empty
    list — that is a valid, non-error outcome.

    Raises ``ValueError`` for an empty path or one containing ``..`` (DEC-014).
    """
    repo_root = resolve_repo_root(repo_root)
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
            if not _is_within(file_path, repo_root):
                # A symlink inside the repo pointing out of it. `resolve_repo_root` cannot see
                # this — the argument was legitimate; the escape is on disk (DEC-014, rule 3).
                logger.warning(
                    "Skipping %s: resolves outside the repository root %s", file_path, repo_root
                )
                continue
            yield file_path


def _is_within(path: Path, repo_root: Path) -> bool:
    """Whether ``path`` really lives under ``repo_root`` once symlinks are followed."""
    try:
        path.resolve().relative_to(repo_root)
    except ValueError:
        return False
    except OSError:  # unreadable/broken link — treat as outside rather than guess
        return False
    return True


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
