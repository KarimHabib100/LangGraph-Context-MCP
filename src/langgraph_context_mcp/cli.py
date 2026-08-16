"""Command line interface: ``index``, ``serve``, ``status``.

A thin front end over the same functions the MCP tools call — `indexer.index_repository` and
`tools.mcp_tools` — so the CLI and a connected client can never describe the same repository
differently. ``argparse`` only, per claude.md's stack rules.

Exit codes follow prd.md's contract, fixed by DEC-019:

    0   ran, and the answer is affirmative   (indexed at least one graph; an index exists)
    1   ran correctly, the answer is negative (no graphs found; no index found)
    2   could not run at all                 (bad path, DEC-014 rejection, unreadable, crash)

``1`` exists so a script can tell "I asked and the answer is no" apart from "I could not ask".
``serve`` has no negative case and returns only 0 or 2.

Output discipline: human-readable summaries go to stdout for ``index`` and ``status``, errors to
stderr, and ``serve`` writes nothing at all to stdout — the MCP stdio transport owns that stream and
a stray byte corrupts it. Nothing here calls ``print()`` on a logging path (claude.md); the
``_out``/``_err`` helpers write user-facing command output, which is what a CLI is for.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import TextIO

from .indexer import index_repository
from .parser.repo_scanner import resolve_repo_root
from .server import configure_logging, run_stdio
from .tools.mcp_tools import get_embedder, repository_status

logger = logging.getLogger(__name__)

PROGRAM_NAME = "langgraph-context-mcp"

# DEC-019. Named rather than inlined so the tests and the code agree on what each means.
EXIT_OK = 0
EXIT_NEGATIVE = 1
EXIT_ERROR = 2

DESCRIPTION = """\
Parse a LangGraph Python codebase into a structural graph model with local semantic search, and
serve it to an MCP client. Runs fully offline after the embedding model's first download.\
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``langgraph-context-mcp`` console script.

    Returns the process exit code rather than calling ``sys.exit`` itself, so tests can assert on
    it directly. ``argparse`` still exits 2 on its own for a malformed command line, which matches
    this contract's "could not run" band.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "index":
        return _run_index(args.path, as_json=args.json)
    if args.command == "status":
        return _run_status(args.path, as_json=args.json)
    if args.command == "serve":
        return _run_serve()

    parser.print_help(sys.stderr)  # argparse enforces `required`, so this is unreachable in practice
    return EXIT_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes: 0 = done, affirmative; 1 = ran fine, negative answer "
            "(no graphs / no index); 2 = could not run"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{index,serve,status}")

    index_parser = subparsers.add_parser(
        "index",
        help="Scan a repository, embed every graph node, and write the index",
        description=(
            "Scan PATH for LangGraph graph definitions, embed each node locally, and store the "
            "index. Replaces any previous index for that path rather than appending to it."
        ),
    )
    index_parser.add_argument("path", help="Directory to index (e.g. '.')")
    index_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured result as JSON instead of a human-readable summary",
    )

    subparsers.add_parser(
        "serve",
        help="Start the MCP server on stdio for a client such as Claude Desktop",
        description=(
            "Start the MCP server on the stdio transport and serve the seven tools. Writes nothing "
            "to stdout — that stream carries the MCP protocol. Runs until the client disconnects "
            "or the process is interrupted."
        ),
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Report whether a repository has an index, and when it was built",
        description=(
            "Report whether PATH has an existing index, when it was last built, how much it "
            "contains, and which storage backend holds it."
        ),
    )
    status_parser.add_argument("path", help="Directory to report on (e.g. '.')")
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structured result as JSON instead of a human-readable summary",
    )

    return parser


# --------------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------------
def _run_index(path: str, as_json: bool = False) -> int:
    """``index <path>`` — 0 when at least one graph was indexed, 1 when none, 2 on failure."""
    try:
        repo_root = resolve_repo_root(path)
        # Same provider the MCP tools use, so a repository indexed from the CLI is queryable by a
        # connected client and vice versa — one index, one model, whichever door it came in by.
        result = index_repository(repo_root, embedder=get_embedder())
    except ValueError as exc:  # DEC-014: empty path, or '..' traversal
        return _fail(f"Invalid path: {exc}")
    except FileNotFoundError:
        return _fail(f"Path does not exist: {path}")
    except NotADirectoryError:
        return _fail(f"Path is not a directory: {path}")
    except OSError as exc:
        return _fail(f"Could not read {path}: {exc}")
    except Exception as exc:  # a CLI must not print a traceback at the user
        logger.exception("index failed")
        return _fail(f"Indexing failed: {type(exc).__name__}: {exc}")

    payload = result.to_dict()
    if as_json:
        _out(json.dumps(payload, indent=2))
    elif result.graphs_found == 0:
        _out(f"No LangGraph graphs found in {repo_root}.")
        _out("Nothing was indexed. Check the path, or that this project uses LangGraph.")
    else:
        _out(f"Indexed {repo_root}")
        _out(
            f"  {result.graphs_found} graph(s), {result.nodes_indexed} node(s), "
            f"{result.edges_indexed} edge(s)"
        )
        if result.partial_nodes:
            _out(
                f"  {result.partial_nodes} node(s) partially resolved "
                f"(function not statically locatable — still indexed)"
            )
        _out(f"  backend: {result.backend}, took {result.duration_ms}ms")

    return EXIT_OK if result.graphs_found > 0 else EXIT_NEGATIVE


def _run_status(path: str, as_json: bool = False) -> int:
    """``status <path>`` — 0 when an index exists, 1 when none, 2 on failure."""
    try:
        status = repository_status(path)
    except Exception as exc:  # same reason as above; repository_status is defensive
        logger.exception("status failed")
        return _fail(f"Status check failed: {type(exc).__name__}: {exc}")

    if "error" in status:
        if as_json:
            _out(json.dumps(status, indent=2))
            return EXIT_ERROR
        return _fail(f"{status['error']}: {status.get('reason') or status.get('path', path)}")

    if as_json:
        _out(json.dumps(status, indent=2))
    elif not status["indexed"]:
        _out(f"No index found for {status['path']}.")
        _out(f"Run: {PROGRAM_NAME} index {path}")
    else:
        _out(f"Index found for {status['path']}")
        _out(
            f"  {status['graph_count']} graph(s), {status['node_count']} node(s), "
            f"{status['edge_count']} edge(s)"
        )
        _out(f"  last indexed: {status['last_indexed_at'] or 'unknown'}")
        _out(f"  backend: {status['backend']}, model: {status['embedding_model']}")

    return EXIT_OK if status["indexed"] else EXIT_NEGATIVE


def _run_serve() -> int:
    """``serve`` — run the MCP server on stdio. 0 on clean shutdown, 2 if it cannot start.

    Never writes to stdout: that stream is the MCP transport (prd.md, claude.md).
    """
    try:
        run_stdio()
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
        return EXIT_OK
    except Exception as exc:  # a failed start must be a message, not a traceback
        logger.exception("serve failed")
        return _fail(f"Could not start the MCP server: {type(exc).__name__}: {exc}")
    return EXIT_OK


# --------------------------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------------------------
def _out(message: str, stream: TextIO | None = None) -> None:
    """Write a line of command output to stdout (never used by ``serve``)."""
    (stream or sys.stdout).write(f"{message}\n")


def _err(message: str) -> None:
    """Write a line of command output to stderr."""
    sys.stderr.write(f"{message}\n")


def _fail(message: str) -> int:
    _err(f"error: {message}")
    return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover — module entry, exercised via the console script
    sys.exit(main())
