"""The seven MCP tool implementations (DEC-004).

Every function here is thin by rule (claude.md): it validates its arguments, calls into
``indexer`` / ``graph_queries`` / ``storage``, and formats the result as a JSON-serializable dict.
No parsing, no embedding, no SQL, and no traversal logic lives in this module — traversal is in
``graph_queries.py`` (DEC-018).

``server.py`` registers these functions directly, so each signature *is* the tool's public schema
and each docstring *is* what a connected LLM reads when deciding which tool to call. Two
consequences worth keeping in mind when editing:

- Do not add parameters for dependency injection; they would surface to the client as tool
  arguments. Tests substitute the embedding provider through ``set_embedder``.
- The docstrings are product surface, not comments. They say what a tool does *and* what it does
  not, because the neighbouring tool answers that other question.

Error handling is total: ``@structured_errors`` guarantees that no exception escapes into the MCP
protocol, converting anything unmapped into ``{"error": "internal_error", ...}`` with the traceback
logged to stderr. That is what closes QA-3-09 (raw ``TypeError``/``AttributeError``) and QA-3-10
(opaque ``sqlite3`` errors) at this boundary.

Resources are per call: a store is opened, used, and closed inside each tool, so no connection ever
crosses a thread boundary when MCP v2 runs handlers on worker threads (RISK-008). The embedding
provider is the one shared object, because loading its model costs seconds — see ``get_embedder``.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable
from pathlib import Path

from ..embeddings.base import EmbeddingProvider
from ..embeddings.nomic_provider import NomicEmbeddingProvider
from ..graph_queries import (
    collect_node_names,
    condition_function_for,
    conditional_destinations,
    find_node,
    find_route,
    has_conditional_edge,
    nodes_with_unenumerated_tools,
    summarize_graph,
    tool_callers,
)
from ..indexer import index_repository
from ..parser.graph_model import END_SENTINEL, START_SENTINEL, GraphDef
from ..parser.repo_scanner import resolve_repo_root
from ..storage.base import make_repo_id
from ..storage.factory import open_existing_store

logger = logging.getLogger(__name__)

# Error codes returned to the client. Fixed in DEC-018 so all seven tools agree, and stable
# strings because a connected client may branch on them.
ERROR_INVALID_PATH = "invalid_path"
ERROR_PATH_NOT_FOUND = "path_not_found"
ERROR_NOT_A_DIRECTORY = "not_a_directory"
ERROR_NOT_INDEXED = "not_indexed"
ERROR_EMPTY_QUERY = "empty_query"
ERROR_EMPTY_TOOL_NAME = "empty_tool_name"
ERROR_INVALID_TOP_K = "invalid_top_k"
ERROR_UNKNOWN_NODE = "unknown_node"
ERROR_NOT_CONDITIONAL = "not_conditional"
ERROR_INTERNAL = "internal_error"

INDEX_FIRST = "call index_repo first"

# Accepted as route endpoints although they are not nodes: LangGraph's own START/END markers are
# real edge endpoints in the parsed model (DEC-006), so tracing from one is a legitimate question.
_SENTINELS = frozenset({START_SENTINEL, END_SENTINEL})

# --------------------------------------------------------------------------------------------
# Shared embedding provider (DEC-018)
# --------------------------------------------------------------------------------------------
_embedder: EmbeddingProvider | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> EmbeddingProvider:
    """The process-wide embedding provider, created on first use.

    Shared rather than built per call because loading the ONNX model takes seconds and would
    otherwise be paid on every query. claude.md's ban on global state names the *vector store
    connection* — what it protects is connection ownership and thread affinity, neither of which
    applies to a stateless-after-load model handle whose own loading is already lock-guarded
    (DEC-018).
    """
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = NomicEmbeddingProvider()
    return _embedder


def set_embedder(provider: EmbeddingProvider | None) -> None:
    """Replace the shared provider; ``None`` resets it so the next call builds the default.

    The seam that lets tests run without the real model, and the hook a future opt-in cloud
    provider would use (DEC-003 — never the default path).
    """
    global _embedder
    with _embedder_lock:
        _embedder = provider


# --------------------------------------------------------------------------------------------
# Error handling (task 4.9)
# --------------------------------------------------------------------------------------------
def structured_errors(function: Callable[..., dict]) -> Callable[..., dict]:
    """Guarantee a tool returns a dict, converting any escaping exception into one.

    The last line of defence, not the first: expected conditions are mapped explicitly by the
    helpers below, so reaching this handler means something genuinely unforeseen happened. It is
    logged with its traceback to stderr and reported as ``internal_error`` — an unhandled exception
    crossing the MCP boundary is a CRITICAL failure, because the protocol has no way to render a
    Python traceback usefully to the client.
    """

    @functools.wraps(function)
    def wrapper(*args: object, **kwargs: object) -> dict:
        try:
            return function(*args, **kwargs)
        except Exception as exc:  # deliberate total boundary guard, see this function's docstring
            logger.exception("%s failed", function.__name__)
            return {"error": ERROR_INTERNAL, "detail": f"{type(exc).__name__}: {exc}"}

    return wrapper


def _resolve_repo(path: str, *, must_exist: bool = True) -> tuple[Path | None, dict | None]:
    """Validate a caller-supplied repo path, returning ``(repo_root, None)`` or ``(None, error)``.

    Applies DEC-014's validation at every tool boundary — the rejection that closes QA-3-07 (an
    empty path silently meaning the current working directory) and QA-3-08 (``..`` escaping the
    named directory). The resulting ``ValueError`` is surfaced as ``invalid_path``, the third
    ``index_repo`` error case DEC-014 created and DEC-018 names.
    """
    try:
        repo_root = resolve_repo_root(path)
    except (ValueError, TypeError) as exc:
        return None, {"error": ERROR_INVALID_PATH, "path": str(path), "reason": str(exc)}

    if must_exist:
        if not repo_root.exists():
            return None, {"error": ERROR_PATH_NOT_FOUND, "path": repo_root.as_posix()}
        if not repo_root.is_dir():
            return None, {"error": ERROR_NOT_A_DIRECTORY, "path": repo_root.as_posix()}
    return repo_root, None


def _not_indexed(repo_root: Path) -> dict:
    return {
        "error": ERROR_NOT_INDEXED,
        "path": repo_root.as_posix(),
        "suggestion": INDEX_FIRST,
    }


def _read_graphs(repo_root: Path) -> tuple[list[GraphDef] | None, dict | None]:
    """Load a repository's stored graphs, or return the ``not_indexed`` error.

    Opens the store, reads, and closes it — nothing is cached between calls (RISK-008). Uses
    ``open_existing_store`` so asking about an unindexed repository never creates an index file in
    it (DEC-018).
    """
    store = open_existing_store(repo_root)
    if store is None:
        return None, _not_indexed(repo_root)
    try:
        repo_id = make_repo_id(repo_root)
        if not store.is_indexed(repo_id):
            return None, _not_indexed(repo_root)
        return store.list_graphs(repo_id), None
    finally:
        store.close()


def _unknown_node(name: str, graphs: list[GraphDef]) -> dict:
    return {
        "error": ERROR_UNKNOWN_NODE,
        "node": name,
        "valid_nodes": collect_node_names(graphs),
    }


# --------------------------------------------------------------------------------------------
# The seven tools (tasks 4.2–4.8), with the docstrings a client reads (task 4.10)
# --------------------------------------------------------------------------------------------
@structured_errors
def index_repo(path: str) -> dict:
    """Build the searchable index for a LangGraph repository. Run this first.

    Scans every Python file under `path`, extracts each LangGraph `StateGraph` it finds — nodes,
    edges, conditional routing, tool bindings — embeds each node locally, and stores the result.
    Every other tool in this server reads that stored index, so calling one before this returns
    `not_indexed`. Indexing a few dozen node functions typically takes seconds; the first ever run
    also downloads the local embedding model once.

    Use this when the user points you at a repository for the first time, or after they say they
    have changed the graph. It is safe to call twice — indexing replaces a repository's previous
    contents rather than duplicating them.

    Do NOT use this to answer a question about the graph; it returns counts, not structure. Ask
    `get_graph_summary` for what is in the repository. To rebuild an index the user has just
    invalidated by editing code, `reindex` says that intent more clearly.

    Args:
        path: Directory to index. Must be an existing directory, and must not contain `..`.

    Returns:
        `{graphs_found, nodes_indexed, edges_indexed, partial_nodes, backend, duration_ms}` on
        success. `graphs_found: 0` means the scan worked and the repository contains no LangGraph
        usage — a real answer, not an error. `partial_nodes` counts nodes whose function could not
        be statically located (dynamic construction, imports beyond the resolver's depth limit);
        they are indexed and reported, never dropped.
        On failure: `{error: "invalid_path" | "path_not_found" | "not_a_directory", ...}`.
    """
    repo_root, error = _resolve_repo(path)
    if error is not None:
        return error
    # The shared provider is passed in explicitly rather than left to the indexer's default, so
    # writing and reading always use the same model. Without this, set_embedder() would change
    # only the query side and a search would fail on a dimension mismatch against its own index.
    result = index_repository(repo_root, embedder=get_embedder())
    logger.info(
        "Indexed %s: %d graph(s), %d node(s) in %dms",
        repo_root,
        result.graphs_found,
        result.nodes_indexed,
        result.duration_ms,
    )
    return result.to_dict()


@structured_errors
def get_graph_summary(path: str) -> dict:
    """List every LangGraph graph in an indexed repository, with its nodes and entry point.

    The structural overview: for each graph found, its variable name, the file it is built in, its
    entry point, its node and edge counts, and the registered name of every node. Answers "what
    graphs are in this project", "what nodes does this agent have", and "where does execution
    start". These come from parsing the actual graph definition, so they are exact rather than
    inferred.

    Use this to orient yourself before any other question — the node names it returns are the exact
    strings `trace_path`, `explain_conditional`, and (as an edge source) the other structural tools
    expect.

    Do NOT use this to find a node by what it *does*; it does not read node logic. That is
    `semantic_search_nodes`. It also does not show how nodes connect — for that, `trace_path`
    between two of these names, or `explain_conditional` on one of them.

    Args:
        path: Directory of an already-indexed repository.

    Returns:
        `{graphs: [{variable_name, file_path, entry_point, node_count, edge_count, nodes}]}`.
        `entry_point` is null when the graph does not state one unambiguously — for example a
        parallel fan-out with several entries — rather than guessing one of them. An empty `graphs`
        list means the repository was indexed and genuinely contains no graphs.
        On failure: `{error: "not_indexed" | "invalid_path" | "path_not_found" |
        "not_a_directory", ...}`.
    """
    repo_root, error = _resolve_repo(path)
    if error is not None:
        return error
    graphs, error = _read_graphs(repo_root)
    if error is not None:
        return error
    return {"graphs": [summarize_graph(graph) for graph in graphs]}


@structured_errors
def semantic_search_nodes(query: str, path: str, top_k: int = 5) -> dict:
    """Find nodes by what their code does, using natural-language meaning rather than exact text.

    Embeds the query locally and ranks every indexed node's function body, decorators, and
    docstring against it. Use this for conceptual questions where the user does not know the node's
    name: "which node handles authentication", "where does this agent call the database", "what
    validates the user's input".

    Do NOT use this when you already know the node's name — that is `get_graph_summary` for the
    list, or `trace_path` / `explain_conditional` for how it connects. Results are ranked guesses,
    not structural facts: report the top hit as the likely node, and prefer the exact tools above
    when the question is about graph structure.

    Args:
        query: Natural-language description of the behaviour to find. Must not be blank.
        path: Directory of an already-indexed repository.
        top_k: How many results to return, best first. Defaults to 5.

    Returns:
        `{results: [{node_id, node_name, graph_id, file_path, line_start, score, docstring}]}`,
        ordered best first. `score` is cosine similarity in 0..1 — treat close scores as a tie
        rather than a ranking. Results may be an empty list when nothing was indexed.
        On failure: `{error: "empty_query" | "invalid_top_k" | "not_indexed" | "invalid_path" |
        "path_not_found" | "not_a_directory", ...}`.
    """
    if not isinstance(query, str) or not query.strip():
        # QA-3-06: an empty query previously returned confident, meaningless hits.
        return {"error": ERROR_EMPTY_QUERY}
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        return {"error": ERROR_INVALID_TOP_K, "top_k": top_k}

    repo_root, error = _resolve_repo(path)
    if error is not None:
        return error

    store = open_existing_store(repo_root)
    if store is None:
        return _not_indexed(repo_root)
    try:
        repo_id = make_repo_id(repo_root)
        if not store.is_indexed(repo_id):
            return _not_indexed(repo_root)
        query_vector = get_embedder().embed_query(query)
        results = store.search(query_vector, top_k=top_k, filters={"repo_id": repo_id})
    finally:
        store.close()

    return {"results": [result.to_dict() for result in results]}


@structured_errors
def trace_path(from_node: str, to_node: str, path: str) -> dict:
    """Find the route execution takes between two named nodes, including conditional branches.

    Walks the parsed graph from `from_node` to `to_node` and returns the shortest sequence of nodes
    connecting them, plus the detail of any conditional edge on the way. Answers "how does the
    agent get from retrieval to output", "does this node ever reach the error handler", "what runs
    between these two steps". `__start__` and `__end__` are accepted as endpoints, so you can trace
    from the graph's entry or to its exit.

    Do NOT use this to ask what a node does — that is `semantic_search_nodes` or
    `get_graph_summary`. For the full set of destinations leaving one conditional node, rather than
    a route between two, use `explain_conditional`.

    Args:
        from_node: Exact registered name of the starting node, as reported by `get_graph_summary`.
        to_node: Exact registered name of the destination node.
        path: Directory of an already-indexed repository.

    Returns:
        `{path_found: true, graph_id, route: [node_name, ...], conditional_branches: [...]}` when a
        route exists. `route` includes both endpoints. Each conditional branch is
        `{source, target, condition_function, condition_value, value_resolution}`.
        `{path_found: false, detail}` is a real answer meaning no directed path exists — not an
        error. Only *one* shortest route is returned; other routes may also exist.
        When `value_resolution` is `"not_derivable"`, `condition_value` is null: the source listed
        destinations without stating what the router returns. Say "routes to X", never "returns 'X'".
        On failure: `{error: "unknown_node", node, valid_nodes}` when a name is not in the graph,
        or the usual path/index errors.
    """
    repo_root, error = _resolve_repo(path)
    if error is not None:
        return error
    graphs, error = _read_graphs(repo_root)
    if error is not None:
        return error

    known = set(collect_node_names(graphs)) | _SENTINELS
    for name in (from_node, to_node):
        if name not in known:
            return _unknown_node(name, graphs)

    for graph in graphs:
        names = {node.name for node in graph.nodes} | _SENTINELS
        if from_node not in names or to_node not in names:
            continue
        route = find_route(graph, from_node, to_node)
        if route is None:
            continue
        return {
            "path_found": True,
            "graph_id": graph.id,
            "route": list(route.node_names),
            "conditional_branches": [
                {
                    "source": step.source,
                    "target": step.target,
                    "condition_function": step.condition_function,
                    "condition_value": step.condition_value,
                    "value_resolution": step.value_resolution,
                }
                for step in route.conditional_steps
            ],
        }

    return {
        "path_found": False,
        "route": [],
        "conditional_branches": [],
        "detail": (
            f"No directed path from {from_node!r} to {to_node!r}. Both names exist, but no graph "
            f"in this repository connects them in that direction."
        ),
    }


@structured_errors
def what_calls_tool(tool_name: str, path: str) -> dict:
    """Find every graph node that binds or calls a given tool.

    Searches the parsed tool bindings — `ToolNode([...])` actions and `.bind_tools([...])` calls —
    for `tool_name`, and returns the nodes that use it. Answers "which node calls the search tool",
    "where is this tool actually used". A tool bound under a module prefix (`toolkit.search`) is
    found by its bare name.

    Do NOT read an empty `callers` list as proof that nothing uses the tool. Only bindings written
    as literal lists can be enumerated statically; a node that passes its tools as a variable is
    reported separately in `unenumerated_tool_nodes`, and on real codebases that is the common case.
    Check that field before telling the user nothing calls the tool.

    Args:
        tool_name: Exact tool name to look for, e.g. `search_web`. Must not be blank.
        path: Directory of an already-indexed repository.

    Returns:
        `{callers: [{node_name, file_path, tool_name, tool_source}],
        unenumerated_tool_nodes: [{node_name, file_path}]}`. `callers` is every node whose binding
        of this tool was statically resolved. `unenumerated_tool_nodes` is every node that binds
        tools which could not be enumerated at all — those nodes may or may not use this tool, and
        the honest answer for them is "cannot tell from the source".
        On failure: `{error: "empty_tool_name" | "not_indexed" | "invalid_path" |
        "path_not_found" | "not_a_directory", ...}`.
    """
    if not isinstance(tool_name, str) or not tool_name.strip():
        return {"error": ERROR_EMPTY_TOOL_NAME}

    repo_root, error = _resolve_repo(path)
    if error is not None:
        return error
    graphs, error = _read_graphs(repo_root)
    if error is not None:
        return error

    return {
        "callers": [
            {
                "node_name": caller.node_name,
                "file_path": caller.file_path,
                "tool_name": caller.tool_name,
                "tool_source": caller.tool_source,
            }
            for caller in tool_callers(graphs, tool_name)
        ],
        "unenumerated_tool_nodes": [
            {"node_name": node.name, "file_path": node.source_file}
            for _graph, node in nodes_with_unenumerated_tools(graphs)
        ],
    }


@structured_errors
def explain_conditional(edge_source: str, path: str) -> dict:
    """List every destination a conditional edge can route to, and the function that decides.

    For a node that ends in `add_conditional_edges`, returns the routing function's name and all of
    its possible destinations. Answers "what happens after this node if the check fails", "where
    can this branch go", "what are the outcomes of this decision".

    Do NOT use this on an ordinary node — one without a conditional edge returns `not_conditional`,
    and its plain successor is better found with `trace_path`. This tool describes one node's
    branches, not a route across the graph.

    Args:
        edge_source: Exact registered name of the node the conditional edge leaves from.
        path: Directory of an already-indexed repository.

    Returns:
        `{condition_function, possible_destinations: [{condition_value, target_node,
        value_resolution}], note?}`.

        Read `value_resolution` before describing a destination. `"known"` means the source mapped
        that return value to that node, so "returns 'X' → routes to Y" is accurate. `"not_derivable"`
        means `condition_value` is null because the source only listed destinations without saying
        what the router returns — often it returns `Send` objects, not names. In that case say
        "routes to Y" and do not state or invent a trigger value.
        On failure: `{error: "not_conditional", node}` when the node exists but has no conditional
        edge, `{error: "unknown_node", node, valid_nodes}` when the name is not in the graph, or the
        usual path/index errors.
    """
    repo_root, error = _resolve_repo(path)
    if error is not None:
        return error
    graphs, error = _read_graphs(repo_root)
    if error is not None:
        return error

    if find_node(graphs, edge_source) is None:
        return _unknown_node(edge_source, graphs)

    for graph in graphs:
        if not has_conditional_edge(graph, edge_source):
            continue
        destinations = conditional_destinations(graph, edge_source)
        payload = {
            "condition_function": condition_function_for(graph, edge_source),
            "possible_destinations": [
                {
                    "condition_value": destination.condition_value,
                    "target_node": destination.target,
                    "value_resolution": destination.value_resolution,
                }
                for destination in destinations
            ],
        }
        if any(destination.condition_value is None for destination in destinations):
            # DEC-017: the source stated destinations, not return values. Say so explicitly so the
            # answer cannot be phrased as a trigger value the router never produces.
            payload["note"] = (
                "Some destinations have no condition_value because the source lists destinations "
                "rather than mapping return values. Describe these as 'routes to X', not as a "
                "returned value."
            )
        return payload

    return {"error": ERROR_NOT_CONDITIONAL, "node": edge_source}


@structured_errors
def reindex(path: str) -> dict:
    """Rebuild a repository's index from scratch after its code has changed.

    Re-scans, re-parses, and re-embeds everything, replacing the stored index. Use this when the
    user says they have edited the graph — added or removed a node, changed routing, edited a node
    function — and the answers from other tools may now be stale. This server never re-indexes on
    its own; nothing detects file changes, so an index is exactly as current as the last explicit
    index call.

    Identical in effect to `index_repo`, which also replaces rather than appends. Prefer this one
    when the intent is "refresh what you already have" and `index_repo` when the intent is "index
    this for the first time" — the distinction is for the reader, not the engine.

    Args:
        path: Directory of the repository to rebuild. Need not have been indexed before.

    Returns:
        The same shape as `index_repo`:
        `{graphs_found, nodes_indexed, edges_indexed, partial_nodes, backend, duration_ms}`.
        On failure: `{error: "invalid_path" | "path_not_found" | "not_a_directory", ...}`.
    """
    return index_repo(path)


# --------------------------------------------------------------------------------------------
# CLI support — not an MCP tool
# --------------------------------------------------------------------------------------------
def repository_status(path: str) -> dict:
    """Report whether ``path`` has an index, when it was built, and with what.

    Backs the CLI's ``status`` subcommand (prd.md). Deliberately **not** registered as an MCP tool:
    DEC-004 fixes the surface at seven, and a connected client learns the same thing from any tool's
    ``not_indexed`` result. It lives here so every read of the store goes through one module.

    Returns ``{indexed: bool, ...}`` — ``indexed: False`` is a negative answer, not an error, and
    the CLI maps it to exit code 1 (DEC-019).
    """
    repo_root, error = _resolve_repo(path)
    if error is not None:
        return error

    store = open_existing_store(repo_root)
    if store is None:
        return {"indexed": False, "path": repo_root.as_posix(), "suggestion": INDEX_FIRST}
    try:
        repo_id = make_repo_id(repo_root)
        repository = store.get_repository(repo_id)
        if repository is None:
            return {"indexed": False, "path": repo_root.as_posix(), "suggestion": INDEX_FIRST}
        graphs = store.list_graphs(repo_id)
    finally:
        store.close()

    return {
        "indexed": True,
        "path": repo_root.as_posix(),
        "graph_count": len(graphs),
        "node_count": sum(len(graph.nodes) for graph in graphs),
        "edge_count": sum(len(graph.edges) for graph in graphs),
        "last_indexed_at": (
            repository.last_indexed_at.isoformat() if repository.last_indexed_at else None
        ),
        "backend": repository.backend_type,
        "embedding_model": repository.embedding_model,
    }
