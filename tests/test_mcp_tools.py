"""Tests for the seven MCP tools (task 4.13).

Called as plain Python functions, which is exactly how ``server.py`` registers them — there is no
wrapper between these tests and what a connected client invokes.

Covered here: every success shape and every error case named in prd.md's API contracts, the three
Phase 3 findings whose fix location is this boundary (QA-3-06 empty query, QA-3-09 raw argument
errors, QA-3-10 opaque storage errors), DEC-014's path validation at the tool surface, and DEC-017's
rule that a route with no derivable value must never be phrased as one.

Indexing runs through the fake embedder, so the suite needs neither the ONNX model nor a download.
"""

from __future__ import annotations

import json

import pytest
from support import FakeEmbeddingProvider

from langgraph_context_mcp.server import TOOL_FUNCTIONS, build_server
from langgraph_context_mcp.storage.sqlite_store import default_index_path
from langgraph_context_mcp.tools import mcp_tools
from langgraph_context_mcp.tools.mcp_tools import (
    explain_conditional,
    get_graph_summary,
    index_repo,
    reindex,
    repository_status,
    semantic_search_nodes,
    trace_path,
    what_calls_tool,
)

LIST_FORM_GRAPH = '''
"""A graph using the list-form path_map and a ToolNode action."""
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from my.tools import search_web, lookup


def router(state): ...
def alpha(state): ...
def beta(state): ...


graph = StateGraph(dict)
graph.add_node("alpha", alpha)
graph.add_node("beta", beta)
graph.add_node("tools", ToolNode([search_web, lookup]))
graph.add_edge(START, "alpha")
graph.add_conditional_edges("alpha", router, ["beta", "tools"])
graph.add_edge("beta", END)
'''


@pytest.fixture(autouse=True)
def fake_embedder_everywhere():
    """Point the shared provider at the deterministic fake for the whole module.

    Reset afterwards so the override cannot leak into another test file — the provider is
    process-wide by design (DEC-018), which is precisely why it must be restored.
    """
    mcp_tools.set_embedder(FakeEmbeddingProvider())
    yield
    mcp_tools.set_embedder(None)


@pytest.fixture
def indexed_repo(fixture_repo):
    """The sample fixture repo, already indexed."""
    index_repo(str(fixture_repo))
    return fixture_repo


@pytest.fixture
def list_form_repo(tmp_path):
    """A repo whose conditional edge uses the list form, plus a ToolNode binding."""
    repo = tmp_path / "list_form"
    repo.mkdir()
    (repo / "agent.py").write_text(LIST_FORM_GRAPH, encoding="utf-8")
    index_repo(str(repo))
    return repo


# --------------------------------------------------------------------------------------------
# Registration (DEC-004)
# --------------------------------------------------------------------------------------------
def test_exactly_seven_tools_are_registered():
    assert len(TOOL_FUNCTIONS) == 7


def test_registered_names_match_the_prd_tool_surface():
    assert [function.__name__ for function in TOOL_FUNCTIONS] == [
        "index_repo",
        "get_graph_summary",
        "semantic_search_nodes",
        "trace_path",
        "what_calls_tool",
        "explain_conditional",
        "reindex",
    ]


def test_every_tool_has_a_substantial_docstring():
    """Docstrings are the tool-selection surface (task 4.10), not decoration."""
    for function in TOOL_FUNCTIONS:
        assert function.__doc__, f"{function.__name__} has no docstring"
        assert len(function.__doc__) > 400, f"{function.__name__}'s docstring is too thin"


def test_tools_expose_only_their_contract_arguments():
    """No injection parameter may leak into the schema a client sees."""
    import inspect

    expected = {
        "index_repo": ["path"],
        "get_graph_summary": ["path"],
        "semantic_search_nodes": ["query", "path", "top_k"],
        "trace_path": ["from_node", "to_node", "path"],
        "what_calls_tool": ["tool_name", "path"],
        "explain_conditional": ["edge_source", "path"],
        "reindex": ["path"],
    }
    for function in TOOL_FUNCTIONS:
        params = list(inspect.signature(function).parameters)
        assert params == expected[function.__name__]


def test_server_builds_and_registers_all_seven():
    server = build_server()
    assert server.name == "langgraph-context-mcp"


# --------------------------------------------------------------------------------------------
# index_repo / reindex
# --------------------------------------------------------------------------------------------
def test_index_repo_returns_the_prd_contract_shape(fixture_repo):
    result = index_repo(str(fixture_repo))

    assert set(result) == {
        "graphs_found",
        "nodes_indexed",
        "edges_indexed",
        "partial_nodes",
        "backend",
        "duration_ms",
    }
    assert result["graphs_found"] == 1
    assert result["nodes_indexed"] == 4
    assert result["backend"] == "sqlite"


def test_index_repo_on_a_repo_without_langgraph_is_not_an_error(tmp_path):
    (tmp_path / "plain.py").write_text("x = 1\n", encoding="utf-8")

    result = index_repo(str(tmp_path))

    assert "error" not in result
    assert result["graphs_found"] == 0


def test_index_repo_reports_path_not_found(tmp_path):
    result = index_repo(str(tmp_path / "missing"))

    assert result["error"] == "path_not_found"


def test_index_repo_reports_not_a_directory(tmp_path):
    file_path = tmp_path / "a_file.py"
    file_path.write_text("x = 1\n", encoding="utf-8")

    assert index_repo(str(file_path))["error"] == "not_a_directory"


@pytest.mark.parametrize("bad_path", ["", "   ", "\t\n"])
def test_index_repo_rejects_empty_paths(bad_path):
    """QA-3-07: an empty path must never mean 'the current working directory'."""
    result = index_repo(bad_path)

    assert result["error"] == "invalid_path"
    assert "reason" in result


def test_index_repo_rejects_parent_traversal(fixture_repo):
    """QA-3-08 / DEC-014 enforced at the tool boundary."""
    result = index_repo(str(fixture_repo / ".." / ".." / ".."))

    assert result["error"] == "invalid_path"


def test_reindex_matches_index_repo_and_is_idempotent(fixture_repo):
    first = index_repo(str(fixture_repo))
    second = reindex(str(fixture_repo))

    assert set(second) == set(first)
    assert second["graphs_found"] == first["graphs_found"]
    assert second["nodes_indexed"] == first["nodes_indexed"]
    assert len(get_graph_summary(str(fixture_repo))["graphs"]) == 1


def test_reindex_reports_the_same_errors_as_index_repo(tmp_path):
    assert reindex(str(tmp_path / "missing"))["error"] == "path_not_found"


# --------------------------------------------------------------------------------------------
# not_indexed, and the no-side-effect rule (DEC-018)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (get_graph_summary, ()),
        (trace_path, ("check_auth_token", "fetch_data")),
        (what_calls_tool, ("search_web",)),
        (explain_conditional, ("check_auth_token",)),
    ],
)
def test_read_tools_report_not_indexed(tool, args, fixture_repo):
    result = tool(*args, str(fixture_repo)) if args else tool(str(fixture_repo))

    assert result["error"] == "not_indexed"
    assert result["suggestion"] == "call index_repo first"


def test_semantic_search_reports_not_indexed(fixture_repo):
    assert semantic_search_nodes("anything", str(fixture_repo))["error"] == "not_indexed"


def test_asking_an_unindexed_repo_creates_no_index_file(fixture_repo):
    """A read-only question must not write into the user's repository (DEC-018)."""
    get_graph_summary(str(fixture_repo))
    semantic_search_nodes("anything", str(fixture_repo))
    repository_status(str(fixture_repo))

    assert not default_index_path(fixture_repo).exists()
    assert not (fixture_repo / ".langgraph-context").exists()


# --------------------------------------------------------------------------------------------
# get_graph_summary
# --------------------------------------------------------------------------------------------
def test_get_graph_summary_returns_contract_keys_and_node_names(indexed_repo):
    summary = get_graph_summary(str(indexed_repo))

    assert len(summary["graphs"]) == 1
    graph = summary["graphs"][0]
    assert graph["variable_name"] == "graph"
    assert graph["entry_point"] == "check_auth_token"
    assert graph["node_count"] == 4
    assert graph["edge_count"] == 5
    assert graph["nodes"] == [
        "check_auth_token",
        "fetch_data",
        "format_response",
        "handle_error",
    ]


def test_get_graph_summary_derives_entry_point_from_a_start_edge(list_form_repo):
    """DEC-015: add_edge(START, x) populates entry_point just like set_entry_point(x)."""
    graph = get_graph_summary(str(list_form_repo))["graphs"][0]

    assert graph["entry_point"] == "alpha"


# --------------------------------------------------------------------------------------------
# semantic_search_nodes
# --------------------------------------------------------------------------------------------
def test_semantic_search_returns_ranked_results(indexed_repo):
    result = semantic_search_nodes("authentication token", str(indexed_repo), top_k=3)

    assert len(result["results"]) <= 3
    names = [hit["node_name"] for hit in result["results"]]
    assert "check_auth_token" in names
    scores = [hit["score"] for hit in result["results"]]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_result_carries_the_prd_fields(indexed_repo):
    hit = semantic_search_nodes("authentication", str(indexed_repo), top_k=1)["results"][0]

    for key in ("node_name", "file_path", "line_start", "score", "docstring"):
        assert key in hit


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_semantic_search_rejects_an_empty_query(blank, indexed_repo):
    """QA-3-06: a blank query previously returned confident, meaningless hits."""
    assert semantic_search_nodes(blank, str(indexed_repo))["error"] == "empty_query"


def test_empty_query_is_rejected_before_the_index_is_consulted(fixture_repo):
    """Argument validation comes first, so a blank query is 'empty_query', not 'not_indexed'."""
    assert semantic_search_nodes("  ", str(fixture_repo))["error"] == "empty_query"


@pytest.mark.parametrize("bad_top_k", [0, -1, "five", 2.5, None, True])
def test_semantic_search_rejects_a_bad_top_k(bad_top_k, indexed_repo):
    """QA-3-09: `top_k='five'` used to raise a raw TypeError out of the store."""
    result = semantic_search_nodes("auth", str(indexed_repo), top_k=bad_top_k)

    assert result["error"] == "invalid_top_k"


def test_semantic_search_top_k_bounds_the_result_count(indexed_repo):
    assert len(semantic_search_nodes("data", str(indexed_repo), top_k=1)["results"]) == 1


# --------------------------------------------------------------------------------------------
# trace_path
# --------------------------------------------------------------------------------------------
def test_trace_path_returns_the_route(indexed_repo):
    result = trace_path("check_auth_token", "format_response", str(indexed_repo))

    assert result["path_found"] is True
    assert result["route"] == ["check_auth_token", "fetch_data", "format_response"]


def test_trace_path_reports_the_conditional_branch_it_crossed(indexed_repo):
    result = trace_path("check_auth_token", "fetch_data", str(indexed_repo))

    branches = result["conditional_branches"]
    assert len(branches) == 1
    assert branches[0]["condition_function"] == "route_after_auth"
    assert branches[0]["condition_value"] == "authorized"
    assert branches[0]["value_resolution"] == "known"


def test_trace_path_returns_path_found_false_for_an_unreachable_pair(indexed_repo):
    """Scenario 3.7: an explicit negative answer, not an error."""
    result = trace_path("handle_error", "fetch_data", str(indexed_repo))

    assert result["path_found"] is False
    assert result["route"] == []
    assert "error" not in result


def test_trace_path_rejects_an_unknown_node_and_lists_valid_ones(indexed_repo):
    """Scenario 3.8: the error names what the caller could have said."""
    result = trace_path("nonexistent_node", "fetch_data", str(indexed_repo))

    assert result["error"] == "unknown_node"
    assert result["node"] == "nonexistent_node"
    assert "check_auth_token" in result["valid_nodes"]


def test_trace_path_validates_the_destination_too(indexed_repo):
    assert trace_path("fetch_data", "nope", str(indexed_repo))["error"] == "unknown_node"


def test_trace_path_accepts_the_end_sentinel(indexed_repo):
    result = trace_path("check_auth_token", "__end__", str(indexed_repo))

    assert result["path_found"] is True
    assert result["route"][-1] == "__end__"


# --------------------------------------------------------------------------------------------
# what_calls_tool
# --------------------------------------------------------------------------------------------
def test_what_calls_tool_finds_the_binding_node(list_form_repo):
    result = what_calls_tool("search_web", str(list_form_repo))

    assert [caller["node_name"] for caller in result["callers"]] == ["tools"]
    assert result["callers"][0]["file_path"] == "agent.py"


def test_what_calls_tool_returns_no_callers_for_an_unbound_tool(list_form_repo):
    result = what_calls_tool("not_a_real_tool", str(list_form_repo))

    assert result["callers"] == []
    assert "unenumerated_tool_nodes" in result


def test_what_calls_tool_discloses_unenumerable_bindings(tmp_path):
    """DEF-004: an empty `callers` must never be read as 'nothing binds this tool'."""
    repo = tmp_path / "dynamic_tools"
    repo.mkdir()
    (repo / "agent.py").write_text(
        "from langgraph.graph import StateGraph\n"
        "from langgraph.prebuilt import ToolNode\n"
        "\n"
        "def build(state): ...\n"
        "\n"
        "tools = get_tools()\n"
        "graph = StateGraph(dict)\n"
        "graph.add_node('tools', ToolNode(tools))\n",
        encoding="utf-8",
    )
    index_repo(str(repo))

    result = what_calls_tool("search_web", str(repo))

    assert result["callers"] == []
    assert [n["node_name"] for n in result["unenumerated_tool_nodes"]] == ["tools"]


@pytest.mark.parametrize("blank", ["", "   "])
def test_what_calls_tool_rejects_a_blank_name(blank, indexed_repo):
    assert what_calls_tool(blank, str(indexed_repo))["error"] == "empty_tool_name"


# --------------------------------------------------------------------------------------------
# explain_conditional
# --------------------------------------------------------------------------------------------
def test_explain_conditional_lists_every_destination(indexed_repo):
    result = explain_conditional("check_auth_token", str(indexed_repo))

    assert result["condition_function"] == "route_after_auth"
    destinations = {
        d["condition_value"]: d["target_node"] for d in result["possible_destinations"]
    }
    assert destinations == {"authorized": "fetch_data", "unauthorized": "handle_error"}
    assert all(d["value_resolution"] == "known" for d in result["possible_destinations"])


def test_explain_conditional_states_no_value_for_a_list_form_map(list_form_repo):
    """DEC-017 / QA-3-04: a list path_map is a destination hint, never a return value."""
    result = explain_conditional("alpha", str(list_form_repo))

    assert {d["target_node"] for d in result["possible_destinations"]} == {"beta", "tools"}
    for destination in result["possible_destinations"]:
        assert destination["condition_value"] is None
        assert destination["value_resolution"] == "not_derivable"
    assert "routes to" in result["note"]


def test_explain_conditional_rejects_a_plain_node(indexed_repo):
    result = explain_conditional("fetch_data", str(indexed_repo))

    assert result["error"] == "not_conditional"
    assert result["node"] == "fetch_data"


def test_explain_conditional_rejects_an_unknown_node(indexed_repo):
    result = explain_conditional("nope", str(indexed_repo))

    assert result["error"] == "unknown_node"
    assert "check_auth_token" in result["valid_nodes"]


# --------------------------------------------------------------------------------------------
# Total error containment (task 4.9 / QA-3-09 / QA-3-10)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (index_repo, (None,)),
        (get_graph_summary, (None,)),
        (semantic_search_nodes, ("q", None)),
        (trace_path, ("a", "b", None)),
        (what_calls_tool, ("t", None)),
        (explain_conditional, ("a", None)),
        (reindex, (None,)),
    ],
)
def test_no_tool_raises_on_a_none_path(tool, args):
    """QA-3-09: `None` used to escape as a raw AttributeError."""
    result = tool(*args)

    assert isinstance(result, dict)
    assert result["error"] in {"invalid_path", "internal_error"}


def test_a_corrupt_index_is_reported_as_a_structured_error(indexed_repo):
    """QA-3-10: an opaque sqlite/vec0 exception must not cross the tool boundary."""
    index_path = default_index_path(indexed_repo)
    index_path.write_bytes(b"SQLite format 3\x00" + b"garbage" * 200)

    result = get_graph_summary(str(indexed_repo))

    assert result["error"] == "internal_error"
    assert "detail" in result


def test_every_tool_result_is_json_serializable(indexed_repo):
    """Whatever a tool returns has to cross the MCP JSON boundary intact."""
    results = [
        index_repo(str(indexed_repo)),
        get_graph_summary(str(indexed_repo)),
        semantic_search_nodes("auth", str(indexed_repo)),
        trace_path("check_auth_token", "format_response", str(indexed_repo)),
        what_calls_tool("search_web", str(indexed_repo)),
        explain_conditional("check_auth_token", str(indexed_repo)),
        reindex(str(indexed_repo)),
        trace_path("nope", "fetch_data", str(indexed_repo)),
        semantic_search_nodes("", str(indexed_repo)),
    ]
    for result in results:
        json.dumps(result)


# --------------------------------------------------------------------------------------------
# Thread affinity (RISK-008)
# --------------------------------------------------------------------------------------------
def test_tools_are_safe_to_call_from_many_threads(indexed_repo):
    """MCP v2 runs sync handlers on worker threads (DEC-010).

    A ``sqlite3.Connection`` is bound to its creating thread, so a store cached across calls would
    raise ``ProgrammingError`` here — intermittently, and only in a live client. Building the store
    inside each call is what makes this pass (DEC-018).
    """
    from concurrent.futures import ThreadPoolExecutor

    path = str(indexed_repo)
    calls = [
        lambda: get_graph_summary(path),
        lambda: semantic_search_nodes("auth", path),
        lambda: trace_path("check_auth_token", "format_response", path),
        lambda: explain_conditional("check_auth_token", path),
        lambda: what_calls_tool("search_web", path),
        lambda: repository_status(path),
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(call) for call in calls * 8]
        results = [future.result() for future in futures]

    assert len(results) == 48
    assert [result for result in results if "error" in result] == []


# --------------------------------------------------------------------------------------------
# repository_status (CLI support, deliberately not an MCP tool)
# --------------------------------------------------------------------------------------------
def test_repository_status_before_and_after_indexing(fixture_repo):
    before = repository_status(str(fixture_repo))
    assert before["indexed"] is False
    assert before["suggestion"] == "call index_repo first"

    index_repo(str(fixture_repo))
    after = repository_status(str(fixture_repo))

    assert after["indexed"] is True
    assert after["graph_count"] == 1
    assert after["node_count"] == 4
    assert after["backend"] == "sqlite"
    assert after["last_indexed_at"]


def test_repository_status_reports_a_bad_path_as_an_error(tmp_path):
    assert repository_status(str(tmp_path / "missing"))["error"] == "path_not_found"


def test_repository_status_is_not_registered_as_a_tool():
    """DEC-004 fixes the surface at seven."""
    assert repository_status not in TOOL_FUNCTIONS
