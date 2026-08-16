"""Tests for the pure graph query layer (DEC-018).

These run on hand-built ``GraphDef`` objects with no store, no embedder, and no MCP layer — the
point of extracting this module is that route-finding can be tested on its own, including shapes
that are awkward to produce from real source (a cycle, a disconnected pair, a multi-graph repo).
"""

from __future__ import annotations

import pytest
from support import build_graph

from langgraph_context_mcp.graph_queries import (
    collect_node_names,
    condition_function_for,
    conditional_destinations,
    find_node,
    find_route,
    has_conditional_edge,
    node_name_from_id,
    nodes_with_unenumerated_tools,
    summarize_graph,
    tool_callers,
)
from langgraph_context_mcp.parser.graph_model import (
    CONDITION_VALUE_KNOWN,
    CONDITION_VALUE_NOT_DERIVABLE,
    EDGE_CONDITIONAL,
    EDGE_NORMAL,
    END_SENTINEL,
    RESOLUTION_FULL,
    START_SENTINEL,
    TOOL_RESOLUTION_NOT_APPLICABLE,
    TOOL_RESOLUTION_PARTIAL,
    ConditionalRoute,
    EdgeDef,
    GraphDef,
    NodeDef,
    ToolBinding,
    make_edge_id,
    make_graph_id,
    make_node_id,
    make_route_id,
    make_tool_binding_id,
)

GRAPH_ID = make_graph_id("agent.py", "graph")


def _node(name: str, *, tool_resolution: str = TOOL_RESOLUTION_NOT_APPLICABLE) -> NodeDef:
    return NodeDef(
        id=make_node_id(GRAPH_ID, name),
        graph_id=GRAPH_ID,
        name=name,
        source_file="agent.py",
        line_start=1,
        line_end=2,
        docstring=None,
        function_body_hash=f"hash-{name}",
        resolution=RESOLUTION_FULL,
        tool_resolution=tool_resolution,
    )


def _edge(source: str, target: str) -> EdgeDef:
    return EdgeDef(
        id=make_edge_id(GRAPH_ID, source, target, EDGE_NORMAL),
        graph_id=GRAPH_ID,
        source_node_id=make_node_id(GRAPH_ID, source),
        target_node_id=make_node_id(GRAPH_ID, target),
        type=EDGE_NORMAL,
        condition_function_name=None,
    )


def _conditional(
    source: str,
    target: str,
    condition_value: str | None,
    value_resolution: str,
    function_name: str = "router",
) -> tuple[EdgeDef, ConditionalRoute]:
    edge_id = make_edge_id(GRAPH_ID, source, target, EDGE_CONDITIONAL, condition_value or target)
    edge = EdgeDef(
        id=edge_id,
        graph_id=GRAPH_ID,
        source_node_id=make_node_id(GRAPH_ID, source),
        target_node_id=make_node_id(GRAPH_ID, target),
        type=EDGE_CONDITIONAL,
        condition_function_name=function_name,
    )
    route = ConditionalRoute(
        id=make_route_id(edge_id),
        edge_id=edge_id,
        condition_value=condition_value,
        target_node_id=make_node_id(GRAPH_ID, target),
        value_resolution=value_resolution,
    )
    return edge, route


def _graph(nodes, edges=(), routes=(), bindings=()) -> GraphDef:
    return GraphDef(
        id=GRAPH_ID,
        repo_id="/repo",
        file_path="agent.py",
        variable_name="graph",
        entry_point=nodes[0].name if nodes else None,
        nodes=tuple(nodes),
        edges=tuple(edges),
        conditional_routes=tuple(routes),
        tool_bindings=tuple(bindings),
    )


# --------------------------------------------------------------------------------------------
# find_route
# --------------------------------------------------------------------------------------------
def test_find_route_returns_nodes_in_order():
    nodes = [_node(n) for n in ("a", "b", "c")]
    graph = _graph(nodes, [_edge("a", "b"), _edge("b", "c")])

    route = find_route(graph, "a", "c")

    assert route is not None
    assert route.node_names == ("a", "b", "c")
    assert len(route.steps) == 2


def test_find_route_returns_none_when_unreachable():
    nodes = [_node(n) for n in ("a", "b", "c")]
    graph = _graph(nodes, [_edge("a", "b")])

    assert find_route(graph, "a", "c") is None


def test_find_route_respects_direction():
    """An edge a->b does not make b->a reachable."""
    graph = _graph([_node("a"), _node("b")], [_edge("a", "b")])

    assert find_route(graph, "a", "b") is not None
    assert find_route(graph, "b", "a") is None


def test_find_route_to_self_is_a_zero_hop_route():
    graph = _graph([_node("a")], [])

    route = find_route(graph, "a", "a")

    assert route is not None
    assert route.node_names == ("a",)
    assert route.steps == ()


def test_find_route_terminates_on_a_cycle():
    """A cycle must not loop forever — the visited set is what guarantees this."""
    nodes = [_node(n) for n in ("a", "b", "c")]
    graph = _graph(nodes, [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")])

    route = find_route(graph, "a", "c")

    assert route is not None
    assert route.node_names == ("a", "b", "c")
    assert find_route(graph, "c", "b").node_names == ("c", "a", "b")


def test_find_route_takes_the_shortest_of_several_paths():
    nodes = [_node(n) for n in ("a", "b", "c", "d")]
    graph = _graph(
        nodes,
        [_edge("a", "b"), _edge("b", "d"), _edge("a", "c"), _edge("c", "b"), _edge("a", "d")],
    )

    route = find_route(graph, "a", "d")

    assert route.node_names == ("a", "d")


def test_find_route_traverses_sentinels():
    nodes = [_node("a")]
    graph = _graph(nodes, [_edge(START_SENTINEL, "a"), _edge("a", END_SENTINEL)])

    route = find_route(graph, START_SENTINEL, END_SENTINEL)

    assert route.node_names == (START_SENTINEL, "a", END_SENTINEL)


def test_find_route_reports_conditional_hops_with_their_branch_facts():
    nodes = [_node(n) for n in ("gate", "ok", "fail")]
    edge, route_def = _conditional("gate", "ok", "authorized", CONDITION_VALUE_KNOWN)
    graph = _graph(nodes, [edge], [route_def])

    route = find_route(graph, "gate", "ok")

    assert len(route.conditional_steps) == 1
    step = route.conditional_steps[0]
    assert step.is_conditional is True
    assert step.condition_function == "router"
    assert step.condition_value == "authorized"
    assert step.value_resolution == CONDITION_VALUE_KNOWN


def test_normal_hops_are_not_reported_as_conditional():
    graph = _graph([_node("a"), _node("b")], [_edge("a", "b")])

    route = find_route(graph, "a", "b")

    assert route.conditional_steps == ()
    assert route.steps[0].value_resolution is None


def test_conditional_step_with_no_derivable_value_reports_none():
    """DEC-017: a list-form path_map states destinations, never a return value."""
    edge, route_def = _conditional("gate", "ok", None, CONDITION_VALUE_NOT_DERIVABLE)
    graph = _graph([_node("gate"), _node("ok")], [edge], [route_def])

    step = find_route(graph, "gate", "ok").conditional_steps[0]

    assert step.condition_value is None
    assert step.value_resolution == CONDITION_VALUE_NOT_DERIVABLE


# --------------------------------------------------------------------------------------------
# conditional_destinations / condition_function_for / has_conditional_edge
# --------------------------------------------------------------------------------------------
def test_conditional_destinations_lists_every_branch():
    edge_ok, route_ok = _conditional("gate", "ok", "authorized", CONDITION_VALUE_KNOWN)
    edge_no, route_no = _conditional("gate", "fail", "denied", CONDITION_VALUE_KNOWN)
    graph = _graph(
        [_node("gate"), _node("ok"), _node("fail")], [edge_ok, edge_no], [route_ok, route_no]
    )

    destinations = conditional_destinations(graph, "gate")

    assert [(d.target, d.condition_value) for d in destinations] == [
        ("ok", "authorized"),
        ("fail", "denied"),
    ]
    assert all(d.value_resolution == CONDITION_VALUE_KNOWN for d in destinations)


def test_conditional_destinations_is_empty_for_a_plain_node():
    graph = _graph([_node("a"), _node("b")], [_edge("a", "b")])

    assert conditional_destinations(graph, "a") == []
    assert has_conditional_edge(graph, "a") is False


def test_condition_function_is_reported():
    edge, route = _conditional("gate", "ok", "yes", CONDITION_VALUE_KNOWN, "decide_next")
    graph = _graph([_node("gate"), _node("ok")], [edge], [route])

    assert condition_function_for(graph, "gate") == "decide_next"
    assert condition_function_for(graph, "ok") is None


def test_destination_without_a_mirroring_route_defaults_to_not_derivable():
    """Missing route metadata must never be reported as a known value."""
    edge, _route = _conditional("gate", "ok", "yes", CONDITION_VALUE_KNOWN)
    graph = _graph([_node("gate"), _node("ok")], [edge], [])  # route deliberately omitted

    destination = conditional_destinations(graph, "gate")[0]

    assert destination.condition_value is None
    assert destination.value_resolution == CONDITION_VALUE_NOT_DERIVABLE


# --------------------------------------------------------------------------------------------
# tool_callers
# --------------------------------------------------------------------------------------------
def _binding(node_name: str, tool_name: str, source: str | None = None) -> ToolBinding:
    node_id = make_node_id(GRAPH_ID, node_name)
    return ToolBinding(
        id=make_tool_binding_id(node_id, tool_name),
        node_id=node_id,
        tool_name=tool_name,
        tool_source=source,
    )


def test_tool_callers_finds_the_binding_node():
    graph = _graph([_node("tools")], bindings=[_binding("tools", "search_web", "my.tools")])

    callers = tool_callers([graph], "search_web")

    assert [(c.node_name, c.tool_source) for c in callers] == [("tools", "my.tools")]


def test_tool_callers_matches_the_final_dotted_segment():
    graph = _graph([_node("tools")], bindings=[_binding("tools", "toolkit.search")])

    assert [c.node_name for c in tool_callers([graph], "search")] == ["tools"]


def test_tool_callers_is_empty_for_an_unbound_tool():
    graph = _graph([_node("tools")], bindings=[_binding("tools", "search_web")])

    assert tool_callers([graph], "nonexistent") == []


def test_nodes_with_unenumerated_tools_are_reported_separately():
    """DEF-004: a node binding a variable is disclosed, not silently absent."""
    graph = _graph(
        [_node("a"), _node("dynamic", tool_resolution=TOOL_RESOLUTION_PARTIAL)],
    )

    partial = nodes_with_unenumerated_tools([graph])

    assert [node.name for _graph, node in partial] == ["dynamic"]


# --------------------------------------------------------------------------------------------
# Naming, lookup, summary
# --------------------------------------------------------------------------------------------
def test_node_name_from_id_round_trips_including_sentinels():
    assert node_name_from_id(GRAPH_ID, make_node_id(GRAPH_ID, "alpha")) == "alpha"
    assert node_name_from_id(GRAPH_ID, make_node_id(GRAPH_ID, START_SENTINEL)) == START_SENTINEL


def test_collect_node_names_is_sorted_deduplicated_and_excludes_sentinels():
    graph_a = build_graph("/repo", "a", ["beta", "alpha"])
    graph_b = build_graph("/repo", "b", ["alpha", "gamma"])

    assert collect_node_names([graph_a, graph_b]) == ["alpha", "beta", "gamma"]


def test_find_node_searches_across_graphs():
    graph_a = build_graph("/repo", "a", ["alpha"])
    graph_b = build_graph("/repo", "b", ["beta"])

    found = find_node([graph_a, graph_b], "beta")

    assert found is not None
    assert found[0].id == graph_b.id
    assert found[1].name == "beta"


def test_find_node_returns_none_when_absent():
    assert find_node([build_graph("/repo", "a", ["alpha"])], "missing") is None


def test_summarize_graph_carries_prd_keys_plus_node_names():
    graph = build_graph("/repo", "a", ["alpha", "beta"])

    summary = summarize_graph(graph)

    assert set(summary) == {
        "variable_name",
        "file_path",
        "entry_point",
        "node_count",
        "edge_count",
        "nodes",
    }
    assert summary["node_count"] == 2
    assert summary["nodes"] == ["alpha", "beta"]


@pytest.mark.parametrize("missing_name", ["", "   ", "Alpha"])
def test_lookups_are_exact_and_do_not_guess(missing_name):
    """Node lookup is case- and whitespace-exact: a near miss is a miss, not a fuzzy match."""
    graph = build_graph("/repo", "a", ["alpha"])

    assert find_node([graph], missing_name) is None
