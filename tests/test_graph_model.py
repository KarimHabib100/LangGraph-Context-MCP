"""Tests for the frozen graph-model dataclasses and their to_dict() output (Phase 1, task 1.7).

Verifies that every dataclass is immutable and that to_dict() returns exactly the prd.md field
set as JSON-serializable primitives — this output crosses the MCP JSON boundary in Phase 4.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from langgraph_context_mcp.parser.graph_model import (
    EDGE_CONDITIONAL,
    EDGE_NORMAL,
    RESOLUTION_FULL,
    TOOL_RESOLUTION_NOT_APPLICABLE,
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


@pytest.fixture
def node() -> NodeDef:
    return NodeDef(
        id="f.py::g::node::n",
        graph_id="f.py::g",
        name="n",
        source_file="f.py",
        line_start=10,
        line_end=20,
        docstring="does a thing",
        function_body_hash="abc123",
        resolution=RESOLUTION_FULL,
        tool_resolution=TOOL_RESOLUTION_NOT_APPLICABLE,
    )


@pytest.fixture
def edge() -> EdgeDef:
    return EdgeDef(
        id="f.py::g::edge::a-->b",
        graph_id="f.py::g",
        source_node_id="f.py::g::node::a",
        target_node_id="f.py::g::node::b",
        type=EDGE_NORMAL,
        condition_function_name=None,
    )


@pytest.fixture
def route() -> ConditionalRoute:
    return ConditionalRoute(
        id="f.py::g::cedge::a--ok-->b::route",
        edge_id="f.py::g::cedge::a--ok-->b",
        condition_value="ok",
        target_node_id="f.py::g::node::b",
    )


@pytest.fixture
def tool_binding() -> ToolBinding:
    return ToolBinding(
        id="f.py::g::node::n::tool::search",
        node_id="f.py::g::node::n",
        tool_name="search",
        tool_source="mypkg.tools",
    )


# --------------------------------------------------------------------------------------------
# to_dict() field shapes
# --------------------------------------------------------------------------------------------
def test_node_to_dict_shape(node: NodeDef):
    assert node.to_dict() == {
        "id": "f.py::g::node::n",
        "graph_id": "f.py::g",
        "name": "n",
        "source_file": "f.py",
        "line_start": 10,
        "line_end": 20,
        "docstring": "does a thing",
        "function_body_hash": "abc123",
        "resolution": "full",
        "tool_resolution": "not_applicable",
    }


def test_edge_to_dict_shape(edge: EdgeDef):
    assert edge.to_dict() == {
        "id": "f.py::g::edge::a-->b",
        "graph_id": "f.py::g",
        "source_node_id": "f.py::g::node::a",
        "target_node_id": "f.py::g::node::b",
        "type": "normal",
        "condition_function_name": None,
    }


def test_route_to_dict_shape(route: ConditionalRoute):
    assert route.to_dict() == {
        "id": "f.py::g::cedge::a--ok-->b::route",
        "edge_id": "f.py::g::cedge::a--ok-->b",
        "condition_value": "ok",
        "target_node_id": "f.py::g::node::b",
    }


def test_tool_binding_to_dict_shape(tool_binding: ToolBinding):
    assert tool_binding.to_dict() == {
        "id": "f.py::g::node::n::tool::search",
        "node_id": "f.py::g::node::n",
        "tool_name": "search",
        "tool_source": "mypkg.tools",
    }


def test_graph_to_dict_nests_children(
    node: NodeDef, edge: EdgeDef, route: ConditionalRoute, tool_binding: ToolBinding
):
    graph = GraphDef(
        id="f.py::g",
        repo_id="/repo",
        file_path="f.py",
        variable_name="g",
        entry_point="n",
        nodes=(node,),
        edges=(edge,),
        conditional_routes=(route,),
        tool_bindings=(tool_binding,),
    )
    result = graph.to_dict()

    assert result["id"] == "f.py::g"
    assert result["repo_id"] == "/repo"
    assert result["file_path"] == "f.py"
    assert result["variable_name"] == "g"
    assert result["entry_point"] == "n"
    assert result["nodes"] == [node.to_dict()]
    assert result["edges"] == [edge.to_dict()]
    assert result["conditional_routes"] == [route.to_dict()]
    assert result["tool_bindings"] == [tool_binding.to_dict()]


def test_graph_defaults_empty_children():
    graph = GraphDef(
        id="f.py::g",
        repo_id="/repo",
        file_path="f.py",
        variable_name="g",
        entry_point=None,
    )
    result = graph.to_dict()
    assert result["entry_point"] is None
    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["conditional_routes"] == []
    assert result["tool_bindings"] == []


# --------------------------------------------------------------------------------------------
# JSON serializability and immutability
# --------------------------------------------------------------------------------------------
def test_full_graph_to_dict_is_json_serializable(
    node: NodeDef, edge: EdgeDef, route: ConditionalRoute, tool_binding: ToolBinding
):
    graph = GraphDef(
        id="f.py::g",
        repo_id="/repo",
        file_path="f.py",
        variable_name="g",
        entry_point="n",
        nodes=(node,),
        edges=(edge,),
        conditional_routes=(route,),
        tool_bindings=(tool_binding,),
    )
    reloaded = json.loads(json.dumps(graph.to_dict()))
    assert reloaded["nodes"][0]["name"] == "n"


@pytest.mark.parametrize("frozen_obj_fixture", ["node", "edge", "route", "tool_binding"])
def test_dataclasses_are_frozen(frozen_obj_fixture, request):
    obj = request.getfixturevalue(frozen_obj_fixture)
    with pytest.raises(dataclasses.FrozenInstanceError):
        obj.id = "mutated"  # type: ignore[misc]


# --------------------------------------------------------------------------------------------
# Deterministic ID helpers
# --------------------------------------------------------------------------------------------
def test_id_helpers_are_deterministic_and_composable():
    graph_id = make_graph_id("f.py", "g")
    assert graph_id == "f.py::g"
    assert make_graph_id("f.py", "g") == graph_id  # deterministic

    node_id = make_node_id(graph_id, "n")
    assert node_id == "f.py::g::node::n"

    normal_edge = make_edge_id(graph_id, "a", "b", EDGE_NORMAL)
    assert normal_edge == "f.py::g::edge::a-->b"

    cond_edge = make_edge_id(graph_id, "a", "b", EDGE_CONDITIONAL, condition_value="ok")
    assert cond_edge == "f.py::g::cedge::a--ok-->b"

    assert make_route_id(cond_edge) == f"{cond_edge}::route"
    assert make_tool_binding_id(node_id, "search") == f"{node_id}::tool::search"
