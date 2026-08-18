"""Immutable graph-model dataclasses produced by the parser layer.

Every class here is a frozen dataclass with a ``to_dict()`` method that returns plain,
JSON-serializable Python primitives (str / int / None / list / dict) only. This output
eventually crosses the MCP JSON boundary in Phase 4, so no custom object may leak into a
``to_dict()`` result.

The logical field set for each entity comes from prd.md's Data Models section. ``GraphDef``
additionally aggregates its own children (nodes, edges, conditional routes, tool bindings) as
tuples so a single graph is one self-contained object — see DEC-006 in decisions.md.

IDs are deterministic strings built by the ``make_*_id`` helpers below, so the same source
always yields the same IDs (required for idempotent reindex in later phases).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# LangGraph's own sentinel node names. START/END are not backed by a NodeDef; edges may still
# reference them as a source or target. These string values match langgraph.graph.START / END.
START_SENTINEL = "__start__"
END_SENTINEL = "__end__"

# Resolution states for a NodeDef — whether the node *function itself* was located:
#   full           — the node function was located and read
#   partial        — genuine uncertainty: the action IS a function but could not be resolved
#                    (imported beyond the depth cap, built in a loop, a factory call, a lambda)
#   not_applicable — the action was never a Python function to resolve (a bare ToolNode([...])
#                    instantiation used directly as the node action)
RESOLUTION_FULL = "full"
RESOLUTION_PARTIAL = "partial"
RESOLUTION_NOT_APPLICABLE = "not_applicable"

# Tool-binding enumeration state for a NodeDef, kept strictly separate from `resolution`:
#   full           — the node binds tools and every tool was statically enumerated
#   partial        — the node binds tools but at least one could not be enumerated
#   not_applicable — the node has no ToolNode/bind_tools call at all (the common case)
TOOL_RESOLUTION_FULL = "full"
TOOL_RESOLUTION_PARTIAL = "partial"
TOOL_RESOLUTION_NOT_APPLICABLE = "not_applicable"

# Routing enumeration state for a NodeDef, kept separate from both `resolution` and
# `tool_resolution` for the same reason those two are separate from each other (DEC-008/DEC-020):
#   full           — the node routes via Command(goto=...) and every destination was resolved
#   partial        — it routes via Command(goto=...) but at least one destination could not be
#                    resolved, OR the node function itself was never located, so its body was
#                    never visible and "does not route" would be a guess
#   not_applicable — the function was read and contains no Command(goto=...) at all, or the action
#                    was never a Python function to read (a bare ToolNode, DEC-009)
ROUTING_RESOLUTION_FULL = "full"
ROUTING_RESOLUTION_PARTIAL = "partial"
ROUTING_RESOLUTION_NOT_APPLICABLE = "not_applicable"

# Certainty of a ConditionalRoute's `condition_value`, kept in its own field for the same reason
# `tool_resolution` is separate from `resolution` (DEC-008):
#   known         — the mapping was a dict literal, so the key IS the router's return value
#   not_derivable — the mapping was a list/tuple destination hint, which states where the router
#                   may route but never what it returns. `condition_value` is None (DEC-017)
CONDITION_VALUE_KNOWN = "known"
CONDITION_VALUE_NOT_DERIVABLE = "not_derivable"

# Edge types.
EDGE_NORMAL = "normal"
EDGE_CONDITIONAL = "conditional"


# --------------------------------------------------------------------------------------------
# Deterministic ID helpers
# --------------------------------------------------------------------------------------------
def make_graph_id(file_path: str, variable_name: str) -> str:
    """Stable ID for a graph: its source file plus the variable it is assigned to."""
    return f"{file_path}::{variable_name}"


def make_node_id(graph_id: str, name: str) -> str:
    """Stable ID for a node within a graph, keyed on the node's registered name."""
    return f"{graph_id}::node::{name}"


def make_edge_id(
    graph_id: str,
    source: str,
    target: str,
    edge_type: str,
    condition_value: str | None = None,
) -> str:
    """Stable ID for an edge.

    Normal edges are keyed on ``source->target``. Conditional edges also embed the condition
    value, because one ``add_conditional_edges`` call produces one edge per destination
    (DEC-006) and those edges share a source but differ by condition value and target.
    """
    if edge_type == EDGE_CONDITIONAL:
        return f"{graph_id}::cedge::{source}--{condition_value}-->{target}"
    return f"{graph_id}::edge::{source}-->{target}"


def make_route_id(edge_id: str) -> str:
    """Stable ID for the ConditionalRoute mirroring a conditional edge."""
    return f"{edge_id}::route"


def make_tool_binding_id(node_id: str, tool_name: str) -> str:
    """Stable ID for a tool binding attached to a node."""
    return f"{node_id}::tool::{tool_name}"


# --------------------------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeDef:
    """A single node registered in a LangGraph ``StateGraph`` via ``add_node``."""

    id: str
    graph_id: str
    name: str
    source_file: str
    line_start: int
    line_end: int
    docstring: str | None
    function_body_hash: str
    resolution: str  # RESOLUTION_FULL | RESOLUTION_PARTIAL — was the node function located
    tool_resolution: str  # TOOL_RESOLUTION_FULL | _PARTIAL | _NOT_APPLICABLE — were its tools enumerated
    # Declared last, with a default, so an index written before DEC-020 still rehydrates through
    # `NodeDef(**node)` in storage/base.py instead of raising TypeError. Such an index reports
    # not_applicable for every node, which is the truthful reading — it carries no routing data.
    routing_resolution: str = ROUTING_RESOLUTION_NOT_APPLICABLE

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "graph_id": self.graph_id,
            "name": self.name,
            "source_file": self.source_file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "docstring": self.docstring,
            "function_body_hash": self.function_body_hash,
            "resolution": self.resolution,
            "tool_resolution": self.tool_resolution,
            "routing_resolution": self.routing_resolution,
        }


@dataclass(frozen=True)
class EdgeDef:
    """A directed edge between two nodes.

    ``type`` is ``"normal"`` or ``"conditional"``. For conditional edges,
    ``condition_function_name`` names the routing function and there is a matching
    ``ConditionalRoute`` (DEC-006). ``source_node_id`` / ``target_node_id`` are node IDs; a
    START/END sentinel target uses ``make_node_id(graph_id, START_SENTINEL | END_SENTINEL)``.
    """

    id: str
    graph_id: str
    source_node_id: str
    target_node_id: str
    type: str  # EDGE_NORMAL | EDGE_CONDITIONAL
    condition_function_name: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "graph_id": self.graph_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "type": self.type,
            "condition_function_name": self.condition_function_name,
        }


@dataclass(frozen=True)
class ConditionalRoute:
    """One branch of a conditional edge: where it routes to, and the value that triggers it.

    ``condition_value`` is ``None`` whenever the source did not state it — a list/tuple
    ``path_map`` is a destination hint, not a return-value mapping, so there is no truthful value
    to report (DEC-017). ``value_resolution`` says which case this is, rather than leaving a bare
    ``None`` to be read as "unknown", "not applicable", or "the parser failed" interchangeably.
    The destination is always known either way.
    """

    id: str
    edge_id: str
    condition_value: str | None
    target_node_id: str
    value_resolution: str  # CONDITION_VALUE_KNOWN | CONDITION_VALUE_NOT_DERIVABLE

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "edge_id": self.edge_id,
            "condition_value": self.condition_value,
            "target_node_id": self.target_node_id,
            "value_resolution": self.value_resolution,
        }


@dataclass(frozen=True)
class ToolBinding:
    """A tool bound to or called by a node.

    Defined per task 1.1. Not populated in Phase 1 — task 1.2 does not enumerate tool-binding
    detection, so ``GraphDef.tool_bindings`` is always empty this phase (see DEC-006).
    """

    id: str
    node_id: str
    tool_name: str
    tool_source: str | None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "tool_name": self.tool_name,
            "tool_source": self.tool_source,
        }


@dataclass(frozen=True)
class GraphDef:
    """A single ``StateGraph`` found in a file, aggregating all of its children (DEC-006)."""

    id: str
    repo_id: str
    file_path: str
    variable_name: str
    entry_point: str | None
    nodes: tuple[NodeDef, ...] = field(default_factory=tuple)
    edges: tuple[EdgeDef, ...] = field(default_factory=tuple)
    conditional_routes: tuple[ConditionalRoute, ...] = field(default_factory=tuple)
    tool_bindings: tuple[ToolBinding, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "file_path": self.file_path,
            "variable_name": self.variable_name,
            "entry_point": self.entry_point,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "conditional_routes": [r.to_dict() for r in self.conditional_routes],
            "tool_bindings": [t.to_dict() for t in self.tool_bindings],
        }
