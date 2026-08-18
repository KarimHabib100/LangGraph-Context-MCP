"""Read-only queries over already-parsed graphs — routes, conditional branches, tool callers.

Pure functions over ``GraphDef`` objects: no file I/O, no storage, no embeddings, no MCP types.
They take parsed graphs in and return plain values out, which is what lets them be unit-tested
without a store, an embedding model, or a server in the way.

This module exists because ``trace_path``, ``explain_conditional`` and ``what_calls_tool`` need
real traversal logic, while claude.md requires the tool functions in ``tools/mcp_tools.py`` to stay
thin. It is an addition to claude.md's FILE & FOLDER STRUCTURE, approved by the developer before
implementation and recorded as DEC-018.

Two conventions inherited from the parser (DEC-006) that shape everything here:

- Edges reference nodes by *ID*, built as ``{graph_id}::node::{name}``. Node names are recovered by
  stripping that prefix — including for ``__start__`` / ``__end__``, which are edge endpoints but
  are deliberately not backed by a ``NodeDef``.
- One ``add_conditional_edges`` call produces one edge *per destination*, each mirrored by a
  ``ConditionalRoute``. Conditional branches are therefore ordinary directed edges and need no
  special case during traversal.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .parser.graph_model import (
    CONDITION_VALUE_NOT_DERIVABLE,
    EDGE_CONDITIONAL,
    ROUTING_RESOLUTION_PARTIAL,
    TOOL_RESOLUTION_PARTIAL,
    ConditionalRoute,
    EdgeDef,
    GraphDef,
    NodeDef,
)


@dataclass(frozen=True)
class RouteStep:
    """One hop of a route: the edge taken, and the branch facts if it was a conditional one."""

    source: str
    target: str
    is_conditional: bool
    condition_function: str | None
    condition_value: str | None
    value_resolution: str | None


@dataclass(frozen=True)
class Route:
    """A found route: the node names in order, plus the hop-by-hop detail behind them."""

    node_names: tuple[str, ...]
    steps: tuple[RouteStep, ...]

    @property
    def conditional_steps(self) -> tuple[RouteStep, ...]:
        """Only the hops that went through a conditional edge."""
        return tuple(step for step in self.steps if step.is_conditional)


@dataclass(frozen=True)
class Destination:
    """One possible outcome of a conditional edge.

    ``condition_value`` is ``None`` exactly when ``value_resolution`` is ``not_derivable`` — the
    source used a list/tuple destination hint, which says where the router may go but never what it
    returns (DEC-017). A caller must not phrase such a destination as a return value.
    """

    target: str
    condition_value: str | None
    value_resolution: str


@dataclass(frozen=True)
class ToolCaller:
    """A node that binds or calls a tool, and where that node lives."""

    node_name: str
    file_path: str
    tool_name: str
    tool_source: str | None


def node_name_from_id(graph_id: str, node_id: str) -> str:
    """Recover a node's registered name from its ID. Works for sentinels too (DEC-006)."""
    return node_id.removeprefix(f"{graph_id}::node::")


def collect_node_names(graphs: list[GraphDef]) -> list[str]:
    """Every registered node name across ``graphs``, sorted and de-duplicated.

    This is what populates the ``valid_nodes`` list on an ``unknown_node`` error, so a caller that
    guessed wrong is told what it could have said. Sentinels are excluded: they are edge endpoints,
    not nodes anyone can name.
    """
    return sorted({node.name for graph in graphs for node in graph.nodes})


def find_node(graphs: list[GraphDef], name: str) -> tuple[GraphDef, NodeDef] | None:
    """First graph containing a node called ``name``, with the node itself."""
    for graph in graphs:
        for node in graph.nodes:
            if node.name == name:
                return graph, node
    return None


def find_route(graph: GraphDef, from_node: str, to_node: str) -> Route | None:
    """Shortest route from ``from_node`` to ``to_node`` within one graph, or ``None``.

    Breadth-first, so the result is a route with the fewest hops. Conditional edges are traversed
    like any other directed edge (DEC-006) and are reported as such in the returned steps.

    Returns *a* shortest route, not every route: a graph with several equally short paths yields
    one of them. ``None`` means no directed path exists — an answer, not a failure.
    """
    if from_node == to_node:
        return Route(node_names=(from_node,), steps=())

    adjacency = _adjacency(graph)
    routes_by_edge = {route.edge_id: route for route in graph.conditional_routes}

    previous: dict[str, tuple[str, EdgeDef]] = {}
    seen = {from_node}
    queue: deque[str] = deque([from_node])

    while queue:
        current = queue.popleft()
        for target, edge in adjacency.get(current, ()):
            if target in seen:
                continue
            seen.add(target)
            previous[target] = (current, edge)
            if target == to_node:
                return _rebuild_route(graph, previous, from_node, to_node, routes_by_edge)
            queue.append(target)

    return None


def conditional_destinations(graph: GraphDef, source_name: str) -> list[Destination]:
    """Every destination of the conditional edge(s) leaving ``source_name``.

    Empty when the node has no conditional edge — the caller distinguishes that from an unknown
    node, which is a different answer.
    """
    routes_by_edge = {route.edge_id: route for route in graph.conditional_routes}
    destinations: list[Destination] = []

    for edge in graph.edges:
        if edge.type != EDGE_CONDITIONAL:
            continue
        if node_name_from_id(graph.id, edge.source_node_id) != source_name:
            continue
        target = node_name_from_id(graph.id, edge.target_node_id)
        route = routes_by_edge.get(edge.id)
        destinations.append(
            Destination(
                target=target,
                condition_value=route.condition_value if route else None,
                value_resolution=_value_resolution(route),
            )
        )
    return destinations


def condition_function_for(graph: GraphDef, source_name: str) -> str | None:
    """The routing function named by the conditional edge(s) leaving ``source_name``."""
    for edge in graph.edges:
        if (
            edge.type == EDGE_CONDITIONAL
            and node_name_from_id(graph.id, edge.source_node_id) == source_name
            and edge.condition_function_name
        ):
            return edge.condition_function_name
    return None


def has_conditional_edge(graph: GraphDef, source_name: str) -> bool:
    """Whether ``source_name`` has at least one conditional edge leaving it."""
    return any(
        edge.type == EDGE_CONDITIONAL
        and node_name_from_id(graph.id, edge.source_node_id) == source_name
        for edge in graph.edges
    )


def tool_callers(graphs: list[GraphDef], tool_name: str) -> list[ToolCaller]:
    """Every node across ``graphs`` that binds or calls ``tool_name``.

    Matches the binding's recorded name exactly, or its final dotted segment — a tool bound as
    ``toolkit.search`` is found by ``search``, because the client asking has the bare name and the
    module prefix is an artifact of how the tool was imported.

    Only statically enumerated bindings can appear here. A node whose tools were passed as a
    variable is not listed (DEC-007/DEF-004); ``nodes_with_unenumerated_tools`` reports those
    separately so an empty result is never mistaken for "nothing binds this tool".
    """
    wanted = tool_name.strip()
    callers: list[ToolCaller] = []

    for graph in graphs:
        nodes_by_id = {node.id: node for node in graph.nodes}
        for binding in graph.tool_bindings:
            if not _tool_name_matches(binding.tool_name, wanted):
                continue
            node = nodes_by_id.get(binding.node_id)
            if node is None:
                continue
            callers.append(
                ToolCaller(
                    node_name=node.name,
                    file_path=node.source_file,
                    tool_name=binding.tool_name,
                    tool_source=binding.tool_source,
                )
            )
    return callers


def nodes_with_unenumerated_tools(graphs: list[GraphDef]) -> list[tuple[GraphDef, NodeDef]]:
    """Nodes that bind tools which could not be fully enumerated (``tool_resolution="partial"``).

    These are the honest caveat on any tool-caller answer: the node does bind tools, but at least
    one of them was supplied in a form static analysis cannot read (DEC-008, DEF-004).
    """
    return [
        (graph, node)
        for graph in graphs
        for node in graph.nodes
        if node.tool_resolution == TOOL_RESOLUTION_PARTIAL
    ]


def nodes_with_unresolved_routing(graphs: list[GraphDef]) -> list[tuple[GraphDef, NodeDef]]:
    """Nodes whose own routing could not be enumerated (``routing_resolution="partial"``).

    The routing counterpart of ``nodes_with_unenumerated_tools``, and the honest caveat on any
    "no path" or "not conditional" answer: such a node may route somewhere we could not see,
    either because its body was never located or because a ``Command(goto=...)`` target is
    computed (DEC-020). Without this, "we could not look" is indistinguishable from "we looked
    and there is nothing" — see RISK-012.
    """
    return [
        (graph, node)
        for graph in graphs
        for node in graph.nodes
        if node.routing_resolution == ROUTING_RESOLUTION_PARTIAL
    ]


def node_routing_resolution(graph: GraphDef, node_name: str) -> str | None:
    """The ``routing_resolution`` of one named node, or ``None`` when it is not in this graph."""
    for node in graph.nodes:
        if node.name == node_name:
            return node.routing_resolution
    return None


def summarize_graph(graph: GraphDef) -> dict:
    """One graph's summary row for ``get_graph_summary``.

    Carries prd.md's five contracted keys plus ``nodes`` (the registered names) — without the names
    no tool in DEC-004's surface can answer prd.md's own success criterion, "what nodes are in this
    graph". See DEC-018.
    """
    return {
        "variable_name": graph.variable_name,
        "file_path": graph.file_path,
        "entry_point": graph.entry_point,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "nodes": [node.name for node in graph.nodes],
    }


# --------------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------------
def _adjacency(graph: GraphDef) -> dict[str, list[tuple[str, EdgeDef]]]:
    """Name-keyed outgoing edges. Built per call: graphs are small and this stays stateless."""
    adjacency: dict[str, list[tuple[str, EdgeDef]]] = {}
    for edge in graph.edges:
        source = node_name_from_id(graph.id, edge.source_node_id)
        target = node_name_from_id(graph.id, edge.target_node_id)
        adjacency.setdefault(source, []).append((target, edge))
    return adjacency


def _rebuild_route(
    graph: GraphDef,
    previous: dict[str, tuple[str, EdgeDef]],
    from_node: str,
    to_node: str,
    routes_by_edge: dict[str, ConditionalRoute],
) -> Route:
    """Walk the BFS parent chain backwards from ``to_node`` and turn it into a Route."""
    names: list[str] = [to_node]
    steps: list[RouteStep] = []

    cursor = to_node
    while cursor != from_node:
        source, edge = previous[cursor]
        route = routes_by_edge.get(edge.id)
        is_conditional = edge.type == EDGE_CONDITIONAL
        steps.append(
            RouteStep(
                source=source,
                target=cursor,
                is_conditional=is_conditional,
                condition_function=edge.condition_function_name,
                condition_value=route.condition_value if route else None,
                value_resolution=_value_resolution(route) if is_conditional else None,
            )
        )
        names.append(source)
        cursor = source

    names.reverse()
    steps.reverse()
    return Route(node_names=tuple(names), steps=tuple(steps))


def _value_resolution(route: ConditionalRoute | None) -> str:
    """A route's certainty about its condition value, defaulting to 'not derivable' when the
    mirroring ConditionalRoute is missing — never claiming a value we do not have (DEC-017)."""
    return route.value_resolution if route else CONDITION_VALUE_NOT_DERIVABLE


def _tool_name_matches(binding_name: str, wanted: str) -> bool:
    if binding_name == wanted:
        return True
    return binding_name.rsplit(".", 1)[-1] == wanted
