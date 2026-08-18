"""Tests for the AST walker, cross-file resolver, and repository scanner (Phase 1).

Covers task 1.6's required cases: a single-file simple graph, a multi-file graph with
cross-file node functions, conditional edges with 3+ branches, a node built inside a loop
(partial, no crash), and a syntax-error file mixed into a repo (skipped, scan continues). Also
covers the RISK-001 (dynamic construction) and RISK-002 (bounded cross-file depth / circular
imports) mitigations, plus general "never crash on malformed input" guarantees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from langgraph_context_mcp.parser.ast_walker import find_graph_definitions
from langgraph_context_mcp.parser.graph_model import (
    CONDITION_VALUE_KNOWN,
    CONDITION_VALUE_NOT_DERIVABLE,
    EDGE_CONDITIONAL,
    EDGE_NORMAL,
    END_SENTINEL,
    RESOLUTION_FULL,
    RESOLUTION_NOT_APPLICABLE,
    RESOLUTION_PARTIAL,
    ROUTING_RESOLUTION_FULL,
    ROUTING_RESOLUTION_NOT_APPLICABLE,
    ROUTING_RESOLUTION_PARTIAL,
    TOOL_RESOLUTION_FULL,
    TOOL_RESOLUTION_NOT_APPLICABLE,
    TOOL_RESOLUTION_PARTIAL,
    make_edge_id,
    make_node_id,
)
from langgraph_context_mcp.parser.repo_scanner import scan_repository

FIXTURE = Path(__file__).parent / "fixtures" / "sample_graphs" / "simple_graph.py"


def _write(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _node_names(graph) -> set[str]:
    return {node.name for node in graph.nodes}


def _target_name(graph_id: str, edge) -> str:
    prefix = f"{graph_id}::node::"
    return edge.target_node_id[len(prefix):]


def _source_name(graph_id: str, edge) -> str:
    prefix = f"{graph_id}::node::"
    return edge.source_node_id[len(prefix):]


# --------------------------------------------------------------------------------------------
# Single-file simple graph (fixture)
# --------------------------------------------------------------------------------------------
def test_simple_fixture_identifies_all_nodes_edges_and_conditionals():
    graphs = find_graph_definitions(FIXTURE)
    assert len(graphs) == 1
    graph = graphs[0]

    assert graph.variable_name == "graph"
    assert graph.entry_point == "check_auth_token"
    assert _node_names(graph) == {
        "check_auth_token",
        "fetch_data",
        "enrich_data",
        "format_response",
        "handle_error",
    }
    # Every node in this single file is fully resolved with real line numbers and a body hash,
    # and none of them bind tools, so tool_resolution is not_applicable.
    for node in graph.nodes:
        assert node.resolution == RESOLUTION_FULL
        assert node.tool_resolution == TOOL_RESOLUTION_NOT_APPLICABLE
        assert node.line_start > 0
        assert node.line_end >= node.line_start
        assert node.function_body_hash
        assert node.docstring is not None

    normal = [e for e in graph.edges if e.type == EDGE_NORMAL]
    conditional = [e for e in graph.edges if e.type == EDGE_CONDITIONAL]
    assert len(normal) == 3
    # 2 from add_conditional_edges + 3 from enrich_data's Command(goto=...) branches, which
    # name the node function itself as the router (DEC-020).
    assert len(conditional) == 5
    assert {e.condition_function_name for e in conditional} == {
        "route_after_auth",
        "enrich_data",
    }

    # The builder-declared conditional edge routes to two distinct destinations via
    # route_after_auth, and its dict-literal mapping still yields real condition values.
    route_targets = {r.condition_value: r.target_node_id for r in graph.conditional_routes}
    assert route_targets["authorized"] == make_node_id(graph.id, "fetch_data")
    assert route_targets["unauthorized"] == make_node_id(graph.id, "handle_error")

    # END is represented by the sentinel, not a fabricated NodeDef.
    end_targets = {_target_name(graph.id, e) for e in normal}
    assert END_SENTINEL in end_targets


def test_fixture_to_dict_is_json_serializable():
    import json

    graph = find_graph_definitions(FIXTURE)[0]
    dumped = json.dumps(graph.to_dict())  # must not raise
    assert '"resolution": "full"' in dumped


# --------------------------------------------------------------------------------------------
# Conditional edges with 3+ branches
# --------------------------------------------------------------------------------------------
def test_conditional_edges_with_four_branches(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph, END


def router(state):
    """Pick a branch."""
    return state["k"]


def a(state):
    return state


def b(state):
    return state


def c(state):
    return state


g = StateGraph(dict)
g.add_node("a", a)
g.add_node("b", b)
g.add_node("c", c)
g.add_conditional_edges(
    "a",
    router,
    {"to_a": "a", "to_b": "b", "to_c": "c", "done": END},
)
'''
    path = _write(tmp_path, "branchy.py", src)
    graph = find_graph_definitions(path)[0]

    conditional = [e for e in graph.edges if e.type == EDGE_CONDITIONAL]
    assert len(conditional) == 4
    assert len(graph.conditional_routes) == 4
    assert {e.condition_function_name for e in conditional} == {"router"}

    route_map = {r.condition_value: _endpoint(graph.id, r.target_node_id) for r in graph.conditional_routes}
    assert route_map == {"to_a": "a", "to_b": "b", "to_c": "c", "done": END_SENTINEL}
    # A dict literal genuinely states the router's return values (DEC-017).
    assert {r.value_resolution for r in graph.conditional_routes} == {CONDITION_VALUE_KNOWN}


def _endpoint(graph_id: str, node_id: str) -> str:
    return node_id[len(f"{graph_id}::node::"):]


# --------------------------------------------------------------------------------------------
# List-form path_map — destination hint, not a return-value mapping (QA-3-04 / DEC-017)
# --------------------------------------------------------------------------------------------
def test_list_form_conditional_records_no_condition_value(tmp_path: Path):
    """The exact shape from `open_deep_research` `src/legacy/graph.py:499`.

    The router returns `Send(...)` objects, so the destination name is emphatically NOT the
    value it returns. Before DEC-017 the parser recorded `condition_value="write_final_sections"`,
    which `explain_conditional` would have stated as the router's return value — a fact the
    function never produces.
    """
    src = '''
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


def gather_completed_sections(state):
    return state


def write_final_sections(state):
    return state


def initiate_final_section_writing(state):
    return [
        Send("write_final_sections", {"topic": state["topic"], "section": s})
        for s in state["sections"]
    ]


builder = StateGraph(dict)
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_conditional_edges(
    "gather_completed_sections", initiate_final_section_writing, ["write_final_sections"]
)
'''
    path = _write(tmp_path, "send_router.py", src)
    graph = find_graph_definitions(path)[0]

    assert len(graph.conditional_routes) == 1
    route = graph.conditional_routes[0]

    # The claim that was fabricated is gone...
    assert route.condition_value is None
    assert route.value_resolution == CONDITION_VALUE_NOT_DERIVABLE
    # ...while the destination, which was always correct, is unchanged.
    assert _endpoint(graph.id, route.target_node_id) == "write_final_sections"

    conditional = [e for e in graph.edges if e.type == EDGE_CONDITIONAL]
    assert len(conditional) == 1
    assert conditional[0].condition_function_name == "initiate_final_section_writing"


def test_list_form_keeps_every_destination(tmp_path: Path):
    """Multiple destinations still produce one route each (DEC-006), all `not_derivable`."""
    src = '''
from langgraph.graph import StateGraph, END


def router(state):
    return "anything at all"


def a(state):
    return state


def b(state):
    return state


g = StateGraph(dict)
g.add_node("a", a)
g.add_node("b", b)
g.add_conditional_edges("a", router, ["a", "b", END])
'''
    path = _write(tmp_path, "listform.py", src)
    graph = find_graph_definitions(path)[0]

    assert len(graph.conditional_routes) == 3
    assert all(r.condition_value is None for r in graph.conditional_routes)
    assert all(
        r.value_resolution == CONDITION_VALUE_NOT_DERIVABLE for r in graph.conditional_routes
    )
    assert {_endpoint(graph.id, r.target_node_id) for r in graph.conditional_routes} == {
        "a",
        "b",
        END_SENTINEL,
    }


def test_list_form_edge_ids_are_unchanged_by_dec_017(tmp_path: Path):
    """Edge IDs must stay byte-identical, or every previously built index is invalidated."""
    src = '''
from langgraph.graph import StateGraph


def router(state):
    return "x"


def a(state):
    return state


def b(state):
    return state


g = StateGraph(dict)
g.add_node("a", a)
g.add_node("b", b)
g.add_conditional_edges("a", router, ["a", "b"])
'''
    path = _write(tmp_path, "ids.py", src)
    graph = find_graph_definitions(path)[0]

    # The pre-DEC-017 ID used the destination name in the condition slot; it still does.
    expected = {
        make_edge_id(graph.id, "a", target, EDGE_CONDITIONAL, target) for target in ("a", "b")
    }
    assert {e.id for e in graph.edges if e.type == EDGE_CONDITIONAL} == expected


# --------------------------------------------------------------------------------------------
# Multi-file graph with cross-file node functions
# --------------------------------------------------------------------------------------------
def test_cross_file_node_functions_resolve_to_full(tmp_path: Path):
    _write(tmp_path, "agent/__init__.py", "")
    _write(
        tmp_path,
        "agent/nodes.py",
        '''
def alpha(state):
    """Alpha node docstring."""
    return state


def beta(state):
    """Beta node docstring."""
    return state
''',
    )
    _write(
        tmp_path,
        "agent/graph.py",
        '''
from langgraph.graph import StateGraph

from .nodes import alpha, beta

g = StateGraph(dict)
g.add_node("alpha", alpha)
g.add_node("beta", beta)
g.add_edge("alpha", "beta")
''',
    )

    graphs = scan_repository(tmp_path)
    assert len(graphs) == 1
    graph = graphs[0]

    by_name = {n.name: n for n in graph.nodes}
    assert set(by_name) == {"alpha", "beta"}
    for name in ("alpha", "beta"):
        node = by_name[name]
        assert node.resolution == RESOLUTION_FULL
        assert node.source_file == "agent/nodes.py"
        assert node.docstring == f"{name.capitalize()} node docstring."
        assert node.line_start > 0
        assert node.function_body_hash


# --------------------------------------------------------------------------------------------
# Node functions built inside a loop -> partial, never a crash (RISK-001)
# --------------------------------------------------------------------------------------------
def test_node_built_in_loop_via_factory_is_partial(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph


def make_node(n):
    def _inner(state):
        return state
    return _inner


g = StateGraph(dict)
for name in ["a", "b", "c"]:
    g.add_node(name, make_node(name))
'''
    path = _write(tmp_path, "loopy.py", src)
    graph = find_graph_definitions(path)[0]  # must not raise

    assert graph.nodes  # the dynamic node is recorded, not dropped
    assert all(n.resolution == RESOLUTION_PARTIAL for n in graph.nodes)


def test_constant_named_node_inside_loop_is_partial(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph


def worker(state):
    return state


g = StateGraph(dict)
for _ in range(3):
    g.add_node("worker", worker)
'''
    path = _write(tmp_path, "loop_const.py", src)
    graph = find_graph_definitions(path)[0]

    worker = next(n for n in graph.nodes if n.name == "worker")
    # Even though `worker` is defined in-file, being constructed inside a loop marks it partial.
    assert worker.resolution == RESOLUTION_PARTIAL


# --------------------------------------------------------------------------------------------
# Syntax-error file mixed into a repo -> skipped, scan continues
# --------------------------------------------------------------------------------------------
def test_syntax_error_file_is_skipped_and_scan_continues(tmp_path: Path):
    _write(tmp_path, "broken.py", "def oops(:\n    return\n")  # invalid syntax
    _write(
        tmp_path,
        "good.py",
        '''
from langgraph.graph import StateGraph


def only_node(state):
    return state


g = StateGraph(dict)
g.add_node("only_node", only_node)
''',
    )

    graphs = scan_repository(tmp_path)  # must not raise despite broken.py
    assert len(graphs) == 1
    assert _node_names(graphs[0]) == {"only_node"}


# --------------------------------------------------------------------------------------------
# UTF-8 BOM handling (DEC-022 / QA-5-06): a BOM is not a syntax error, and must not hide one
# --------------------------------------------------------------------------------------------
_UTF8_BOM = b"\xef\xbb\xbf"

_VALID_SOURCE = """
from langgraph.graph import StateGraph


def only_node(state):
    return state


g = StateGraph(dict)
g.add_node("only_node", only_node)
"""


def test_bom_prefixed_valid_file_parses_correctly(tmp_path: Path):
    """A file saved with a UTF-8 BOM is valid Python (`python file.py` runs it fine); the
    parser must not misreport it as a syntax error and drop its graph (QA-5-06)."""
    path = tmp_path / "bom_ok.py"
    path.write_bytes(_UTF8_BOM + _VALID_SOURCE.encode("utf-8"))

    graphs = find_graph_definitions(path)

    assert len(graphs) == 1
    assert _node_names(graphs[0]) == {"only_node"}
    assert graphs[0].nodes[0].resolution == RESOLUTION_FULL


def test_bom_prefixed_file_matches_non_bom_output_exactly(tmp_path: Path):
    """Same source, with and without a BOM, must produce identical graph output — the BOM
    carries no information, so it must not change what is extracted."""
    plain_path = tmp_path / "plain.py"
    plain_path.write_text(_VALID_SOURCE, encoding="utf-8")
    bom_path = tmp_path / "bom.py"
    bom_path.write_bytes(_UTF8_BOM + _VALID_SOURCE.encode("utf-8"))

    plain_graphs = find_graph_definitions(plain_path)
    bom_graphs = find_graph_definitions(bom_path)

    assert len(plain_graphs) == len(bom_graphs) == 1
    plain_node, bom_node = plain_graphs[0].nodes[0], bom_graphs[0].nodes[0]
    # function_body_hash must match: the BOM must not leak into what gets hashed/embedded.
    assert plain_node.function_body_hash == bom_node.function_body_hash
    assert plain_node.line_start == bom_node.line_start
    assert plain_node.line_end == bom_node.line_end


def test_bom_prefixed_file_with_a_genuine_syntax_error_is_still_skipped(tmp_path: Path):
    """Stripping the BOM must not swallow a real syntax error elsewhere in the same file —
    the fix removes BOM noise only, it must not widen what counts as 'valid'."""
    path = tmp_path / "bom_broken.py"
    path.write_bytes(_UTF8_BOM + b"def oops(:\n    return\n")

    graphs = find_graph_definitions(path)

    assert graphs == []  # still reported as invalid, not silently treated as empty-but-fine


def test_non_utf8_source_file_is_skipped_not_crashed(tmp_path: Path):
    """A genuinely non-UTF-8-encoded file (e.g. Windows-1252 with an accented byte outside
    UTF-8's valid ranges) must still be skipped gracefully — utf-8-sig only changes BOM
    handling, not multi-byte validation, so this must behave exactly as before DEC-022."""
    path = tmp_path / "cp1252.py"
    # 'é' as a single cp1252 byte (0xE9) is not valid UTF-8 on its own.
    path.write_bytes(b"# caf\xe9\ndef node(state):\n    return state\n")

    graphs = find_graph_definitions(path)  # must not raise

    assert graphs == []


def test_bom_prefixed_cross_file_module_resolves_to_full(tmp_path: Path):
    """The fix lives in the one shared `_parse_file()` helper, so a BOM in a module reached
    through cross-file resolution (resolver.py) must resolve to `full`, not `partial` — not
    just same-file scanning."""
    _write(tmp_path, "agent/__init__.py", "")
    nodes_path = tmp_path / "agent" / "nodes.py"
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_path.write_bytes(
        _UTF8_BOM
        + b'''
def alpha(state):
    """Alpha node docstring."""
    return state
'''
    )
    _write(
        tmp_path,
        "agent/graph.py",
        '''
from langgraph.graph import StateGraph

from .nodes import alpha

g = StateGraph(dict)
g.add_node("alpha", alpha)
''',
    )

    graphs = scan_repository(tmp_path)

    assert len(graphs) == 1
    node = graphs[0].nodes[0]
    assert node.name == "alpha"
    assert node.resolution == RESOLUTION_FULL  # not "partial" — the BOM must not defeat resolution
    assert node.docstring == "Alpha node docstring."


# --------------------------------------------------------------------------------------------
# Malformed conditional mapping -> partial/best-effort, never a crash (RISK-001)
# --------------------------------------------------------------------------------------------
def test_conditional_edges_with_non_literal_mapping_records_unresolved(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph


def router(state):
    return "x"


def a(state):
    return state


mapping = {"x": "a"}
g = StateGraph(dict)
g.add_node("a", a)
g.add_conditional_edges("a", router, mapping)
'''
    path = _write(tmp_path, "dyn_map.py", src)
    graph = find_graph_definitions(path)[0]  # must not raise

    conditional = [e for e in graph.edges if e.type == EDGE_CONDITIONAL]
    assert len(conditional) == 1
    assert conditional[0].condition_function_name == "router"
    # The branches could not be statically enumerated, so no routes are asserted.
    assert graph.conditional_routes == ()


# --------------------------------------------------------------------------------------------
# Bounded cross-file depth / circular imports (RISK-002)
# --------------------------------------------------------------------------------------------
def test_circular_import_does_not_recurse_forever(tmp_path: Path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/a.py", "from .b import node_fn\n")
    _write(tmp_path, "pkg/b.py", "from .a import node_fn\n")
    _write(
        tmp_path,
        "pkg/graph.py",
        '''
from langgraph.graph import StateGraph

from .a import node_fn

g = StateGraph(dict)
g.add_node("n", node_fn)
''',
    )

    graphs = scan_repository(tmp_path)  # must terminate, not hang or crash
    graph = next(g for g in graphs if g.variable_name == "g")
    node = next(n for n in graph.nodes if n.name == "n")
    # node_fn is never actually defined; the depth cap + visited set leave it partial.
    assert node.resolution == RESOLUTION_PARTIAL


# --------------------------------------------------------------------------------------------
# General "never crash" guarantees
# --------------------------------------------------------------------------------------------
def test_empty_file_returns_no_graphs(tmp_path: Path):
    path = _write(tmp_path, "empty.py", "")
    assert find_graph_definitions(path) == []


def test_file_without_langgraph_returns_no_graphs(tmp_path: Path):
    path = _write(tmp_path, "plain.py", "x = 1\n\ndef f():\n    return x\n")
    assert find_graph_definitions(path) == []


def test_missing_file_returns_no_graphs(tmp_path: Path):
    assert find_graph_definitions(tmp_path / "does_not_exist.py") == []


def test_repo_with_no_graphs_returns_empty_list(tmp_path: Path):
    _write(tmp_path, "a.py", "x = 1\n")
    _write(tmp_path, "b.py", "def g():\n    return 2\n")
    assert scan_repository(tmp_path) == []


def test_gitignore_excludes_matching_files(tmp_path: Path):
    _write(tmp_path, ".gitignore", "ignored_dir/\n")
    _write(
        tmp_path,
        "ignored_dir/hidden.py",
        '''
from langgraph.graph import StateGraph

g = StateGraph(dict)
g.add_node("x", lambda s: s)
''',
    )
    _write(
        tmp_path,
        "visible.py",
        '''
from langgraph.graph import StateGraph


def visible_node(state):
    return state


g = StateGraph(dict)
g.add_node("visible_node", visible_node)
''',
    )

    graphs = scan_repository(tmp_path)
    assert len(graphs) == 1
    assert graphs[0].file_path == "visible.py"


# --------------------------------------------------------------------------------------------
# Tool-binding detection: ToolNode([...]) and .bind_tools([...]) (DEC-007, Phase 1 addendum)
# --------------------------------------------------------------------------------------------
def _bindings_by_node(graph) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    prefix = f"{graph.id}::node::"
    for binding in graph.tool_bindings:
        node_name = binding.node_id[len(prefix):]
        result.setdefault(node_name, set()).add(binding.tool_name)
    return result


def test_toolnode_action_binds_tools_to_node(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from mypkg.tools import search, calculator

g = StateGraph(dict)
g.add_node("tools", ToolNode([search, calculator]))
'''
    path = _write(tmp_path, "toolnode.py", src)
    graph = find_graph_definitions(path)[0]

    assert _bindings_by_node(graph) == {"tools": {"search", "calculator"}}
    # tool_source is resolved from the file's imports.
    sources = {b.tool_name: b.tool_source for b in graph.tool_bindings}
    assert sources == {"search": "mypkg.tools", "calculator": "mypkg.tools"}
    # The ToolNode was never a Python function to locate (resolution not_applicable, not
    # partial — DEC-009) yet its tools are fully enumerated (tool_resolution full); the two axes
    # are independent (DEC-008).
    tools_node = next(n for n in graph.nodes if n.name == "tools")
    assert tools_node.resolution == RESOLUTION_NOT_APPLICABLE
    assert tools_node.tool_resolution == TOOL_RESOLUTION_FULL


def test_bind_tools_in_function_body_binds_to_node(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph

from mypkg.tools import search

def agent(state):
    """Agent that can call a tool."""
    model = llm.bind_tools([search])
    return model.invoke(state)

g = StateGraph(dict)
g.add_node("agent", agent)
'''
    path = _write(tmp_path, "binder.py", src)
    graph = find_graph_definitions(path)[0]

    assert _bindings_by_node(graph) == {"agent": {"search"}}
    binding = graph.tool_bindings[0]
    assert binding.tool_source == "mypkg.tools"
    # The node function resolves fully AND its tools are fully enumerated.
    agent = next(n for n in graph.nodes if n.name == "agent")
    assert agent.resolution == RESOLUTION_FULL
    assert agent.tool_resolution == TOOL_RESOLUTION_FULL


def test_toolnode_import_alias_is_detected(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode as TN

from mypkg.tools import search

g = StateGraph(dict)
g.add_node("tools", TN([search]))
'''
    path = _write(tmp_path, "aliased.py", src)
    graph = find_graph_definitions(path)[0]
    assert _bindings_by_node(graph) == {"tools": {"search"}}


def test_cross_file_bind_tools_populates_tool_bindings(tmp_path: Path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/nodes.py",
        '''
from mypkg.tools import lookup


def worker(state):
    """Worker node that binds a tool."""
    model = chat.bind_tools([lookup])
    return model.invoke(state)
''',
    )
    _write(
        tmp_path,
        "pkg/graph.py",
        '''
from langgraph.graph import StateGraph

from .nodes import worker

g = StateGraph(dict)
g.add_node("worker", worker)
''',
    )

    graph = scan_repository(tmp_path)[0]
    assert _bindings_by_node(graph) == {"worker": {"lookup"}}
    # tool_source is resolved from the *defining* module's imports (pkg/nodes.py).
    assert graph.tool_bindings[0].tool_source == "mypkg.tools"
    assert next(n for n in graph.nodes if n.name == "worker").resolution == RESOLUTION_FULL


def test_non_literal_tools_argument_yields_no_bindings(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

my_tools = [1, 2]
g = StateGraph(dict)
g.add_node("tools", ToolNode(my_tools))
'''
    path = _write(tmp_path, "dynamic_tools.py", src)
    graph = find_graph_definitions(path)[0]  # must not raise
    assert graph.tool_bindings == ()


def test_variable_tools_argument_sets_tool_resolution_partial(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph

from mypkg.tools import search

def agent(state):
    """Agent binding tools from a variable."""
    my_tools = load_tools()
    model = llm.bind_tools(my_tools)
    return model

g = StateGraph(dict)
g.add_node("agent", agent)
'''
    path = _write(tmp_path, "unresolved_bind.py", src)
    graph = find_graph_definitions(path)[0]

    node = next(n for n in graph.nodes if n.name == "agent")
    # The function itself is located (resolution full), but its tools come from a variable, not
    # a list literal — so tool_resolution is partial and no bindings are enumerated. The tool
    # state does not leak into `resolution` (DEC-008).
    assert node.resolution == RESOLUTION_FULL
    assert node.tool_resolution == TOOL_RESOLUTION_PARTIAL
    assert graph.tool_bindings == ()


def test_cross_file_variable_bind_tools_sets_tool_resolution_partial(tmp_path: Path):
    _write(tmp_path, "pkg/__init__.py", "")
    _write(
        tmp_path,
        "pkg/nodes.py",
        '''
def worker(state):
    """Worker binding tools from a variable."""
    my_tools = build_tools()
    model = chat.bind_tools(my_tools)
    return model.invoke(state)
''',
    )
    _write(
        tmp_path,
        "pkg/graph.py",
        '''
from langgraph.graph import StateGraph

from .nodes import worker

g = StateGraph(dict)
g.add_node("worker", worker)
''',
    )

    graph = scan_repository(tmp_path)[0]
    node = next(n for n in graph.nodes if n.name == "worker")
    # The function was located cross-file (resolution full), but its tools come from a variable,
    # so tool_resolution is partial — consistent with the same-file behaviour (DEC-008).
    assert node.resolution == RESOLUTION_FULL
    assert node.tool_resolution == TOOL_RESOLUTION_PARTIAL
    assert graph.tool_bindings == ()


def test_bare_toolnode_action_is_resolution_not_applicable(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from mypkg.tools import search

g = StateGraph(dict)
g.add_node("tools", ToolNode([search]))
'''
    path = _write(tmp_path, "bare_toolnode.py", src)
    graph = find_graph_definitions(path)[0]
    node = next(n for n in graph.nodes if n.name == "tools")
    # The action was never a Python function, so resolution is not_applicable, NOT partial —
    # partial is reserved for genuine uncertainty about a function that exists (DEC-009).
    assert node.resolution == RESOLUTION_NOT_APPLICABLE


def test_factory_call_action_stays_partial_not_not_applicable(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph

g = StateGraph(dict)
g.add_node("built", make_node("built"))
'''
    path = _write(tmp_path, "factory_action.py", src)
    graph = find_graph_definitions(path)[0]
    node = next(n for n in graph.nodes if n.name == "built")
    # A factory call IS meant to produce a function we could not resolve — genuine uncertainty,
    # so it stays partial and is not reclassified as not_applicable.
    assert node.resolution == RESOLUTION_PARTIAL


def test_node_without_tool_calls_is_tool_resolution_not_applicable(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph


def plain(state):
    """A node that binds no tools at all — the common case."""
    return {**state, "x": 1}


g = StateGraph(dict)
g.add_node("plain", plain)
'''
    path = _write(tmp_path, "plain_node.py", src)
    graph = find_graph_definitions(path)[0]
    node = next(n for n in graph.nodes if n.name == "plain")
    # A node with no ToolNode/bind_tools call is not_applicable, distinct from full/partial.
    assert node.resolution == RESOLUTION_FULL
    assert node.tool_resolution == TOOL_RESOLUTION_NOT_APPLICABLE
    assert graph.tool_bindings == ()


@pytest.mark.parametrize(
    ("label", "list_expr"),
    [
        ("starred", "[*base_tools, search]"),
        ("nested_list", "[search, [a, b]]"),
        ("dict", '[search, {"k": v}]'),
        ("non_string_constant", "[search, 42]"),
    ],
)
def test_non_resolvable_list_element_flags_partial_without_fabricating(
    tmp_path: Path, label: str, list_expr: str
):
    """A non-literal element inside an otherwise-literal tools list (all four leak categories).

    Each such element must be skipped (never fabricated into a bogus tool name via unparse) and
    must set tool_resolution=partial, while resolution stays full because the function is
    located.
    """
    src = f'''
from langgraph.graph import StateGraph

from mypkg.tools import search, base_tools, a, b, v


def agent(state):
    """Agent binding a mix of resolvable and non-resolvable tools."""
    model = llm.bind_tools({list_expr})
    return model


g = StateGraph(dict)
g.add_node("agent", agent)
'''
    path = _write(tmp_path, f"leak_{label}.py", src)
    graph = find_graph_definitions(path)[0]
    node = next(n for n in graph.nodes if n.name == "agent")

    assert node.resolution == RESOLUTION_FULL
    assert node.tool_resolution == TOOL_RESOLUTION_PARTIAL
    # Only the resolvable element (search) is emitted; the leak element is skipped, not turned
    # into a bogus tool name like "*base_tools", "[a, b]", "{'k': v}" or "42".
    assert _bindings_by_node(graph) == {"agent": {"search"}}


def test_tool_name_from_attribute_and_string_forms(tmp_path: Path):
    src = '''
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

import mypkg.tools as toolkit

g = StateGraph(dict)
g.add_node("tools", ToolNode([toolkit.search, "web_search"]))
'''
    path = _write(tmp_path, "attr_tools.py", src)
    graph = find_graph_definitions(path)[0]
    by_name = {b.tool_name: b.tool_source for b in graph.tool_bindings}
    # Attribute form resolves the base module; a string tool name has no import source.
    assert by_name == {"search": "mypkg.tools", "web_search": None}


# --------------------------------------------------------------------------------------------
# line_start includes decorators (DEC-013, task 1.11)
# --------------------------------------------------------------------------------------------
def test_decorated_node_line_start_includes_its_decorators(tmp_path: Path):
    """A decorated node begins at its first `@`, not at its `def` (DEC-013).

    `ast` reports a FunctionDef's lineno on the `def` line and keeps decorators above it, so
    without this the node's reported span silently excludes every decorator — which is both a
    wrong answer for any consumer of line_start and the reason DEC-005's "docstring + decorators
    + body" chunk had to be reconstructed heuristically at chunk time.
    """
    src = '''from langgraph.graph import StateGraph


def trace(func):
    return func


@trace
@trace
def decorated(state):
    """Decorated node."""
    return state


def plain(state):
    """Undecorated node."""
    return state


g = StateGraph(dict)
g.add_node("decorated", decorated)
g.add_node("plain", plain)
'''
    path = _write(tmp_path, "decorated.py", src)
    lines = src.splitlines()
    graph = find_graph_definitions(path)[0]
    by_name = {n.name: n for n in graph.nodes}

    decorated = by_name["decorated"]
    assert decorated.resolution == RESOLUTION_FULL
    # line_start lands on the FIRST of the two decorators, not the second and not the def.
    assert lines[decorated.line_start - 1] == "@trace"
    assert lines[decorated.line_start] == "@trace"
    assert lines[decorated.line_start + 1] == "def decorated(state):"
    # The full span therefore carries decorators, docstring, and body together (DEC-005).
    span = "\n".join(lines[decorated.line_start - 1 : decorated.line_end])
    assert span.count("@trace") == 2
    assert '"""Decorated node."""' in span
    assert "return state" in span

    # An undecorated node is unaffected: it still starts at its def line.
    plain = by_name["plain"]
    assert lines[plain.line_start - 1] == "def plain(state):"


def test_multiline_decorator_line_start_is_the_at_sign(tmp_path: Path):
    """A decorator whose arguments wrap across lines still anchors on its own `@` line."""
    src = '''from langgraph.graph import StateGraph


def tagged(**kwargs):
    def wrap(func):
        return func
    return wrap


@tagged(
    name="wrapped",
    retries=3,
)
def wrapped(state):
    """Wrapped node."""
    return state


g = StateGraph(dict)
g.add_node("wrapped", wrapped)
'''
    path = _write(tmp_path, "multiline_decorator.py", src)
    lines = src.splitlines()
    node = find_graph_definitions(path)[0].nodes[0]

    assert lines[node.line_start - 1] == "@tagged("
    span = "\n".join(lines[node.line_start - 1 : node.line_end])
    assert 'name="wrapped"' in span
    assert "def wrapped(state):" in span


def test_cross_file_decorated_node_line_start_includes_decorators(tmp_path: Path):
    """The resolver must agree with the walker — same construct, same span, either file."""
    _write(tmp_path, "agent/__init__.py", "")
    nodes_src = '''def trace(func):
    return func


@trace
def remote(state):
    """Remote node."""
    return state
'''
    _write(tmp_path, "agent/nodes.py", nodes_src)
    _write(
        tmp_path,
        "agent/graph.py",
        '''from langgraph.graph import StateGraph

from .nodes import remote

g = StateGraph(dict)
g.add_node("remote", remote)
''',
    )

    node = scan_repository(tmp_path)[0].nodes[0]

    assert node.resolution == RESOLUTION_FULL
    assert node.source_file == "agent/nodes.py"
    assert nodes_src.splitlines()[node.line_start - 1] == "@trace"


def test_decorators_do_not_change_the_function_body_hash(tmp_path: Path):
    """Widening the span must not disturb `function_body_hash`.

    The hash comes from the `def`-onward source segment, so an existing index stays valid — a
    changed hash would make every decorated node look edited on the next reindex.
    """
    undecorated = _write(
        tmp_path,
        "undecorated.py",
        '''from langgraph.graph import StateGraph


def node(state):
    """A node."""
    return state


g = StateGraph(dict)
g.add_node("node", node)
''',
    )
    decorated = _write(
        tmp_path,
        "decorated_hash.py",
        '''from langgraph.graph import StateGraph


def trace(func):
    return func


@trace
def node(state):
    """A node."""
    return state


g = StateGraph(dict)
g.add_node("node", node)
''',
    )

    plain_node = find_graph_definitions(undecorated)[0].nodes[0]
    decorated_node = find_graph_definitions(decorated)[0].nodes[0]

    assert decorated_node.function_body_hash == plain_node.function_body_hash
    assert decorated_node.line_start < decorated_node.line_end


# --------------------------------------------------------------------------------------------
# entry_point derivation (DEC-015 / QA-3-02)
#
# LangGraph documents set_entry_point(x) as sugar for add_edge(START, x), and real code
# overwhelmingly writes the latter — QA-3-02 found entry_point null on all 7 graphs of a real
# repo for exactly that reason. Both idioms must now normalize to the same answer, and the
# ambiguous shapes must stay None rather than guess.
# --------------------------------------------------------------------------------------------
_ENTRY_PREAMBLE = """\
from langgraph.graph import START, END, StateGraph

def alpha(state):
    \"\"\"First.\"\"\"
    return state

def beta(state):
    \"\"\"Second.\"\"\"
    return state

def router(state):
    \"\"\"Routes.\"\"\"
    return "a"

g = StateGraph(dict)
g.add_node("alpha", alpha)
g.add_node("beta", beta)
"""


def _entry_point_of(tmp_path: Path, body: str, name: str = "entry.py") -> str | None:
    path = _write(tmp_path, name, _ENTRY_PREAMBLE + body)
    graphs = find_graph_definitions(path)
    assert len(graphs) == 1
    return graphs[0].entry_point


def test_entry_point_derived_from_a_lone_start_edge(tmp_path: Path):
    """Exactly one normal START edge — the case QA-3-02 hit on 7/7 real graphs."""
    assert _entry_point_of(tmp_path, 'g.add_edge(START, "alpha")\n') == "alpha"


def test_entry_point_is_none_with_no_start_edge(tmp_path: Path):
    """Zero START edges and no set_entry_point — nothing to derive from."""
    assert _entry_point_of(tmp_path, 'g.add_edge("alpha", "beta")\n') is None


def test_entry_point_is_none_with_parallel_entries(tmp_path: Path):
    """Two START edges is a fan-out; a singular field cannot express it, so it must not guess."""
    body = 'g.add_edge(START, "alpha")\ng.add_edge(START, "beta")\n'
    assert _entry_point_of(tmp_path, body) is None


def test_explicit_set_entry_point_still_wins(tmp_path: Path):
    """Derivation only fills a field that is otherwise None — it never overwrites a stated value."""
    body = 'g.set_entry_point("beta")\ng.add_edge(START, "alpha")\n'
    assert _entry_point_of(tmp_path, body) == "beta"


def test_both_idioms_produce_the_same_entry_point(tmp_path: Path):
    """The equivalence DEC-015 exists to enforce, asserted directly."""
    explicit = _entry_point_of(tmp_path, 'g.set_entry_point("alpha")\n', "explicit.py")
    implicit = _entry_point_of(tmp_path, 'g.add_edge(START, "alpha")\n', "implicit.py")

    assert explicit == implicit == "alpha"


def test_conditional_start_edge_does_not_derive_an_entry_point(tmp_path: Path):
    """A conditional entry's entry is a routing function, not a node — there is no name to report."""
    body = 'g.add_conditional_edges(START, router, {"a": "alpha"})\n'
    assert _entry_point_of(tmp_path, body) is None


def test_start_to_end_edge_does_not_derive_a_sentinel_as_entry_point(tmp_path: Path):
    """A degenerate START->END edge must not report '__end__' as the entry node."""
    assert _entry_point_of(tmp_path, "g.add_edge(START, END)\n") is None


def test_derivation_is_independent_of_call_order(tmp_path: Path):
    """Derived at build time over the finished edge set, so source ordering cannot change it."""
    before = _entry_point_of(
        tmp_path, 'g.add_edge("alpha", "beta")\ng.add_edge(START, "alpha")\n', "before.py"
    )
    after = _entry_point_of(
        tmp_path, 'g.add_edge(START, "alpha")\ng.add_edge("alpha", "beta")\n', "after.py"
    )

    assert before == after == "alpha"


# --------------------------------------------------------------------------------------------
# DEC-020 — Command(goto=...) routing declared in a node function body (QA-5-03 / RISK-011)
# --------------------------------------------------------------------------------------------
_COMMAND_HEADER = """from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send
from typing import Literal
"""


def test_command_goto_literal_becomes_a_normal_edge(tmp_path: Path):
    """A single `return Command(goto="x")` is an unconditional hand-off, so one normal edge."""
    src = _COMMAND_HEADER + '''
def plan(state) -> Command[Literal["execute"]]:
    """Decide the plan."""
    return Command(goto="execute", update={"plan": 1})

def execute(state):
    return state

g = StateGraph(dict)
g.add_node("plan", plan)
g.add_node("execute", execute)
g.add_edge(START, "plan")
g.add_edge("execute", END)
'''
    path = _write(tmp_path, "literal_goto.py", src)
    graph = find_graph_definitions(path)[0]

    edge = next(
        e
        for e in graph.edges
        if e.source_node_id.endswith("::plan") and e.target_node_id.endswith("::execute")
    )
    assert edge.type == EDGE_NORMAL
    assert edge.condition_function_name is None

    plan = next(n for n in graph.nodes if n.name == "plan")
    assert plan.resolution == RESOLUTION_FULL
    assert plan.routing_resolution == ROUTING_RESOLUTION_FULL


def test_command_goto_branching_becomes_conditional_edges_with_no_invented_value(tmp_path: Path):
    """Several goto targets from different branches: one conditional edge each.

    The node function is itself the router, and no `condition_value` may be invented — the
    source says where it can go, never what return value triggers it (DEC-017's rule).
    """
    src = _COMMAND_HEADER + '''
def triage(state) -> Command[Literal["urgent", "normal", "__end__"]]:
    """Route by severity."""
    if state["sev"] > 8:
        return Command(goto="urgent")
    if state["sev"] > 3:
        return Command(goto="normal")
    return Command(goto=END)

def urgent(state):
    return state

def normal(state):
    return state

g = StateGraph(dict)
g.add_node("triage", triage)
g.add_node("urgent", urgent)
g.add_node("normal", normal)
g.add_edge(START, "triage")
'''
    path = _write(tmp_path, "branching_goto.py", src)
    graph = find_graph_definitions(path)[0]

    from_triage = [e for e in graph.edges if e.source_node_id.endswith("::triage")]
    targets = {e.target_node_id.rsplit("::", 1)[-1] for e in from_triage}
    assert targets == {"urgent", "normal", END_SENTINEL}
    assert all(e.type == EDGE_CONDITIONAL for e in from_triage)
    assert all(e.condition_function_name == "triage" for e in from_triage)

    routes = [
        r for r in graph.conditional_routes if r.edge_id in {e.id for e in from_triage}
    ]
    assert len(routes) == 3
    assert all(r.condition_value is None for r in routes)
    assert all(r.value_resolution == CONDITION_VALUE_NOT_DERIVABLE for r in routes)

    triage = next(n for n in graph.nodes if n.name == "triage")
    assert triage.routing_resolution == ROUTING_RESOLUTION_FULL


def test_command_goto_non_literal_is_disclosed_not_dropped(tmp_path: Path):
    """A computed goto yields no edge and marks routing partial — never a silent 'no routing'.

    This is the honesty requirement: an unknown must not look identical to a verified fact.
    `resolution` stays FULL because the *function* was located; only the routing axis degrades.
    """
    src = _COMMAND_HEADER + '''
def dispatch(state):
    """Route to a computed destination."""
    target = state["next_step"]
    return Command(goto=target)

g = StateGraph(dict)
g.add_node("dispatch", dispatch)
g.add_edge(START, "dispatch")
'''
    path = _write(tmp_path, "dynamic_goto.py", src)
    graph = find_graph_definitions(path)[0]

    dispatch = next(n for n in graph.nodes if n.name == "dispatch")
    assert dispatch.resolution == RESOLUTION_FULL, "the function itself was located"
    assert dispatch.routing_resolution == ROUTING_RESOLUTION_PARTIAL
    assert not [e for e in graph.edges if e.source_node_id.endswith("::dispatch")], (
        "an unresolvable destination must not be fabricated into an edge"
    )


def test_command_goto_mixed_literal_and_variable_keeps_the_known_half(tmp_path: Path):
    """One resolvable branch and one not: keep the real edge, still flag the node partial."""
    src = _COMMAND_HEADER + '''
def partly(state):
    """Half-knowable routing."""
    if state["done"]:
        return Command(goto="finish")
    return Command(goto=state["fallback"])

def finish(state):
    return state

g = StateGraph(dict)
g.add_node("partly", partly)
g.add_node("finish", finish)
'''
    path = _write(tmp_path, "mixed_goto.py", src)
    graph = find_graph_definitions(path)[0]

    targets = {
        e.target_node_id.rsplit("::", 1)[-1]
        for e in graph.edges
        if e.source_node_id.endswith("::partly")
    }
    assert targets == {"finish"}
    node = next(n for n in graph.nodes if n.name == "partly")
    assert node.routing_resolution == ROUTING_RESOLUTION_PARTIAL


def test_command_goto_send_fanout_resolves_to_the_send_destination(tmp_path: Path):
    """`goto=[Send("worker", ...) for ...]` is parallel fan-out at one known destination."""
    src = _COMMAND_HEADER + '''
def spread(state):
    """Fan out to workers."""
    return Command(goto=[Send("worker", {"item": i}) for i in state["items"]])

def worker(state):
    return state

g = StateGraph(dict)
g.add_node("spread", spread)
g.add_node("worker", worker)
'''
    path = _write(tmp_path, "fanout_goto.py", src)
    graph = find_graph_definitions(path)[0]

    edge = next(e for e in graph.edges if e.source_node_id.endswith("::spread"))
    assert edge.target_node_id.endswith("::worker")
    assert edge.type == EDGE_NORMAL  # one distinct destination
    node = next(n for n in graph.nodes if n.name == "spread")
    assert node.routing_resolution == ROUTING_RESOLUTION_FULL


def test_node_without_command_reports_routing_not_applicable(tmp_path: Path):
    """A body that was read and contains no Command(goto=...) is not_applicable, not partial."""
    src = _COMMAND_HEADER + '''
def plain(state):
    """No routing of its own."""
    return {"x": 1}

g = StateGraph(dict)
g.add_node("plain", plain)
g.add_edge(START, "plain")
g.add_edge("plain", END)
'''
    path = _write(tmp_path, "plain_node.py", src)
    graph = find_graph_definitions(path)[0]
    node = next(n for n in graph.nodes if n.name == "plain")
    assert node.routing_resolution == ROUTING_RESOLUTION_NOT_APPLICABLE


def test_unresolved_node_reports_routing_partial_not_a_false_negative(tmp_path: Path):
    """If the body was never located, routing state is an honest unknown.

    This is the shape that produced QA-5-03: `research_supervisor` is a compiled subgraph whose
    body we cannot read, and reporting "does not route" for it would be a guess.
    """
    src = _COMMAND_HEADER + '''
from .elsewhere import make_stage

g = StateGraph(dict)
g.add_node("stage", make_stage("a"))
'''
    path = _write(tmp_path, "unresolved_node.py", src)
    graph = find_graph_definitions(path)[0]
    node = next(n for n in graph.nodes if n.name == "stage")
    assert node.resolution == RESOLUTION_PARTIAL
    assert node.routing_resolution == ROUTING_RESOLUTION_PARTIAL


def test_command_resume_without_goto_is_not_routing(tmp_path: Path):
    """`Command(resume=True)` expresses resumption, not routing — it must create no edge."""
    src = _COMMAND_HEADER + '''
def gate(state):
    """Human-in-the-loop gate."""
    return Command(resume=True)

g = StateGraph(dict)
g.add_node("gate", gate)
g.add_edge(START, "gate")
'''
    path = _write(tmp_path, "resume_only.py", src)
    graph = find_graph_definitions(path)[0]
    node = next(n for n in graph.nodes if n.name == "gate")
    assert node.routing_resolution == ROUTING_RESOLUTION_NOT_APPLICABLE
    assert not [e for e in graph.edges if e.source_node_id.endswith("::gate")]


def test_command_edge_does_not_duplicate_an_already_declared_edge(tmp_path: Path):
    """A goto restating an add_edge-declared edge collapses onto the same deterministic ID."""
    src = _COMMAND_HEADER + '''
def a(state):
    """Routes onward."""
    return Command(goto="b")

def b(state):
    return state

g = StateGraph(dict)
g.add_node("a", a)
g.add_node("b", b)
g.add_edge("a", "b")
'''
    path = _write(tmp_path, "dupe_edge.py", src)
    graph = find_graph_definitions(path)[0]
    a_to_b = [
        e
        for e in graph.edges
        if e.source_node_id.endswith("::a") and e.target_node_id.endswith("::b")
    ]
    assert len(a_to_b) == 1


def test_cross_file_command_routing_is_detected(tmp_path: Path):
    """Routing declared in a node function that lives in another module still yields edges."""
    _write(
        tmp_path,
        "stages.py",
        '''from langgraph.types import Command
from typing import Literal


def review(state) -> Command[Literal["approve", "reject"]]:
    """Route by score."""
    if state["score"] > 5:
        return Command(goto="approve")
    return Command(goto="reject")
''',
    )
    _write(
        tmp_path,
        "app.py",
        """from langgraph.graph import StateGraph, START
from stages import review


def approve(state):
    return state


def reject(state):
    return state


g = StateGraph(dict)
g.add_node("review", review)
g.add_node("approve", approve)
g.add_node("reject", reject)
g.add_edge(START, "review")
""",
    )
    graphs = scan_repository(tmp_path)
    graph = next(g for g in graphs if g.variable_name == "g")

    review = next(n for n in graph.nodes if n.name == "review")
    assert review.resolution == RESOLUTION_FULL
    assert review.routing_resolution == ROUTING_RESOLUTION_FULL

    targets = {
        e.target_node_id.rsplit("::", 1)[-1]
        for e in graph.edges
        if e.source_node_id.endswith("::review")
    }
    assert targets == {"approve", "reject"}


def test_command_goto_end_uses_the_sentinel_under_an_alias(tmp_path: Path):
    """`goto=END` must resolve to the same sentinel `add_edge(x, END)` produces, alias included."""
    src = '''from langgraph.graph import StateGraph, START
from langgraph.graph import END as FINISH
from langgraph.types import Command


def stop(state):
    """Ends the run."""
    return Command(goto=FINISH)


g = StateGraph(dict)
g.add_node("stop", stop)
g.add_edge(START, "stop")
'''
    path = _write(tmp_path, "aliased_end.py", src)
    graph = find_graph_definitions(path)[0]
    edge = next(e for e in graph.edges if e.source_node_id.endswith("::stop"))
    assert edge.target_node_id == make_node_id(graph.id, END_SENTINEL)
