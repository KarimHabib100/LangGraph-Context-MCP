"""AST-based detection of LangGraph ``StateGraph`` definitions.

All parsing here uses the Python standard library ``ast`` module only — no tree-sitter, no
libcst, no regex over source (COR-001 / DEC-001). The single most important guarantee of this
module is that it NEVER raises on malformed input: a syntax error, an unreadable file, a
dynamically constructed node, or a malformed conditional mapping degrades to a skipped file or
a ``resolution="partial"`` node, never an exception (RISK-001 / RISK-002).

Public API:
    find_graph_definitions(file_path, repo_root=None) -> list[GraphDef]
    collect_node_function_refs(file_path, repo_root=None) -> dict[str, dict[str, FuncRef]]

``collect_node_function_refs`` exposes, per graph and per node name, the raw callable reference
the walker saw in ``add_node``. The cross-file resolver (resolver.py) uses it to locate node
functions that live in other modules without re-implementing the walk.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from .graph_model import (
    CONDITION_VALUE_KNOWN,
    CONDITION_VALUE_NOT_DERIVABLE,
    EDGE_CONDITIONAL,
    EDGE_NORMAL,
    END_SENTINEL,
    RESOLUTION_FULL,
    RESOLUTION_NOT_APPLICABLE,
    RESOLUTION_PARTIAL,
    START_SENTINEL,
    TOOL_RESOLUTION_FULL,
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

logger = logging.getLogger(__name__)

# Method names on a StateGraph builder variable that this phase detects (task 1.2).
_ADD_NODE = "add_node"
_ADD_EDGE = "add_edge"
_ADD_CONDITIONAL = "add_conditional_edges"
_SET_ENTRY_POINT = "set_entry_point"

# Tool-binding constructs detected per DEC-007: the LangGraph prebuilt ``ToolNode([...])`` and
# any ``<x>.bind_tools([...])`` call.
_TOOL_NODE = "ToolNode"
_BIND_TOOLS = "bind_tools"

# The only argument shapes we can statically enumerate tools from. A tool call whose argument
# is anything else (a variable / ast.Name, a call, etc.) is a "tools present but not
# enumerable" flag that downgrades the node to partial (DEC-007) — never traced.
_TOOL_LIST_LITERALS = (ast.List, ast.Tuple)

# AST node types whose presence in a call's ancestry means the call is inside a loop /
# comprehension / lambda, i.e. the node is constructed dynamically (RISK-001).
_DYNAMIC_ANCESTORS = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
)

_UNRESOLVED_TARGET = "<unresolved>"

# Exceptions that a malformed / unexpected AST shape can raise while we read a call's arguments.
# We catch these narrowly (per claude.md: always catch specific exception types) so one odd call
# degrades to a skip instead of crashing the whole file parse.
_EXTRACTION_ERRORS = (AttributeError, IndexError, KeyError, TypeError, ValueError)


@dataclass(frozen=True)
class FuncRef:
    """The callable reference the walker saw for a node's ``add_node`` action argument.

    ``kind`` is one of ``"name"``, ``"attr"``, ``"call"``, ``"lambda"``, ``"none"``, ``"other"``.
    ``name`` is the simple identifier when ``kind == "name"`` (the resolver's cross-file hook),
    otherwise ``None``.
    """

    kind: str
    name: str | None


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------
def find_graph_definitions(
    file_path: Path, repo_root: Path | None = None
) -> list[GraphDef]:
    """Parse ``file_path`` and return every ``StateGraph`` defined in it.

    Same-file node functions are resolved directly (task 1.3). Any node whose function cannot
    be resolved in this file — an import from another module, a factory call, a lambda, or a
    node built inside a loop — is returned with ``resolution="partial"`` for the cross-file
    resolver to attempt to upgrade. A syntax error or read error yields an empty list plus a
    logged warning; it never raises.
    """
    parsed = _parse_file(file_path)
    if parsed is None:
        return []
    source, tree = parsed

    display_path = _display_path(file_path, repo_root)
    repo_id = _repo_id(file_path, repo_root)

    aliases = _import_aliases(tree)
    stategraph_names = _names_for(aliases, "StateGraph")
    start_names = _names_for(aliases, "START")
    end_names = _names_for(aliases, "END")
    toolnode_names = _names_for(aliases, "ToolNode")
    import_modules = _import_modules(tree)

    parents = _parent_map(tree)
    local_functions = _local_functions(tree)

    graph_vars = _find_state_graph_vars(tree, stategraph_names)
    if not graph_vars:
        return []

    builders: dict[str, _GraphBuilder] = {
        var: _GraphBuilder(
            graph_id=make_graph_id(display_path, var),
            repo_id=repo_id,
            file_path=display_path,
            variable_name=var,
        )
        for var in graph_vars
    }

    for var, method, call in _iter_builder_calls(tree, set(graph_vars)):
        builder = builders[var]
        try:
            _apply_call(
                builder=builder,
                method=method,
                call=call,
                source=source,
                local_functions=local_functions,
                parents=parents,
                start_names=start_names,
                end_names=end_names,
                toolnode_names=toolnode_names,
                import_modules=import_modules,
            )
        except _EXTRACTION_ERRORS as exc:  # never let one odd call abort the whole file
            logger.warning(
                "Skipping malformed %s call in %s (line %s): %s",
                method,
                display_path,
                getattr(call, "lineno", "?"),
                exc,
            )

    return [b.build() for b in builders.values()]


def collect_node_function_refs(
    file_path: Path, repo_root: Path | None = None
) -> dict[str, dict[str, FuncRef]]:
    """Return ``{graph_id: {node_name: FuncRef}}`` for every ``add_node`` in ``file_path``.

    Used by the cross-file resolver to find where an unresolved node function is imported from,
    without re-implementing the graph-variable discovery walk. Never raises.
    """
    parsed = _parse_file(file_path)
    if parsed is None:
        return {}
    _source, tree = parsed

    display_path = _display_path(file_path, repo_root)
    aliases = _import_aliases(tree)
    stategraph_names = _names_for(aliases, "StateGraph")
    graph_vars = _find_state_graph_vars(tree, stategraph_names)
    if not graph_vars:
        return {}

    result: dict[str, dict[str, FuncRef]] = {
        make_graph_id(display_path, var): {} for var in graph_vars
    }
    for var, method, call in _iter_builder_calls(tree, set(graph_vars)):
        if method != _ADD_NODE:
            continue
        try:
            name, _dynamic_name, func_arg = _parse_add_node_args(call)
        except _EXTRACTION_ERRORS as exc:
            logger.debug("Could not read add_node args in %s: %s", display_path, exc)
            continue
        if name is None:
            continue
        graph_id = make_graph_id(display_path, var)
        result[graph_id][name] = _func_ref(func_arg)
    return result


# --------------------------------------------------------------------------------------------
# Builder accumulator
# --------------------------------------------------------------------------------------------
class _GraphBuilder:
    """Mutable accumulator for one graph; frozen into a GraphDef at the end."""

    def __init__(self, graph_id: str, repo_id: str, file_path: str, variable_name: str) -> None:
        self.graph_id = graph_id
        self.repo_id = repo_id
        self.file_path = file_path
        self.variable_name = variable_name
        self.entry_point: str | None = None
        self._nodes: dict[str, NodeDef] = {}
        self._edges: dict[str, EdgeDef] = {}
        self._routes: dict[str, ConditionalRoute] = {}
        self._tool_bindings: dict[str, ToolBinding] = {}

    def add_node(self, node: NodeDef) -> None:
        self._nodes.setdefault(node.name, node)  # first registration wins

    def add_edge(self, edge: EdgeDef) -> None:
        self._edges.setdefault(edge.id, edge)

    def add_route(self, route: ConditionalRoute) -> None:
        self._routes.setdefault(route.id, route)

    def add_tool_binding(self, binding: ToolBinding) -> None:
        self._tool_bindings.setdefault(binding.id, binding)

    def build(self) -> GraphDef:
        return GraphDef(
            id=self.graph_id,
            repo_id=self.repo_id,
            file_path=self.file_path,
            variable_name=self.variable_name,
            entry_point=self.entry_point or self._derive_entry_point(),
            nodes=tuple(self._nodes.values()),
            edges=tuple(self._edges.values()),
            conditional_routes=tuple(self._routes.values()),
            tool_bindings=tuple(self._tool_bindings.values()),  # DEC-007
        )

    def _derive_entry_point(self) -> str | None:
        """The entry point implied by a lone normal ``add_edge(START, x)`` (DEC-015).

        LangGraph documents ``set_entry_point(x)`` as sugar for ``add_edge(START, x)``, and real
        code overwhelmingly writes the latter — so both must normalize to the same answer rather
        than leaving this field null for the idiom that is actually used (QA-3-02).

        Derived only when the answer is unambiguous. Two or more START edges is a parallel
        fan-out, and a single *conditional* START edge is ``set_conditional_entry_point``, whose
        entry is a routing function rather than a node — neither has a single node name to report,
        so both stay ``None`` instead of guessing. Runs at build time, over the finished edge set,
        so the result never depends on the order calls appeared in the source.
        """
        start_id = make_node_id(self.graph_id, START_SENTINEL)
        from_start = [edge for edge in self._edges.values() if edge.source_node_id == start_id]
        if len(from_start) != 1:
            return None

        edge = from_start[0]
        if edge.type != EDGE_NORMAL:
            return None

        target = edge.target_node_id.removeprefix(f"{self.graph_id}::node::")
        if target in (START_SENTINEL, END_SENTINEL):
            return None
        return target


# --------------------------------------------------------------------------------------------
# Per-call handling
# --------------------------------------------------------------------------------------------
def _apply_call(
    builder: _GraphBuilder,
    method: str,
    call: ast.Call,
    source: str,
    local_functions: dict[str, ast.AST],
    parents: dict[ast.AST, ast.AST],
    start_names: set[str],
    end_names: set[str],
    toolnode_names: set[str],
    import_modules: dict[str, str],
) -> None:
    if method == _ADD_NODE:
        _handle_add_node(
            builder, call, source, local_functions, parents, toolnode_names, import_modules
        )
    elif method == _ADD_EDGE:
        _handle_add_edge(builder, call, start_names, end_names)
    elif method == _ADD_CONDITIONAL:
        _handle_add_conditional(builder, call, start_names, end_names)
    elif method == _SET_ENTRY_POINT:
        _handle_set_entry_point(builder, call)


def _handle_add_node(
    builder: _GraphBuilder,
    call: ast.Call,
    source: str,
    local_functions: dict[str, ast.AST],
    parents: dict[ast.AST, ast.AST],
    toolnode_names: set[str],
    import_modules: dict[str, str],
) -> None:
    name, dynamic_name, func_arg = _parse_add_node_args(call)
    if name is None:
        # Could not determine a node name at all — record it, do not drop it.
        name = f"<dynamic:{_safe_unparse(call)}>"
        dynamic_name = True

    in_loop = _is_dynamically_constructed(call, parents)
    funcdef = _resolve_local_func(func_arg, local_functions)
    node_id = make_node_id(builder.graph_id, name)
    # A bare ToolNode([...]) used inline as the action was never a Python function to resolve.
    action_is_toolnode = isinstance(func_arg, ast.Call) and _is_toolnode(
        func_arg.func, toolnode_names
    )

    # `resolution` reflects ONLY whether the node function itself was located (DEC-008); tool
    # state is tracked separately in `tool_resolution` below and never affects `resolution`.
    if funcdef is not None and not dynamic_name and not in_loop:
        resolution = RESOLUTION_FULL
        line_start = function_line_start(funcdef)
        line_end = funcdef.end_lineno or funcdef.lineno
        docstring = ast.get_docstring(funcdef)
        body_hash = _hash_segment(source, funcdef)
    else:
        # not_applicable = there was no function to resolve (a bare ToolNode action); partial is
        # reserved for genuine uncertainty about a function that does exist (DEC-009).
        resolution = (
            RESOLUTION_NOT_APPLICABLE if action_is_toolnode else RESOLUTION_PARTIAL
        )
        line_start = getattr(call, "lineno", 0)
        line_end = getattr(call, "end_lineno", None) or line_start
        docstring = None
        body_hash = ""

    # Tool bindings (DEC-007): a ToolNode([...]) used directly as the node's action, plus any
    # bind_tools([...]) / ToolNode([...]) called inside the node's same-file function body.
    # Cross-file function bodies are handled separately in resolver.py.
    tool_args = _iter_tool_lists(funcdef, toolnode_names) if funcdef is not None else []
    if action_is_toolnode and func_arg.args:
        tool_args.append(func_arg.args[0])
    bindings, tool_resolution = _extract_tools(node_id, tool_args, import_modules)

    builder.add_node(
        NodeDef(
            id=node_id,
            graph_id=builder.graph_id,
            name=name,
            source_file=builder.file_path,
            line_start=line_start,
            line_end=line_end,
            docstring=docstring,
            function_body_hash=body_hash,
            resolution=resolution,
            tool_resolution=tool_resolution,
        )
    )

    for binding in bindings:
        builder.add_tool_binding(binding)


def _handle_add_edge(
    builder: _GraphBuilder,
    call: ast.Call,
    start_names: set[str],
    end_names: set[str],
) -> None:
    if len(call.args) < 2:
        return
    source_name = _resolve_node_ref(call.args[0], start_names, end_names)
    target_name = _resolve_node_ref(call.args[1], start_names, end_names)
    edge_id = make_edge_id(builder.graph_id, source_name, target_name, EDGE_NORMAL)
    builder.add_edge(
        EdgeDef(
            id=edge_id,
            graph_id=builder.graph_id,
            source_node_id=make_node_id(builder.graph_id, source_name),
            target_node_id=make_node_id(builder.graph_id, target_name),
            type=EDGE_NORMAL,
            condition_function_name=None,
        )
    )


def _handle_add_conditional(
    builder: _GraphBuilder,
    call: ast.Call,
    start_names: set[str],
    end_names: set[str],
) -> None:
    if not call.args:
        return
    source_name = _resolve_node_ref(call.args[0], start_names, end_names)
    condition_name = _callable_name(call.args[1]) if len(call.args) >= 2 else None

    mapping = call.args[2] if len(call.args) >= 3 else _keyword(call, "path_map")
    destinations = _extract_mapping(mapping, start_names, end_names)

    if not destinations:
        # Conditional edge whose branches cannot be statically enumerated (mapping is a
        # variable, omitted, or the router returns node names directly). Record the edge's
        # existence so it is never silently dropped, but with no enumerable routes.
        destinations = [(_UNRESOLVED_TARGET, _UNRESOLVED_TARGET, CONDITION_VALUE_NOT_DERIVABLE)]

    for condition_value, target_name, value_resolution in destinations:
        # The ID keeps using the destination name when there is no condition value, so every
        # edge ID generated before DEC-017 is byte-identical and no existing index is invalidated.
        # This is an opaque uniqueness key, not a semantic claim.
        edge_id = make_edge_id(
            builder.graph_id,
            source_name,
            target_name,
            EDGE_CONDITIONAL,
            condition_value if condition_value is not None else target_name,
        )
        builder.add_edge(
            EdgeDef(
                id=edge_id,
                graph_id=builder.graph_id,
                source_node_id=make_node_id(builder.graph_id, source_name),
                target_node_id=make_node_id(builder.graph_id, target_name),
                type=EDGE_CONDITIONAL,
                condition_function_name=condition_name,
            )
        )
        if target_name != _UNRESOLVED_TARGET:
            builder.add_route(
                ConditionalRoute(
                    id=make_route_id(edge_id),
                    edge_id=edge_id,
                    condition_value=condition_value,
                    target_node_id=make_node_id(builder.graph_id, target_name),
                    value_resolution=value_resolution,
                )
            )


def _handle_set_entry_point(builder: _GraphBuilder, call: ast.Call) -> None:
    if not call.args:
        return
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        builder.entry_point = arg.value


# --------------------------------------------------------------------------------------------
# Argument parsing helpers
# --------------------------------------------------------------------------------------------
def _parse_add_node_args(call: ast.Call) -> tuple[str | None, bool, ast.AST | None]:
    """Return ``(node_name, dynamic_name, func_arg)`` for an ``add_node`` call.

    Handles the common forms: ``add_node("name", func)``, ``add_node(func)`` (name inferred
    from the function identifier), and the keyword forms ``node=``/``action=``. When the node
    name is not a static string constant, ``dynamic_name`` is True and ``node_name`` is a best
    effort (or None if nothing usable could be derived).
    """
    positional = call.args
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg}

    name_arg: ast.AST | None = None
    func_arg: ast.AST | None = None

    if positional:
        first = positional[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            name_arg = first
            func_arg = positional[1] if len(positional) >= 2 else None
        else:
            # Single callable positional: add_node(func) -> name inferred from func identifier.
            func_arg = first
    if name_arg is None and "node" in kwargs:
        name_arg = kwargs["node"]
    if name_arg is None and "name" in kwargs:
        name_arg = kwargs["name"]
    if func_arg is None:
        func_arg = kwargs.get("action")

    if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str):
        return name_arg.value, False, func_arg

    if name_arg is None and isinstance(func_arg, ast.Name):
        # add_node(some_function) — LangGraph uses the function's name as the node name.
        return func_arg.id, False, func_arg

    if name_arg is not None:
        return f"<dynamic:{_safe_unparse(name_arg)}>", True, func_arg

    return None, True, func_arg


def _extract_mapping(
    mapping: ast.AST | None, start_names: set[str], end_names: set[str]
) -> list[tuple[str | None, str, str]]:
    """Extract ``(condition_value, target_name, value_resolution)`` triples from a mapping.

    The two supported forms mean different things, and DEC-017 keeps them distinct:

    - A dict literal ``{"a": "node_a"}`` is a real mapping — the key *is* the router's return
      value, so it is recorded with ``CONDITION_VALUE_KNOWN``.
    - A list/tuple ``["node_a", "node_b"]`` is a *destination hint*. It states where the router
      may route, never what it returns (a real router may return ``Send`` objects, which are not
      node names at all), so ``condition_value`` is ``None`` and the route is marked
      ``CONDITION_VALUE_NOT_DERIVABLE`` rather than labelled with the destination's own name.

    Anything else yields an empty list, which the caller records as an unresolved conditional edge.
    """
    if isinstance(mapping, ast.Dict):
        pairs: list[tuple[str | None, str, str]] = []
        for key, value in zip(mapping.keys, mapping.values):
            if key is None:  # dict unpacking (**other) — cannot enumerate
                continue
            condition_value = _constant_str(key)
            target_name = _resolve_node_ref(value, start_names, end_names)
            if condition_value is None:
                # A non-literal key is still genuinely the compared value, just not a bare
                # constant — so it stays `known`, unparsed for display.
                condition_value = _safe_unparse(key)
            pairs.append((condition_value, target_name, CONDITION_VALUE_KNOWN))
        return pairs
    if isinstance(mapping, (ast.List, ast.Tuple)):
        pairs = []
        for element in mapping.elts:
            target_name = _resolve_node_ref(element, start_names, end_names)
            pairs.append((None, target_name, CONDITION_VALUE_NOT_DERIVABLE))
        return pairs
    return []


def _func_ref(func_arg: ast.AST | None) -> FuncRef:
    if isinstance(func_arg, ast.Name):
        return FuncRef("name", func_arg.id)
    if isinstance(func_arg, ast.Attribute):
        return FuncRef("attr", None)
    if isinstance(func_arg, ast.Call):
        return FuncRef("call", None)
    if isinstance(func_arg, ast.Lambda):
        return FuncRef("lambda", None)
    if func_arg is None:
        return FuncRef("none", None)
    return FuncRef("other", None)


def function_line_start(funcdef: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Return the first line of ``funcdef``, counting its decorators (DEC-013).

    ``ast`` puts a ``FunctionDef``'s ``lineno`` on the ``def`` line and keeps decorators in
    ``decorator_list``, whose entries sit above it — so ``funcdef.lineno`` alone under-reports
    where a decorated node actually begins. A decorator expression's own ``lineno`` is already
    the line carrying its ``@`` (verified against CPython for both ``@name`` and a multi-line
    ``@call(...)`` form), so the earliest of those lines is the first line of the node.

    Shared by the same-file walker and the cross-file resolver so a decorated node reports the
    same span wherever its function is defined.
    """
    earliest = funcdef.lineno
    for decorator in funcdef.decorator_list:
        decorator_line = getattr(decorator, "lineno", None)
        if decorator_line is not None:
            earliest = min(earliest, decorator_line)
    return earliest


def _resolve_local_func(
    func_arg: ast.AST | None, local_functions: dict[str, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if isinstance(func_arg, ast.Name):
        found = local_functions.get(func_arg.id)
        if isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return found
    return None


def _resolve_node_ref(
    value: ast.AST, start_names: set[str], end_names: set[str]
) -> str:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    tail = _trailing_name(value)
    if tail is not None:
        if tail in start_names:
            return START_SENTINEL
        if tail in end_names:
            return END_SENTINEL
    return _safe_unparse(value)


def _callable_name(value: ast.AST | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Lambda):
        return "<lambda>"
    return _safe_unparse(value)


# --------------------------------------------------------------------------------------------
# Tool-binding extraction (DEC-007)
# --------------------------------------------------------------------------------------------
def extract_tools_from_funcdef(
    node_id: str, funcdef: ast.AST, module_tree: ast.AST
) -> tuple[list[ToolBinding], str]:
    """Extract a node function's tool bindings and its ``tool_resolution`` state (DEC-008).

    Public entry point used by the cross-file resolver: given a node function located in another
    module, scan its body for ``bind_tools([...])`` / ``ToolNode([...])`` calls, resolve each
    tool's source against ``module_tree``'s own imports, and return the bindings plus whether
    they were fully enumerable. Never raises.
    """
    aliases = _import_aliases(module_tree)
    toolnode_names = _names_for(aliases, "ToolNode")
    import_modules = _import_modules(module_tree)
    tool_args = _iter_tool_lists(funcdef, toolnode_names)
    return _extract_tools(node_id, tool_args, import_modules)


def _extract_tools(
    node_id: str, tool_args: list[ast.AST], import_modules: dict[str, str]
) -> tuple[list[ToolBinding], str]:
    """Turn the argument nodes of tool calls into bindings plus a ``tool_resolution`` state.

    - No tool calls at all -> ``not_applicable``.
    - Every argument is a list literal and every element is a resolvable tool -> ``full``.
    - Any argument is a non-literal (a variable), or any list element is not a resolvable tool
      (Starred, nested list/tuple, dict, non-string constant, ...) -> ``partial``. The
      unresolvable piece is skipped, never fabricated into a bogus tool name (DEC-007/DEC-008).
      Detection only — a variable's contents are never traced.
    """
    if not tool_args:
        return [], TOOL_RESOLUTION_NOT_APPLICABLE

    bindings: dict[str, ToolBinding] = {}
    fully_enumerable = True
    for arg in tool_args:
        if not isinstance(arg, _TOOL_LIST_LITERALS):
            fully_enumerable = False  # a variable / non-literal argument
            continue
        for element in arg.elts:
            if not _is_resolvable_tool_element(element):
                fully_enumerable = False  # e.g. *starred, nested list, dict, number
                continue
            tool_name, base = _tool_name_and_base(element)
            binding = ToolBinding(
                id=make_tool_binding_id(node_id, tool_name),
                node_id=node_id,
                tool_name=tool_name,
                tool_source=import_modules.get(base) if base else None,
            )
            bindings[binding.id] = binding

    state = TOOL_RESOLUTION_FULL if fully_enumerable else TOOL_RESOLUTION_PARTIAL
    return list(bindings.values()), state


def _is_resolvable_tool_element(node: ast.AST) -> bool:
    """A tools-list element we can statically turn into a tool name: Name/Attribute/Call/str."""
    if isinstance(node, (ast.Name, ast.Attribute, ast.Call)):
        return True
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _iter_tool_lists(scope: ast.AST, toolnode_names: set[str]) -> list[ast.AST]:
    """Return the list-literal argument of every bind_tools / ToolNode call within ``scope``."""
    tool_lists: list[ast.AST] = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        is_bind_tools = isinstance(func, ast.Attribute) and func.attr == _BIND_TOOLS
        if is_bind_tools or _is_toolnode(func, toolnode_names):
            tool_lists.append(node.args[0])
    return tool_lists


def _is_toolnode(func: ast.AST, toolnode_names: set[str]) -> bool:
    if isinstance(func, ast.Name):
        return func.id in toolnode_names
    if isinstance(func, ast.Attribute):
        return func.attr == _TOOL_NODE
    return False


def _tool_name_and_base(node: ast.AST) -> tuple[str, str | None]:
    """Return ``(tool_name, base_symbol)`` for one resolvable element of a tools list.

    Only called on elements that passed ``_is_resolvable_tool_element`` (Name/Attribute/Call/
    str constant). ``base_symbol`` is the identifier whose import determines ``tool_source``
    (e.g. for ``tools.search`` it is ``tools``; for a bare ``search`` it is ``search``); ``None``
    when no import-resolvable base exists (e.g. a string constant).
    """
    if isinstance(node, ast.Name):
        return node.id, node.id
    if isinstance(node, ast.Attribute):
        base = node.value.id if isinstance(node.value, ast.Name) else None
        return node.attr, base
    if isinstance(node, ast.Call):
        return _tool_name_and_base(node.func)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, None
    return _safe_unparse(node), None


# --------------------------------------------------------------------------------------------
# Tree discovery helpers
# --------------------------------------------------------------------------------------------
def _find_state_graph_vars(tree: ast.AST, stategraph_names: set[str]) -> list[str]:
    """Find variables assigned a ``StateGraph(...)`` instance, in source order."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _is_state_graph_call(node.value, stategraph_names):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id not in found:
                found.append(target.id)
    return found


def _is_state_graph_call(value: ast.AST, stategraph_names: set[str]) -> bool:
    if not isinstance(value, ast.Call):
        return False
    func = value.func
    if isinstance(func, ast.Name):
        return func.id in stategraph_names
    if isinstance(func, ast.Attribute):
        return func.attr == "StateGraph"
    return False


def _iter_builder_calls(
    tree: ast.AST, graph_vars: set[str]
) -> list[tuple[str, str, ast.Call]]:
    """Yield ``(variable_name, method_name, call)`` for builder method calls, in source order."""
    calls: list[tuple[str, str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        base = func.value
        if isinstance(base, ast.Name) and base.id in graph_vars:
            calls.append((base.id, func.attr, node))
    calls.sort(key=lambda item: (getattr(item[2], "lineno", 0), getattr(item[2], "col_offset", 0)))
    return calls


def _local_functions(tree: ast.AST) -> dict[str, ast.AST]:
    """Map function name -> its def node. Module-level defs take precedence over nested ones."""
    functions: dict[str, ast.AST] = {}
    module_body = getattr(tree, "body", [])
    for node in module_body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, node)
    return functions


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_dynamically_constructed(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = parents.get(node)
    while current is not None:
        if isinstance(current, _DYNAMIC_ANCESTORS):
            return True
        current = parents.get(current)
    return False


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    """Map ``local_name -> original_name`` for ``import``/``from ... import`` statements."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            for alias in node.names:
                aliases[alias.asname or alias.name] = alias.name
    return aliases


def _names_for(aliases: dict[str, str], original: str) -> set[str]:
    names = {local for local, orig in aliases.items() if orig == original}
    names.add(original)
    return names


def _import_modules(tree: ast.AST) -> dict[str, str]:
    """Map ``local_name -> module_path`` for imports, for resolving a tool's ``tool_source``.

    ``from pkg.sub import x`` maps ``x -> "pkg.sub"``; a relative ``from .tools import x`` maps
    ``x -> ".tools"`` (leading dots preserved); ``import pkg.sub as p`` maps ``p -> "pkg.sub"``.
    """
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_path = "." * node.level + (node.module or "")
            for alias in node.names:
                if alias.name == "*":
                    continue
                modules[alias.asname or alias.name] = module_path
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name] = alias.name
    return modules


# --------------------------------------------------------------------------------------------
# Low-level utilities
# --------------------------------------------------------------------------------------------
def _parse_file(file_path: Path) -> tuple[str, ast.Module] | None:
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping %s: cannot read as UTF-8 (%s)", file_path, exc)
        return None
    try:
        tree = ast.parse(source, filename=str(file_path))
    except SyntaxError as exc:
        logger.warning("Skipping %s: syntax error (%s)", file_path, exc)
        return None
    return source, tree


def _display_path(file_path: Path, repo_root: Path | None) -> str:
    if repo_root is not None:
        try:
            return file_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            pass
    return file_path.as_posix()


def _repo_id(file_path: Path, repo_root: Path | None) -> str:
    if repo_root is not None:
        return repo_root.resolve().as_posix()
    return file_path.resolve().parent.as_posix()


def _hash_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        segment = _safe_unparse(node)
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _constant_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        return str(node.value)
    return None


def _trailing_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (ValueError, AttributeError, RecursionError):
        return "<unparseable>"
