"""Bounded-depth cross-file resolution of node functions.

When ``add_node("x", some_func)`` references a function imported from another module, the
same-file walker leaves that node ``resolution="partial"``. This module follows the file's
``from ... import ...`` statements up to a hard depth of 3 module hops to locate the function's
real definition and upgrade the node to ``resolution="full"`` (real line numbers, docstring,
body hash).

The depth cap is a hard limit, not a suggestion (RISK-002): a re-export chain longer than 3
hops, or a circular import, stops at the cap and the node stays ``partial`` rather than
recursing without bound. This module never raises on malformed input.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import replace
from pathlib import Path

from .ast_walker import (
    _display_path,
    _hash_segment,
    _parse_file,
    collect_node_function_refs,
    extract_tools_from_funcdef,
)
from .graph_model import (
    RESOLUTION_FULL,
    RESOLUTION_PARTIAL,
    GraphDef,
)

logger = logging.getLogger(__name__)

# Hard cap on how many module hops the resolver will follow (RISK-002). Not configurable.
MAX_IMPORT_DEPTH = 3

# Placeholder marker used by the walker for nodes with a non-static name; never resolvable.
_DYNAMIC_PREFIX = "<dynamic:"


def resolve_cross_file_references(graph_def: GraphDef, repo_root: Path) -> GraphDef:
    """Upgrade partial nodes whose functions live in other modules, within the depth cap.

    Returns a new ``GraphDef``. Nodes that cannot be resolved within 3 import hops — or whose
    reference is dynamic, a factory call, a lambda, or an attribute — are left ``partial`` and
    never dropped. Never raises: any failure degrades to the unchanged input.
    """
    partial_names = {
        node.name for node in graph_def.nodes if node.resolution == RESOLUTION_PARTIAL
    }
    if not partial_names:
        return graph_def

    real_file = _real_file(graph_def, repo_root)
    if real_file is None:
        return graph_def

    parsed = _parse_file(real_file)
    if parsed is None:
        return graph_def
    _source, tree = parsed

    refs = collect_node_function_refs(real_file, repo_root).get(graph_def.id, {})
    from_imports = _from_imports(tree)

    upgrades = {}
    new_bindings = []
    for node in graph_def.nodes:
        if node.resolution != RESOLUTION_PARTIAL:
            continue
        if node.name.startswith(_DYNAMIC_PREFIX):
            continue  # dynamically constructed — not a cross-file resolution candidate
        ref = refs.get(node.name)
        if ref is None or ref.kind != "name" or ref.name not in from_imports:
            continue
        match = _resolve_symbol(
            symbol=ref.name,
            current_file=real_file,
            repo_root=repo_root,
            from_imports=from_imports,
            depth=1,
            visited={real_file.resolve()},
        )
        if match is None:
            continue
        display_path, funcdef, source, module_tree = match
        # The function was located, so `resolution` is FULL regardless of tools (DEC-008). Tool
        # enumeration state is tracked independently in `tool_resolution`: a cross-file body may
        # bind tools via a variable or a non-resolvable element, which makes it partial.
        bindings, tool_resolution = extract_tools_from_funcdef(node.id, funcdef, module_tree)
        upgrades[node.name] = replace(
            node,
            source_file=display_path,
            line_start=funcdef.lineno,
            line_end=funcdef.end_lineno or funcdef.lineno,
            docstring=ast.get_docstring(funcdef),
            function_body_hash=_hash_segment(source, funcdef),
            resolution=RESOLUTION_FULL,
            tool_resolution=tool_resolution,
        )
        new_bindings.extend(bindings)

    if not upgrades and not new_bindings:
        return graph_def

    new_nodes = tuple(upgrades.get(node.name, node) for node in graph_def.nodes)
    merged_bindings = {binding.id: binding for binding in graph_def.tool_bindings}
    for binding in new_bindings:
        merged_bindings.setdefault(binding.id, binding)
    return replace(
        graph_def, nodes=new_nodes, tool_bindings=tuple(merged_bindings.values())
    )


# --------------------------------------------------------------------------------------------
# Internal resolution machinery
# --------------------------------------------------------------------------------------------
def _resolve_symbol(
    symbol: str,
    current_file: Path,
    repo_root: Path,
    from_imports: dict[str, tuple[str | None, int, str]],
    depth: int,
    visited: set[Path],
) -> tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, str, ast.Module] | None:
    """Follow ``symbol``'s import to its defining module and locate its FunctionDef.

    Returns ``(display_path, funcdef, source, module_tree)`` or ``None``. Bounded by
    ``MAX_IMPORT_DEPTH`` module hops and a ``visited`` set so a circular import cannot recurse
    forever. ``module_tree`` is the defining module's AST, used to resolve tool_source for any
    tool bindings in the function body (DEC-007).
    """
    if depth > MAX_IMPORT_DEPTH:
        return None
    info = from_imports.get(symbol)
    if info is None:
        return None
    module, level, original_name = info

    for candidate in _candidate_files(module, level, current_file, repo_root):
        if candidate is None:
            continue
        resolved = candidate.resolve()
        if not candidate.exists() or resolved in visited:
            continue
        parsed = _parse_file(candidate)
        if parsed is None:
            continue
        source, tree = parsed

        funcdef = _module_level_func(tree, original_name)
        if funcdef is not None:
            return _display_path(candidate, repo_root), funcdef, source, tree

        # Not defined here directly — maybe re-exported. Follow one more hop.
        sub_imports = _from_imports(tree)
        if original_name in sub_imports:
            deeper = _resolve_symbol(
                symbol=original_name,
                current_file=candidate,
                repo_root=repo_root,
                from_imports=sub_imports,
                depth=depth + 1,
                visited=visited | {resolved},
            )
            if deeper is not None:
                return deeper
    return None


def _candidate_files(
    module: str | None, level: int, current_file: Path, repo_root: Path
) -> list[Path | None]:
    """Return possible files for an imported module, as ``module.py`` and ``module/__init__.py``.

    Handles both absolute imports (resolved against ``repo_root``) and relative imports
    (resolved against the importing file's package, honouring the import ``level``).
    """
    if level and level > 0:
        base = current_file.parent
        for _ in range(level - 1):
            base = base.parent
        parts = module.split(".") if module else []
        if not parts:
            return [base / "__init__.py"]
        target = base.joinpath(*parts)
    else:
        if not module:
            return []
        target = repo_root.joinpath(*module.split("."))
    return [target.with_suffix(".py"), target / "__init__.py"]


def _from_imports(tree: ast.AST) -> dict[str, tuple[str | None, int, str]]:
    """Map ``local_name -> (module, level, original_name)`` for ``from ... import`` statements."""
    result: dict[str, tuple[str | None, int, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            result[local] = (node.module, node.level, alias.name)
    return result


def _module_level_func(
    tree: ast.AST, name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _real_file(graph_def: GraphDef, repo_root: Path) -> Path | None:
    """Recover the on-disk file for a GraphDef whose file_path may be repo-relative."""
    candidate = repo_root / graph_def.file_path
    if candidate.exists():
        return candidate
    alt = Path(graph_def.file_path)
    if alt.exists():
        return alt
    return None
