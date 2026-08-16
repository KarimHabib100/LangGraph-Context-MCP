"""The MCP server: one ``MCPServer`` instance with all seven tools registered on it.

The tool *implementations* live in ``tools/mcp_tools.py``; this module only creates the server,
registers them, and runs the stdio transport. Registration is deliberately
``mcp.tool()(function)`` rather than a decorator on a locally-defined wrapper, so the exposed tool
name is the Python function's own name and the description the client reads is that function's own
docstring — claude.md's naming convention holds by construction rather than by convention
(DEC-018).

Two transport rules matter here and are easy to break:

- **stdout belongs to the protocol.** Anything else written there corrupts the JSON-RPC stream, so
  logging is pinned to stderr and this package never calls ``print()`` (claude.md).
- **Sync handlers, on purpose.** MCP SDK v2 runs a synchronous ``@mcp.tool()`` function on a worker
  thread instead of the event loop (DEC-010), so the blocking storage reads inside each tool cannot
  stall the server. That is also why no store is ever cached across calls — see RISK-008 and
  DEC-018.
"""

from __future__ import annotations

import logging
import os
import sys

from mcp.server import MCPServer

from .tools.mcp_tools import (
    explain_conditional,
    get_graph_summary,
    index_repo,
    reindex,
    semantic_search_nodes,
    trace_path,
    what_calls_tool,
)

logger = logging.getLogger(__name__)

SERVER_NAME = "langgraph-context-mcp"

# claude.md's ENVIRONMENT VARIABLES table. Unset means INFO.
LOG_LEVEL_ENV_VAR = "LANGGRAPH_CONTEXT_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"

# What a connected client is told the server is for, before it reads any individual tool.
INSTRUCTIONS = """\
Answers structural and semantic questions about a LangGraph Python codebase by parsing its graph
definitions — nodes, edges, conditional routing, and tool bindings — rather than by reading files.

Call index_repo once per repository path before anything else; the other tools read that index and
will tell you to index first if it is missing. Structural answers (get_graph_summary, trace_path,
what_calls_tool, explain_conditional) come from the parsed graph and are exact. Semantic answers
(semantic_search_nodes) come from local embeddings and are ranked guesses.

Where the source does not state something, these tools say so instead of inferring it: a node whose
function could not be located reports resolution="partial", and a conditional route whose trigger
value the source never states reports value_resolution="not_derivable" with a null condition_value.
Report those as unknown rather than filling them in.\
"""

# The 7 tools of DEC-004. This tuple is the registration list and the count asserted by the tests —
# adding an 8th tool requires a new DEC entry and developer approval, not an edit here.
TOOL_FUNCTIONS = (
    index_repo,
    get_graph_summary,
    semantic_search_nodes,
    trace_path,
    what_calls_tool,
    explain_conditional,
    reindex,
)


def configure_logging(stream=None) -> None:
    """Send this package's logs to stderr at ``LANGGRAPH_CONTEXT_LOG_LEVEL`` (default INFO).

    stderr specifically: on the stdio transport stdout carries the MCP protocol, and a log line
    written there would corrupt it. Installs a handler only if the root logger has none, so a host
    application that already configured logging keeps its own setup.
    """
    level_name = (os.environ.get(LOG_LEVEL_ENV_VAR) or DEFAULT_LOG_LEVEL).strip().upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level)


def build_server() -> MCPServer:
    """Create the ``MCPServer`` and register all seven tools on it.

    Returns a new instance per call rather than a module-level singleton, so tests can build one
    without side effects. Building the server is cheap: no store is opened and no embedding model
    is loaded here.
    """
    mcp = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    for function in TOOL_FUNCTIONS:
        mcp.tool()(function)
    return mcp


def run_stdio() -> None:
    """Run the server on the stdio transport until the client disconnects or we are interrupted.

    Produces no stdout output of its own — see this module's docstring.
    """
    configure_logging()
    logger.info("Starting %s on stdio transport", SERVER_NAME)
    build_server().run("stdio")
