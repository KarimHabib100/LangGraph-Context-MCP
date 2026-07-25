# PRD.MD — PRODUCT REQUIREMENTS DOCUMENT
# Project: LangGraph Context MCP
# Version: 1.0
# Scope: MVP ONLY

## PROBLEM STATEMENT

Developers building agents with LangGraph work with codebases where the actual behavior lives
in a graph of nodes, edges, and conditional routing functions — not in a flat file tree. When
they use an AI coding assistant like Claude Code to work on that codebase, the assistant has
no native understanding of the graph: it can grep for a function name, but it cannot answer
"what happens after this node if the condition fails" or "which nodes can reach the output
node" without reading and reasoning over many files by hand, which burns tokens and produces
unreliable answers on anything but a trivial graph. Generic code-search and code-intelligence
MCP tools solve the language-parsing problem (find a function, find a class) but are explicitly
not framework-aware, so this specific gap is unaddressed by any existing installable tool as
of this writing. If unsolved, developers keep paying the token and reliability cost of
file-by-file exploration every time they or their AI assistant touch a LangGraph codebase.

## GOAL

Ship a Model Context Protocol server that a LangGraph developer can install with one command,
point at a repository, and immediately ask structural and semantic questions about the graph
through any MCP-compatible client (Claude Code, Claude Desktop). Success for the MVP means:
- Zero-configuration first run — no database server, no API key required
- Structural questions (nodes, edges, conditional routing, tool bindings) are answered
  exactly, not approximately, because they come from parsing the actual graph definition
- Semantic questions ("which node handles X") are answered usefully via local embeddings
- The tool works against real, unmodified open-source LangGraph repositories, not just
  synthetic examples

## TARGET USER

A backend or AI engineer who has already built at least one non-trivial LangGraph application
(multiple nodes, at least one conditional edge) and uses an AI coding assistant daily. They
are comfortable with the command line and with `pip`/`uv`, but do not want to stand up
infrastructure (Postgres, Docker, a cloud vector DB) just to try a developer tool. They likely
already pay for or use Claude Code, Cursor, or another MCP-compatible client, and are
frustrated by watching their assistant re-read the same graph-construction files every
session. They may separately run production Postgres with `pgvector` for their own application
and would use that same instance for this tool if it were supported, but will not install
Postgres solely for this tool.

## MVP SCREENS & FLOWS

This project has no graphical UI. Its "screens" are the CLI surface and the MCP tool surface.

### CLI: `langgraph-context-mcp index <path>`
- What it does: scans `<path>` for LangGraph graph definitions, chunks each node, generates
  local embeddings, and writes the index to the configured storage backend.
- Actions available: none interactive — a single command with one required positional argument.
- Conditional states:
  - Success: prints a summary — graphs found, nodes indexed, edges indexed, time taken, backend used
  - Empty: if no LangGraph usage is found in `<path>`, prints a clear "no graphs found" message and exits 0, not an error
  - Error: if `<path>` does not exist or is not readable, prints a clear error and exits non-zero
  - Partial: if some files fail to parse (syntax errors, dynamic construction beyond the resolver's depth limit), prints which files/nodes were skipped and why, but still completes indexing of everything else

### CLI: `langgraph-context-mcp serve`
- What it does: starts the MCP server over stdio transport, ready for a client like Claude
  Desktop or Claude Code to connect.
- Actions available: none interactive once started — it is a long-running process until the
  client disconnects or the process receives an interrupt.
- Conditional states:
  - Success: server starts silently (per MCP stdio convention, no stdout noise) and responds
    to `tools/list`
  - Error: if the configured storage backend is unreachable (e.g. `DATABASE_URL` set but
    Postgres is down), prints a clear startup error and exits non-zero rather than starting
    in a broken state

### CLI: `langgraph-context-mcp status <path>`
- What it does: reports whether `<path>` has an existing index, when it was last built, and
  which backend it uses.
- Conditional states:
  - Empty: if no index exists yet for `<path>`, says so and suggests running `index`
  - Success: prints graph count, node count, last-indexed timestamp, backend type

### MCP Tool Surface (what a connected client sees and calls)
1. `index_repo(path)` — same behavior as the CLI `index` command, callable by the AI client itself
2. `get_graph_summary(path)` — returns all graphs found, their nodes, edges, and entry points
3. `semantic_search_nodes(query, path, top_k=5)` — natural-language search over node logic
4. `trace_path(from_node, to_node, path)` — returns the route (including conditional branches) between two named nodes, or an explicit "no path" result
5. `what_calls_tool(tool_name, path)` — returns every node that binds or calls a given tool name
6. `explain_conditional(edge_source, path)` — returns the routing function and every possible destination of a conditional edge
7. `reindex(path)` — forces a full re-index, used after the user has edited the graph

Every tool call that references a `path` not yet indexed returns a clear "not indexed — call
index_repo first" result rather than an empty or misleading answer.

## USER FLOW (HAPPY PATH)

1. Developer runs `uvx langgraph-context-mcp index .` inside their LangGraph project
2. Tool scans the repo, finds graph definitions, embeds each node locally, writes to
   `.langgraph-context/index.db` (SQLite default)
3. Developer adds the server to their Claude Code / Claude Desktop MCP config, pointing at
   `uvx langgraph-context-mcp serve`
4. Developer asks Claude Code a question like "what does this agent do if the retrieval node
   comes back empty"
5. Claude Code calls `explain_conditional` or `trace_path` against the already-built index
6. Developer gets a structurally accurate answer without Claude re-reading the whole codebase
7. Developer edits the graph, later runs `reindex` (manually or asks their assistant to call it)

## NON-GOALS (EXPLICIT)

1. No support for agent frameworks other than LangGraph in v1 (no CrewAI, AutoGen, Google ADK, OpenAI Agents SDK)
2. No support for LangGraph.js / non-Python LangGraph in v1 — Python only
3. No automatic background file-watching or live incremental re-index in v1 — `reindex` is a manual/explicit call only
4. No cloud-hosted or multi-tenant version, no user accounts, no auth — this is a local, single-user developer tool
5. No IDE plugin, browser extension, or web UI in v1 — MCP server and CLI only
6. No LLM API calls from inside the server, and no AI-generated summaries or explanations — all answers are derived from parsing and embedding retrieval, not generation, to keep the tool deterministic and free to run
7. No code modification, refactoring, or autofix capability of any kind — this is a strictly read-only analysis tool
8. No automatic secret redaction or security scanning of indexed source — documented as a user responsibility, not a v1 feature

## DATA MODELS (LOGICAL)

**Repository**
- `id`: string, required, derived from resolved absolute path
- `root_path`: string, required
- `last_indexed_at`: datetime, nullable
- `backend_type`: enum(`sqlite`, `pgvector`), required

**GraphDef**
- `id`: string, required
- `repo_id`: string, required, references Repository
- `file_path`: string, required — where the `StateGraph(...)` instantiation lives
- `variable_name`: string, required — the Python variable the graph is assigned to
- `entry_point`: string, nullable — the node name set via `set_entry_point`

**NodeDef**
- `id`: string, required
- `graph_id`: string, required, references GraphDef
- `name`: string, required — the node's registered name in the graph
- `source_file`: string, required
- `line_start`: integer, required
- `line_end`: integer, required
- `docstring`: string, nullable
- `function_body_hash`: string, required — used to detect unchanged nodes on reindex
- `resolution`: enum(`full`, `partial`), required — whether the node function was fully resolved

**EdgeDef**
- `id`: string, required
- `graph_id`: string, required, references GraphDef
- `source_node_id`: string, required, references NodeDef
- `target_node_id`: string, required, references NodeDef
- `type`: enum(`normal`, `conditional`), required
- `condition_function_name`: string, nullable — set only when type is `conditional`

**ConditionalRoute**
- `id`: string, required
- `edge_id`: string, required, references EdgeDef
- `condition_value`: string, required — the return value of the condition function that triggers this route
- `target_node_id`: string, required, references NodeDef

**ToolBinding**
- `id`: string, required
- `node_id`: string, required, references NodeDef
- `tool_name`: string, required
- `tool_source`: string, nullable — module path the tool was imported from, if resolvable

**EmbeddingChunk**
- `id`: string, required
- `node_id`: string, required, references NodeDef, one-to-one
- `vector`: float array, required
- `embedding_model`: string, required
- `dimension`: integer, required

## API / BACKEND CONTRACTS (MCP TOOLS)

**`index_repo`**
- Input: `path: str` (required)
- Auth: none — local tool
- Output: `{graphs_found: int, nodes_indexed: int, edges_indexed: int, partial_nodes: int, backend: str, duration_ms: int}`
- Error cases: path does not exist → `{error: "path_not_found", path: str}`; path exists but is not a directory → `{error: "not_a_directory"}`

**`get_graph_summary`**
- Input: `path: str` (required)
- Output: `{graphs: [{variable_name, file_path, entry_point, node_count, edge_count}]}`
- Error cases: not yet indexed → `{error: "not_indexed", suggestion: "call index_repo first"}`

**`semantic_search_nodes`**
- Input: `query: str` (required), `path: str` (required), `top_k: int` (default 5)
- Output: `{results: [{node_name, file_path, line_start, score, docstring}]}`
- Error cases: empty query → `{error: "empty_query"}`; not indexed → `{error: "not_indexed"}`

**`trace_path`**
- Input: `from_node: str`, `to_node: str`, `path: str`
- Output: `{path_found: bool, route: [node_name, ...], conditional_branches: [...]}`
- Error cases: unknown node name → `{error: "unknown_node", node: str, valid_nodes: [...]}`

**`what_calls_tool`**
- Input: `tool_name: str`, `path: str`
- Output: `{callers: [{node_name, file_path}]}`
- Error cases: not indexed → `{error: "not_indexed"}`

**`explain_conditional`**
- Input: `edge_source: str`, `path: str`
- Output: `{condition_function: str, possible_destinations: [{condition_value, target_node}]}`
- Error cases: source node has no conditional edge → `{error: "not_conditional", node: str}`

**`reindex`**
- Input: `path: str`
- Output: same shape as `index_repo`
- Error cases: same as `index_repo`

## SUCCESS CRITERIA

- [ ] User can run `uvx langgraph-context-mcp index .` on a real LangGraph repo and get a populated index in under 30 seconds for a typical-sized repo
- [ ] User can ask Claude Code "what nodes are in this graph" and get a structurally accurate answer via the MCP tools
- [ ] A semantic query like "which node handles authentication" returns the correct node in the top 3 results on the test fixture
- [ ] `trace_path` correctly returns the full route between two nodes, including conditional branches, on a graph with 3+ levels of conditional routing
- [ ] `explain_conditional` correctly lists every possible destination of a conditional edge
- [ ] The tool works with zero configuration using the SQLite backend — no Postgres, no API key, no network access required for a first run
- [ ] The tool works correctly against pgvector when `DATABASE_URL` is set, including automatic HNSW index creation
- [ ] The server returns a clear partial/best-effort result — never a crash — when it encounters a dynamically constructed or malformed graph
- [ ] `uvx langgraph-context-mcp serve` registers correctly as an MCP server in Claude Desktop and responds to `tools/list`
- [ ] The full test suite passes against at least two real, unmodified open-source LangGraph repositories in addition to the synthetic fixtures
