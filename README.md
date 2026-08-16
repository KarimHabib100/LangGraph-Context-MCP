# LangGraph Context MCP

**Status: In Development**

LangGraph Context MCP is a Model Context Protocol server that parses a LangGraph Python
codebase into a structural graph model (nodes, edges, conditional routing, tool bindings) and
layers local semantic search on top of it, so AI coding assistants can answer questions about
an agent's graph structure and logic without grepping or reading every file.

## Install

```bash
uvx langgraph-context-mcp
```

## CLI

Three subcommands. Index a repository once, then serve it to an MCP client.

```bash
langgraph-context-mcp index <path>     # scan, embed, and store the index
langgraph-context-mcp serve            # run the MCP server on stdio
langgraph-context-mcp status <path>    # report whether a path has an index
```

### `index <path>`

Scans `<path>` for LangGraph graph definitions, builds one embedding per graph node, and writes
the index to `<path>/.langgraph-context/index.db` (or to PostgreSQL — see
[Storage backends](#storage-backends)). Re-running replaces that repository's previous index
rather than appending to it.

```bash
cd my-langgraph-project
langgraph-context-mcp index .
```

### `serve`

Starts the MCP server on the stdio transport and waits for a client. It writes **nothing** to
stdout — that stream carries the MCP protocol — and logs to stderr. Runs until the client
disconnects or the process is interrupted.

### `status <path>`

Reports whether `<path>` has an index, when it was last built, how much it contains, and which
backend holds it. Read-only: it never creates an index.

### `--json`

`index` and `status` accept `--json`, which prints the same structured result as machine-readable
JSON instead of the human-readable summary. Exit codes are unaffected, so a script can read the
code and the payload together. `serve` has no `--json` flag — extra output there would corrupt
the MCP transport.

```bash
langgraph-context-mcp index . --json
```

### Exit codes

Every subcommand uses the same three codes:

| Code | Meaning | When you get it |
|---|---|---|
| `0` | Ran, and the answer is **affirmative** | `index` indexed at least one graph; `status` found an index |
| `1` | Ran correctly, but the answer is **negative** | `index` found no LangGraph usage in the path; `status` found no index |
| `2` | **Could not run at all** | Path missing, not a directory, rejected by path validation, unreadable, or an unexpected failure |

`1` is not an error. It exists so a script can tell "I asked, and the answer is no" apart from
"I could not ask". A caller that does not need the distinction can test for `>= 2`:

```bash
langgraph-context-mcp index . || [ $? -lt 2 ] || exit 1   # fail only on a real error
```

`serve` has no negative-answer case, so it returns only `0` (clean shutdown) or `2` (could not
start — for example `DATABASE_URL` is set but PostgreSQL is unreachable).

Paths containing `..` are rejected with exit `2`, as is an empty path. Pass an absolute path, or
`cd` into the repository and use `.`.

## Connecting an MCP client

Point your client at `langgraph-context-mcp serve`. For Claude Code, a project-level `.mcp.json`:

```json
{
  "mcpServers": {
    "langgraph-context": {
      "command": "uvx",
      "args": ["--from", ".", "langgraph-context-mcp", "serve"]
    }
  }
}
```

For Claude Desktop, use the published package and an **absolute** path in the tool's arguments —
the Desktop app does not launch the server from your project directory:

```json
{
  "mcpServers": {
    "langgraph-context": {
      "command": "uvx",
      "args": ["langgraph-context-mcp", "serve"]
    }
  }
}
```

Index a repository first (`langgraph-context-mcp index /abs/path/to/repo`); the tools report
`not_indexed` until you do, and the client can also call `index_repo` itself.

### The seven tools

| Tool | Answers |
|---|---|
| `index_repo` | Build the index for a repository. Run this first |
| `get_graph_summary` | What graphs and nodes exist, and where execution starts |
| `semantic_search_nodes` | Which node does X, by meaning rather than by name |
| `trace_path` | How execution gets from one node to another, including conditional branches |
| `what_calls_tool` | Which nodes bind or call a given tool |
| `explain_conditional` | Every destination a conditional edge can route to |
| `reindex` | Rebuild the index after the graph has changed |

## Storage backends

By default the index is a single SQLite file at `<repo>/.langgraph-context/index.db` — no server,
no configuration. Set `DATABASE_URL` to a PostgreSQL connection string with the `pgvector`
extension installed to use that instead; the tables and the HNSW index are created automatically
on first connection.

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | Use PostgreSQL + `pgvector` instead of SQLite | unset (SQLite) |
| `LANGGRAPH_CONTEXT_EMBEDDING_MODEL` | Override the embedding model | `nomic-embed-text-v1.5` |
| `LANGGRAPH_CONTEXT_LOG_LEVEL` | Log verbosity (stderr) | `INFO` |

Embeddings run locally on CPU. After the model downloads once on first use, the tool makes no
network calls at all — apart from your own `DATABASE_URL`, if you set one.

## Known limitations

This tool reads code statically and reports what the source actually states. Where it cannot tell,
it says so rather than guessing:

- **Dynamically built nodes.** A node registered in a loop, from a factory, or through a lambda
  cannot be resolved statically. It is still indexed, marked `resolution: "partial"`, and never
  dropped.
- **Tools passed as a variable.** Only tool bindings written as literal lists —
  `ToolNode([search, lookup])`, `.bind_tools([...])` — can be enumerated. A node that passes a
  variable is reported under `unenumerated_tool_nodes`, so an empty `callers` list is never
  mistaken for "nothing uses this tool".
- **List-form conditional routing.** `add_conditional_edges(src, fn, ["a", "b"])` states possible
  destinations, not what the router returns. Those routes report `condition_value: null` with
  `value_resolution: "not_derivable"` instead of inventing a trigger value.
- **Cross-file imports** are followed three levels deep; anything beyond that stays `partial`.
- **No file watching.** An index is exactly as current as the last `index`/`reindex` call.
- **Python only**, and LangGraph only.

Indexed source is stored in plaintext in your chosen backend. If you point `DATABASE_URL` at a
shared database, anything in your node functions and docstrings — including a credential
accidentally committed to source — goes there too. There is no automatic secret redaction.

## License

MIT
