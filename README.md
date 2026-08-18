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

## Status & limits

*As of 2026-08-18. Pre-launch: the engine and the MCP surface are built and tested; packaging and
the final pre-launch QA pass are not finished.*

### What has actually been verified

Measured against real, unmodified open-source LangGraph repositories, not only synthetic fixtures:

- **Parses real code.** `langchain-ai/open_deep_research` (@ `1b7d2e8`) yields 7 graphs, 23 nodes
  and 23 edges, 3 of them partially resolved, with no crash. Tool-binding shapes were additionally
  surveyed across 4 unmodified repositories.
- **Indexes inside the target.** That repository indexes end to end in ~19s on CPU, including the
  one-time model load — against a design target of 30s. A generated 250-node graph indexes in ~19s.
- **Retrieval holds up adversarially.** On five queries deliberately phrased so a common verb
  lexically matches the *wrong* node's name, the correct node was in the top 3 every time and
  ranked first in 4 of 5.
- **Both backends agree.** SQLite and PostgreSQL + `pgvector` return the same ranking order on
  identical data — zero ordering differences, worst score delta ~6e-07 — because both are pinned
  to cosine on unit-length vectors rather than left on their differing defaults.
- **Genuinely offline.** With outbound sockets blocked at the OS level, a full query completes with
  zero connection attempts once the model is cached.
- **Re-indexing is idempotent.** Five consecutive runs leave exactly one repository row, one graph
  row and four chunk rows — no duplicates, and rows for deleted nodes are dropped.
- **Works with real MCP clients.** All seven tools list and execute under the MCP Inspector, the
  `mcp` SDK's own stdio client, and Claude Desktop.
- **Test suite:** 265 passing. A further 25 `pgvector` tests skip without `DATABASE_URL` and run in
  CI against PostgreSQL 16 + `pgvector`.

Not yet exercised: Claude Code's bundled client (Claude Desktop is verified), and installation from
PyPI, which does not exist until launch.

### Where it stops

This tool reports what the source *states*. Where the source does not state something, it says so
instead of inferring — every limit below is a deliberate choice to stay silent rather than guess.

**Scope.** Python LangGraph only: no LangGraph.js, and no other agent framework (CrewAI, AutoGen,
Google ADK, OpenAI Agents SDK). It is strictly read-only — it never edits, refactors, or fixes code.
It makes no LLM calls and writes no generated prose: every answer is parsed structure or retrieved
text, which is what makes the structural answers exact and free to run.

**Freshness.** Nothing watches your files. An index is exactly as current as the last `index` or
`reindex`.

**What static analysis cannot reach.**

- *Dynamically built nodes* — registered in a loop, returned from a factory, or bound as a lambda —
  are indexed and marked `resolution: "partial"`, never dropped. Nodes registered inside a loop
  additionally collapse into a single entry carrying a synthesized placeholder name, so the
  individual iterations are not enumerable.
- *Cross-file node functions* are followed three import hops; beyond that the node stays `partial`.
- *Tool bindings* are read only from literal lists — `ToolNode([search, lookup])`,
  `.bind_tools([...])`. A binding passed as a variable is not traced, and on the four repositories
  surveyed only one of eight real binding sites was a literal, so expect `callers` to be sparse on
  real code. Those nodes appear in `unenumerated_tool_nodes` so an empty result is never mistaken
  for "nothing uses this tool". Within a list, elements that cannot be resolved statically — a
  `*spread`, a nested collection, a computed value — are skipped rather than turned into a
  plausible-looking tool name, and the node is flagged `tool_resolution: "partial"`.
- *List-form conditional routing* — `add_conditional_edges(src, fn, ["a", "b"])` — names
  destinations without saying what the router returns, so those routes report
  `condition_value: null` with `value_resolution: "not_derivable"`. Destinations are still exact.
- *Entry points* come from `set_entry_point(...)`, or are derived from a single `add_edge(START, x)`,
  which LangGraph treats as the same thing. A graph with several parallel entry edges, or a
  conditional entry whose "entry" is a router rather than a node, reports `null` — a singular field
  cannot honestly describe either.
- *Line spans* for a node whose function could not be located point at its `add_node(...)`
  registration rather than a function body. For a resolved node the span starts at its first
  decorator, not the `def`.

**Ranking.** Semantic search is a ranked guess, not a structural fact. A partially-resolved node's
chunk is a single registration line, and its brevity can push a real function body from first to
second place; measured on real code this cost one rank position and never pushed the correct node
out of the top 3.

**Index integrity.** A damaged index fails loudly rather than returning wrong results, but the
message still carries SQLite's own wording rather than plainly saying the index is corrupt — and a
zero-length index file reports as "not indexed" instead of damaged, because SQLite cannot tell the
two apart. If a repository you just indexed insists it is not indexed, delete
`.langgraph-context/` and re-run `index`.

**Storage and privacy.** Indexed source is stored in plaintext in whichever backend you choose.
Point `DATABASE_URL` at a shared database and your node bodies and docstrings — including any
credential that was committed to source — go there too. There is no automatic secret redaction, and
this is a single-user local tool with no authentication or multi-tenancy.

## Design lineage

The planning pass for this project studied the existing "code intelligence for AI agents" servers —
notably **claude-context** and **codebase-memory-mcp** — along with academic work on statically
extracting agent graphs from LangGraph source. Two things were taken directly: from
codebase-memory-mcp, the choice of many narrow, verb-first, typed tools over one generic
`query(question)` tool, because tool selection then carries most of the intent instead of prompt
parsing; and from the academic work, the framing that a framework-aware extractor should recover
nodes, edges and conditional routing as first-class structure rather than as text. Where this
project diverges is parsing: those tools use tree-sitter because they support many languages, while
this one supports exactly one, so Python's own `ast` is sufficient and keeps the install free of a
compiled grammar. The **Status & limits** section above follows the format used by **GraphARC** —
dated, specific about what is actually verified, and treating limitations as first-class content
rather than a footnote — which remains the clearest presentation of this kind we found.

## License

MIT
