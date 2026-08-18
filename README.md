<div align="center">

# LangGraph Context MCP

**Ask your LangGraph codebase what it actually does — structurally, not by grepping.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-307-brightgreen.svg)](#status--limits)
[![CI](https://github.com/KarimHabib100/LangGraph-Context-MCP/actions/workflows/ci.yml/badge.svg)](https://github.com/KarimHabib100/LangGraph-Context-MCP/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-server-000000.svg)](https://modelcontextprotocol.io)

</div>

---

LangGraph Context MCP is a Model Context Protocol server that parses a LangGraph Python codebase
into a structural graph model — nodes, edges, conditional routing, tool bindings — and layers local
semantic search on top, so an AI coding assistant can answer questions about your agent's
architecture without reading every file.

Structural answers come from parsing the actual graph definition, so they are exact rather than
inferred. Everything runs locally: no API key, and no network access after the embedding model
downloads once.

```
"how does execution get from clarify_with_user to the final report?"

  clarify_with_user → write_research_brief → research_supervisor → final_report_generation
  (branch at clarify_with_user, routed by the node itself)
```

## Requirements

- **Python 3.11 or newer**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**, only if you use the `uv`
  install path below. `pip` works without it.
- Roughly **500 MB of disk** for the embedding model, downloaded once on first use.
- No API key. No database server. No network access at query time.

## Install

> **Not yet on PyPI.** The package name is registered as available but publication happens at
> launch, so the two commands below are the correct forms and will work once it is published.
> Until then, use the *from a local clone* option, which works today.

Install it as a standalone command-line tool (recommended — keeps it out of your project's
dependencies):

```bash
uv tool install langgraph-context-mcp
```

Or install it into a Python environment with pip:

```bash
pip install langgraph-context-mcp
```

**From a local clone** (works today, before publication):

```bash
git clone https://github.com/KarimHabib100/LangGraph-Context-MCP.git
cd LangGraph-Context-MCP
uv tool install .          # or: pip install .
```

### Find where it was installed

You will need the **absolute path** to the installed command for the MCP client configuration
below, because desktop clients do not launch it through your shell. After installing:

```bash
# macOS / Linux
which langgraph-context-mcp

# Windows (PowerShell)
Get-Command langgraph-context-mcp | Select-Object -ExpandProperty Source
```

`uv tool install` places it in `~/.local/bin` (Windows: `C:\Users\<you>\.local\bin`). uv will warn
you if that directory is not on your `PATH`; `uv tool update-shell` adds it, and the tool still
works by absolute path either way.

## Quick start

```bash
cd my-langgraph-project
langgraph-context-mcp index .
```

That scans the repository, embeds every graph node locally, and writes the index to
`.langgraph-context/index.db`. Then point an MCP client at it — see
[Connecting an MCP client](#connecting-an-mcp-client).

The first run also downloads the embedding model, which takes a few minutes. Later runs are fast:
a mid-sized repository indexes in roughly 20 seconds.

## CLI

Three subcommands. Index a repository once, then serve it to an MCP client.

```bash
langgraph-context-mcp index <path>     # scan, embed, and store the index
langgraph-context-mcp serve            # run the MCP server on stdio
langgraph-context-mcp status <path>    # report whether a path has an index
```

### `index <path>`

Scans `<path>` for LangGraph graph definitions, builds one embedding per graph node, and writes the
index to `<path>/.langgraph-context/index.db` (or to PostgreSQL — see
[Storage backends](#storage-backends)). Re-running replaces that repository's previous index rather
than appending to it.

### `serve`

Starts the MCP server on the stdio transport and waits for a client. It writes **nothing** to
stdout — that stream carries the MCP protocol — and logs to stderr. Runs until the client
disconnects or the process is interrupted.

### `status <path>`

Reports whether `<path>` has an index, when it was last built, how much it contains, and which
backend holds it. Read-only: it never creates an index.

### `--json`

`index` and `status` accept `--json`, which prints the same structured result as machine-readable
JSON instead of the human-readable summary. Exit codes are unaffected, so a script can read the code
and the payload together. `serve` has no `--json` flag — extra output there would corrupt the MCP
transport.

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

`1` is not an error. It exists so a script can tell "I asked, and the answer is no" apart from "I
could not ask". A caller that does not need the distinction can test for `>= 2`:

```bash
langgraph-context-mcp index . || [ $? -lt 2 ] || exit 1   # fail only on a real error
```

`serve` has no negative-answer case, so it returns only `0` (clean shutdown) or `2` (could not
start — for example `DATABASE_URL` is set but PostgreSQL is unreachable).

Paths containing `..` are rejected with exit `2`, as is an empty path. Pass an absolute path, or
`cd` into the repository and use `.`.

## Connecting an MCP client

> **Use an absolute path to the command.** Desktop MCP clients spawn servers directly, without a
> login shell, so they do not see the `PATH` your terminal sees. A bare `"command":
> "langgraph-context-mcp"` or `"command": "uvx"` fails with `spawn ... ENOENT` /
> `FileNotFoundError [WinError 2]` unless that directory happens to be on the *system* `PATH`.
> Every example below uses an absolute path for that reason. Get yours with the
> [`which` / `Get-Command`](#find-where-it-was-installed) commands above.

Two forms work. The examples use the first:

| Form | Use it when | Trade-off |
|---|---|---|
| **Installed console script** (absolute path) | Default. You ran `uv tool install` or `pip install` | Fastest startup — the server is already installed. You update it deliberately |
| **`uvx` by absolute path** | You would rather not install anything permanently | No install step, and always the latest published version, but every launch resolves the environment first, so startup is slower. Requires the absolute path to `uvx` itself, not bare `uvx` |

An absolute path is machine-specific, which is the cost of this approach: a config file committed to
a shared repository will need each contributor to adjust the path, or to keep it in their own
user-level client config instead.

Index a repository first (`langgraph-context-mcp index /abs/path/to/repo`); the tools report
`not_indexed` until you do, and the client can also call `index_repo` itself.

### Claude Code

A project-level `.mcp.json` in your repository root:

```json
{
  "mcpServers": {
    "langgraph-context": {
      "command": "/Users/you/.local/bin/langgraph-context-mcp",
      "args": ["serve"]
    }
  }
}
```

On Windows, use the full executable path with escaped backslashes:

```json
{
  "mcpServers": {
    "langgraph-context": {
      "command": "C:\\Users\\you\\.local\\bin\\langgraph-context-mcp.exe",
      "args": ["serve"]
    }
  }
}
```

### Claude Desktop

Edit `claude_desktop_config.json` (Settings → Developer → Edit Config), using the same shape:

```json
{
  "mcpServers": {
    "langgraph-context": {
      "command": "/Users/you/.local/bin/langgraph-context-mcp",
      "args": ["serve"]
    }
  }
}
```

Restart Claude Desktop, then confirm the server appears with its seven tools.

<details>
<summary>Alternative: <code>uvx</code> without installing</summary>

Use the absolute path to `uvx`, not a bare `uvx`:

```json
{
  "mcpServers": {
    "langgraph-context": {
      "command": "/Users/you/.local/bin/uvx",
      "args": ["langgraph-context-mcp", "serve"]
    }
  }
}
```

</details>

### Cursor

Cursor uses the same `mcpServers` schema. Put it in `.cursor/mcp.json` for a single project, or
`~/.cursor/mcp.json` to make it available everywhere:

```json
{
  "mcpServers": {
    "langgraph-context": {
      "command": "/Users/you/.local/bin/langgraph-context-mcp",
      "args": ["serve"]
    }
  }
}
```

**MCP support is behind a settings toggle that can default to off.** If the server never appears,
open Settings → MCP (or Settings → Tools & Integrations, depending on version), confirm MCP is
enabled, and check that `langgraph-context` is toggled on in the server list.

### Codex CLI / ChatGPT desktop

> **Caveat, stated because the rest of this section is not hedged:** the Codex syntax below was
> confirmed against current OpenAI documentation but **not run on the machine this README was
> written on**, because the `codex` CLI is not installed there. Every other client configuration in
> this README was actually spawned and verified. The launch command itself is the same verified
> absolute-path form; what is unverified here is Codex's own `mcp add` flags, TOML shape, and config
> file locations.

Codex uses TOML, not JSON. The supported path is the CLI, which writes the config for you:

```bash
codex mcp add langgraph-context -- /Users/you/.local/bin/langgraph-context-mcp serve
```

Everything after `--` is the launch command. Then start a session and run `/mcp` to confirm the
server connected and is listing its tools.

This writes to `~/.codex/config.toml`, which the Codex CLI, the IDE extension, and the ChatGPT
desktop app all share. Trusted projects may also use a project-scoped `.codex/config.toml`. The
resulting block looks like this, if you prefer to write or review it by hand:

```toml
[mcp_servers.langgraph-context]
command = "/Users/you/.local/bin/langgraph-context-mcp"
args = ["serve"]
```

**Without touching a config file:** in the ChatGPT desktop app, go to Settings → MCP servers → Add
server, give it a name, choose **STDIO**, enter the same absolute command, and restart.

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

Two of these deliberately distinguish "there is nothing" from "we could not tell", because
conflating them is how a static analyser ends up stating something false:

- **`explain_conditional`** returns `not_conditional` when a node's body was read and genuinely has
  no conditional edge — but `routing_not_resolvable` when the node has no *declared* conditional
  edge and its routing could not be enumerated. That happens when the node's function could not be
  located, or when it routes with `Command(goto=...)` whose destination is computed at runtime
  rather than written as a literal. The second result means "this node may well branch, and we
  cannot say where" — never treat it as "this node does not branch".
- **`what_calls_tool`** returns `unenumerated_tool_nodes` alongside `callers`, listing nodes that
  bind tools which could not be read statically, so an empty `callers` list is never mistaken for
  "nothing uses this tool".

`trace_path` follows the same rule: when it finds no route, it also returns
`unresolved_routing_nodes`, so "no declared path" is distinguishable from "definitely not
connected".

## Storage backends

By default the index is a single SQLite file at `<repo>/.langgraph-context/index.db` — no server, no
configuration. Set `DATABASE_URL` to a PostgreSQL connection string with the `pgvector` extension
installed to use that instead; the tables and the HNSW index are created automatically on first
connection.

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | Use PostgreSQL + `pgvector` instead of SQLite | unset (SQLite) |
| `LANGGRAPH_CONTEXT_EMBEDDING_MODEL` | Override the embedding model | `nomic-embed-text-v1.5` |
| `LANGGRAPH_CONTEXT_LOG_LEVEL` | Log verbosity (stderr) | `INFO` |

Embeddings run locally on CPU. After the model downloads once on first use, the tool makes no
network calls at all — apart from your own `DATABASE_URL`, if you set one.

## Status & limits

*As of 2026-08-18. Pre-launch: the engine and the MCP surface are built and tested; publication to
PyPI has not happened yet.*

### What has actually been verified

Measured against real, unmodified open-source LangGraph repositories, not only synthetic fixtures:

- **Parses real code.** `langchain-ai/open_deep_research` (@ `1b7d2e8`) yields 7 graphs, 23 nodes
  and 39 edges, 3 partially resolved, with no crash. A 450-file monorepo
  (`langchain-ai/langgraph` @ `644815f`) yields 125 graphs, 468 nodes and 568 edges in ~56s.
- **Indexes inside the target.** That first repository indexes end to end in ~20s on CPU, including
  the one-time model load — against a design target of 30s.
- **Retrieval holds up adversarially.** On five queries deliberately phrased so a common verb
  lexically matches the *wrong* node's name, the correct node was in the top 3 every time and
  ranked first in 4 of 5.
- **Both backends agree.** SQLite and PostgreSQL + `pgvector` return the same ranking order on
  identical data — zero ordering differences, worst score delta ~6e-07 — because both are pinned to
  cosine on unit-length vectors rather than left on their differing defaults.
- **Genuinely offline.** With outbound sockets blocked at the OS level, a full query completes with
  zero connection attempts once the model is cached.
- **Re-indexing is idempotent.** Repeated runs leave exactly one repository row and one graph row
  per graph — no duplicates, and rows for deleted nodes are dropped.
- **Survives interruption.** Killing an index mid-run leaves an index that passes SQLite's
  `integrity_check` with no half-written rows; `status` reports it honestly and re-indexing recovers.
- **Works with real MCP clients.** All seven tools list and execute under the MCP Inspector, the
  `mcp` SDK's own stdio client, and Claude Desktop.
- **Test suite:** 307 tests. 282 pass on a default machine with 25 `pgvector` tests skipped; all 307
  run in CI against PostgreSQL 16 + `pgvector`, where 305 pass and 2 platform-specific tests skip.

Not yet exercised: Cursor's and Codex's bundled clients (Claude Desktop is verified), and
installation from PyPI, which does not exist until launch.

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

- *Routing declared in a node's body* — `return Command(goto="next")` — is parsed, including
  branching across several `if` arms, the `END` sentinel, and `Send(...)` fan-out. A `goto` whose
  destination is a **computed value** cannot be, so that node yields no edge for it and is reported
  as `routing_not_resolvable` rather than as having no routing. Destinations declared only in a
  `Command[Literal[...]]` return annotation are deliberately not treated as edges, since an
  annotation states intent rather than a route actually taken.
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
two apart. If a repository you just indexed insists it is not indexed, delete `.langgraph-context/`
and re-run `index`.

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
