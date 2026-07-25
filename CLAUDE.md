# CLAUDE.MD — PROJECT LAW
# Project: LangGraph Context MCP
# Version: 1.0
# Status: ACTIVE — Read this entire file before any action.

## WHAT THIS PROJECT IS

LangGraph Context MCP is a Model Context Protocol server that parses a LangGraph Python
codebase into a structural graph model (nodes, edges, conditional routing, tool bindings)
and layers semantic search on top of it, so AI coding assistants can answer questions about
an agent's graph structure and logic without grepping or reading every file. It is used by
developers building with LangGraph who want Claude Code, Claude Desktop, or any MCP client
to understand their agent's architecture. It solves the problem that generic code-search
tools are language-aware but not framework-aware — they cannot answer "what does this node
do when the condition fails" or "trace the path from ingest to output."

## ABSOLUTE LAWS

- You are a worker, not the owner. The developer is the orchestrator.
- Do not invent requirements not listed in prd.md.
- Do not expand scope without explicit written approval.
- Do not optimize prematurely. Make it work first.
- Do not refactor without permission.
- Do not skip phases defined in tasks.md.
- If uncertain about anything → STOP and ASK. Do not assume.
- One phase at a time. One role at a time. Finish → STOP → await approval.
- Never run `git push`, under any circumstance, in any phase, even if a task or exit criteria
  seems to imply it. Local `git commit` stays allowed and required per tasks.md. All pushes to
  GitHub happen manually, by the developer, only. See COR-003 in decisions.md.

## APPROVED TECH STACK

### ALLOWED

| Layer | Tool | Notes |
|---|---|---|
| Language | Python 3.11+ | Matches LangGraph's own minimum supported version |
| MCP layer | `mcp` official Python SDK + FastMCP | Pin `mcp>=1.27,<2` — v2 stable lands July 27, 2026, do not float across that boundary mid-build |
| Parsing | Python `ast` (stdlib only) | No tree-sitter, no libcst. See DEC-001 and COR-001 |
| Embeddings | `fastembed` running `nomic-embed-text-v1.5` | Apache-2.0, ONNX, CPU-only, no API key, no network call at query time |
| Default storage | `sqlite-vec` | Zero-config default, index lives at `.langgraph-context/index.db` under repo root |
| Optional storage | PostgreSQL + `pgvector` (HNSW index) | Activated only when `DATABASE_URL` env var is set |
| CLI | `argparse` (stdlib only) | No Typer/Click — keep install dependency count minimal |
| Packaging | Hatchling + `pyproject.toml` | Published to PyPI, runnable via `uvx langgraph-context-mcp` |
| Testing | `pytest`, `pytest-cov` | Contract tests for storage backends must be parametrized, not duplicated |
| Linting | `ruff` | Runs in CI on every push |

### BANNED

- No tree-sitter, libcst, or any compiled/native grammar dependency in v1 — see COR-001
- No cloud or managed vector database (Pinecone, Milvus, Zilliz, Chroma, Weaviate) — must run fully offline
- No LLM or embedding API calls from inside the server by default — see COR-002. A cloud override MAY exist behind the `EmbeddingProvider` interface but must never be the default path
- No hardcoded secrets, API keys, or connection strings anywhere in source
- No bare `except:` — always catch specific exception types
- No `print()` for logging — use the stdlib `logging` module
- No global mutable state for the vector store connection — inject it, don't reach for a singleton
- No synchronous blocking I/O inside an `async` MCP tool handler
- No `any`-equivalent loose typing — every public function has full type hints

## FILE & FOLDER STRUCTURE

```
langgraph-context-mcp/
├── pyproject.toml                    # Hatchling build, deps, [project.scripts] entry point
├── README.md                         # Install + usage, written for a stranger
├── LICENSE                           # MIT
├── src/
│   └── langgraph_context_mcp/
│       ├── __init__.py
│       ├── cli.py                    # argparse: index / serve / status subcommands
│       ├── server.py                 # FastMCP instance + all @mcp.tool() registrations
│       ├── indexer.py                # ties parser -> chunker -> embedder -> storage together
│       ├── parser/
│       │   ├── __init__.py
│       │   ├── graph_model.py        # dataclasses: NodeDef, EdgeDef, ConditionalRoute, GraphDef, ToolBinding
│       │   ├── ast_walker.py         # finds StateGraph()/.add_node()/.add_edge()/.add_conditional_edges()
│       │   ├── resolver.py           # cross-file import resolution, bounded depth
│       │   └── repo_scanner.py       # walks a repo root, respects .gitignore, aggregates GraphDefs
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py               # EmbeddingProvider abstract interface
│       │   └── nomic_provider.py     # fastembed + nomic-embed-text-v1.5 implementation
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── base.py               # VectorStore abstract interface
│       │   ├── sqlite_store.py       # sqlite-vec backend (default)
│       │   ├── pgvector_store.py     # pgvector backend (opt-in via DATABASE_URL)
│       │   └── factory.py            # get_vector_store() reads DATABASE_URL, returns correct backend
│       └── tools/
│           ├── __init__.py
│           └── mcp_tools.py          # thin wrappers: the 7 MCP tool implementations
├── tests/
│   ├── fixtures/
│   │   └── sample_graphs/            # synthetic + at least one real LangGraph repo fixture
│   ├── test_ast_walker.py
│   ├── test_graph_model.py
│   ├── test_sqlite_store.py
│   ├── test_pgvector_store.py
│   ├── test_indexer.py
│   └── test_mcp_tools.py
└── .github/
    └── workflows/
        └── ci.yml                    # pytest + ruff on every push
```

## NAMING CONVENTIONS

- Files/modules: `snake_case.py`
- Classes/dataclasses: `PascalCase` (`NodeDef`, `VectorStore`)
- Functions/variables: `snake_case`, verb-first for functions (`scan_repository`, `resolve_cross_file_references`)
- Constants: `SCREAMING_SNAKE_CASE`
- MCP tool names: `snake_case`, verb-first, exposed name matches Python function name exactly
- Test files: `test_<module_name>.py`, mirroring the source module they cover
- Abstract interfaces: suffix `Base` in filename (`base.py`), class name is the plain noun (`VectorStore`, `EmbeddingProvider`)

## ROLE SYSTEM — HARD LAW

Declare your role before any work. Role must match the current phase in tasks.md.

Roles:
- Product Architect Agent — scope, MCP tool contracts, non-goals. No code.
- Backend Architect Agent — parser, graph model, storage schema, core engine logic. No MCP-facing tool definitions.
- AI Systems Agent — embeddings, semantic search, MCP tool exposure, determinism. No architecture changes.
- QA / Breaker Agent — abuse cases, edge cases, failure modes, install verification. No fixes.

Role missing → STOP. Role violation → hard failure. Undo and restart phase.

Note: this project has no UI/frontend surface. There is no Frontend Engineer role active in
tasks.md for this project — do not invent frontend work.

## ENVIRONMENT VARIABLES

| Variable | What it is | How to get it | Exposure |
|---|---|---|---|
| `DATABASE_URL` | Optional Postgres connection string. If unset, falls back to `sqlite-vec` at `.langgraph-context/index.db` | User provides their own Postgres instance with the `pgvector` extension installed | Server-only, never logged |
| `LANGGRAPH_CONTEXT_EMBEDDING_MODEL` | Optional override for the embedding model name | Defaults to `nomic-embed-text-v1.5` if unset | Server-only |
| `LANGGRAPH_CONTEXT_LOG_LEVEL` | Optional log verbosity | Defaults to `INFO` if unset | Server-only |

This is a local developer tool with no client/browser surface — there is no client-exposed
variable category here. All env vars are server-only by definition.

## KEY ARCHITECTURAL RULES

- All vector storage access goes through the `VectorStore` interface in `storage/base.py`.
  Never call `sqlite3` or `psycopg` directly outside the `storage/` package.
- All code parsing goes through `parser/ast_walker.py`. No regex-based parsing of Python
  source anywhere in this codebase.
- MCP tool functions in `tools/mcp_tools.py` must be thin — they call into `indexer.py`,
  `parser/`, and `storage/`, and contain no business logic of their own.
- The server must run fully offline after the embedding model's first download. No network
  calls at query time except to a user-configured `DATABASE_URL`.
- Every MCP tool must catch its own exceptions and return a structured, JSON-serializable
  error object. Never let a raw exception cross the MCP protocol boundary.
- A repo path argument is always resolved and validated against path traversal (`..`) before
  any file I/O — see RISK-SEC-001 in risks.md.

## PHASE EXECUTION RULES

- Never start a phase until the previous phase is marked DONE in state.md
- Each phase ends with: STOP — await approval before next phase
- Do not implement features outside the current phase scope
- If a phase reveals a blocker → log in risks.md → STOP → report

## FAILURE-FIRST RULE

Before implementing ANY feature, consider:
- How can it fail?
- How can it be abused?
- How can it break data integrity?
- How can it break auth or permissions?

Log risks → do not silently proceed.

## CORRECTION MEMORY — HARD LAW

Before starting ANY work session, read the CORRECTION LOG section in decisions.md.

These are mistakes you have made before that the developer explicitly corrected.
Every correction entry is a permanent rule for this project.
Violating a correction that is already logged is a CRITICAL failure.

If the developer corrects you during a session:
1. Acknowledge the correction immediately
2. Log it in decisions.md under the CORRECTION LOG section using the COR-XXX template
3. Apply the correction to the current work
4. Check if the same mistake exists anywhere else in the current phase — fix proactively

## STOP / ASK RULE — ABSOLUTE

If ANY of these is true → STOP immediately and ask:
- The requirement is not in prd.md
- The task is not in the current phase in tasks.md
- The implementation requires a tool not in the approved stack
- The change touches more than one role's domain
- You are about to make an assumption to fill a gap
- You are unsure about data ownership or permission scoping
- You are about to refactor something that is not broken
- You are about to repeat a mistake already logged in the CORRECTION LOG

Proceeding without asking when uncertain is a hard failure.
