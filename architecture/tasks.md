# TASKS.MD — PHASE EXECUTION PLAN
# Rule: Execute ONE phase at a time. STOP after each phase. Await approval.
# This project is deliberately compressed to 6 phases (developer constraint). Bootstrap,
# mid-build QA, and pre-launch QA are non-negotiable; the 3 build phases between them are
# scoped tightly so each is achievable in roughly one day of focused work.

## PHASE OVERVIEW

| Phase | Name | Role | Status |
|---|---|---|---|
| 0 | Bootstrap | Backend Architect Agent | NOT STARTED |
| 1 | Core Parser & Graph Model | Backend Architect Agent | NOT STARTED |
| 2 | Embeddings, Storage & Semantic Search | AI Systems Agent | NOT STARTED |
| 3 | QA / Breaker — Mid-Build | QA / Breaker Agent | NOT STARTED |
| 4 | MCP Server & Tool Exposure | AI Systems Agent | NOT STARTED |
| 5 | QA / Breaker — Pre-Launch | QA / Breaker Agent | NOT STARTED |

---

## PHASE 0 — BOOTSTRAP

**Role: Backend Architect Agent**

### Tasks
- [x] 0.1 Run `git init` in the project root `langgraph-context-mcp/`
- [x] 0.2 Check PyPI name availability for `langgraph-context-mcp` (visit `https://pypi.org/project/langgraph-context-mcp/`). If taken, fall back to `lg-context-mcp` and use that name consistently in every file from this point forward — log this as DEC in decisions.md if the fallback is used
- [x] 0.3 Create `pyproject.toml` using the Hatchling build backend:
  - `requires-python = ">=3.11"`
  - Core dependencies: `mcp>=1.27,<2`, `fastembed>=0.4`, `sqlite-vec>=0.1`
  - Optional extra `[project.optional-dependencies] pgvector = ["psycopg[binary]>=3.2", "pgvector>=0.3"]`
  - Optional extra `dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6"]`
  - `[project.scripts]` entry: `langgraph-context-mcp = "langgraph_context_mcp.cli:main"`
- [x] 0.4 Create the src layout exactly as specified in claude.md's File & Folder Structure section, including empty `__init__.py` in every package directory (`parser/`, `embeddings/`, `storage/`, `tools/`)
- [x] 0.5 Create `tests/fixtures/sample_graphs/simple_graph.py` — a minimal synthetic LangGraph example with at least 3 nodes, 1 normal edge, and 1 conditional edge, to be used by every later test phase
- [x] 0.6 Run `uv venv` then `uv pip install -e ".[dev,pgvector]"`
- [x] 0.7 Create `.gitignore` covering standard Python patterns plus `.langgraph-context/` (the local SQLite index directory)
- [x] 0.8 Create `README.md` with: project name, one-paragraph description, install command placeholder (`uvx langgraph-context-mcp`), and a "Status: In Development" banner
- [x] 0.9 Create `LICENSE` — MIT
- [x] 0.10 Create `.github/workflows/ci.yml` that runs `ruff check .` and `pytest` on every push to any branch

### Exit Criteria
- `pip install -e .` completes with zero errors
- `python -c "import langgraph_context_mcp"` runs with no output and exit code 0
- `pytest` runs and reports "no tests collected" cleanly, not a collection error
- `git log` shows at least one commit
- `pyproject.toml` contains no dependency outside the ALLOWED table in claude.md

### STOP — Await approval before Phase 1

---

## PHASE 1 — CORE PARSER & GRAPH MODEL

**Role: Backend Architect Agent**

### Tasks
- [ ] 1.1 Create `src/langgraph_context_mcp/parser/graph_model.py` defining frozen dataclasses `NodeDef`, `EdgeDef`, `ConditionalRoute`, `GraphDef`, `ToolBinding` with the exact fields listed in prd.md's Data Models section, plus a `to_dict()` method on each for JSON serialization
- [ ] 1.2 Create `src/langgraph_context_mcp/parser/ast_walker.py` with `find_graph_definitions(file_path: Path) -> list[GraphDef]`:
  - Detect `StateGraph(...)` instantiation via `ast.walk`
  - Detect `.add_node(name, func)` calls chained or called on the graph variable
  - Detect `.add_edge(source, target)` calls
  - Detect `.add_conditional_edges(source, condition_func, mapping)` calls, populating `ConditionalRoute` entries from the mapping dict literal
  - Detect `.set_entry_point(name)` calls
  - Every unresolvable pattern must set `resolution="partial"` on the affected `NodeDef`, never raise
- [ ] 1.3 In `ast_walker.py`, handle same-file node function resolution directly (function is defined in the file being walked)
- [ ] 1.4 Create `src/langgraph_context_mcp/parser/resolver.py` with `resolve_cross_file_references(graph_def: GraphDef, repo_root: Path) -> GraphDef`:
  - Follows `import` and `from ... import` statements up to 3 levels deep to locate node functions defined in other modules
  - Anything beyond depth 3, or an unresolvable dynamic pattern, is marked `resolution="partial"` and left in the result — never dropped silently
- [ ] 1.5 Create `src/langgraph_context_mcp/parser/repo_scanner.py` with `scan_repository(repo_root: Path) -> list[GraphDef]`:
  - Walks all `.py` files under `repo_root`, respecting `.gitignore` (use the `pathspec` library only if strictly needed — prefer a minimal manual `.gitignore` pattern match first and only add the dependency if that proves insufficient during this phase)
  - A file with a syntax error is skipped with a logged warning; scanning continues for the rest of the repo
- [ ] 1.6 Write `tests/test_ast_walker.py` covering: single-file simple graph, multi-file graph with cross-file node functions, conditional edges with 3+ branches, a node function built inside a loop (must return `resolution="partial"`, must not crash), and a file with a syntax error mixed into the repo (must be skipped, must not halt the scan)
- [ ] 1.7 Write `tests/test_graph_model.py` covering `to_dict()` output shape for every dataclass

### Exit Criteria
- `scan_repository()` correctly identifies all nodes, edges, and conditional edges in `tests/fixtures/sample_graphs/simple_graph.py`
- A malformed or dynamically-constructed graph produces a partial result with `resolution="partial"` on the affected nodes, never an exception
- `pytest tests/test_ast_walker.py tests/test_graph_model.py -v` passes with every written test green

### STOP — Await approval before Phase 2

---

## PHASE 2 — EMBEDDINGS, STORAGE & SEMANTIC SEARCH

**Role: AI Systems Agent**

### Tasks
- [ ] 2.1 Create `src/langgraph_context_mcp/embeddings/base.py` with an abstract `EmbeddingProvider` class: `embed(texts: list[str]) -> list[list[float]]`, property `dimension: int`, property `model_name: str`
- [ ] 2.2 Create `src/langgraph_context_mcp/embeddings/nomic_provider.py` implementing `EmbeddingProvider` using `fastembed` with `nomic-embed-text-v1.5`. Load the model lazily on first `embed()` call, not at import time
- [ ] 2.3 Create `src/langgraph_context_mcp/storage/base.py` with an abstract `VectorStore` class: `upsert_chunks(chunks: list[EmbeddingChunk]) -> None`, `search(query_vector: list[float], top_k: int, filters: dict) -> list[SearchResult]`, `get_graph(graph_id: str) -> GraphDef | None`, `delete_repo(repo_id: str) -> None`
- [ ] 2.4 Create `src/langgraph_context_mcp/storage/sqlite_store.py` implementing `VectorStore` using `sqlite-vec`. Default index path: `.langgraph-context/index.db` relative to the indexed repo root. Create this directory if it does not exist
- [ ] 2.5 Create `src/langgraph_context_mcp/storage/pgvector_store.py` implementing `VectorStore` using `psycopg` and `pgvector`. On first connection, create the required table and an HNSW index on the embedding column if they do not already exist
- [ ] 2.6 Create `src/langgraph_context_mcp/storage/factory.py` with `get_vector_store(repo_root: Path) -> VectorStore`: reads `DATABASE_URL` from the environment; returns `PgvectorStore` if set, otherwise `SqliteStore`
- [ ] 2.7 Create `src/langgraph_context_mcp/indexer.py` with `index_repository(repo_root: Path) -> IndexResult`:
  - Calls `scan_repository()`, then for each `NodeDef` builds one chunk of text = docstring + immediate decorators + function body
  - Calls the configured `EmbeddingProvider` to embed all chunks in a single batched call
  - Calls the configured `VectorStore.upsert_chunks()` to persist
  - Returns counts matching the `index_repo` MCP tool's output contract in prd.md
- [ ] 2.8 Write `tests/test_sqlite_store.py` and `tests/test_pgvector_store.py` as a **shared parametrized contract test suite** — the same test functions run against both backends via `pytest.mark.parametrize`, not duplicated logic. The pgvector tests must `pytest.skip()` cleanly if `DATABASE_URL` is not set locally, but must run in CI where a Postgres service container is available
- [ ] 2.9 Write `tests/test_indexer.py` covering the full pipeline against `tests/fixtures/sample_graphs/simple_graph.py`

### Exit Criteria
- `index_repository()` succeeds against the fixture repo using the default SQLite backend with zero configuration
- A semantic query for "authentication" against a fixture node named `check_auth_token` returns that node in the top 3 results
- The shared contract test suite passes against both SQLite and pgvector backends (pgvector skip is acceptable locally, must pass in CI)
- `pytest tests/ -v` passes fully with no failures

### STOP — Await approval before Phase 3

---

## PHASE 3 — QA / BREAKER — MID-BUILD

**Role: QA / Breaker Agent**

**Mandate: find and document failures. Fix nothing. Every finding goes in risks.md under
QA FINDINGS — PHASE 3 (MID BUILD) with a severity rating.**

### Test Scenarios (minimum 15 — this phase has 16)
- [ ] 3.1 Clone and index a real, unmodified open-source LangGraph repository end-to-end
- [ ] 3.2 Index a repository with zero LangGraph usage — expect a clean "no graphs found," not an error
- [ ] 3.3 Index a repo with one syntactically invalid `.py` file mixed in among valid files — expect that file skipped with a warning, scan continues
- [ ] 3.4 Index a graph where node functions are constructed inside a `for` loop — expect `resolution="partial"`, not a crash
- [ ] 3.5 Index a graph with 3+ levels of nested conditional edges — verify `trace_path` output is correct
- [ ] 3.6 Call `semantic_search_nodes` with an empty query string — expect a clean validation error
- [ ] 3.7 Call `trace_path` between two nodes with no route between them — expect an explicit `path_found: false` result
- [ ] 3.8 Call `trace_path` with a node name that does not exist — expect an error listing valid node names
- [ ] 3.9 Re-index the same repository twice in a row — expect no duplicate rows in storage (idempotent upsert keyed on node identity)
- [ ] 3.10 Index a synthetic graph with 200+ nodes — measure and record indexing time; flag if it exceeds 60 seconds
- [ ] 3.11 Run the same semantic query against SQLite and pgvector backends on identical fixture data — verify near-identical top-k ranking
- [ ] 3.12 Simulate a corrupted or partially-written SQLite index file — expect a clear error on next access, never silent wrong results
- [ ] 3.13 Point `DATABASE_URL` at an unreachable Postgres host — expect a clear connection error at startup, never a hang
- [ ] 3.14 Index a repo with circular imports between two modules — verify the resolver's depth limit prevents infinite recursion
- [ ] 3.15 Call every one of the 7 planned MCP tool functions directly (pre-Phase-4, test the underlying functions) with missing required arguments — expect clean validation errors, never stack traces
- [ ] 3.16 Confirm no network call is made during `semantic_search_nodes` after the embedding model has been cached locally once (fully offline operation)

### Exit Criteria
- All 16 scenarios executed and logged in risks.md QA FINDINGS — PHASE 3 table with a severity per finding
- Zero CRITICAL severity findings remain open — CRITICAL findings must be fixed with developer approval before Phase 4 begins, or explicitly moved to risks.md Deferred Risks with developer sign-off

### STOP — Await approval before Phase 4

---

## PHASE 4 — MCP SERVER & TOOL EXPOSURE

**Role: AI Systems Agent**

### Tasks
- [ ] 4.1 Create `src/langgraph_context_mcp/server.py` instantiating `FastMCP("langgraph-context-mcp")`
- [ ] 4.2 Implement `@mcp.tool() index_repo(path: str) -> dict` — thin wrapper around `indexer.index_repository`
- [ ] 4.3 Implement `@mcp.tool() get_graph_summary(path: str) -> dict`
- [ ] 4.4 Implement `@mcp.tool() semantic_search_nodes(query: str, path: str, top_k: int = 5) -> dict`
- [ ] 4.5 Implement `@mcp.tool() trace_path(from_node: str, to_node: str, path: str) -> dict`
- [ ] 4.6 Implement `@mcp.tool() what_calls_tool(tool_name: str, path: str) -> dict`
- [ ] 4.7 Implement `@mcp.tool() explain_conditional(edge_source: str, path: str) -> dict`
- [ ] 4.8 Implement `@mcp.tool() reindex(path: str) -> dict`
- [ ] 4.9 Wrap every tool handler body in a try/except that returns a structured `{error: str, ...}` dict instead of raising, per claude.md's architectural rule
- [ ] 4.10 Write clear, specific docstrings on every `@mcp.tool()` function — these docstrings are what the connected AI client reads to decide which tool to call, so ambiguity here directly causes wrong tool selection
- [ ] 4.11 Create `src/langgraph_context_mcp/cli.py` with `argparse` subcommands `index <path>`, `serve`, `status <path>`, matching the CLI contracts in prd.md
- [ ] 4.12 Wire the `[project.scripts]` entry point in `pyproject.toml` (created in Phase 0) to `langgraph_context_mcp.cli:main`
- [ ] 4.13 Write `tests/test_mcp_tools.py` testing all 7 tool functions directly at the function level, including the error cases specified in prd.md's API contracts
- [ ] 4.14 Manually verify: create a local `.mcp.json` pointing at `uvx --from . langgraph-context-mcp serve`, connect Claude Desktop, confirm `tools/list` shows all 7 tools

### Exit Criteria
- All 7 MCP tools are registered and individually callable via the MCP Inspector (`npx @modelcontextprotocol/inspector`)
- `uvx langgraph-context-mcp index .` works against the fixture repo from a clean virtual environment
- `uvx langgraph-context-mcp serve` starts with no stdout noise and responds correctly to a `tools/list` request
- `pytest tests/test_mcp_tools.py -v` passes

### STOP — Await approval before Phase 5

---

## PHASE 5 — QA / BREAKER — PRE-LAUNCH

**Role: QA / Breaker Agent**

**Mandate: find and document failures at the full-system level, plus verify launch readiness.
Fix nothing except CRITICAL blockers with explicit developer approval. Findings go in risks.md
under QA FINDINGS — PHASE 5 (PRE-LAUNCH).**

### Test Scenarios (minimum 15 — this phase has 16)
- [ ] 5.1 Fresh-machine install test: `pip install langgraph-context-mcp` (or `uvx`) in a brand-new virtualenv with nothing else installed
- [ ] 5.2 Perform the entire README walkthrough literally, step by step, exactly as written — flag any command that does not work as documented
- [ ] 5.3 Connect the server to Claude Desktop following only the README instructions, with no out-of-band knowledge
- [ ] 5.4 Ask Claude Code five realistic natural-language questions against a real indexed LangGraph repository — verify every answer is structurally correct
- [ ] 5.5 Verify `LICENSE` and `pyproject.toml` metadata (name, version, author, description) are correct and internally consistent
- [ ] 5.6 Grep the entire codebase for hardcoded secrets, API keys, or absolute local file paths — must find none
- [ ] 5.7 Run `pytest --cov=src --cov-report=term-missing` and confirm coverage on `parser/` and `storage/` is at or above 80%
- [ ] 5.8 Run `ruff check .` and confirm zero errors
- [ ] 5.9 Test indexing a directory that is not a git repository — expect graceful handling, not an assumption that `.git` exists
- [ ] 5.10 Test indexing a repo path containing spaces and non-ASCII characters
- [ ] 5.11 Verify all 8 non-goals in prd.md are genuinely NOT implemented anywhere in the codebase — explicit scope-creep check
- [ ] 5.12 Spot-check the final code against DEC-001 through DEC-005 in decisions.md — confirm each decision was actually followed
- [ ] 5.13 Confirm every active entry in the decisions.md CORRECTION LOG was followed throughout the build
- [ ] 5.14 If a large real-world LangGraph monorepo is available, index it and confirm no crash and a reasonable completion time
- [ ] 5.15 Ask 5 deliberately ambiguous natural-language questions through Claude Code and confirm it selects the correct MCP tool without hinting — validates that tool docstrings from Phase 4 are clear enough
- [ ] 5.16 Verify graceful shutdown of the MCP server on interrupt — no orphaned processes, no corrupted SQLite index file

### Final Summary Table
Document every scenario from Phase 3 and Phase 5 in one combined table: ID, Scenario, Severity
of any finding, Status (RESOLVED / DEFERRED / OPEN).

### LAUNCH DECISION
- **APPROVED** if: zero CRITICAL or HIGH severity findings remain OPEN, README walkthrough
  succeeds with zero undocumented steps, and package installs cleanly from a fresh environment
- **BLOCKED** if: any CRITICAL or HIGH finding is OPEN, or the fresh-install test fails

### Exit Criteria
- Zero CRITICAL or HIGH severity findings remain OPEN in risks.md
- README walkthrough completes successfully by a fresh reader with zero undocumented steps
- Package installs and runs correctly from a clean environment
- LAUNCH DECISION = APPROVED

### STOP — End of project plan. Ready for PyPI publish, MCP directory submission (glama.ai,
mcp.so, Smithery), and public launch (Show HN, r/LangChain, r/ClaudeAI).
