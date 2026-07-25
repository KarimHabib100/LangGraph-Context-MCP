# STATE.MD — PROJECT STATE TRACKER
# Rule: Update this file at the END of every phase. This is the source of truth.

## CURRENT STATUS
Phase:         Phase 0 — Bootstrap
Status:        DONE
Last Updated:  2026-07-25
Active Role:   Backend Architect Agent

## PHASE CHECKLIST

| Phase | Name | Status | Completed Date |
|---|---|---|---|
| 0 | Bootstrap | DONE | 2026-07-25 |
| 1 | Core Parser & Graph Model | NOT STARTED | |
| 2 | Embeddings, Storage & Semantic Search | NOT STARTED | |
| 3 | QA / Breaker — Mid-Build | NOT STARTED | |
| 4 | MCP Server & Tool Exposure | NOT STARTED | |
| 5 | QA / Breaker — Pre-Launch | NOT STARTED | |

## ENVIRONMENT VARIABLES

| Variable | Local (YES/NO) | Production (YES/NO) | Notes |
|---|---|---|---|
| `DATABASE_URL` | NO | N/A — local tool | Optional. Unset = SQLite default. Set to a Postgres connection string with `pgvector` installed to use that backend instead |
| `LANGGRAPH_CONTEXT_EMBEDDING_MODEL` | NO | N/A | Optional override, defaults to `nomic-embed-text-v1.5` |
| `LANGGRAPH_CONTEXT_LOG_LEVEL` | NO | N/A | Optional, defaults to `INFO` |

## DEPENDENCIES

| Package | Installed (YES/NO) | Version | Notes |
|---|---|---|---|
| `mcp` (with FastMCP) | YES | 1.28.1 | Pin `>=1.27,<2` — see RISK-005 |
| `fastembed` | YES | 0.8.0 | Runs `nomic-embed-text-v1.5` locally |
| `sqlite-vec` | YES | 0.1.9 | Default storage backend |
| `psycopg[binary]` | YES | 3.3.4 | Optional extra `[pgvector]` |
| `pgvector` | YES | 0.5.0 | Optional extra `[pgvector]` |
| `pytest` | YES | 9.1.1 | Dev extra |
| `pytest-cov` | YES | 7.1.0 | Dev extra |
| `ruff` | YES | 0.16.0 | Dev extra |

Note: `uv` (build tool, not a project dependency) was not present on this machine and was
installed via `pip install --user uv` (v0.11.32) with the developer's explicit approval before
task 0.6 could run.

## KNOWN BLOCKERS

| ID | Description | Blocking Phase | Raised Date | Resolution |
|---|---|---|---|---|

## NEXT ACTION
Action:   Begin Phase 1 — Core Parser & Graph Model (define dataclasses, then build the ast-based walker)
Command:  Create `src/langgraph_context_mcp/parser/graph_model.py`
Role:     Backend Architect Agent
Phase:    Phase 1, Task 1.1

## DECISIONS SUMMARY

| ID | Decision | Phase | Status |
|---|---|---|---|
| DEC-001 | Use stdlib `ast` instead of tree-sitter/libcst for parsing | Phase 1 | ACTIVE |
| DEC-002 | Dual storage: `sqlite-vec` default, `pgvector` opt-in via `DATABASE_URL` | Phase 2 | ACTIVE |
| DEC-003 | Local `nomic-embed-text-v1.5` embeddings by default, no cloud API call | Phase 2 | ACTIVE |
| DEC-004 | Seven narrow typed MCP tools instead of one generic query tool | Phase 4 | ACTIVE |
| DEC-005 | One embedding chunk per graph node, not fixed-token chunking | Phase 2 | ACTIVE |

## CORRECTION LOG STATUS

| ID | Correction | Phase Logged | Status |
|---|---|---|---|
| COR-001 | No tree-sitter or compiled parsing dependency — stdlib `ast` only | Pre-build | ACTIVE |
| COR-002 | No external LLM/embedding API call by default — local `fastembed` only | Pre-build | ACTIVE |
| COR-003 | Never run `git push`, any phase, any circumstance — local `git commit` only, developer pushes manually | Phase 0 | ACTIVE |

## RISK STATUS SUMMARY

| ID | Description | Severity | Status |
|---|---|---|---|
| RISK-001 | Dynamically constructed nodes cannot be statically resolved | HIGH | OPEN |
| RISK-002 | Cross-file resolution may fail on complex import patterns | MEDIUM | OPEN |
| RISK-003 | Local embedding model may rank trivial nodes poorly | MEDIUM | OPEN |
| RISK-004 | SQLite and pgvector backends could diverge in ranking behavior | HIGH | OPEN |
| RISK-005 | MCP SDK v2 stable release lands during build window | MEDIUM | OPEN |
| RISK-006 | Package name collision on PyPI | LOW | OPEN |
| RISK-SEC-001 | Path traversal outside intended repo root | MEDIUM | OPEN |
| RISK-SEC-002 | Proprietary code stored in plaintext if pointed at shared Postgres | LOW | OPEN |
| RISK-SEC-003 | Accidental credentials embedded in indexed source | LOW | OPEN |

## CHANGE LOG

| Date | Phase | Change Description | By |
|---|---|---|---|
| Pre-build | — | Project files initialized (claude.md, prd.md, tasks.md, decisions.md, risks.md, state.md, prompts.md, agents/) | Planning session |
| 2026-07-25 | Phase 0 | Task 0.1: `git init` run in project root | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.2: Verified `langgraph-context-mcp` unregistered on PyPI (HTTP 404 on JSON API) — no fallback name needed. RISK-006 updated to MITIGATED | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.3: Created `pyproject.toml` (Hatchling backend, core deps `mcp>=1.27,<2`/`fastembed>=0.4`/`sqlite-vec>=0.1`, optional extras `pgvector`/`dev`, `[project.scripts]` entry point) | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.4: Scaffolded `src/langgraph_context_mcp/` src layout exactly per claude.md (parser/, embeddings/, storage/, tools/ packages with empty `__init__.py` and empty module stub files) — no logic written, scaffolding only | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.5: Created `tests/fixtures/sample_graphs/simple_graph.py` — 4 nodes (incl. `check_auth_token` for Phase 2's semantic-search exit criterion), 1 conditional edge with 2 destinations, 3 normal edges | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Ordering note: task 0.6's `uv pip install -e .` failed because hatchling validates `pyproject.toml`'s `readme = "README.md"` field exists at build time. Created README.md (task 0.8's exact deliverable) out of numeric order to unblock 0.6, since it is a mechanical dependency, not a scope change. Task 0.8 marked done at this point rather than later | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.8: Created `README.md` (project name, description, `uvx langgraph-context-mcp` install placeholder, "Status: In Development" banner) | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.6: Developer approved installing `uv` (not present on machine) via `pip install --user uv`. Ran `uv venv` (Python 3.12.10) and `uv pip install -e ".[dev,pgvector]"` — 65 packages resolved and installed with zero errors. Verified `python -c "import langgraph_context_mcp"` produces no output, exit code 0 | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.7: Created `.gitignore` (standard Python patterns + `.langgraph-context/`) | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.9: Created `LICENSE` (MIT) | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Task 0.10: Created `.github/workflows/ci.yml` — runs `ruff check .` and `pytest` on push to any branch, Python 3.11 | Backend Architect Agent |
| 2026-07-25 | Phase 0 | Phase 0 exit criteria verified all PASS. Initial commit `f73dcbf` created (35 files). Phase 0 marked DONE | Backend Architect Agent |
| 2026-07-25 | Post-Phase 0 | Developer request: repo-wide name audit found one stray reference (`pyproject.toml` author name "Yassin Hassan Habib") corrected to "Karim Habib". Added COR-003 (never `git push`) to decisions.md, claude.md Absolute Laws, and this file's Correction Log Status. Not yet committed | Backend Architect Agent |
