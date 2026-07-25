# RISKS.MD — RISK REGISTER
# Severity: CRITICAL | HIGH | MEDIUM | LOW
# Rule: Log risks before they happen. Fixes go in decisions.md, not here.

## INITIAL ASSUMPTIONS

| ID | Assumption | Must Validate By | Status |
|---|---|---|---|
| ASM-001 | Most real-world LangGraph codebases construct their `StateGraph` via static, top-level calls, not deep runtime metaprogramming | Phase 3 (mid-build QA against real repos) | UNVALIDATED |
| ASM-002 | `nomic-embed-text-v1.5` retrieval quality is sufficient for short code + docstring chunks to rank usefully | Phase 3 | UNVALIDATED |
| ASM-003 | `sqlite-vec` brute-force search performs acceptably at typical repo scale (hundreds to low-thousands of nodes) | Phase 3 | UNVALIDATED |
| ASM-004 | LangGraph's public API (`add_node`, `add_edge`, `add_conditional_edges`, `set_entry_point`) remains stable through the build window with no breaking upstream change | Ongoing, re-check at Phase 0 and Phase 5 | UNVALIDATED |

## TECHNICAL RISKS

**RISK-001 — Dynamically constructed node functions cannot be statically resolved**
- Severity: HIGH
- Phase introduced: Phase 1
- Description: `ast`-based parsing cannot resolve node functions built inside loops, returned
  from factory functions, or assigned via closures/lambdas at runtime. This is a real pattern
  in more advanced LangGraph codebases.
- Mitigation: Detect these cases explicitly and mark the affected `NodeDef.resolution` as
  `"partial"` in the output rather than silently dropping the node or raising an exception.
  Document this as a known v1 limitation in the README.
- Status: OPEN

**RISK-002 — Cross-file resolution may fail on complex import patterns**
- Severity: MEDIUM
- Phase introduced: Phase 1
- Description: Relative imports, `__init__.py` re-exports, and aliased imports can defeat a
  naive import-following resolver.
- Mitigation: Bound resolution depth to 3 import levels; fall back to `resolution="partial"`
  rather than infinite recursion, a crash, or a silently wrong answer.
- Status: OPEN

**RISK-003 — Local embedding model may rank trivial/boilerplate nodes poorly**
- Severity: MEDIUM
- Phase introduced: Phase 2
- Description: Very short or generic function bodies (e.g. simple passthrough nodes) may not
  carry enough signal for `nomic-embed-text-v1.5` to rank them accurately against a natural
  language query.
- Mitigation: Always include the node's docstring and immediate decorator context in the
  embedded chunk, not just the raw function body, to give the embedding more signal to work
  with.
- Status: OPEN

**RISK-004 — SQLite and pgvector backends could silently diverge in ranking behavior**
- Severity: HIGH
- Phase introduced: Phase 2
- Description: `sqlite-vec`'s brute-force exact search and `pgvector`'s HNSW approximate search
  use different underlying mechanics; without an explicit shared contract, the two backends
  could return meaningfully different top-k results for the same query, breaking the promise
  that either backend "just works."
- Mitigation: Normalize both backends to cosine similarity explicitly. Task 2.8 requires a
  shared parametrized contract test that both backends must pass against identical fixture
  data with near-identical top-k results.
- Status: OPEN

**RISK-005 — MCP Python SDK v2 stable release lands during the build window**
- Severity: MEDIUM
- Phase introduced: Phase 0
- Description: The `mcp` SDK's v2 stable release is targeted for July 27, 2026, which falls
  inside this project's build window. An unplanned mid-build upgrade could break
  `@mcp.tool()` registration syntax or transport behavior.
- Mitigation: Pin `mcp>=1.27,<2` explicitly in `pyproject.toml` at Phase 0. Do not upgrade to
  v2 until a deliberate, separate v1.1 milestone evaluates the migration guide.
- Status: OPEN

**RISK-006 — Package name collision on PyPI**
- Severity: LOW
- Phase introduced: Phase 0
- Description: `langgraph-context-mcp` may already be registered on PyPI by another project.
- Mitigation: Verify availability during Phase 0 task 0.2 before writing any publish-related
  config. Fallback name `lg-context-mcp` is pre-approved if needed.
- Status: MITIGATED — verified 2026-07-25, `https://pypi.org/pypi/langgraph-context-mcp/json`
  returns HTTP 404 (unregistered). Name `langgraph-context-mcp` confirmed available, no
  fallback needed.

## AUTH & SECURITY RISKS

**RISK-SEC-001 — Path traversal outside the intended repo root**
- Severity: MEDIUM
- Description: The tool reads arbitrary source files from the filesystem based on a `path`
  argument passed through an MCP tool call. A malformed or malicious `path` value could
  attempt to read outside the intended repo root.
- Mitigation: Every `path` argument is resolved to an absolute path and validated to reject
  `..` traversal before any file I/O occurs.
- Status: OPEN

**RISK-SEC-002 — Proprietary source code stored in plaintext if pointed at a shared Postgres instance**
- Severity: LOW
- Description: If a user sets `DATABASE_URL` to a shared or production Postgres instance,
  indexed source code (including proprietary business logic in docstrings and function bodies)
  is stored there in plaintext.
- Mitigation: Document this clearly and prominently in the README. The tool never transmits
  anything over the network beyond the local SQLite file or the user's own explicitly
  configured `DATABASE_URL` — no telemetry, no external calls.
- Status: OPEN

**RISK-SEC-003 — Accidental credentials embedded in indexed source**
- Severity: LOW
- Description: No secrets are typically present in source code embeddings, but a node's
  docstring or inline comment could accidentally contain a hardcoded credential that then
  gets stored in the vector index alongside everything else.
- Mitigation: Document this as a user responsibility, identical to the same risk in any
  code-indexing tool. Automatic secret redaction is explicitly out of scope for v1 (see
  DEF-003) to avoid creating a false sense of security with a half-built scanner.
- Status: OPEN

## QA FINDINGS — PHASE 3 (MID BUILD)

| ID | Finding | Severity | Status |
|---|---|---|---|

## QA FINDINGS — PHASE 5 (PRE-LAUNCH)

| ID | Finding | Severity | Status |
|---|---|---|---|

## DEFERRED RISKS

| ID | Description | Reason Deferred |
|---|---|---|
| DEF-001 | Automatic file-watching / live incremental re-index on save | Explicitly out of v1 scope per prd.md non-goals; manual `reindex` is acceptable for initial release. Candidate for v1.1 |
| DEF-002 | Support for agent frameworks other than LangGraph (CrewAI, AutoGen, Google ADK) | v1 focuses narrowly on LangGraph to fit the 6-phase build window; multi-framework support is a v2 direction |
| DEF-003 | Automatic secret redaction on indexed content | Needs its own design pass; shipping a partial/unreliable scanner in v1 would create a false sense of security, which is worse than not having one |
