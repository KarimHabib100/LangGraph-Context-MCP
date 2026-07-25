# LangGraph Context MCP — Claude Code Prompts
# Usage: Send the CONTEXT PRIMER first in every new conversation.
#        Then send the relevant PHASE PROMPT immediately after.
#        Do not combine them into one message.

---

## SECTION 1: CONTEXT PRIMER

You are starting a work session on the LangGraph Context MCP project — a Model Context
Protocol server that parses LangGraph Python codebases into a structural graph model and
layers local semantic search on top, exposed as MCP tools for AI coding assistants.

Before doing anything else, read all 7 project files in this exact order:
1. CLAUDE.md     — project law, stack rules, naming conventions, absolute constraints
2. prd.md        — product requirements, CLI/tool contracts, data models, success criteria
3. tasks.md      — phase execution plan, find current phase and its tasks
4. decisions.md  — architectural decisions already made, do not re-open these
5. risks.md      — known risks and their mitigations
6. state.md      — current project state, what is done, what env vars are set, next action
7. prompts.md    — you are reading from this file right now

Then read the agent file for your assigned role:
8. agents/[your-role].md — your role-specific rules, patterns, boundaries, and quality standards

Read every word. Do not skim. The files contain constraints that will affect every action you
take, including the six-phase compression, the "no tree-sitter" rule, and the "no cloud API by
default" rule — these are easy to violate by defaulting to common patterns from outside this
project's context.

CRITICAL — CORRECTION LOG:
After reading decisions.md, pay special attention to the CORRECTION LOG section.
Every COR-XXX entry is a mistake that was made before and corrected by the developer.
These are PERMANENT rules. Violating any of them is a CRITICAL failure.
List all active corrections in your status report below.

After reading all 7 files, output a status report in this exact format:

---
PROJECT: LangGraph Context MCP
CURRENT PHASE: [Phase number and name from state.md]
PHASE STATUS: [Status from state.md]
ACTIVE ROLE: [Role from state.md]
LAST COMPLETED PHASE: [Most recent DONE phase, or "None"]
OPEN BLOCKERS: [Any items in the Known Blockers table, or "None"]
CRITICAL OPEN RISKS: [Count of CRITICAL status risks from risks.md]
ACTIVE CORRECTIONS: [List all COR-XXX IDs and their one-line rules, or "None"]
NEXT ACTION: [Exact Next Action from state.md]
ENV VARS CONFIGURED: [Count configured / total]
DEPENDENCIES INSTALLED: [Count installed / total]
---

Then stop. Do not start any work. Wait for the phase prompt.

FILE UPDATE PROTOCOL — memorize this before receiving the phase prompt:

Throughout every session, you must update these files as work progresses:

tasks.md:
  — Mark each task DONE by changing `- [ ]` to `- [x]` as you complete it
  — Do NOT mark a task done until it is fully implemented and verified
  — Mark phase status as DONE only when ALL exit criteria are confirmed

state.md:
  — Update CURRENT STATUS block whenever phase or status changes
  — Update Last Updated date
  — Mark dependencies as Installed=YES with their version when installed
  — Mark env vars as YES when confirmed present when applicable
  — Update Known Blockers table immediately when a blocker is found
  — Update NEXT ACTION block whenever the next action changes
  — Update the Change Log with a row for every significant change made

decisions.md:
  — Add a new DEC-00X entry BEFORE implementing any significant architectural choice
  — A choice is significant if it affects more than one file or picks between two approaches
  — Never modify existing entries. Append only.
  — If the developer corrects you: IMMEDIATELY add a COR-XXX entry to the CORRECTION LOG

risks.md:
  — Add a new RISK-XXX entry whenever you discover a failure mode not already listed
  — Update the Status field of a risk from OPEN to MITIGATED when its mitigation is implemented
  — Fill QA FINDINGS tables during QA phases

These file updates are not optional. They are part of completing a task, not separate from it.

---

## SECTION 2: PHASE PROMPTS

---

# ════════════════════════════════════════════════════════════
# PHASE 0 — Bootstrap
# ════════════════════════════════════════════════════════════

ROLE DECLARATION: You are now the Backend Architect Agent.
Read agents/backend-architect.md before proceeding.

CORRECTION CHECK: Review all active COR-XXX entries in decisions.md before starting.
COR-001 (no tree-sitter) and COR-002 (no cloud embedding API) are not yet relevant to
bootstrap work, but confirm no dependency you add in this phase violates either.

SCOPE: Only the tasks listed under PHASE 0 in tasks.md. Nothing else. Do not write any parser,
embedding, storage, or MCP tool code in this phase — this phase is scaffolding only.

EXECUTION ORDER:
Work through tasks 0.1 through 0.10 sequentially. Do not skip tasks. Do not reorder them.
Each task must be fully complete before starting the next.

After each task is complete:
  — Change its checkbox in tasks.md from [ ] to [x]
  — Add a row to the Change Log in state.md describing what was created/changed

PHASE-SPECIFIC RULES:
- Verify the PyPI name (task 0.2) before writing pyproject.toml's `name` field — do not
  guess and fix later.
- pyproject.toml dependencies must exactly match the ALLOWED table in claude.md. Do not add
  Typer, Click, requests, or any convenience library "just in case" — this project uses
  argparse and stdlib wherever possible by design.
- The `[project.optional-dependencies]` extras (`pgvector`, `dev`) must be separate extras,
  not bundled into the core install — a user who only wants the SQLite default should not be
  forced to install `psycopg`.
- `tests/fixtures/sample_graphs/simple_graph.py` (task 0.5) is load-bearing for every later
  phase's tests. Give it at least 3 nodes, 1 normal edge, and 1 conditional edge with at least
  2 possible destinations — a graph with only 1 node or no conditional edge is not sufficient
  fixture coverage for Phase 1 and Phase 2 tests.

SPECIFIC FILE UPDATES FOR THIS PHASE:

tasks.md:
  — Mark tasks 0.1–0.10 as [x] when each is complete

state.md:
  — Update CURRENT STATUS → Phase: "Phase 0 — Bootstrap", Status: "IN PROGRESS", then "DONE"
  — Mark each dependency in the Dependencies table as Installed=YES with its resolved version
    once `uv pip install -e ".[dev,pgvector]"` succeeds
  — Update NEXT ACTION to point at Phase 1, Task 1.1 once this phase is DONE

decisions.md:
  — If the PyPI name fallback (`lg-context-mcp`) is used instead of `langgraph-context-mcp`,
    log this as a new DEC entry explaining why, since it changes the package name referenced
    throughout every other file
  — No other new decisions expected in this phase

risks.md:
  — Validate RISK-006 (package name collision) — update its status once task 0.2 completes

MANUAL TEST (task 0.6):
Run: `python -c "import langgraph_context_mcp"`
Expected: no output, exit code 0

EXIT CRITERIA VERIFICATION:
Before marking Phase 0 DONE, verify every item:

PHASE 0 EXIT CRITERIA CHECK:
[ ] `pip install -e .` completes with zero errors — [PASS/FAIL]
[ ] `python -c "import langgraph_context_mcp"` succeeds — [PASS/FAIL]
[ ] `pytest` runs and reports no collection errors — [PASS/FAIL]
[ ] `git log` shows at least one commit — [PASS/FAIL]
[ ] `pyproject.toml` contains no dependency outside claude.md's ALLOWED table — [PASS/FAIL]

If any item is FAIL, fix it before declaring the phase done.

STOP. After Phase 0 exit criteria all pass, output:
"Phase 0 complete. state.md updated. Awaiting approval to begin Phase 1."
Do not begin Phase 1.

---

# ════════════════════════════════════════════════════════════
# PHASE 1 — Core Parser & Graph Model
# ════════════════════════════════════════════════════════════

ROLE DECLARATION: You are now the Backend Architect Agent.
Read agents/backend-architect.md before proceeding.

CORRECTION CHECK: Review all active COR-XXX entries in decisions.md before starting.
COR-001 is directly relevant to this entire phase: use only the stdlib `ast` module. If any
task in this phase feels like it needs tree-sitter or libcst, STOP and ask instead of adding
the dependency.

PREREQUISITE CHECK: Confirm Phase 0 is marked DONE in tasks.md and state.md before starting.
If Phase 0 is not DONE — STOP and report: "Phase 0 is not marked complete. Cannot begin Phase 1."

SCOPE: Only the tasks listed under PHASE 1 in tasks.md. Nothing else. Do not touch embeddings,
storage, or MCP tools in this phase — those are Phase 2 and Phase 4.

EXECUTION ORDER:
Work through tasks 1.1 through 1.7 sequentially. Do not skip tasks. Do not reorder them.
Build `graph_model.py` before `ast_walker.py` — the walker returns instances of the dataclasses
defined in the model, so the model must exist first.

PHASE-SPECIFIC RULES:
- Every dataclass in graph_model.py must be frozen (immutable) and must implement `to_dict()`
  returning plain Python primitives only — no custom objects — since this output eventually
  crosses the MCP JSON boundary in Phase 4.
- The parser must NEVER raise an unhandled exception on malformed input. Every failure mode —
  a syntax error in a scanned file, a node function that cannot be resolved, a conditional
  edge with a malformed mapping — must degrade to a partial result with `resolution="partial"`
  or a skipped-file warning, never a crash. This is the single most important rule of this
  phase because Phase 3's QA scenarios are specifically designed to try to break this.
  See RISK-001 and RISK-002.
- The cross-file resolver's depth limit (3 levels) is a hard cap, not a suggestion — do not
  implement unbounded recursion "to be more thorough." Bounded, predictable, partial results
  beat unbounded recursion that could hang on a circular import.
- `repo_scanner.py` must respect `.gitignore`. Start with a minimal manual pattern match
  (most repos only ignore `.venv/`, `__pycache__/`, `node_modules/` — handle the common cases
  directly) rather than immediately reaching for the `pathspec` dependency. Only add
  `pathspec` if you hit a real case the manual approach cannot handle, and log that as a new
  DEC entry if you do.

SPECIFIC FILE UPDATES FOR THIS PHASE:

tasks.md:
  — Mark tasks 1.1–1.7 as [x] when each is complete

state.md:
  — Update CURRENT STATUS → Phase: "Phase 1 — Core Parser & Graph Model", Status: "IN PROGRESS"
  — When phase complete: mark Phase 1 DONE with date, update NEXT ACTION to Phase 2 Task 2.1

decisions.md:
  — No new decisions expected. If you hit an unlisted choice (e.g. whether to add `pathspec`),
    log it before proceeding rather than deciding silently.

risks.md:
  — Update RISK-001 and RISK-002 status once their mitigations (partial resolution, depth cap)
    are implemented and covered by a passing test

MANUAL TEST (task 1.6):
Run: `pytest tests/test_ast_walker.py tests/test_graph_model.py -v`
Expected: every written test passes, including the malformed-input and partial-resolution cases

EXIT CRITERIA VERIFICATION:
Before marking Phase 1 DONE, verify every item:

PHASE 1 EXIT CRITERIA CHECK:
[ ] `scan_repository()` correctly identifies all nodes/edges/conditional edges in the fixture — [PASS/FAIL]
[ ] Malformed/dynamic graph input produces `resolution="partial"`, never an exception — [PASS/FAIL]
[ ] `pytest tests/test_ast_walker.py tests/test_graph_model.py -v` passes fully — [PASS/FAIL]

If any item is FAIL, fix it before declaring the phase done.

STOP. After Phase 1 exit criteria all pass, output:
"Phase 1 complete. state.md updated. Awaiting approval to begin Phase 2."
Do not begin Phase 2.

---

# ════════════════════════════════════════════════════════════
# PHASE 2 — Embeddings, Storage & Semantic Search
# ════════════════════════════════════════════════════════════

ROLE DECLARATION: You are now the AI Systems Agent.
Read agents/ai-systems.md before proceeding.

CORRECTION CHECK: Review all active COR-XXX entries in decisions.md before starting.
COR-002 is directly relevant to this entire phase: the default `EmbeddingProvider` must call
`fastembed` locally. Do not add an OpenAI, Voyage, or any cloud embedding call as the default
path, even temporarily "to get something working faster."

PREREQUISITE CHECK: Confirm Phase 1 is marked DONE in tasks.md and state.md before starting.
If Phase 1 is not DONE — STOP and report: "Phase 1 is not marked complete. Cannot begin Phase 2."

SCOPE: Only the tasks listed under PHASE 2 in tasks.md. Nothing else. Do not touch the MCP
server or CLI in this phase — that is Phase 4.

EXECUTION ORDER:
Work through tasks 2.1 through 2.9 sequentially. Build the two abstract interfaces
(`EmbeddingProvider` in 2.1, `VectorStore` in 2.3) before either concrete implementation —
every concrete class must satisfy its interface, not the other way around.

PHASE-SPECIFIC RULES:
- Per DEC-005, chunking is one chunk per `NodeDef`: docstring + immediate decorators + full
  function body, concatenated. Do not implement generic fixed-token chunking — the whole
  point of this decision is reusing Phase 1's parser output as the chunk boundary.
- Per DEC-002, both storage backends must sit behind the same `VectorStore` interface and be
  covered by the same parametrized contract test (task 2.8) — do not write two separate test
  files with duplicated logic that could silently drift apart.
- Normalize both backends to cosine similarity explicitly per RISK-004's mitigation. Do not
  assume `sqlite-vec`'s default distance metric matches `pgvector`'s HNSW default — verify and
  configure both explicitly.
- The SQLite index path is `.langgraph-context/index.db` relative to the repo root being
  indexed, not relative to this tool's own installation directory. Create the directory if it
  does not exist.
- `nomic_provider.py` must load the ONNX model lazily on first `embed()` call, not at module
  import time — importing `langgraph_context_mcp` must stay fast and side-effect-free.
- The pgvector backend must create its table and HNSW index automatically on first connection
  if they do not already exist — do not require the user to run manual SQL migrations for a
  v1 tool this small.

SPECIFIC FILE UPDATES FOR THIS PHASE:

tasks.md:
  — Mark tasks 2.1–2.9 as [x] when each is complete

state.md:
  — Update CURRENT STATUS → Phase: "Phase 2 — Embeddings, Storage & Semantic Search", Status: "IN PROGRESS"
  — Mark `fastembed`, `sqlite-vec`, `psycopg[binary]`, `pgvector` as Installed=YES with version
  — When phase complete: mark Phase 2 DONE with date, update NEXT ACTION to Phase 3

decisions.md:
  — No new decisions expected beyond DEC-002, DEC-003, DEC-005 which are already logged and
    must be followed exactly as written, not reinterpreted

risks.md:
  — Update RISK-003 and RISK-004 status once their mitigations are implemented and tested
  — If pgvector tests must skip locally due to no `DATABASE_URL`, note this in state.md Known
    Blockers as informational, not a real blocker, and confirm the CI workflow provides a
    Postgres service container so these tests do run somewhere before Phase 5

MANUAL TEST (task 2.9):
Run: `pytest tests/ -v`
Expected: all tests pass; a semantic query for "authentication" against the fixture's
`check_auth_token` node returns it in the top 3 results

EXIT CRITERIA VERIFICATION:
Before marking Phase 2 DONE, verify every item:

PHASE 2 EXIT CRITERIA CHECK:
[ ] `index_repository()` succeeds against the fixture with zero configuration (SQLite) — [PASS/FAIL]
[ ] Semantic query for "authentication" returns `check_auth_token` in top 3 — [PASS/FAIL]
[ ] Shared contract test suite passes on both backends (pgvector may skip locally, must run in CI) — [PASS/FAIL]
[ ] `pytest tests/ -v` passes fully — [PASS/FAIL]

If any item is FAIL, fix it before declaring the phase done.

STOP. After Phase 2 exit criteria all pass, output:
"Phase 2 complete. state.md updated. Awaiting approval to begin Phase 3."
Do not begin Phase 3.

---

# ════════════════════════════════════════════════════════════
# PHASE 3 — QA / Breaker — Mid-Build
# ════════════════════════════════════════════════════════════

ROLE DECLARATION: You are now the QA / Breaker Agent.
Read agents/qa-breaker.md before proceeding.

CORRECTION CHECK: Review all active COR-XXX entries in decisions.md before starting.
Your job this phase is to actively try to trigger violations of COR-001 and COR-002's
guarantees (no crash on malformed input, no network call at query time) — treat them as
things to attack, not just things to remember.

PREREQUISITE CHECK: Confirm Phase 2 is marked DONE in tasks.md and state.md before starting.
If Phase 2 is not DONE — STOP and report: "Phase 2 is not marked complete. Cannot begin Phase 3."

SCOPE: Only the 16 test scenarios listed under PHASE 3 in tasks.md. Your mandate is to find
and document failures. You do NOT fix anything in this phase, including trivial-looking
fixes — every finding goes into risks.md and waits for explicit developer approval to address.

EXECUTION ORDER:
Work through scenarios 3.1 through 3.16 sequentially. For each scenario, use this exact output
format:

```
SCENARIO 3.X: [name]
Action: [exactly what you did]
Expected: [what should happen per prd.md / claude.md / risks.md mitigations]
Actual: [what actually happened]
Result: [PASS / FAIL]
Finding: [if FAIL, a specific description suitable for risks.md; if PASS, "None"]
```

PHASE-SPECIFIC RULES:
- This phase tests the parser and storage layers built in Phase 1 and Phase 2. The MCP tool
  layer does not exist yet (that's Phase 4) — scenario 3.15 tests the underlying Python
  functions directly, not through the MCP protocol.
- Scenario 3.1 requires a real, unmodified open-source LangGraph repository, not another
  synthetic fixture — clone one from GitHub for this test specifically.
- Scenario 3.10's 200+ node synthetic graph should be generated programmatically for this test,
  not hand-written — it does not need to be a meaningful graph, just structurally valid at
  that scale.
- Severity assignment guide for this project: CRITICAL = crashes, data loss, or a wrong answer
  presented with false confidence (no partial/error signal). HIGH = a documented guarantee is
  violated (e.g. a network call happens when it shouldn't, RISK-004's backend parity breaks).
  MEDIUM = a rough edge that degrades quality but doesn't violate a stated guarantee. LOW =
  cosmetic or purely informational.

SPECIFIC FILE UPDATES FOR THIS PHASE:

tasks.md:
  — Mark tasks 3.1–3.16 as [x] once each scenario has been executed and documented (executed,
    not necessarily passed — a documented FAIL still means the task is complete)

state.md:
  — Update CURRENT STATUS → Phase: "Phase 3 — QA / Breaker — Mid-Build", Status: "IN PROGRESS"
  — When phase complete: mark Phase 3 DONE with date, update NEXT ACTION to Phase 4

decisions.md:
  — Do not log new architectural decisions in this phase — that is not this role's job.
    If a finding implies a decision is needed, flag it in your findings and let the developer
    decide before Phase 4 begins.

risks.md:
  — Add every finding to the QA FINDINGS — PHASE 3 (MID BUILD) table with ID, Finding,
    Severity, Status (OPEN)
  — Do not change Status to MITIGATED yourself — that happens only after an approved fix

EXIT CRITERIA VERIFICATION:
Before marking Phase 3 DONE, verify every item:

PHASE 3 EXIT CRITERIA CHECK:
[ ] All 16 scenarios executed and logged in risks.md with a severity — [PASS/FAIL]
[ ] Zero CRITICAL findings remain open (fixed with approval, or explicitly deferred with sign-off) — [PASS/FAIL]

If any item is FAIL, do not proceed — report the open CRITICAL findings and wait for developer
direction.

FINAL SUMMARY — PHASE 3:
Output a table: Scenario ID | Result | Severity (if FAIL) | Finding ID in risks.md (if FAIL)

STOP. After Phase 3 exit criteria all pass, output:
"Phase 3 complete. state.md updated. Awaiting approval to begin Phase 4."
Do not begin Phase 4.

---

# ════════════════════════════════════════════════════════════
# PHASE 4 — MCP Server & Tool Exposure
# ════════════════════════════════════════════════════════════

ROLE DECLARATION: You are now the AI Systems Agent.
Read agents/ai-systems.md before proceeding.

CORRECTION CHECK: Review all active COR-XXX entries in decisions.md before starting.
Confirm any CRITICAL findings from Phase 3 that were fixed are actually reflected in the
current code before building the MCP layer on top of it.

PREREQUISITE CHECK: Confirm Phase 3 is marked DONE in tasks.md and state.md before starting.
If Phase 3 is not DONE — STOP and report: "Phase 3 is not marked complete. Cannot begin Phase 4."

SCOPE: Only the tasks listed under PHASE 4 in tasks.md. Nothing else. This phase exposes the
engine built in Phases 1–2 as MCP tools and a CLI — it does not modify parser, embedding, or
storage logic itself. If you find a bug in those layers while wiring the tools, log it in
risks.md and ask before fixing it, do not silently patch it mid-phase.

EXECUTION ORDER:
Work through tasks 4.1 through 4.14 sequentially. Build `server.py`'s FastMCP instance (4.1)
before any individual tool (4.2–4.8). Write the docstrings (4.10) as part of writing each tool,
not as a separate pass at the end.

PHASE-SPECIFIC RULES:
- Per claude.md's architectural rule and DEC-004: every `@mcp.tool()` function must be thin —
  it validates its arguments, calls into `indexer.py`/`parser/`/`storage/`, and formats the
  result. It must not contain parsing, embedding, or storage logic inline.
- Per claude.md: every tool handler must catch its own exceptions and return a structured
  `{error: ..., ...}` dict. An unhandled exception crossing the MCP boundary is a CRITICAL
  failure in this phase, not a minor bug — the MCP protocol has no good way to surface a raw
  Python traceback to the connected client.
- Tool docstrings (task 4.10) are the actual product surface for tool selection — a connected
  LLM reads these to decide which of the 7 tools to call. Write them the way you would write
  a well-documented API, not a one-line comment. Include what the tool does NOT do where that
  disambiguates it from a neighboring tool (e.g. `trace_path` finds a route between two named
  nodes; it does not answer "what does this node do" — that's `semantic_search_nodes` or
  `get_graph_summary`).
- The CLI (`cli.py`, task 4.11) and the MCP tools should call the same underlying functions in
  `indexer.py` — do not duplicate the indexing logic between the CLI's `index` subcommand and
  the MCP `index_repo` tool.
- `serve` must produce zero stdout output on success per the MCP stdio transport convention —
  any print statement here can corrupt the protocol stream. Use `logging` configured to stderr
  only, per claude.md's banned-`print()` rule.

SPECIFIC FILE UPDATES FOR THIS PHASE:

tasks.md:
  — Mark tasks 4.1–4.14 as [x] when each is complete

state.md:
  — Update CURRENT STATUS → Phase: "Phase 4 — MCP Server & Tool Exposure", Status: "IN PROGRESS"
  — When phase complete: mark Phase 4 DONE with date, update NEXT ACTION to Phase 5

decisions.md:
  — No new decisions expected — DEC-004 already specifies the 7-tool surface. If you find a
    genuine need for an 8th tool while building, STOP and ask rather than silently adding one.

risks.md:
  — Any bug found in Phase 1/2 logic while wiring tools gets a new RISK-XXX entry, not a
    silent fix

MANUAL TEST (task 4.14):
Run: `npx @modelcontextprotocol/inspector uvx --from . langgraph-context-mcp serve`
Expected: MCP Inspector connects and lists all 7 tools with their docstrings visible

EXIT CRITERIA VERIFICATION:
Before marking Phase 4 DONE, verify every item:

PHASE 4 EXIT CRITERIA CHECK:
[ ] All 7 tools registered and individually callable via MCP Inspector — [PASS/FAIL]
[ ] `uvx langgraph-context-mcp index .` works from a clean environment — [PASS/FAIL]
[ ] `uvx langgraph-context-mcp serve` starts with no stdout noise, responds to `tools/list` — [PASS/FAIL]
[ ] `pytest tests/test_mcp_tools.py -v` passes — [PASS/FAIL]

If any item is FAIL, fix it before declaring the phase done.

STOP. After Phase 4 exit criteria all pass, output:
"Phase 4 complete. state.md updated. Awaiting approval to begin Phase 5."
Do not begin Phase 5.

---

# ════════════════════════════════════════════════════════════
# PHASE 5 — QA / Breaker — Pre-Launch
# ════════════════════════════════════════════════════════════

ROLE DECLARATION: You are now the QA / Breaker Agent.
Read agents/qa-breaker.md before proceeding.

CORRECTION CHECK: Review all active COR-XXX entries in decisions.md before starting.
This is the last checkpoint before this project is presented as a finished, launchable tool —
verify both corrections hold at the full-system level, not just in unit tests.

PREREQUISITE CHECK: Confirm Phase 4 is marked DONE in tasks.md and state.md before starting.
If Phase 4 is not DONE — STOP and report: "Phase 4 is not marked complete. Cannot begin Phase 5."

SCOPE: Only the 16 test scenarios listed under PHASE 5 in tasks.md, plus the launch-readiness
checks. Your mandate is to find and document failures, and separately to verify the project is
genuinely ready for a stranger to install and use. Fix nothing except a CRITICAL blocker with
explicit developer approval.

EXECUTION ORDER:
Work through scenarios 5.1 through 5.16 sequentially, using the same output format as Phase 3:

```
SCENARIO 5.X: [name]
Action: [exactly what you did]
Expected: [what should happen]
Actual: [what actually happened]
Result: [PASS / FAIL]
Finding: [if FAIL, a specific description suitable for risks.md; if PASS, "None"]
```

PHASE-SPECIFIC RULES:
- Scenario 5.1 (fresh-machine install) and 5.2 (literal README walkthrough) must be performed
  exactly as a first-time stranger would experience them — do not use any knowledge of the
  codebase's internals to work around a broken or missing README step; if a step is unclear or
  missing, that is itself a FAIL finding.
- Scenario 5.11 (non-goals scope-creep check) requires reading prd.md's Non-Goals list and
  actively searching the codebase for violations — e.g. grep for any LLM API client library,
  any multi-framework abstraction beyond LangGraph, any file-watcher dependency. Finding one of
  these is a FAIL even if it "seemed like a good idea" during a build phase.
- Scenario 5.15 (ambiguous natural-language tool selection) is the real test of whether Phase
  4's docstring work succeeded — if Claude Code picks the wrong tool or hesitates, that is a
  FAIL on the docstring, not on this phase's testing.
- CRITICAL SCENARIOS — give these the most scrutiny: 5.1, 5.3, 5.6, 5.16. A failure in a fresh
  install, in Claude Desktop connection, in a leaked secret, or in a corrupted index on
  shutdown is disqualifying for launch regardless of how well everything else performs.

SPECIFIC FILE UPDATES FOR THIS PHASE:

tasks.md:
  — Mark tasks 5.1–5.16 as [x] once each scenario has been executed and documented

state.md:
  — Update CURRENT STATUS → Phase: "Phase 5 — QA / Breaker — Pre-Launch", Status: "IN PROGRESS"
  — When phase complete and APPROVED: mark Phase 5 DONE with date, update NEXT ACTION to
    "Publish to PyPI and submit to MCP directories"

decisions.md:
  — No new decisions in this phase

risks.md:
  — Add every finding to QA FINDINGS — PHASE 5 (PRE-LAUNCH)
  — Update the Final Summary Table (per tasks.md Phase 5) combining Phase 3 and Phase 5
    findings with their final Status (RESOLVED / DEFERRED / OPEN)

FINAL SUMMARY — PHASE 5:
Output the combined Phase 3 + Phase 5 findings table (ID, Scenario, Severity, Status), plus
severity counts (CRITICAL: N, HIGH: N, MEDIUM: N, LOW: N).

LAUNCH DECISION:
Output exactly one of:
- "LAUNCH DECISION: APPROVED — zero CRITICAL/HIGH findings open, fresh install verified,
  README walkthrough clean."
- "LAUNCH DECISION: BLOCKED — [list the specific open CRITICAL/HIGH findings by ID]."

EXIT CRITERIA VERIFICATION:
Before marking Phase 5 DONE, verify every item:

PHASE 5 EXIT CRITERIA CHECK:
[ ] Zero CRITICAL or HIGH severity findings remain OPEN — [PASS/FAIL]
[ ] README walkthrough completes with zero undocumented steps — [PASS/FAIL]
[ ] Package installs and runs correctly from a clean environment — [PASS/FAIL]
[ ] LAUNCH DECISION = APPROVED — [PASS/FAIL]

If any item is FAIL, the project is not done — report blockers and wait for developer direction.

STOP. This is the final phase. After Phase 5 exit criteria all pass and LAUNCH DECISION is
APPROVED, output:
"Phase 5 complete. Project ready for launch. state.md updated."
Do not take any further action — publishing to PyPI, submitting to MCP directories, and public
launch posts are the developer's decision to trigger, not this session's.
