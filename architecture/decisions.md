# DECISIONS.MD — ARCHITECTURAL REASONING LOG & CORRECTION MEMORY
# Rule: Log significant decisions BEFORE implementation. Past entries are immutable.
#       To change direction, add a new decision that supersedes the old one.
#       Correction entries are PERMANENT. They represent developer-enforced rules.

## HOW TO USE THIS FILE

Before implementing any significant architectural choice (anything that affects more than one
file, or that picks between two real approaches), add a new `DEC-00X` entry using the template
below BEFORE writing the code. Never edit a past entry — if direction changes, add a new entry
that says so and references the one it supersedes.

Template:
```
## DEC-00X — [Short Title]

Date: [fill in when building]
Phase: Phase X
Status: ACTIVE

### Context
[What problem or constraint led to this decision?]

### Options Considered
| Option | Pros | Cons |
|--------|------|------|

### Decision
[What was chosen and why]

### Consequences
[What this locks in. What it makes harder.]
```

---

## DEC-001 — Python `ast` (stdlib) instead of tree-sitter or libcst for parsing

Date: Pre-build
Phase: Phase 1
Status: ACTIVE

### Context
Every comparable "code intelligence for AI agents" MCP server reviewed during planning
(codebase-memory-mcp, claude-context, semantic-code-mcp, CodeGrok MCP) uses tree-sitter for
parsing, because they all support many languages. This project supports exactly one language
(Python) and exactly one framework (LangGraph), so the multi-language justification for
tree-sitter does not apply here.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| tree-sitter | Industry standard for this category, robust, incremental parsing | Native binary/compiled grammar dependency, heavier install, solves a multi-language problem this project does not have |
| libcst | Preserves exact formatting, good for codemods | Heavier dependency, we do not modify code so format preservation is unused |
| stdlib `ast` | Zero dependency, ships with Python, sufficient for structural extraction of call chains | Python-only (already an accepted v1 boundary), no format preservation (not needed — read-only tool) |

### Decision
Use the stdlib `ast` module exclusively for all parsing.

### Consequences
Locks this project to Python-only in v1, which is already a stated non-goal boundary, not a
new limitation. In exchange, install is a pure-Python `pip install` with no compiled grammar
download step — a genuine differentiator against every competitor reviewed, all of which
require a tree-sitter grammar bundle.

---

## DEC-002 — Dual storage backend: sqlite-vec default, pgvector opt-in

Date: Pre-build
Phase: Phase 2
Status: ACTIVE

### Context
Every competing tool's adoption friction, visible in their own README setup sections, comes
from requiring either a cloud API key or a running database server before the tool can be
tried. Separately, the developer's own production stack for other projects already standardizes
on PostgreSQL + pgvector, and larger LangGraph monorepos will outgrow SQLite's write
concurrency.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| pgvector only | Matches developer's existing production skill and infrastructure | Requires Postgres setup before a first-time user can try the tool at all — the single biggest adoption blocker seen across every competitor reviewed |
| sqlite-vec only | Simplest possible install, single file, zero config | Does not scale to large monorepos, does not exercise pgvector/HNSW skill in the portfolio |
| Dual backend behind a `VectorStore` interface | Zero-config default removes the adoption blocker; pgvector remains available for scale and for the portfolio story | More code surface, requires a shared contract test suite to keep both backends behaving identically |

### Decision
Implement both behind a common `VectorStore` interface. `sqlite-vec` is the default with zero
configuration. `pgvector` with an HNSW index activates only when `DATABASE_URL` is set.

### Consequences
Requires the `storage/base.py` interface to be genuinely backend-agnostic from the start, and
requires a shared parametrized contract test (Phase 2, task 2.8) rather than two independent
test suites that could silently drift apart in behavior.

---

## DEC-003 — No LLM or cloud embedding API calls inside the server by default

Date: Pre-build
Phase: Phase 2
Status: ACTIVE

### Context
Several competing tools reviewed require an OpenAI or Voyage API key purely to generate
embeddings, which shows up as a recurring friction point in their own documentation and issue
trackers. This project's node-level chunks are short (a single function body plus docstring),
which does not require a frontier-scale embedding model to rank well.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| Cloud embedding API (OpenAI/Voyage) | Highest raw retrieval quality | Requires a paid API key, adds a network dependency, adds a per-index cost, breaks the "try it in one command" promise |
| Local ONNX model (`nomic-embed-text-v1.5` via `fastembed`) | Free, fully offline, Apache-2.0 licensed, ~274MB, runs on CPU with no GPU requirement | Slightly lower ceiling on retrieval quality than the best cloud models |
| Hybrid — local default, cloud as an opt-in override | Best of both | Requires the `EmbeddingProvider` interface to be designed for swapping from day one |

### Decision
Default to `nomic-embed-text-v1.5` via `fastembed`, running entirely locally on CPU. The
`EmbeddingProvider` interface allows a future cloud override without touching any caller code,
but no cloud provider ships in v1.

### Consequences
v1 ships fully free and fully offline after the one-time model download, which is a direct,
verifiable claim for the README and for the Anthropic Ecosystem Impact application. The
retrieval quality ceiling is slightly below top cloud models, judged acceptable given how short
and structurally clean the per-node chunks are (see DEC-005).

---

## DEC-004 — Seven narrow, typed MCP tools instead of one generic query tool

Date: Pre-build
Phase: Phase 4
Status: ACTIVE

### Context
Reviewed MCP servers split into two patterns: a single generic `query(question: str)` tool
that requires the connected LLM to infer structured intent from free text every call, or many
narrow, verb-first tools with explicit typed arguments. The structural-graph tools reviewed
(codebase-memory-mcp) use the many-narrow-tools pattern and report more reliable agent
behavior as a result, because tool selection carries most of the intent instead of prompt
parsing.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| Single generic `query(question)` tool | Small API surface, simple to implement | Connected LLM must infer structured intent from free text every time, less reliable, harder to test deterministically |
| Many narrow typed tools | Precise, matches the pattern used by the most successful comparable tool, easy to unit test each in isolation | Larger surface to design and document; each tool's docstring quality directly determines whether the LLM picks it correctly |

### Decision
Ship exactly 7 narrow, verb-first tools as specified in prd.md: `index_repo`,
`get_graph_summary`, `semantic_search_nodes`, `trace_path`, `what_calls_tool`,
`explain_conditional`, `reindex`.

### Consequences
Locks the initial MCP tool surface. Adding an 8th tool post-launch is low-risk; removing or
renaming one of the 7 after adoption is a breaking change for anyone with it configured, so
Phase 4's docstring quality (task 4.10) is treated as launch-critical, not cosmetic.

---

## DEC-005 — One embedding chunk per graph node, not fixed-token or whole-file chunking

Date: Pre-build
Phase: Phase 2
Status: ACTIVE

### Context
Generic RAG systems chunk by token count or by generic AST boundaries because they don't know
anything about the document's structure. This project already has the LangGraph parser from
Phase 1, which means the "correct" semantic unit — a single graph node's function — is known
for free before the embedding step ever runs.

### Options Considered
| Option | Pros | Cons |
|--------|------|------|
| Fixed-token chunking | Standard, simple, well-understood | Cuts function bodies mid-statement, has no relationship to the actual graph structure |
| Whole-file chunking | Simple | Wastes tokens and dilutes relevance with unrelated functions in the same file |
| One chunk per graph node (docstring + decorators + function body) | Chunk boundary equals semantic boundary equals graph node, reuses Phase 1's parser output directly | Very long node functions may eventually need secondary splitting — deferred, not solved in v1 |

### Decision
Each `EmbeddingChunk` corresponds to exactly one `NodeDef`: its docstring, immediate
decorators, and full function body, concatenated.

### Consequences
Directly couples the correctness of the embedding layer (Phase 2) to the correctness of the
parser (Phase 1) — this is why Phase 1 has its own exit criteria and must be solid before
Phase 2 begins, and why the phase ordering in tasks.md is not arbitrary.

---

## [Next decision — copy the template above]

---

# ════════════════════════════════════════════════════════════
# CORRECTION LOG — AI LEARNING MEMORY
# ════════════════════════════════════════════════════════════

# PURPOSE: When the developer corrects the AI during a build session, log it here.
# These corrections are PERMANENT project rules. The AI must read this section at the
# start of every work session and treat each entry as a hard constraint.
#
# WHEN TO LOG:
# - The developer says "don't do X" or "always do Y instead"
# - The developer undoes something the AI did and explains why
# - The developer corrects a pattern, approach, tool usage, or style choice
# - The AI used a banned tool, wrong pattern, or incorrect assumption
#
# RULES:
# - Corrections are append-only. Never modify or delete a correction.
# - Each correction becomes a permanent rule for this project.
# - Before starting any task, scan this log for relevant corrections.
# - Violating a logged correction is a CRITICAL failure.

## COR-001 — Do not reach for tree-sitter or any compiled parsing dependency

Date: Pre-build
Phase: Pre-build
Triggered By: Preventive — common AI default given the surrounding ecosystem

### What Happened
Nearly every reference implementation of "code intelligence MCP server" that an AI coding
tool has likely seen in training data or would find while searching for guidance uses
tree-sitter, because most of those tools support many languages. An AI assistant working on
this project is likely to default to adding tree-sitter the moment parsing gets even slightly
difficult (e.g. handling decorators or nested calls), because that is the dominant pattern in
the surrounding literature.

### Developer Correction
This project supports exactly one language and one framework. Use only the stdlib `ast` module.
Do not add tree-sitter, libcst, or any other parsing dependency for any reason without an
explicit new DEC entry and developer sign-off first.

### Rule Going Forward
All parsing goes through `parser/ast_walker.py` using only `ast` from the Python standard
library. If a parsing task seems to require a heavier tool, STOP and ask before adding any
dependency.

### Applies To
`parser/ast_walker.py`, `parser/resolver.py`, `parser/repo_scanner.py`, and any future parsing
code in this project.

---

## COR-002 — Do not call an external LLM or embedding API by default

Date: Pre-build
Phase: Pre-build
Triggered By: Preventive — common AI default when asked to "generate embeddings"

### What Happened
The most common pattern in AI coding tool training data for "generate embeddings" is
`openai.Embeddings.create(...)` or an equivalent cloud call, because most tutorials and
reference projects assume a cloud API key is available. An AI assistant is likely to default
to this pattern the first time it implements the `EmbeddingProvider`, even though this project's
explicit goal is a fully offline, zero-API-key default.

### Developer Correction
The default and only v1 embedding path is the local `nomic-embed-text-v1.5` model via
`fastembed`, running on CPU with no network call and no API key. A cloud provider MAY be added
later strictly as an opt-in override behind the `EmbeddingProvider` interface, never as the
default.

### Rule Going Forward
Never add code that calls an external LLM or embedding API as the default path. Any cloud
provider integration requires its own DEC entry and explicit developer approval before
implementation begins.

### Applies To
`embeddings/nomic_provider.py`, `embeddings/base.py`, `indexer.py`, and any future embedding
provider added to this project.

---

## [Next correction — copy the template below when needed]

## COR-00X — [Short description of what was wrong]

Date: [date]
Phase: Phase X
Triggered By: [What the AI did wrong]

### What Happened
[Describe the specific mistake]

### Developer Correction
[Exact instruction from the developer]

### Rule Going Forward
[Clear, enforceable rule]

### Applies To
[Which files, patterns, or phases this affects]

## [Empty slot — ready for use during build]

## [Empty slot — ready for use during build]
