"""Synthetic LangGraph fixture used by the parser, embedding, and storage test suites.

Deliberately exercises BOTH ways LangGraph declares routing, because a fixture that only used
the builder-call idiom is what let QA-5-03 / RISK-011 go undetected until Phase 5: the parser
was blind to `Command(goto=...)` while every fixture-based test passed. See DEC-020.

Structure:
- 5 nodes: check_auth_token, fetch_data, enrich_data, format_response, handle_error
- Builder-declared routing:
  - 1 conditional edge (check_auth_token -> route_after_auth -> {authorized, unauthorized}),
    dict form, so `condition_value` is `known`
  - 3 normal edges: fetch_data -> enrich_data, format_response -> END, handle_error -> END
- Body-declared routing (`Command(goto=...)`, DEC-020) from `enrich_data`, which branches three
  ways and so yields 3 conditional edges naming `enrich_data` itself as the router, each with
  `condition_value=None` / `value_resolution="not_derivable"`. Covers the three shapes the
  DEC-020 census found in real code: a literal destination, a branch, and the bare END sentinel.
- 8 edges total.

The four original node functions are kept byte-identical on purpose: their embeddings, and the
`check_auth_token` vs `handle_error` ranking margin recorded under RISK-003, are referenced by
other tests and by the risk register, so this fixture grew rather than changed.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command


class AgentState(TypedDict):
    token: str
    authorized: bool
    data: dict
    response: str
    error: str


def check_auth_token(state: AgentState) -> AgentState:
    """Validate the caller's auth token before allowing any data access."""
    token = state.get("token", "")
    is_valid = token.startswith("valid-")
    return {**state, "authorized": is_valid}


def fetch_data(state: AgentState) -> AgentState:
    """Fetch the requested data now that the caller is authorized."""
    return {**state, "data": {"result": "ok"}}


def format_response(state: AgentState) -> AgentState:
    """Format the fetched data into the final response payload."""
    return {**state, "response": f"Data: {state['data']}"}


def handle_error(state: AgentState) -> AgentState:
    """Handle an unauthorized request by producing an error response."""
    return {**state, "error": "unauthorized: invalid or missing token"}


def enrich_data(state: AgentState) -> Command[Literal["format_response", "handle_error", "__end__"]]:
    """Enrich the fetched data, then continue, divert to the error path, or stop early.

    Routes from its own body rather than through a builder call, which is the idiom DEC-020
    added support for. Three branches, one per destination shape worth covering: a plain literal
    node name, a second literal reached from a different arm, and the bare END sentinel.
    """
    data = state.get("data") or {}
    if not data:
        return Command(goto="handle_error")
    if data.get("cached"):
        return Command(goto=END)
    return Command(goto="format_response", update={"data": {**data, "enriched": True}})


def route_after_auth(state: AgentState) -> Literal["authorized", "unauthorized"]:
    """Route to the data path if authorized, otherwise to the error path."""
    return "authorized" if state.get("authorized") else "unauthorized"


graph = StateGraph(AgentState)

graph.add_node("check_auth_token", check_auth_token)
graph.add_node("fetch_data", fetch_data)
graph.add_node("enrich_data", enrich_data)
graph.add_node("format_response", format_response)
graph.add_node("handle_error", handle_error)

graph.set_entry_point("check_auth_token")

graph.add_conditional_edges(
    "check_auth_token",
    route_after_auth,
    {
        "authorized": "fetch_data",
        "unauthorized": "handle_error",
    },
)

# fetch_data hands off to enrich_data here; enrich_data's own onward routing is declared in its
# body via Command(goto=...) instead, so this graph exercises both mechanisms at once.
graph.add_edge("fetch_data", "enrich_data")
graph.add_edge("format_response", END)
graph.add_edge("handle_error", END)

app = graph.compile()
