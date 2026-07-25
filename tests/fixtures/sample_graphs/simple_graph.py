"""Synthetic LangGraph fixture used by the parser, embedding, and storage test suites.

Structure:
- 4 nodes: check_auth_token, fetch_data, format_response, handle_error
- 1 conditional edge (check_auth_token -> route_after_auth -> {authorized, unauthorized})
- 3 normal edges: fetch_data -> format_response, format_response -> END, handle_error -> END
"""

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph


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


def route_after_auth(state: AgentState) -> Literal["authorized", "unauthorized"]:
    """Route to the data path if authorized, otherwise to the error path."""
    return "authorized" if state.get("authorized") else "unauthorized"


graph = StateGraph(AgentState)

graph.add_node("check_auth_token", check_auth_token)
graph.add_node("fetch_data", fetch_data)
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

graph.add_edge("fetch_data", "format_response")
graph.add_edge("format_response", END)
graph.add_edge("handle_error", END)

app = graph.compile()
