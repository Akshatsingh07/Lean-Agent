from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Agentstate(TypedDict):
    user_query: str
    retrieved_tools: list[dict[str, Any]]
    retrieval_metadata: dict[str, Any]
    messages: Annotated[list[BaseMessage], add_messages]
    tool_calls_made: list[str]
    tool_results: list[dict[str, Any]]
    final_answer: str
    memory_context: str
    error: str | None


def create_initial_state(user_query: str) -> Agentstate:
    return Agentstate(
        user_query=user_query,
        retrieved_tools=[],
        retrieval_metadata={},
        messages=[],
        tool_calls_made=[],
        tool_results=[],
        final_answer="",
        memory_context="",
        error=None,
    )
