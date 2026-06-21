from langchain_core.messages import AIMessage

from agents.router_node import save_memory
from graph.state import Agentstate


def respond_node(state: Agentstate) -> dict:
    print("[NODE 3 - RESPOND]")
    messages = state.get("messages", [])
    final_message = None

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_message = msg
            break
        
    if not final_message:
        final_answer = state.get("final_answer") or "I encountered an issue and could not complete your request."
    else:
        content = final_message.content
        if isinstance(content, list):
            final_answer = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            final_answer = str(content)

    final_answer = final_answer.strip()

    tools_used = state.get("tool_calls_made", [])
    query = state.get("user_query", "")
 
    if tools_used:
        memory_text = (
            f"User request: '{query[:80]}'. "
            f"Tools used: {', '.join(tools_used)}."
        )
        save_memory(memory_text)
        
    print(f"  Query: {query}")
 
    retrieved = state.get("retrieved_tools", [])
    print(f"  Tools retrieved: {len(retrieved)}/50")
    for t in retrieved:
        print(f"    - {t['name']} (rerank={t.get('rerank_score', '?')})")
 
    print(f"  Tools called:   {tools_used}")
 
    meta = state.get("retrieval_metadata", {})
    timing = meta.get("timing_ms", {})
    if timing:
        print(f"  Retrieval time: {timing.get('total', '?')}ms")
        print(f"    BM25:    {timing.get('bm25', '?')}ms")
        print(f"    Vector:  {timing.get('vector', '?')}ms")
        print(f"    Rerank:  {timing.get('reranking', '?')}ms")
 
    if final_answer and query:
        memory_string = f"User asked: '{query}'. Agent answered: '{final_answer}'"
        save_memory(text=memory_string)
 
    return {"final_answer": final_answer}
