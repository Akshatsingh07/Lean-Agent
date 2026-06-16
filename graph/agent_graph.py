from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
 
from graph.state import Agentstate, create_initial_state
from agents.router_node import router_node
from agents.worker_node import worker_node
from agents.respond_node import respond_node
from agents.router_node import save_memory

def _should_continue(state: Agentstate) -> str:
    messages = state.get("messages", [])
    if not messages:
        return "respond"
 
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        print(f"\nRouting → worker (LLM has {len(last.tool_calls)} pending tool call(s))")
        return "worker"
 
    print(f"\nRouting → respond (LLM is done)")
    return "respond"

def build_graph():
    graph=StateGraph(Agentstate)
    graph.add_node("router",router_node)
    graph.add_node("worker",worker_node)
    graph.add_node("respond",respond_node)
    
    graph.add_edge(START,"router")
    graph.add_edge("router","worker")
    
    graph.add_conditional_edges(
        "worker",
        _should_continue,
        {
            "worker":"worker",
            "respond":"respond"
        }
    )
    graph.add_edge("respond",END)
    compile=graph.compile()
    return compile
    
def run_agent(query: str):
    app = build_graph()

    initial_state = create_initial_state(query)

    return app.invoke(initial_state)