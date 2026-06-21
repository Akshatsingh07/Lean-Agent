import time

from graph.state import Agentstate
from vector_store.hybrid_retriever import hybrid_retrieve

try:
    from mem0 import Memory
    MEMO_AVAILABLE = True
except ImportError:
    MEMO_AVAILABLE = False
    print("mem0 not installed. Run: pip install mem0ai")
    print("Continuing without memory - agent will have no personalisation.")
    
_memory = None


def get_memory():
    global _memory
    if not MEMO_AVAILABLE:
        return None

    if _memory is not None:
        return _memory

    config = {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": "llama3.1",
                "temperature": 0
            }
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "embedding_dims": 768  
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": "./mem0_db_768",      
                "embedding_model_dims": 768  
            }
        }
    }
    
    _memory = Memory.from_config(config_dict=config)
    return _memory


def fetch_memory_context(query: str, user_id: str = "default_user") -> str:
    mem = get_memory()
    if mem is None:
        return ""
    
    try:
        search_results = mem.search(query=query, filters={"user_id": user_id})
        if not search_results:
            return ""
        # memory_lines=[]
        # for result in search_results:
        #     memory_text=result.get("memory", "")
        #     if memory_text:
        #         memory_lines.append(f"-{memory_text}")
        # if not memory_lines:
        #     return ""
        # context="Relevant past context:\n"+"\n".join(memory_lines)
        memories = []
        if isinstance(search_results, list):
            for r in search_results:
                if isinstance(r, dict) and "memory" in r:
                    memories.append(r["memory"])
                elif isinstance(r, str):
                    memories.append(r)
        elif isinstance(search_results, str):
            memories.append(search_results)
            
        if memories:
            return "Relevant History:\n" + "\n".join(memories[:5])
            
        return ""
    except Exception as e:
        error_msg = str(e)
        if "'str' object has no attribute 'get'" in error_msg:
            pass 
        else:
            print(f"  [Mem0 Error]: {error_msg}")
            
        return ""

def save_memory(text: str, user_id: str = "default_user") -> None:
    mem = get_memory()
    if not mem:
        return

    try:
        mem.add(text, user_id=user_id)
        print(" [Mem0] Memory saved successfully.")
    except Exception as e:
        print(f"  [Mem0 Warning]: Save failed: {e}")

def router_node(state: Agentstate) -> dict:
    print("[NODE 1 - ROUTER]")
    query = state["user_query"]
    user_id = "default_user"
    print(f"Query : {query}")
    
    t_start = time.time()
    print("\n[1/2] Fetching memory context from mem0")
    memory_context = fetch_memory_context(query, user_id=user_id)
    if memory_context:
        print(f"  Memory: {memory_context[:100]}...")
    else:
        print(f"  Memory: (none found for this query)")
    
    print("\n  [2/2] Running hybrid retrieval (BM25 + Vector + RRF + Rerank)")
    try:
        retrieval_result = hybrid_retrieve(query)
        tools = retrieval_result["tools"]
        metadata = retrieval_result["metadata"]
    except Exception as e:
        error_msg = f"Hybrid retrieval failed: {e}"
        print(f"{error_msg}")
        return {
            "retrieved_tools": [],
            "retrieval_metadata": {"error": error_msg},
            "memory_context": memory_context,
            "error": error_msg,
        }
    total_ms = round((time.time() - t_start) * 1000, 1)
    print(f"\nRouter complete in {total_ms}ms")
    print(f"  Tools retrieved ({len(tools)}/50):")
    for t in tools:
        print(
            f"    [{t['final_rank']}] {t['name']:35}"
            f"  rerank={t.get('rerank_score', 0):.3f}"
            f"  [{t['category']}]"
        )
    print(f"\n  Worker will see ONLY these {len(tools)} tools (not all 50).")
    
    return {
        "retrieved_tools": tools,
        "retrieval_metadata": {
            **metadata,
            "total_router_ms": total_ms,
            "memory_retrieved": bool(memory_context),
        },
        "memory_context": memory_context,
    }
    
