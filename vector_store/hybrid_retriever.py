# import time
# from vector_store.bm25_store import get_bm25_store
# from vector_store.chroma_store import vector_search
# from sentence_transformers import CrossEncoder

# N_CANDIDATES=10
# N_FINAL=3
# RRF_K=60
# RERANKER_MODEL="cross-encoder/ms-macro-MiniLM-L-6-v2"
# _reranker: CrossEncoder | None=None

# def get_reranker() -> CrossEncoder:
#     global _reranker
#     if _reranker is None:
#         print(f"Loading reranker model: {RERANKER_MODEL}")
#         _reranker=CrossEncoder(RERANKER_MODEL)
#     return _reranker

# def reciprocal_rank_fusion(bm25_results: list[dict],vector_results: list[dict],k: int=RRF_K) -> list[dict]:
#     merged: dict[str,dict]={}
    
#     for tool in bm25_results:
#         name=tool["name"]
#         if name not in merged:
#             merged[name]={
#                 "name":tool["name"],
#                 "category":tool["category"],
#                 "description":tool["description"],
#                 "parameters":tool["parameters"],
#                 "return":tool["returns"],
#                 "bm25_rank":None,
#                 "vector_rank":None,
#                 "rrf_score": 0.0,
#             }
#         merged[name]["bm25_rank"]=tool["bm25_rank"]
#         merged[name]["rrf_score"]+=1.0/(k+tool["bm25_rank"])
        
#     for tool in vector_results:
#         name=tool["name"]
#         if name not in merged:
#             merged[name]={
#                 "name":tool["name"],
#                 "category":tool["category"],
#                 "description":tool["description"],
#                 "parameters":tool["parameters"],
#                 "return":tool["returns"],
#                 "bm25_rank":None,
#                 "vector_rank":None,
#                 "rrf_score": 0.0,
#             }
#         merged[name]["vector_rank"]=tool["vector_rank"]
#         merged[name]["rrf_score"]+=1.0/(k+tool["vector_rank"])
        
#     fused_list=list(merged.values())
#     fused_list.sort(key= lambda x:x["rrf_score"],reverse=True)
    
#     for i,tool in enumerate(fused_list):
#         tool["rrf_rank"]=i+1
#         tool["rrf_score"]=round(tool["rrf_score"],6)
#     return fused_list

# def rerank_with_crossencoder(query:str, candidates: list[dict],n_final:int =N_FINAL) -> list[dict]:
#     if not candidates:
#         return []
    
#     reranker=get_reranker()
    
#     pairs=[
#         (query,f"{tool['name']}: {tool['description']}") for tool in candidates
#     ]
#     scores=reranker.predict(pairs)
#     for i,tool in enumerate(candidates):
#         tool["rerank_score"]=round(float(scores[i]),4)
    
#     candidates.sort(key=lambda x:x["rerank_score"],reverse=True)
#     final=candidates[:n_final]
#     for i,tool in enumerate(final):
#         tool["final_rank"]=i+1
#     return final

# def hybrid_retrieve(query: str) -> dict:
#     print(f"Query:{query}")
#     t_start=time.time()
#     t0=time.time()
#     bm25_store=get_bm25_store()
#     bm25_results=bm25_store.search(query, n_results=N_CANDIDATES)
#     t_bm25=round((time.time()-t0)*1000,1)
#     print(f"Top 3 of bm25: {[r['name'] for r in bm25_results[:3]]} ({t_bm25}ms)")
    
#     t0=time.time()
#     vector_results=vector_search(query, n_results=N_CANDIDATES)
#     t_vector=round((time.time()-t0)*1000,1)
    
#     print(f"Top 3 of Vector: {[r['name'] for r in vector_results[:3]]} ({t_vector}ms)")
    
#     t0=time.time()
#     fused=reciprocal_rank_fusion(bm25_results,vector_results,k=RRF_K)
#     t_rrf=round((time.time() -t0)*1000,1)
#     print(f"  RRF top-3: {[r['name'] for r in fused[:3]]} ({t_rrf}ms)")
    
#     t0 = time.time()
#     top_candidates = fused[:N_CANDIDATES] 
#     final_tools = rerank_with_crossencoder(query, top_candidates, n_final=N_FINAL)
#     t_rerank = round((time.time() - t0) * 1000, 1)
 
#     t_total = round((time.time() - t_start) * 1000, 1)
 
#     print(f"  Re-ranked final: {[t['name'] for t in final_tools]} ({t_rerank}ms)")
#     print(f"  Total retrieval: {t_total}ms")
    
    
#     return {
#         "tools": final_tools,
#         "metadata": {
#             "query": query,
#             "n_bm25_candidates": len(bm25_results),
#             "n_vector_candidates": len(vector_results),
#             "n_fused": len(fused),
#             "n_final": len(final_tools),
#             "final_tool_names": [t["name"] for t in final_tools],
#             "rerank_scores": {t["name"]: t.get("rerank_score") for t in final_tools},
#             "rrf_scores": {t["name"]: t.get("rrf_score") for t in fused[:5]},
#             "timing_ms": {
#                 "bm25": t_bm25,
#                 "vector": t_vector,
#                 "rrf_fusion": t_rrf,
#                 "reranking": t_rerank,
#                 "total": t_total,
#             }
#         }
#     }
    
# if __name__ == "__main__":
#     from vector_store.chroma_store import build_vector_store
 
#     print("Setting up vector store...")
#     build_vector_store()
 
#     queries = [
#         "Create a Jira bug ticket for the login crash",
#         "What are our revenue metrics for last quarter?",
#         "Check production error logs for the auth service",
#     ]
#     for q in queries:
#         result = hybrid_retrieve(q)
#         print(f"\n{'─'*50}")
#         print(f"Final top-{N_FINAL} tools for: '{q}'")
#         for t in result["tools"]:
#             print(
#                 f"  [{t['final_rank']}] {t['name']:35}"
#                 f"  rerank={t.get('rerank_score','?')}"
#                 f"  rrf={t.get('rrf_score','?')}"
#             )

import time
from vector_store.bm25_store import get_bm25_store
from vector_store.chroma_store import vector_search
from sentence_transformers import CrossEncoder

N_CANDIDATES = 10
N_FINAL = 3
RRF_K = 60
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None

def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker

def reciprocal_rank_fusion(bm25_results: list[dict], vector_results: list[dict], k: int = RRF_K) -> list[dict]:
    """
    Combines BM25 (sparse) and Vector (dense) results using Reciprocal Rank Fusion.
    """
    merged: dict[str, dict] = {}

    for i, tool in enumerate(bm25_results):
        name = tool["name"]
        rank = i + 1
        if name not in merged:
            merged[name] = {
                "name": tool["name"],
                "category": tool.get("category", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "returns": tool.get("returns", ""),
                "bm25_rank": None,
                "vector_rank": None,
                "rrf_score": 0.0,
            }
        merged[name]["bm25_rank"] = rank
        merged[name]["rrf_score"] += 1.0 / (k + rank)
        
    for i, tool in enumerate(vector_results):
        name = tool["name"]
        rank = i + 1
        if name not in merged:
            merged[name] = {
                "name": tool["name"],
                "category": tool.get("category", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "returns": tool.get("returns", ""),
                "bm25_rank": None,
                "vector_rank": None,
                "rrf_score": 0.0,
            }
        merged[name]["vector_rank"] = rank
        merged[name]["rrf_score"] += 1.0 / (k + rank)

    fused_list = list(merged.values())
    fused_list.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    for i, tool in enumerate(fused_list):
        tool["rrf_rank"] = i + 1
        tool["rrf_score"] = round(tool["rrf_score"], 6)
        
    return fused_list

def rerank_with_crossencoder(query: str, candidates: list[dict], n_final: int = N_FINAL) -> list[dict]:
    if not candidates:
        return []
    
    reranker = get_reranker()
    
    pairs = [
        (query, f"{tool['name']}: {tool['description']}") for tool in candidates
    ]
    scores = reranker.predict(pairs)
    for i, tool in enumerate(candidates):
        tool["rerank_score"] = round(float(scores[i]), 4)
    
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    final = candidates[:n_final]
    for i, tool in enumerate(final):
        tool["final_rank"] = i + 1
    return final

def hybrid_retrieve(query: str) -> dict:
    print(f"Query: {query}")
    t_start = time.time()
    
    t0 = time.time()
    bm25_store = get_bm25_store()
    bm25_results = bm25_store.search(query, n_results=N_CANDIDATES)
    t_bm25 = round((time.time() - t0) * 1000, 1)
    print(f"Top 3 of bm25: {[r['name'] for r in bm25_results[:3]]} ({t_bm25}ms)")
    
    t0 = time.time()
    vector_results = vector_search(query, n_results=N_CANDIDATES)
    t_vector = round((time.time() - t0) * 1000, 1)
    print(f"Top 3 of Vector: {[r['name'] for r in vector_results[:3]]} ({t_vector}ms)")
    
    t0 = time.time()
    fused = reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)
    t_rrf = round((time.time() - t0) * 1000, 1)
    print(f"  RRF top-3: {[r['name'] for r in fused[:3]]} ({t_rrf}ms)")
    
    t0 = time.time()
    top_candidates = fused[:N_CANDIDATES] 
    final_tools = rerank_with_crossencoder(query, top_candidates, n_final=N_FINAL)
    t_rerank = round((time.time() - t0) * 1000, 1)
 
    t_total = round((time.time() - t_start) * 1000, 1)
 
    print(f"  Re-ranked final: {[t['name'] for t in final_tools]} ({t_rerank}ms)")
    print(f"  Total retrieval: {t_total}ms")
    
    return {
        "tools": final_tools,
        "metadata": {
            "query": query,
            "n_bm25_candidates": len(bm25_results),
            "n_vector_candidates": len(vector_results),
            "n_fused": len(fused),
            "n_final": len(final_tools),
            "final_tool_names": [t["name"] for t in final_tools],
            "rerank_scores": {t["name"]: t.get("rerank_score") for t in final_tools},
            "rrf_scores": {t["name"]: t.get("rrf_score") for t in fused[:5]},
            "timing_ms": {
                "bm25": t_bm25,
                "vector": t_vector,
                "rrf_fusion": t_rrf,
                "reranking": t_rerank,
                "total": t_total,
            }
        }
    }
    
if __name__ == "__main__":
    from vector_store.chroma_store import embed_all_tools
 
    print("Setting up vector store...")
    embed_all_tools()
 
    queries = [
        "Create a Jira bug ticket for the login crash",
        "What are our revenue metrics for last quarter?",
        "Check production error logs for the auth service",
    ]
    for q in queries:
        result = hybrid_retrieve(q)
        print(f"\n{'─'*50}")
        print(f"Final top-{N_FINAL} tools for: '{q}'")
        for t in result["tools"]:
            print(
                f"  [{t['final_rank']}] {t['name']:35}"
                f"  rerank={t.get('rerank_score','?')}"
                f"  rrf={t.get('rrf_score','?')}"
            )