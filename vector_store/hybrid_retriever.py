import time
from typing import Any

from vector_store.bm25_store import get_bm25_store
from vector_store.chroma_store import vector_search

N_CANDIDATES = 10
N_FINAL = 3
RRF_K = 60
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: Any | None = None


def get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        print(f"Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _tool_payload(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "category": tool.get("category", ""),
        "description": tool.get("description", ""),
        "parameters": tool.get("parameters", {}),
        "returns": tool.get("returns", ""),
        "bm25_rank": None,
        "vector_rank": None,
        "rrf_score": 0.0,
    }


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse sparse and dense rankings without assuming comparable score scales."""
    merged: dict[str, dict[str, Any]] = {}

    for rank, tool in enumerate(bm25_results, start=1):
        name = tool["name"]
        merged.setdefault(name, _tool_payload(tool))
        merged[name]["bm25_rank"] = tool.get("bm25_rank", rank)
        merged[name]["bm25_score"] = tool.get("bm25_score")
        merged[name]["rrf_score"] += 1.0 / (k + rank)

    for rank, tool in enumerate(vector_results, start=1):
        name = tool["name"]
        merged.setdefault(name, _tool_payload(tool))
        merged[name]["vector_rank"] = tool.get("vector_rank", rank)
        merged[name]["similarity_score"] = tool.get("similarity_score")
        merged[name]["rrf_score"] += 1.0 / (k + rank)

    fused = sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)
    for rank, tool in enumerate(fused, start=1):
        tool["rrf_rank"] = rank
        tool["rrf_score"] = round(tool["rrf_score"], 6)
    return fused


def rerank_with_crossencoder(
    query: str,
    candidates: list[dict[str, Any]],
    n_final: int = N_FINAL,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, f"{tool['name']}: {tool['description']}") for tool in candidates]
    scores = reranker.predict(pairs)

    for index, tool in enumerate(candidates):
        tool["rerank_score"] = round(float(scores[index]), 4)

    ranked = sorted(candidates, key=lambda item: item["rerank_score"], reverse=True)
    final = ranked[:n_final]
    for rank, tool in enumerate(final, start=1):
        tool["final_rank"] = rank
    return final


def _rank_without_crossencoder(
    candidates: list[dict[str, Any]],
    n_final: int = N_FINAL,
) -> list[dict[str, Any]]:
    final = candidates[:n_final]
    for rank, tool in enumerate(final, start=1):
        tool["final_rank"] = rank
        tool["rerank_score"] = tool.get("rrf_score", 0.0)
    return final


def hybrid_retrieve(query: str, use_reranker: bool = True) -> dict[str, Any]:
    print(f"Query: {query}")
    t_start = time.time()
    warnings: list[str] = []

    t0 = time.time()
    bm25_store = get_bm25_store()
    bm25_results = bm25_store.search(query, n_results=N_CANDIDATES)
    t_bm25 = round((time.time() - t0) * 1000, 1)
    print(f"Top 3 BM25: {[r['name'] for r in bm25_results[:3]]} ({t_bm25}ms)")

    t0 = time.time()
    try:
        vector_results = vector_search(query, n_results=N_CANDIDATES)
        print(f"Top 3 Vector: {[r['name'] for r in vector_results[:3]]}")
    except Exception as exc:
        vector_results = []
        warnings.append(f"Vector search unavailable: {exc}")
        print(f"Vector search unavailable: {exc}")
    t_vector = round((time.time() - t0) * 1000, 1)

    t0 = time.time()
    fused = reciprocal_rank_fusion(bm25_results, vector_results, k=RRF_K)
    t_rrf = round((time.time() - t0) * 1000, 1)
    print(f"RRF top 3: {[r['name'] for r in fused[:3]]} ({t_rrf}ms)")

    t0 = time.time()
    top_candidates = fused[:N_CANDIDATES]
    if use_reranker:
        try:
            final_tools = rerank_with_crossencoder(query, top_candidates, n_final=N_FINAL)
        except Exception as exc:
            warnings.append(f"Cross-encoder unavailable: {exc}")
            print(f"Cross-encoder unavailable: {exc}")
            final_tools = _rank_without_crossencoder(top_candidates, n_final=N_FINAL)
    else:
        final_tools = _rank_without_crossencoder(top_candidates, n_final=N_FINAL)
    t_rerank = round((time.time() - t0) * 1000, 1)

    t_total = round((time.time() - t_start) * 1000, 1)
    print(f"Final tools: {[t['name'] for t in final_tools]} ({t_rerank}ms rerank)")
    print(f"Total retrieval: {t_total}ms")

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
            "warnings": warnings,
            "timing_ms": {
                "bm25": t_bm25,
                "vector": t_vector,
                "rrf_fusion": t_rrf,
                "reranking": t_rerank,
                "total": t_total,
            },
        },
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
        print(f"\n{'-' * 50}")
        print(f"Final top-{N_FINAL} tools for: '{q}'")
        for t in result["tools"]:
            print(
                f"  [{t['final_rank']}] {t['name']:35}"
                f" rerank={t.get('rerank_score', '?')}"
                f" rrf={t.get('rrf_score', '?')}"
            )
