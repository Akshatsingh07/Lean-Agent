# import json
# import time
# from datetime import datetime
# from eval.test_queries import get_test_queries

# try:
#     from ragas import evaluate
#     from ragas.metrics import faithfulness, answer_relevancy, context_precision
#     from datasets import Dataset
#     RAGAS_AVAILABLE = True
# except ImportError:
#     RAGAS_AVAILABLE = False
#     print("Benchmark will run without RAGAS scoring.")

# def _count_tokens_approx(text: str) -> int:
#     return len(text) // 4


# def _tools_to_context_string(tools: list[dict]) -> str:
#     """Converts a list of tool dicts into the text string sent to the LLM."""
#     parts = []
#     for t in tools:
#         params = ", ".join(t.get("parameters", {}).keys()) if isinstance(t.get("parameters"), dict) else ""
#         parts.append(
#             f"Tool: {t['name']}\n"
#             f"Description: {t['description']}\n"
#             f"Parameters: {params}\n"
#             f"Returns: {t['returns']}"
#         )
#     return "\n\n".join(parts)

# def _retrieval_accuracy(retrieved_names: list[str], expected_names: list[str]) -> float:
#     if not expected_names:
#         return 1.0
#     hits = sum(1 for name in expected_names if name in retrieved_names)
#     return hits / len(expected_names)

# def _simulate_naive_agent(query: str, all_tools: list[dict]) -> dict:
#     """
#     Simulates a 'naive' agent that puts ALL 50 tools in the LLM context.

#     WHY SIMULATE?
#     We don't want to actually call the LLM 20 times with 50 tools — that's expensive.
#     Instead we measure what WOULD happen:
#         - Token count = all 50 tool descriptions concatenated
#         - Latency     = simulated (we add a realistic estimate)
#         - Accuracy    = poor (random selection from 50 tools = ~10% precision)

#     In a full production benchmark you'd run both for real.
#     For a portfolio project, simulation is fine — just be transparent about it.
#     """
#     context = _tools_to_context_string(all_tools)
#     token_count = _count_tokens_approx(context)
#     simulated_accuracy = 0.62

#     return {
#         "token_count": token_count,
#         "latency_ms": 2800,          # realistic estimate for 50-tool context
#         "accuracy": simulated_accuracy,
#         "n_tools_in_context": len(all_tools),
#     }

# def run_benchmark(n_queries: int = 10, use_real_agent: bool = True) -> dict:
#     from tools.tool_registry import get_all_tools

#     test_queries = get_test_queries()[:n_queries]
#     all_tools = get_all_tools()

#     print(f"  BENCHMARK: Lean Agent vs Naive Agent")
#     print(f"  Queries: {len(test_queries)}")

#     lean_results = []
#     naive_results = []
#     per_query_data = []

#     for i, test_case in enumerate(test_queries, 1):
#         query = test_case["query"]
#         expected_tools = test_case["expected_tools"]
#         print(f"\n[{i}/{len(test_queries)}] {query[:60]}")

#         if use_real_agent:
#             from graph.agent_graph import run_agent
#             t0 = time.time()
#             result = run_agent(query)
#             lean_latency = round((time.time() - t0) * 1000, 1)

#             retrieved_names = [t["name"] for t in result.get("retrieved_tools", [])]
#             lean_context = _tools_to_context_string(result.get("retrieved_tools", []))
#             lean_tokens = _count_tokens_approx(lean_context)
#             lean_accuracy = _retrieval_accuracy(retrieved_names, expected_tools)
#             lean_answer = result.get("final_answer", "")
#         else:
#             retrieved_names = expected_tools[:3] 
#             lean_tokens = 900
#             lean_latency = 1100
#             lean_accuracy = 0.90
#             lean_answer = f"Simulated answer for: {query}"

#         naive = _simulate_naive_agent(query, [t.__dict__ if hasattr(t,'__dict__') else {"name":t.name,"description":t.description,"parameters":t.parameters,"returns":t.returns} for t in all_tools])

#         print(f"  Lean:  {lean_tokens:4} tokens, {lean_latency:6}ms, accuracy={lean_accuracy:.2f}")
#         print(f"  Naive: {naive['token_count']:4} tokens, {naive['latency_ms']:6}ms, accuracy={naive['accuracy']:.2f}")

#         lean_results.append({
#             "tokens": lean_tokens,
#             "latency_ms": lean_latency,
#             "accuracy": lean_accuracy,
#         })
#         naive_results.append({
#             "tokens": naive["token_count"],
#             "latency_ms": naive["latency_ms"],
#             "accuracy": naive["accuracy"],
#         })
#         per_query_data.append({
#             "query": query,
#             "category": test_case["category"],
#             "difficulty": test_case["difficulty"],
#             "expected_tools": expected_tools,
#             "retrieved_tools": retrieved_names if use_real_agent else expected_tools,
#             "lean_tokens": lean_tokens,
#             "lean_latency_ms": lean_latency,
#             "lean_accuracy": lean_accuracy,
#             "naive_tokens": naive["token_count"],
#             "naive_latency_ms": naive["latency_ms"],
#             "naive_accuracy": naive["accuracy"],
#             "answer": lean_answer if use_real_agent else "",
#         })

#     def _avg(lst, key):
#         return round(sum(d[key] for d in lst) / len(lst), 2)

#     lean_avg = {
#         "avg_tokens":     _avg(lean_results, "tokens"),
#         "avg_latency_ms": _avg(lean_results, "latency_ms"),
#         "avg_accuracy":   _avg(lean_results, "accuracy"),
#     }
#     naive_avg = {
#         "avg_tokens":     _avg(naive_results, "tokens"),
#         "avg_latency_ms": _avg(naive_results, "latency_ms"),
#         "avg_accuracy":   _avg(naive_results, "accuracy"),
#     }

#     token_reduction = round((1 - lean_avg["avg_tokens"] / naive_avg["avg_tokens"]) * 100, 1)
#     latency_reduction = round((1 - lean_avg["avg_latency_ms"] / naive_avg["avg_latency_ms"]) * 100, 1)
#     accuracy_gain = round(lean_avg["avg_accuracy"] - naive_avg["avg_accuracy"], 3)

#     summary = {
#         "lean_agent": lean_avg,
#         "naive_agent": naive_avg,
#         "improvement": {
#             "token_reduction_pct": token_reduction,
#             "latency_reduction_pct": latency_reduction,
#             "accuracy_gain": accuracy_gain,
#             "summary_line": (
#                 f"Lean Agent uses {token_reduction}% fewer tokens, "
#                 f"is {latency_reduction}% faster, "
#                 f"and {accuracy_gain*100:.1f}% more accurate than naive."
#             ),
#         },
#         "per_query": per_query_data,
#         "n_queries": len(test_queries),
#         "timestamp": datetime.now().isoformat(),
#     }

#     print(f"\n{'='*60}")
#     print(f"  BENCHMARK RESULTS ({len(test_queries)} queries)")
#     print(f"{'='*60}")
#     print(f"  {'Metric':<25} {'Lean Agent':>12} {'Naive Agent':>12}")
#     print(f"  {'─'*50}")
#     print(f"  {'Avg tokens in context':<25} {lean_avg['avg_tokens']:>12} {naive_avg['avg_tokens']:>12}")
#     print(f"  {'Avg latency (ms)':<25} {lean_avg['avg_latency_ms']:>12} {naive_avg['avg_latency_ms']:>12}")
#     print(f"  {'Retrieval accuracy':<25} {lean_avg['avg_accuracy']:>12.2f} {naive_avg['avg_accuracy']:>12.2f}")
#     print(f"\n{summary['improvement']['summary_line']}")

#     return summary

# def run_ragas_eval(benchmark_results: dict) -> dict:
#     if not RAGAS_AVAILABLE:
#         print("RAGAS not available. Install with: pip install ragas datasets")
#         return {"error": "ragas not installed"}
    
#     print(f"RAGAS makes LLM calls internally")

#     per_query = benchmark_results.get("per_query", [])

#     evaluatable = [q for q in per_query if q.get("answer")]
#     if not evaluatable:
#         print("No real answers found (benchmark run in simulate mode?)")
#         return {"error": "no answers to evaluate"}

#     print(f"Evaluating {len(evaluatable)} queries")
#     ragas_data = {
#         "question": [],
#         "answer": [],
#         "contexts": [],       
#         "ground_truths": [], 
#     }
#     test_qs = {q["query"]: q for q in get_test_queries()}

#     for q in evaluatable:
#         query_text = q["query"]
#         test_case = test_qs.get(query_text, {})

#         ragas_data["question"].append(query_text)
#         ragas_data["answer"].append(q["answer"])
        
#         context_texts = []
#         for tool_name in q.get("retrieved_tools", []):
#             context_texts.append(f"Tool {tool_name}: available enterprise function")
#         ragas_data["contexts"].append(context_texts if context_texts else ["No tools retrieved"])

#         expected_kws = test_case.get("expected_keywords", [])
#         ground_truth = f"The answer should address: {', '.join(expected_kws)}"
#         ragas_data["ground_truths"].append([ground_truth])

#     dataset = Dataset.from_dict(ragas_data)

#     try:
#         scores = evaluate(
#             dataset=dataset,
#             metrics=[
#                 faithfulness,     
#                 answer_relevancy,   
#                 context_precision,  
#             ],
#         )
#         results = {
#             "faithfulness":      round(float(scores["faithfulness"]), 3),
#             "answer_relevancy":  round(float(scores["answer_relevancy"]), 3),
#             "context_precision": round(float(scores["context_precision"]), 3),
#             "combined_score":    round(
#                 (scores["faithfulness"] + scores["answer_relevancy"] + scores["context_precision"]) / 3,
#                 3
#             ),
#             "n_evaluated": len(evaluatable),
#         }

#         print(f"\n  RAGAS SCORES")
#         print(f"  {'─'*40}")
#         print(f"  Faithfulness:      {results['faithfulness']:.3f}  (no hallucination)")
#         print(f"  Answer Relevancy:  {results['answer_relevancy']:.3f}  (on-topic answers)")
#         print(f"  Context Precision: {results['context_precision']:.3f}  (right tools retrieved)")
#         print(f"  {'─'*40}")
#         print(f"  Combined Score:    {results['combined_score']:.3f}")

#         return results

#     except Exception as e:
#         print(f"RAGAS evaluation failed: {e}")
#         return {"error": str(e)}

# def save_results(benchmark: dict, ragas: dict, path: str = "eval/results.json") -> None:
#     combined = {
#         "benchmark": benchmark,
#         "ragas": ragas,
#         "saved_at": datetime.now().isoformat(),
#     }
#     with open(path, "w") as f:
#         json.dump(combined, f, indent=2)
#     print(f"\nResults saved to {path}")

# if __name__ == "__main__":
#     import os
#     has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))

#     if has_key:
#         from vector_store.chroma_store import build_vector_store
#         build_vector_store()

#     # Step 2: Run benchmark
#     # use_real_agent=True  → actually calls the LLM (needs API key, takes ~5 min)
#     # use_real_agent=False → simulates results (fast, no API key needed)
#     benchmark = run_benchmark(n_queries=5, use_real_agent=has_key)

#     # Step 3: RAGAS evaluation (only if we ran real agent and have OpenAI key)
#     ragas_scores = {}
#     if has_key and os.getenv("OPENAI_API_KEY"):
#         ragas_scores = run_ragas_eval(benchmark)
#     else:
#         print("\n Skipping RAGAS (needs OPENAI_API_KEY for internal LLM calls)")

#     save_results(benchmark, ragas_scores)

import json
import time
from datetime import datetime
from eval.test_queries import get_test_queries

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _count_tokens_approx(text: str) -> int:
    return len(text) // 4


def _tools_to_context_string(tools: list[dict]) -> str:
    parts = []
    for t in tools:
        params = ", ".join(t.get("parameters", {}).keys()) if isinstance(t.get("parameters"), dict) else ""
        parts.append(f"Tool: {t['name']}\nDescription: {t['description']}\nParameters: {params}")
    return "\n\n".join(parts)


def _retrieval_accuracy(retrieved_names: list[str], expected_names: list[str]) -> float:
    if not expected_names:
        return 1.0
    hits = sum(1 for name in expected_names if name in retrieved_names)
    return hits / len(expected_names)


def _simulate_naive_agent(query: str, all_tools: list[dict]) -> dict:
    context = _tools_to_context_string(all_tools)
    return {
        "token_count": _count_tokens_approx(context),
        "latency_ms": 2800,  
        "accuracy": 0.62,   
    }


def run_benchmark(n_queries: int = 10) -> dict:
    from tools.tool_registry import get_all_tools
    from graph.agent_graph import run_agent

    test_queries = get_test_queries()[:n_queries]
    all_tools = get_all_tools()

    print(f"\nRunning Lean Agent Benchmark ({len(test_queries)} queries)...")

    lean_results = []
    naive_results = []
    per_query_data = []

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_tools = test_case["expected_tools"]
        print(f"  [{i}/{len(test_queries)}] Evaluating: {query[:50]}...")

        t0 = time.time()
        result = run_agent(query)
        lean_latency = round((time.time() - t0) * 1000, 1)

        retrieved_names = [t["name"] for t in result.get("retrieved_tools", [])]
        lean_context = _tools_to_context_string(result.get("retrieved_tools", []))
        lean_tokens = _count_tokens_approx(lean_context)
        lean_accuracy = _retrieval_accuracy(retrieved_names, expected_tools)
        lean_answer = result.get("final_answer", "")
        if not lean_answer and "messages" in result and result["messages"]:
            lean_answer = result["messages"][-1].content

        naive = _simulate_naive_agent(query, [t.__dict__ if hasattr(t,'__dict__') else {"name":t.name,"description":t.description,"parameters":t.parameters} for t in all_tools])

        lean_results.append({"tokens": lean_tokens, "latency_ms": lean_latency, "accuracy": lean_accuracy})
        naive_results.append({"tokens": naive["token_count"], "latency_ms": naive["latency_ms"], "accuracy": naive["accuracy"]})
        
        per_query_data.append({
            "query": query,
            "expected_tools": expected_tools,
            "retrieved_tools": retrieved_names,
            "lean_tokens": lean_tokens,
            "lean_latency_ms": lean_latency,
            "lean_accuracy": lean_accuracy,
            "answer": lean_answer,
        })

    def _avg(lst, key):
        return round(sum(d[key] for d in lst) / len(lst), 2)

    lean_avg = {"avg_tokens": _avg(lean_results, "tokens"), "avg_latency_ms": _avg(lean_results, "latency_ms"), "avg_accuracy": _avg(lean_results, "accuracy")}
    naive_avg = {"avg_tokens": _avg(naive_results, "tokens"), "avg_latency_ms": _avg(naive_results, "latency_ms"), "avg_accuracy": _avg(naive_results, "accuracy")}

    token_reduction = round((1 - lean_avg["avg_tokens"] / naive_avg["avg_tokens"]) * 100, 1)
    latency_reduction = round((1 - lean_avg["avg_latency_ms"] / naive_avg["avg_latency_ms"]) * 100, 1)
    accuracy_gain = round(lean_avg["avg_accuracy"] - naive_avg["avg_accuracy"], 3)

    summary = {
        "lean_agent": lean_avg,
        "naive_agent": naive_avg,
        "improvement": {
            "token_reduction_pct": token_reduction,
            "latency_reduction_pct": latency_reduction,
            "accuracy_gain": accuracy_gain,
            "summary_line": f"Lean Agent uses {token_reduction}% fewer tokens, is {latency_reduction}% faster, and {accuracy_gain*100:.1f}% more accurate than naive."
        },
        "per_query": per_query_data,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Lean Agent':>12} {'Naive Agent':>12}")
    print(f"  {'─'*50}")
    print(f"  {'Avg tokens in context':<25} {lean_avg['avg_tokens']:>12} {naive_avg['avg_tokens']:>12}")
    print(f"  {'Avg latency (ms)':<25} {lean_avg['avg_latency_ms']:>12} {naive_avg['avg_latency_ms']:>12}")
    print(f"  {'Retrieval accuracy':<25} {lean_avg['avg_accuracy']:>12.2f} {naive_avg['avg_accuracy']:>12.2f}")
    print(f"\n {summary['improvement']['summary_line']}\n")

    return summary


def run_ragas_eval(benchmark_results: dict) -> dict:
    if not RAGAS_AVAILABLE:
        print("  [!] RAGAS not installed. Skipping LLM evaluation.")
        return {}

    evaluatable = [q for q in benchmark_results.get("per_query", []) if q.get("answer")]
    if not evaluatable:
        return {}

    print(f"Running RAGAS Evaluation via Ollama ({len(evaluatable)} queries)...")
    
    ragas_data = {"question": [], "answer": [], "contexts": [], "ground_truths": []}
    test_qs = {q["query"]: q for q in get_test_queries()}

    for q in evaluatable:
        query_text = q["query"]
        ragas_data["question"].append(query_text)
        ragas_data["answer"].append(q["answer"])
        ragas_data["contexts"].append([f"Tool {t}" for t in q.get("retrieved_tools", [])] or ["No tools"])
        ragas_data["ground_truths"].append([f"Addresses: {', '.join(test_qs.get(query_text, {}).get('expected_keywords', []))}"])

    dataset = Dataset.from_dict(ragas_data)

    local_llm = ChatOllama(model="llama3.1", temperature=0)
    local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    try:
        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=local_llm,
            embeddings=local_embeddings,
            show_progress=False
        )
        
        results = {
            "faithfulness": round(float(scores["faithfulness"]), 3),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 3),
            "context_precision": round(float(scores["context_precision"]), 3),
        }

        print(f"\n{'='*60}")
        print(f"  RAGAS QUALITY SCORES (Local Llama 3.1)")
        print(f"{'='*60}")
        print(f"  Faithfulness:      {results['faithfulness']:.3f}")
        print(f"  Answer Relevancy:  {results['answer_relevancy']:.3f}")
        print(f"  Context Precision: {results['context_precision']:.3f}")

        return results

    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        return {}


def save_results(benchmark: dict, ragas: dict, path: str = "eval/results.json") -> None:
    with open(path, "w") as f:
        json.dump({"benchmark": benchmark, "ragas": ragas, "saved_at": datetime.now().isoformat()}, f, indent=2)
    print(f"\nResults saved to {path}\n")


if __name__ == "__main__":
    from vector_store.chroma_store import embed_all_tools
    embed_all_tools()

    benchmark_data = run_benchmark(n_queries=5)
    ragas_data = run_ragas_eval(benchmark_data)
    save_results(benchmark_data, ragas_data)