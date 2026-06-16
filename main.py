"""
main.py
--------
RUN THIS FILE to:
    1. Build the ChromaDB vector store (one-time, persists to disk)
    2. Build the BM25 index (in-memory, rebuilt each startup in ~50ms)
    3. Run test queries through the full pipeline
    4. Optionally run the benchmark
"""
import sys
import argparse
import requests

def check_ollama():

    print("Checking Local Services:")
    try:
        response = requests.get("http://localhost:11434/")
        if response.status_code == 200:
            print("  Ollama: ✓ running locally")
            return True
    except requests.exceptions.ConnectionError:
        print("Ollama: not found")
        print("\nERROR: Cannot connect to Ollama.")
        sys.exit(1)


def setup(force_refresh: bool = False):
    """
    One-time setup: build ChromaDB vector store + BM25 index.

    IDEMPOTENT: Safe to call every run.
    Vector store is only rebuilt if force_refresh=True or it doesn't exist.
    BM25 index is always rebuilt (fast, ~50ms).
    """
    print("\n" + "="*55)
    print("  SETUP (LOCAL HYBRID RAG)")
    print("="*55)

    print("\n[1/2] Building ChromaDB vector store (Ollama Embeddings)...")
    from vector_store.chroma_store import embed_all_tools
    embed_all_tools(force_refresh=force_refresh)

    print("\n[2/2] Building BM25 index...")
    from vector_store.bm25_store import get_bm25_store
    get_bm25_store()

    print("\nSetup complete.")


def run_test_queries():
    from graph.agent_graph import run_agent

    test_queries = [
        "Create a high priority bug ticket 'Login page crashes on Safari iOS 17' in MOBILE project",
        "What were our MRR and ARR metrics last quarter?",
        "Check if production deployment is healthy and look for recent error logs in auth service",
        "Send a Slack message to #engineering that the v2.14.3 deployment is complete",
    ]
    print("  TEST QUERIES")

    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}/{len(test_queries)}] Running...")
        result = run_agent(query)

        user_query = result.get('user_query', query)
        tools = result.get('retrieved_tools', [])
        calls = result.get('tool_calls_made', [])
        answer = result.get('final_answer', 'No final answer provided.')

        print(f"  Query:   {user_query}")
        print(f"  Tools:   {[t['name'] for t in tools] if tools else 'None'}")
        print(f"  Called:  {calls}")
        print(f"  Answer:  {answer[:200]}...")


def run_benchmark_suite():
    print("  RUNNING BENCHMARK")

    from eval.benchmark import run_benchmark, save_results
    from eval.benchmark import run_ragas_eval

    benchmark = run_benchmark(n_queries=5, use_real_agent=True)

    ragas = run_ragas_eval(benchmark)

    save_results(benchmark, ragas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean Agent — Dynamic RAG Tool Selection")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark and local RAGAS eval after test queries")
    parser.add_argument("--refresh", action="store_true",
                        help="Force rebuild the vector store from scratch")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║   Lean Agent — Dynamic RAG Tool Selection            ║
║   BM25 + Vector → RRF → Local Ollama LLM + mem0      ║
╚══════════════════════════════════════════════════════╝
    """)

    check_ollama()
    setup(force_refresh=args.refresh)
    run_test_queries()

    if args.benchmark:
        run_benchmark_suite()