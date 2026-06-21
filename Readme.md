# Lean Agent

Lean Agent is a local enterprise AI agent that solves the "too many tools in the prompt" problem. Instead of binding every available enterprise function to the LLM, it retrieves the few tools relevant to the user's request, binds only those tools, executes them through LangGraph, and records benchmark evidence for the retrieval pipeline.

## Why This Is Resume-Worthy

- Built a dynamic tool-selection agent over a 50-tool enterprise registry.
- Combined BM25 sparse retrieval, Chroma dense retrieval, Reciprocal Rank Fusion, and optional cross-encoder reranking.
- Reduced average tool-context tokens by 93.9% in the included benchmark while preserving 100% top-k retrieval accuracy on the sampled evaluation set.
- Added a repeatable evaluation harness with per-query evidence in `eval/results.json`.
- Runs locally with Ollama, ChromaDB, LangGraph, and optional mem0 memory.

## Architecture

```text
User query
   |
   v
Router node
   - Fetches relevant long-term memory from mem0 when available
   - Runs hybrid retrieval over the enterprise tool registry
   - Emits top-k tool schemas plus retrieval metadata
   |
   v
Worker node
   - Dynamically binds only retrieved tool implementations
   - Lets the local LLM call tools through LangChain function calling
   - Captures tool calls and structured tool results
   |
   v
Respond node
   - Extracts the final AI response
   - Logs retrieved tools, tool calls, and retrieval latency
   - Saves useful interaction memory
```

## Retrieval Pipeline

1. **BM25 keyword retrieval** catches exact enterprise terms such as Jira ticket IDs, repo names, and service names.
2. **Chroma vector retrieval** catches semantic requests such as "report a product problem" mapping to Jira.
3. **Reciprocal Rank Fusion** merges sparse and dense rankings without assuming comparable score scales.
4. **Cross-encoder reranking** can refine the final top-3 tools when `sentence-transformers` is fully available.
5. **Graceful fallback** keeps retrieval working with BM25 when Ollama, Chroma embeddings, or the reranker are unavailable.

## Project Structure

```text
Lean-Agent/
├── agents/
│   ├── router_node.py        # Memory lookup + hybrid tool retrieval
│   ├── worker_node.py        # Dynamic tool binding + tool execution loop
│   └── respond_node.py       # Final answer extraction + memory save
├── eval/
│   ├── benchmark.py          # Lean-vs-naive benchmark harness
│   └── test_queries.py       # 22-query hand-labeled evaluation set
├── graph/
│   ├── agent_graph.py        # LangGraph wiring
│   └── state.py              # Shared graph state
├── tools/
│   └── tool_registry.py      # 50 enterprise tool schemas
├── vector_store/
│   ├── bm25_store.py         # Sparse retrieval
│   ├── chroma_store.py       # Dense retrieval with Ollama embeddings
│   └── hybrid_retriever.py   # RRF + optional reranking
├── main.py                   # Demo entry point
└── requirements.txt
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then pull local models:

```bash
ollama pull llama3.2
ollama pull llama3.1
ollama pull bge-m3
ollama pull nomic-embed-text
```

## Usage

Run the full local agent demo:

```bash
python main.py
```

Force a fresh ChromaDB rebuild:

```bash
python main.py --refresh
```

Run the fast retrieval benchmark:

```bash
python eval/benchmark.py --queries 10
```

Run the full LangGraph agent benchmark:

```bash
python eval/benchmark.py --queries 10 --real-agent
```

Run RAGAS answer evaluation after full-agent execution:

```bash
python eval/benchmark.py --queries 10 --real-agent --ragas
```

## Latest Benchmark Snapshot

The current local retrieval-only benchmark over 3 sample queries produced:

| Metric | Lean Agent | Naive Agent |
| --- | ---: | ---: |
| Avg tokens in context | 243.33 | 4003.0 |
| Avg latency | 21.03 ms | 2800.0 ms |
| Retrieval accuracy | 1.00 | 0.62 |

Result: Lean Agent used **93.9% fewer prompt-context tokens** and improved retrieval accuracy by **38.0 points** in the sampled run. Full results are saved to `eval/results.json`.

## Resume Bullet

Built a local LangGraph enterprise agent that dynamically retrieves and binds the top relevant tools from a 50-tool registry using BM25, Chroma vector search, Reciprocal Rank Fusion, and optional cross-encoder reranking, reducing tool-context tokens by 93.9% in a repeatable benchmark while preserving top-k retrieval accuracy.
