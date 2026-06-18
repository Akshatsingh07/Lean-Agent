# Lean Agent

A 100% local, air-gapped AI agent designed to execute enterprise tools efficiently. By utilizing an advanced Hybrid RAG pipeline and a LangGraph architecture, Lean Agent dynamically retrieves and binds only the necessary tools for each query—drastically reducing the LLM context window, eliminating hallucinations, and cutting out cloud API latency.

## 🚀 The Proof (Benchmark Results)

Compared to a "naive" agent that loads all 50+ enterprise tools into the prompt context, Lean Agent performs significantly better on local hardware (tested with `llama3.1`):

* **Context Window:** 📉 93.7% fewer tokens (~190 vs 3,000+)
* **Retrieval Accuracy:** 🎯 100% accuracy (vs 62% naive guessing)
* **Latency:** ⚡ ~2–5 seconds per query (post cold-start)
* **Privacy:** 🔒 100% local execution. Zero data leaves your machine.

## 🧠 Architecture & Tech Stack

Lean Agent operates on a deterministic graph loop:
1. **Memory Retrieval:** Extracts long-term user context.
2. **Hybrid Search:** Combines semantic search with exact keyword matching.
3. **Re-ranking:** Scores and isolates the top 3 tools from a registry of 50+.
4. **Execution:** Binds the isolated tools to a local LLM for strict JSON schema output and function calling.

**The Stack:**
* **Orchestration:** LangGraph, LangChain
* **LLM & Embeddings:** Ollama (`llama3.1`, `nomic-embed-text`)
* **Vector Database:** ChromaDB
* **Sparse Retrieval:** BM25 (FastEmbed)
* **Cross-Encoder:** HuggingFace Sentence-Transformers (`ms-marco-MiniLM-L-6-v2`)
* **Long-Term Memory:** Mem0 (Qdrant)

## 📂 Project Structure

```text
lean-agent/
├── tools/
│   ├── tool_registry.py       # 50+ mock enterprise tool schemas
│   └── tool_embedder.py       # Embeds schemas into ChromaDB
├── vector_store/
│   ├── chroma_store.py        # Dense vector search
│   ├── bm25_store.py          # Sparse keyword search
│   └── hybrid_retriever.py    # RRF fusion + Cross-Encoder reranking
├── agents/
│   ├── router_node.py         # Mem0 retrieval + Hybrid Tool RAG
│   └── worker_node.py         # Dynamic LLM binding + Tool Execution
│   └── respond_node.py         #Final Answer 
├── graph/
│   └── agent_graph.py         # LangGraph state wiring
├── eval/
│   └── test_queries.py        # Evaluation datasets
│   └── benchmark.py            # Evaluation using Ragas with lean agent and naive agent
├── main.py                    # Entry point & execution
└── requirements.txt
```

⚙️ Prerequisites & Setup
Install Ollama and pull the required local models:

Bash
ollama pull llama3.1
ollama pull nomic-embed-text
Clone and Install:

Bash
git clone (https://github.com/Akshatsingh07/Lean-Agent.git)
cd Lean-Agent
pip install -r requirements.txt
(Note: The first run will automatically download the local BM25 and Cross-Encoder weights from HuggingFace).

💻 Usage
Run the Agent Interactive Pipeline:

Bash
python main.py
Run the Local Evaluator (Benchmark + RAGAS):
Test the hybrid retrieval accuracy and evaluate the local LLM's faithfulness and context precision without any OpenAI keys.

Bash
python main.py --benchmark