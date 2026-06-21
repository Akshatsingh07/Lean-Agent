import json

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from tools.tool_registry import get_all_tools

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "enterprise_tools"
EMBEDDING_MODEL = "bge-m3"


def get_embedding_func():
    return embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        url="http://localhost:11434/api/embeddings"
    )
    
def chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_needs_refresh(collection: chromadb.Collection) -> bool:
    tools = get_all_tools()
    if collection.count() != len(tools):
        return True

    try:
        sample = collection.peek(1)
    except Exception:
        return True

    metadatas = sample.get("metadatas") or []
    if not metadatas:
        return True

    return "returns" not in metadatas[0]


def embed_all_tools(force_refresh: bool = False) -> chromadb.Collection:
    client = chroma_client()
    embedding_fnc = get_embedding_func()
    
    if force_refresh:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted the existing collection")
        except Exception:
            pass
        
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fnc,
        metadata={"hnsw:space": "cosine"}
    )
    existing_count = collection.count()
    if existing_count > 0 and not force_refresh and not _collection_needs_refresh(collection):
        print(f"ChromaDB already contains {existing_count} embedded tools.")
        return collection

    if existing_count > 0 and not force_refresh:
        print("Refreshing ChromaDB tool collection due to stale or incomplete metadata.")
    
    tools = get_all_tools()
    ids = []
    documents = []
    metadatas = []
    
    for tool in tools:
        document = (
            f"{tool.name} | "
            f"{tool.name.replace('_', ' ')} | "
            f"{tool.description} | "
            f"{tool.category} | "
            f"params: {tool.parameters} |"
            f"returns: {tool.returns}"
        )
        ids.append(tool.name)
        documents.append(document)
        metadatas.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": json.dumps(tool.parameters),
            "returns": tool.returns,
            "category": tool.category,
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Successfully embedded {len(tools)} tools into ChromaDB.")
    print(f"Saved to: {CHROMA_PERSIST_DIR}/")
    return collection


def _parse_parameters(raw_parameters: str | dict | None) -> dict:
    if isinstance(raw_parameters, dict):
        return raw_parameters
    if not raw_parameters:
        return {}
    return json.loads(raw_parameters)
    

def vector_search(query: str, n_results: int = 10) -> list[dict]:
    collection = embed_all_tools()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas", "distances", "documents"]
    )
    
    retrieved_tools = []
    for i in range(len(results["ids"][0])):
        metadata = results["metadatas"][0][i]
        similarity_score = 1 - results["distances"][0][i]
        tool_data = {
            "name": metadata["name"],
            "description": metadata["description"],
            "category": metadata["category"],
            "parameters": _parse_parameters(metadata.get("parameters")),
            "returns": metadata.get("returns", metadata.get("return", "")),
            "similarity_score": round(similarity_score, 4),
            "vector_rank": i + 1,
        }
        retrieved_tools.append(tool_data)
    return retrieved_tools

if __name__ == "__main__":
    embed_all_tools(force_refresh=True)
    
    test_queries = [
        "I need to create a bug report for a login page crash",
        "What are our revenue metrics for last quarter?",
        "Is our production deployment healthy? Check for errors",
    ]
    
    for query in test_queries:
        tools_found = vector_search(query, n_results=3)
        for t in tools_found:
            print(f"[{t['vector_rank']}] {t['name']:35} (score: {t['similarity_score']:.3f}) [{t['category']}]")
