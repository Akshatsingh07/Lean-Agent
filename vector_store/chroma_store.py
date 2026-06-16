import os
import chromadb
from chromadb.utils import embedding_functions
from tools.tool_registry import get_all_tools, ToolSchema
import json
from langchain_ollama import OllamaEmbeddings

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME= "enterprise_tools"

def get_embedding_func():
    return embedding_functions.OllamaEmbeddingFunction(
        model_name="bge-m3",
        url="http://localhost:11434/api/embeddings"
    )
    
def chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def embed_all_tools(force_refresh: bool=False) -> chromadb.Collection:
    client=chroma_client()
    embedding_fnc=get_embedding_func()
    
    if force_refresh:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Deleted the existing collection")
        except Exception:
            pass
        
    collection=client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fnc,
        metadata={"hnsw:space":"cosine"}
    )
    existing_count=collection.count()
    if existing_count>0 and not force_refresh:
        return collection
    
    tools=get_all_tools()
    ids=[]
    documents=[]
    metadatas=[]
    
    for tool in tools:
        new_text=(
            f"{tool.name} |"
            f"{tool.description} |"
            f"{tool.category} |"
            f"params: {tool.parameters} |"
            f"returns: {tool.returns}"
        )
        ids.append(tool.name)
        documents.append(new_text)
        metadatas.append({
            "name":tool.name,
            "description":tool.description,
            "parameters":json.dumps(tool.parameters),
            "return":tool.returns,
            "category":tool.category,
        })
        print(f"Embedding {len(tools)}")
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"✓ Successfully embedded {len(tools)} tools into ChromaDB!")
        print(f"  Saved to: {CHROMA_PERSIST_DIR}/")
        return collection
    
def vector_search(query: str,n_results: int=10) -> list[dict]:
    client=chroma_client()
    embedding_funcs=get_embedding_func()
    collection=client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_funcs
    )
    results=collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["metadatas","distances","documents"]
    )
    
    retrieved_tools=[]
    for i in range(len(results["ids"][0])):
        similarity_score=1-results["distances"][0][i]
        tool_data={
            "name":results["metadatas"][0][i]["name"],
            "description":results["metadatas"][0][i]["description"],
            "category":results["metadatas"][0][i]["category"],
            "parameters":json.loads(results["metadatas"][0][i]["parameters"]),
            "returns":results["metadatas"][0][i]["return"],
            "similarity_score":round(similarity_score,4),
            "rank":i+1
        }
        retrieved_tools.append(tool_data)
    return retrieved_tools

if __name__=="__main__":
    embed_all_tools(force_refresh=True)
    
    test_queries = [
        "I need to create a bug report for a login page crash",
        "What are our revenue metrics for last quarter?",
        "Is our production deployment healthy? Check for errors",
    ]
    
    for query in test_queries:
        tools_found=vector_search(query,n_results=3)
        for t in tools_found:
             print(f"[{t['rank']}] {t['name']:35} (score: {t['similarity_score']:.3f}) [{t['category']}]")