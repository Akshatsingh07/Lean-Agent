import re 
from rank_bm25 import BM25Okapi
from tools.tool_registry import get_all_tools,ToolSchema

def tokenize(text: str) -> list[str]:
    text=text.lower()
    text=re.sub(r'[^a-z0-9\s_]',' ',text)
    tokens=text.split()
    return [t for t in tokens if len(t)>1]

class BM25ToolStore:
    def __init__(self):
        self.tools: list[ToolSchema] = get_all_tools()
        self._raw_documents: list[str] = []
        for tool in self.tools:
            doc=(
                f"{tool.name} {tool.name.replace('_',' ')}"
                f"{tool.category}"
                f"{tool.description}"
                f"{' '.join(tool.parameters.keys())}"
                f"{tool.returns}"            
            )
            self._raw_documents.append(doc)
    
        tokenized_docs=[tokenize(doc) for doc in self._raw_documents]
        self._bm25=BM25Okapi(tokenized_docs)
        print(f"BM25 index built over{len(self.tools)}")
        
    def search(self, query:str, n_results: int=10) -> list[dict]:
        query_tokens=tokenize(query)
        scores=self._bm25.get_scores(query_tokens)
        scored=[(self.tools[i], float(scores[i])) for i in range(len(self.tools))]
        scored.sort(key=lambda x:x[1], reverse=True)
        results = []
        for rank, (tool, score) in enumerate(scored[:n_results], start=1):
            results.append({
                "name": tool.name,
                "category": tool.category,
                "description": tool.description,
                "parameters": tool.parameters,
                "returns": tool.returns,
                "bm25_score": round(score, 4),
                "bm25_rank": rank,
                "source": "bm25",
            })
 
        return results
_store:BM25ToolStore | None=None

def get_bm25_store() -> BM25ToolStore:
    global _store
    if _store is None:
        _store=BM25ToolStore()
    return _store

if __name__ == "__main__":
    store = get_bm25_store()
    test_queries = [
        "create jira bug ticket project",
        "revenue metrics last quarter ARR MRR",
        "production deployment errors logs",
    ]
    for q in test_queries:
        print(f"\n🔍 BM25 Query: '{q}'")
        for r in store.search(q, n_results=3):
            print(f"   {r['bm25_rank']}. {r['name']:35} score={r['bm25_score']:.3f}")