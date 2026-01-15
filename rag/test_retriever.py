# rag/test_retriever.py
from rag.retriever import RagRetriever

if __name__ == "__main__":
    r = RagRetriever()

    hits = r.search("inclusion criteria diabetes exclusion pregnancy", top_k=3, where={"doc_type": "protocol"})
    for h in hits:
        print("---")
        print("SCORE:", h.score)
        print("META :", h.meta)
        print("TEXT :", h.text[:300])
