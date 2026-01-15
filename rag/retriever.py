# rag/retriever.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


RAG_DIR = Path(__file__).resolve().parent
STORE_DIR = RAG_DIR / "rag_store"

INDEX_PATH = STORE_DIR / "faiss.index"
DOCS_PATH = STORE_DIR / "docs.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SearchHit:
    score: float
    text: str
    meta: Dict[str, Any]


class RagRetriever:
    """
    Loads:
    - FAISS index (vectors)
    - docs.jsonl (text + metadata aligned by row id)
    Then supports semantic search.
    """

    def __init__(self) -> None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"Missing index: {INDEX_PATH}. Run: python -m rag.index_all")
        if not DOCS_PATH.exists():
            raise FileNotFoundError(f"Missing docs file: {DOCS_PATH}. Run: python -m rag.index_all")

        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))

        # Load docs (order must match FAISS insertion order)
        self.docs: List[Dict[str, Any]] = []
        with DOCS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))

        if self.index.ntotal != len(self.docs):
            raise RuntimeError(
                f"Index/doc mismatch: FAISS has {self.index.ntotal} vectors but docs.jsonl has {len(self.docs)} rows."
            )

    def _embed(self, query: str) -> np.ndarray:
        vec = self.model.encode([query], normalize_embeddings=True)
        return vec.astype("float32")

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, str]] = None,
    ) -> List[SearchHit]:
        """
        Semantic search over all documents.
        Optional 'where' filter by meta fields (client-side filter).
        Example:
          where={"doc_type": "protocol", "protocol_id": "NCT05928572_CGM_Initiation"}
        """
        qv = self._embed(query)

        # We may need to fetch more than top_k if we plan to filter
        fetch_k = max(top_k, 30) if where else top_k

        scores, ids = self.index.search(qv, fetch_k)
        scores = scores[0].tolist()
        ids = ids[0].tolist()

        hits: List[SearchHit] = []
        for score, idx in zip(scores, ids):
            if idx == -1:
                continue
            d = self.docs[idx]
            meta = d.get("meta", {}) or {}

            if where:
                ok = True
                for k, v in where.items():
                    if str(meta.get(k, "")) != str(v):
                        ok = False
                        break
                if not ok:
                    continue

            hits.append(SearchHit(score=float(score), text=d.get("text", ""), meta=meta))
            if len(hits) >= top_k:
                break

        return hits

    def get_protocol_chunks(self, protocol_id: str, top_k: int = 8) -> List[SearchHit]:
        """
        Pull the most relevant chunks for a specific protocol (by protocol_id).
        """
        return self.search(
            query=f"Eligibility criteria inclusion exclusion for protocol {protocol_id}",
            top_k=top_k,
            where={"doc_type": "protocol", "protocol_id": protocol_id},
        )

    def search_patients(self, query: str, top_k: int = 10) -> List[SearchHit]:
        """
        Search across patient structured + patient note docs.
        """
        # We do not filter doc_type here so it can return both structured and notes,
        # but we can restrict if you want later.
        return self.search(query=query, top_k=top_k, where=None)
