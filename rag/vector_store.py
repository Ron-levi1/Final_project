import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class StoredDoc:
    doc_id: str
    source: str              # e.g., "protocols/DECLARE_TIMI58.md" or "patients/patients.csv"
    kind: str                # "protocol" or "patient"
    title: str               # short human title
    text: str                # the chunk text
    meta: Dict[str, Any]     # any extra metadata (patient_id, protocol_id, etc.)


class VectorStore:
    """
    Simple local FAISS vector store (cosine similarity via normalized embeddings).
    Persists:
      - index.faiss
      - meta.json  (list of StoredDoc dicts aligned with FAISS ids)
      - config.json (model name, dim)
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.docs: List[StoredDoc] = []
        self.dim: Optional[int] = None

    def _ensure_model(self) -> None:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        # Normalize rows to unit length for cosine similarity
        norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / norm

    def build_new(self, dim: int) -> None:
        # Cosine similarity via inner product on normalized vectors
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.docs = []

    def add_documents(self, docs: List[StoredDoc]) -> None:
        if not docs:
            return

        self._ensure_model()
        texts = [d.text for d in docs]
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)

        if self.index is None:
            self.build_new(dim=emb.shape[1])

        if self.dim != emb.shape[1]:
            raise ValueError(f"Embedding dimension mismatch: store dim={self.dim}, new dim={emb.shape[1]}")

        emb = self._normalize(emb.astype("float32"))
        self.index.add(emb)
        self.docs.extend(docs)

    def search(self, query: str, k: int = 5) -> List[Tuple[StoredDoc, float]]:
        if self.index is None or not self.docs:
            return []

        self._ensure_model()
        q = self.model.encode([query], convert_to_numpy=True)
        q = q.astype("float32")
        q = self._normalize(q)

        scores, ids = self.index.search(q, k)
        results: List[Tuple[StoredDoc, float]] = []

        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.docs[int(idx)], float(score)))

        return results

    def save(self, folder: str) -> None:
        if self.index is None:
            raise RuntimeError("Nothing to save: index is empty / not built.")

        os.makedirs(folder, exist_ok=True)

        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))

        meta_path = os.path.join(folder, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([self._doc_to_dict(d) for d in self.docs], f, ensure_ascii=False, indent=2)

        cfg = {"model_name": self.model_name, "dim": self.dim}
        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder: str) -> "VectorStore":
        cfg_path = os.path.join(folder, "config.json")
        meta_path = os.path.join(folder, "meta.json")
        index_path = os.path.join(folder, "index.faiss")

        if not (os.path.exists(cfg_path) and os.path.exists(meta_path) and os.path.exists(index_path)):
            raise FileNotFoundError(f"Missing store files in {folder}. Expected index.faiss, meta.json, config.json")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        store = cls(model_name=cfg["model_name"])
        store.dim = cfg.get("dim")

        store.index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        store.docs = [cls._dict_to_doc(x) for x in raw]

        return store

    @staticmethod
    def _doc_to_dict(d: StoredDoc) -> Dict[str, Any]:
        return {
            "doc_id": d.doc_id,
            "source": d.source,
            "kind": d.kind,
            "title": d.title,
            "text": d.text,
            "meta": d.meta,
        }

    @staticmethod
    def _dict_to_doc(x: Dict[str, Any]) -> StoredDoc:
        return StoredDoc(
            doc_id=x["doc_id"],
            source=x["source"],
            kind=x["kind"],
            title=x.get("title", ""),
            text=x["text"],
            meta=x.get("meta", {}),
        )
