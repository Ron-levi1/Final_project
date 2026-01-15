# rag/matcher.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from agent.gemini_judge import judge_with_gemini, LLMMatchResult


# -------------------------
# Paths (MUST match your structure)
# -------------------------
RAG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = RAG_DIR.parent

STORE_DIR = RAG_DIR / "rag_store"
INDEX_PATH = STORE_DIR / "faiss.index"
DOCS_PATH = STORE_DIR / "docs.jsonl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class DocItem:
    text: str
    meta: Dict[str, Any]


def _load_docs() -> List[DocItem]:
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Missing docs.jsonl at: {DOCS_PATH}. Run: python -m rag.index_all")

    docs: List[DocItem] = []
    with DOCS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(DocItem(text=obj["text"], meta=obj.get("meta", {})))
    return docs


def _load_index() -> faiss.Index:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index at: {INDEX_PATH}. Run: python -m rag.index_all")
    return faiss.read_index(str(INDEX_PATH))


def _embed(model: SentenceTransformer, text: str) -> np.ndarray:
    v = model.encode([text], normalize_embeddings=True)
    return v.astype("float32")


def _search(index: faiss.Index, qvec: np.ndarray, top_n: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(qvec, top_n)
    return scores[0], ids[0]


# -------------------------
# Public helpers for Streamlit
# -------------------------
def list_protocols() -> List[Dict[str, str]]:
    """
    Returns list of protocols for dropdown:
    [{"protocol_id": "...", "label": "..."}]
    """
    docs = _load_docs()
    seen = {}
    for d in docs:
        if d.meta.get("doc_type") == "protocol":
            pid = d.meta.get("protocol_id", "")
            if not pid:
                continue
            if pid not in seen:
                # build a nice label from first chunk
                first_line = (d.text.splitlines()[0] if d.text else "").strip()
                label = f"{pid} — {first_line[:80]}" if first_line else pid
                seen[pid] = {"protocol_id": pid, "label": label}
    return sorted(seen.values(), key=lambda x: x["protocol_id"])


def _protocol_evidence_for(pid: str, docs: List[DocItem], model: SentenceTransformer, index: faiss.Index) -> List[str]:
    """
    Get top protocol snippets relevant to itself (lightweight) for evidence.
    """
    # Take a “protocol query” from its own chunks: ID + keywords
    proto_chunks = [d for d in docs if d.meta.get("doc_type") == "protocol" and d.meta.get("protocol_id") == pid]
    if not proto_chunks:
        return []

    # Use first 2 chunks as query seed
    seed = "\n".join([c.text for c in proto_chunks[:2]])
    q = f"Protocol {pid}. Key eligibility criteria. {seed[:800]}"

    qvec = _embed(model, q)
    scores, ids = _search(index, qvec, top_n=30)

    ev = []
    for s, i in zip(scores, ids):
        if i == -1:
            continue
        d = docs[i]
        if d.meta.get("doc_type") == "protocol" and d.meta.get("protocol_id") == pid:
            snippet = d.text.strip().replace("\n", " ")
            ev.append(snippet[:250])
        if len(ev) >= 6:
            break
    return ev


def match_patients_to_protocol(protocol_id: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Main function Streamlit should call.

    Returns list of dict rows:
    {
      patient_id, age, sex,
      decision, confidence,
      reason_short,
      missing_info,
      evidence_protocol: [...],
      evidence_patient: [...]
    }
    """
    docs = _load_docs()
    index = _load_index()
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 1) Build query from protocol chunks (RAG query)
    proto_chunks = [d for d in docs if d.meta.get("doc_type") == "protocol" and d.meta.get("protocol_id") == protocol_id]
    if not proto_chunks:
        return []

    proto_text = "\n".join([c.text for c in proto_chunks[:3]])[:1200]
    query = f"Find patient candidates for protocol {protocol_id}. Use inclusion and exclusion criteria. {proto_text}"

    # 2) Retrieve top hits, then keep only patient docs (notes + structured)
    qvec = _embed(model, query)
    scores, ids = _search(index, qvec, top_n=80)

    # Aggregate by patient_id (so we return unique patients)
    patient_best: Dict[str, float] = {}
    patient_evidence: Dict[str, List[str]] = {}

    for s, i in zip(scores, ids):
        if i == -1:
            continue
        d = docs[i]
        dt = d.meta.get("doc_type")
        if dt not in ("patient_note", "patient_structured"):
            continue

        pid = d.meta.get("patient_id", "UNKNOWN")
        patient_best[pid] = max(patient_best.get(pid, -1.0), float(s))

        snippet = d.text.strip().replace("\n", " ")
        patient_evidence.setdefault(pid, []).append(snippet[:250])

    # pick top-k patients
    ranked = sorted(patient_best.items(), key=lambda x: x[1], reverse=True)[: max(k, 1)]

    # 3) Build protocol evidence (quotes)
    proto_evidence = _protocol_evidence_for(protocol_id, docs, model, index)

    results: List[Dict[str, Any]] = []
    for pid, _score in ranked:
        # patient summary from structured doc if exists
        summary = {"patient_id": pid, "age": None, "sex": None, "has_note": False}

        for d in docs:
            if d.meta.get("doc_type") == "patient_structured" and d.meta.get("patient_id") == pid:
                # Try extract Age/Sex lines from structured text
                lines = d.text.splitlines()
                for ln in lines:
                    if ln.lower().startswith("age:"):
                        summary["age"] = ln.split(":", 1)[1].strip()
                    if ln.lower().startswith("sex:"):
                        summary["sex"] = ln.split(":", 1)[1].strip()
                break

        # has note?
        for d in docs:
            if d.meta.get("doc_type") == "patient_note" and d.meta.get("patient_id") == pid:
                summary["has_note"] = True
                break

        # Basic missing warnings (your requirement)
        missing_critical = []
        if not summary.get("age"):
            missing_critical.append("age")
        if not summary.get("sex"):
            missing_critical.append("sex")

        patient_ev = patient_evidence.get(pid, [])[:6]

        llm: LLMMatchResult = judge_with_gemini(
            protocol_id=protocol_id,
            protocol_evidence=proto_evidence,
            patient_id=pid,
            patient_summary=summary,
            patient_evidence=patient_ev,
        )

        # merge missing info
        merged_missing = list(dict.fromkeys(missing_critical + (llm.missing_info or [])))

        results.append(
            {
                "patient_id": pid,
                "age": summary.get("age") or "",
                "sex": summary.get("sex") or "",
                "decision": llm.decision,
                "confidence": round(llm.confidence, 2),
                "reason_short": llm.reason_short,
                "missing_info": merged_missing,
                "evidence_protocol": proto_evidence,
                "evidence_patient": patient_ev,
                "criteria": [c.model_dump() for c in llm.criteria],
            }
        )

    return results
