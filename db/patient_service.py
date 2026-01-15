# db/patient_service.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

# RAG retriever
from rag.retriever import RagRetriever


# ============================================================
# Paths (fits your structure)
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../Final_project
RAG_DATA_DIR = PROJECT_DIR / "rag_data"
PATIENTS_DIR = RAG_DATA_DIR / "patients"
PROTOCOLS_DIR = RAG_DATA_DIR / "protocols"

DB_DIR = PROJECT_DIR / "db_store"
DB_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS_JSON = DB_DIR / "patients.json"
PROTOCOLS_JSON = DB_DIR / "protocols.json"


# ============================================================
# Helpers: IO
# ============================================================

def _load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _ensure_dirs() -> None:
    PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)

def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _normalize_sex(value: Any) -> str:
    if value is None:
        return ""
    v = str(value).strip().lower()
    if v in ["m", "male"]:
        return "Male"
    if v in ["f", "female"]:
        return "Female"
    return str(value).strip()

def _is_missing(value: Any) -> bool:
    v = str(value).strip().lower()
    return v in ["", "none", "nan", "null"]

def _read_bytes_as_text(file_bytes: bytes, filename: str) -> str:
    """
    TXT/MD: decode
    PDF/DOCX: we don't parse here; we index via rag/index_all (already handles those)
    For extraction (age/sex), we only do reliable extraction from TXT/MD.
    """
    suf = Path(filename).suffix.lower()
    if suf in [".txt", ".md"]:
        return _clean_text(file_bytes.decode("utf-8", errors="ignore"))
    return ""

def _extract_age_sex_from_text(text: str) -> Tuple[Optional[int], str]:
    """
    Simple heuristic extraction (non-blocking):
    - Age: look for patterns like "Age: 54", "54 years old", "בן 54", "בת 54"
    - Sex: look for "sex: male/female", "gender: m/f", "Male/Female"
    """
    if not text:
        return None, ""

    t = text.lower()

    # age
    age = None
    age_patterns = [
        r"\bage\s*[:\-]\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*(?:years old|y/o|yo)\b",
        r"\bבן\s*(\d{1,3})\b",
        r"\bבת\s*(\d{1,3})\b",
    ]
    for pat in age_patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            try:
                age = int(m.group(1))
                break
            except:
                pass

    # sex
    sex = ""
    sex_patterns = [
        r"\bsex\s*[:\-]\s*(male|female|m|f)\b",
        r"\bgender\s*[:\-]\s*(male|female|m|f)\b",
        r"\b(male|female)\b",
    ]
    for pat in sex_patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            sex = _normalize_sex(m.group(1))
            break

    return age, sex


# ============================================================
# Auto IDs
# ============================================================

def _next_patient_id(existing_ids: List[str]) -> str:
    """
    Generate P001, P002...
    """
    max_n = 0
    for pid in existing_ids:
        m = re.match(r"^P(\d+)$", str(pid).strip().upper())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"P{max_n + 1:03d}"


# ============================================================
# Gemini LLM
# ============================================================

def _get_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY environment variable.\n"
            "Set it in PyCharm: Run -> Edit Configurations -> Environment variables\n"
            "Example: GEMINI_API_KEY=YOUR_KEY"
        )
    return key

def _init_gemini():
    genai.configure(api_key=_get_gemini_key())
    # model you already use / can use:
    return genai.GenerativeModel("gemini-2.5-flash")

def _safe_json_load(s: str) -> Dict[str, Any]:
    """
    Gemini sometimes wraps JSON in ```json ... ```
    """
    if not s:
        return {}
    s = s.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except:
        # last resort: find first { ... } block
        m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except:
                return {}
        return {}


def _llm_match_one(
    model,
    protocol_id: str,
    protocol_context: str,
    patient_id: str,
    patient_context: str,
) -> Dict[str, Any]:
    """
    Returns:
    {
      decision: "Eligible"|"Not eligible"|"Uncertain",
      confidence: float 0..1,
      reason: str (one sentence, evidence-based),
      missing_fields: ["age","sex"] optional,
      evidence_protocol: [snippets],
      evidence_patient: [snippets]
    }
    """

    prompt = f"""
You are a clinical trial recruitment assistant.

TASK:
Given a clinical trial protocol context and a patient context, decide if the patient is Eligible, Not eligible, or Uncertain.
You must be conservative: if key data is missing, choose "Uncertain" and list missing fields.

OUTPUT RULES:
Return ONLY valid JSON with exactly these keys:
- decision: "Eligible" | "Not eligible" | "Uncertain"
- confidence: number between 0 and 1
- reason: one short sentence explaining WHY (must cite evidence from the provided contexts)
- missing_fields: list of strings (e.g., ["age","sex"]) or []
- evidence_protocol: list of 1-3 short snippets copied from PROTOCOL context
- evidence_patient: list of 1-3 short snippets copied from PATIENT context

IMPORTANT:
- Do not invent facts.
- Use only the text given below.
- If unsure or missing data, decision must be "Uncertain".

PROTOCOL_ID: {protocol_id}

PROTOCOL CONTEXT:
{protocol_context}

PATIENT_ID: {patient_id}

PATIENT CONTEXT:
{patient_context}
""".strip()

    resp = model.generate_content(prompt)
    text = resp.text if hasattr(resp, "text") else str(resp)
    data = _safe_json_load(text)

    # harden defaults
    decision = data.get("decision", "Uncertain")
    if decision not in ["Eligible", "Not eligible", "Uncertain"]:
        decision = "Uncertain"

    conf = data.get("confidence", 0.0)
    try:
        conf = float(conf)
    except:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    out = {
        "decision": decision,
        "confidence": conf,
        "reason": str(data.get("reason", "")).strip(),
        "missing_fields": data.get("missing_fields") if isinstance(data.get("missing_fields"), list) else [],
        "evidence_protocol": data.get("evidence_protocol") if isinstance(data.get("evidence_protocol"), list) else [],
        "evidence_patient": data.get("evidence_patient") if isinstance(data.get("evidence_patient"), list) else [],
    }
    return out


# ============================================================
# Public API used by Streamlit
# ============================================================

def list_protocols() -> List[Dict[str, Any]]:
    _ensure_dirs()
    return _load_json(PROTOCOLS_JSON, [])

def list_patients_summary() -> List[Dict[str, Any]]:
    _ensure_dirs()
    patients = _load_json(PATIENTS_JSON, [])
    # summary only
    out = []
    for p in patients:
        out.append({
            "patient_id": p.get("patient_id", ""),
            "age": p.get("age", ""),
            "sex": _normalize_sex(p.get("sex", "")),
        })
    return out

def add_protocol(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Save protocol file under rag_data/protocols and trigger re-index (RAG).
    protocol_id is derived from file name stem (no manual ID).
    """
    _ensure_dirs()

    p = Path(filename)
    protocol_id = p.stem.strip()

    # save file
    save_path = PROTOCOLS_DIR / filename
    save_path.write_bytes(file_bytes)

    # title heuristic: first markdown heading, else protocol_id
    title = protocol_id
    if p.suffix.lower() in [".md", ".txt"]:
        txt = _read_bytes_as_text(file_bytes, filename)
        m = re.search(r"^#\s+(.+)$", txt, flags=re.MULTILINE)
        if m:
            title = m.group(1).strip()

    protocols = _load_json(PROTOCOLS_JSON, [])
    # upsert
    protocols = [x for x in protocols if x.get("protocol_id") != protocol_id]
    protocols.append({"protocol_id": protocol_id, "title": title, "file": filename})
    protocols.sort(key=lambda x: x["protocol_id"])
    _save_json(PROTOCOLS_JSON, protocols)

    # re-index all (simple + reliable)
    from rag.index_all import index_all
    index_all()

    return {"protocol_id": protocol_id, "title": title, "chunks_added": "reindexed"}

def add_patient_note(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Create patient automatically (P001...) and save note under rag_data/patients/.
    No manual fields.
    Extract age/sex only if TXT/MD has them; otherwise keep empty and warn.
    """
    _ensure_dirs()

    patients = _load_json(PATIENTS_JSON, [])
    existing_ids = [p.get("patient_id", "") for p in patients if p.get("patient_id")]
    patient_id = _next_patient_id(existing_ids)

    suf = Path(filename).suffix.lower()
    # save file (prefix with patient id so we can later associate easily)
    safe_name = f"{patient_id}__{Path(filename).name}"
    save_path = PATIENTS_DIR / safe_name
    save_path.write_bytes(file_bytes)

    # extract (only from text types)
    txt = _read_bytes_as_text(file_bytes, filename)
    age, sex = _extract_age_sex_from_text(txt)
    sex = _normalize_sex(sex)

    missing_fields = []
    if age is None:
        missing_fields.append("age")
    if not sex:
        missing_fields.append("sex")

    # store patient record (structured “behind the scenes”)
    patients.append({
        "patient_id": patient_id,
        "age": age if age is not None else "",
        "sex": sex,
        "files": [safe_name],
    })
    _save_json(PATIENTS_JSON, patients)

    # index
    from rag.index_all import index_all
    index_all()

    # tiny preview only if TXT
    txt_preview = ""
    if suf == ".txt" and txt:
        txt_preview = txt[:500].strip()

    return {
        "patient_id": patient_id,
        "chunks_added": "reindexed",
        "missing_fields": missing_fields,
        "txt_preview": txt_preview,
    }


def find_candidates(protocol_id: str, top_k: int = 5) -> Dict[str, Any]:
    """
    LLM + RAG pipeline:
    1) RAG prefilter: retrieve patient chunks relevant to protocol
    2) Aggregate by patient_id -> candidate list
    3) For each candidate: gather patient context + protocol context
    4) Gemini decides + produces reason + evidence snippets
    """
    _ensure_dirs()

    protocols = _load_json(PROTOCOLS_JSON, [])
    prot = next((p for p in protocols if p.get("protocol_id") == protocol_id), None)
    if not prot:
        return {"results": []}

    # 1) init retriever (loads FAISS + docs.jsonl)
    retriever = RagRetriever()

    title = prot.get("title", protocol_id)
    protocol_query = f"{protocol_id} {title} inclusion criteria exclusion criteria eligibility"

    # 2) prefilter patient chunks using RAG (no LLM yet)
    patient_hits = retriever.search(
        query=protocol_query,
        top_n=max(80, top_k * 25),
        filters={"doc_type": ["patient_structured", "patient_note"]},
    )

    # aggregate per patient_id
    agg: Dict[str, Dict[str, Any]] = {}
    for h in patient_hits:
        meta = h.get("meta", {}) or {}
        pid = meta.get("patient_id", "")
        if not pid:
            continue
        if pid not in agg:
            agg[pid] = {
                "patient_id": pid,
                "score": 0.0,
                "evidence_patient": [],
            }
        agg[pid]["score"] = max(agg[pid]["score"], float(h.get("score", 0.0)))
        txt = (h.get("text") or "").strip()
        if txt:
            agg[pid]["evidence_patient"].append(txt[:260])

    if not agg:
        return {"results": []}

    # rank candidates by retrieval score
    ranked = sorted(agg.values(), key=lambda x: x["score"], reverse=True)
    # take more than needed for LLM rerank
    preselect_n = min(len(ranked), max(top_k * 3, 15))
    candidates = ranked[:preselect_n]

    # 3) protocol context from RAG (protocol chunks only)
    protocol_hits = retriever.search(
        query=f"{protocol_id} inclusion exclusion criteria",
        top_n=8,
        filters={"doc_type": ["protocol"], "protocol_id": [protocol_id]},
    )
    protocol_context = "\n\n".join([(x.get("text") or "")[:500] for x in protocol_hits if x.get("text")])[:2500]
    protocol_evidence_snips = [(x.get("text") or "")[:220] for x in protocol_hits if x.get("text")][:3]

    # 4) LLM evaluate each candidate
    model = _init_gemini()

    patients_db = _load_json(PATIENTS_JSON, [])
    pat_map = {p.get("patient_id"): p for p in patients_db}

    results: List[Dict[str, Any]] = []

    for c in candidates:
        pid = c["patient_id"]

        # Build patient context:
        # (a) use the top retrieved evidence we already have
        patient_context = "\n\n".join(c.get("evidence_patient", [])[:6])

        # (b) add structured summary if exists in db
        p_rec = pat_map.get(pid, {}) or {}
        age = p_rec.get("age", "")
        sex = _normalize_sex(p_rec.get("sex", ""))

        structured_lines = [f"Patient ID: {pid}"]
        if not _is_missing(age):
            structured_lines.append(f"Age: {age}")
        if sex:
            structured_lines.append(f"Sex: {sex}")
        structured_block = "\n".join(structured_lines)

        patient_context = (structured_block + "\n\n" + patient_context).strip()[:2500]

        llm_out = _llm_match_one(
            model=model,
            protocol_id=protocol_id,
            protocol_context=protocol_context,
            patient_id=pid,
            patient_context=patient_context,
        )

        # Ensure missing fields also reflect stored record
        missing_fields = llm_out.get("missing_fields", [])
        if _is_missing(age) and "age" not in missing_fields:
            missing_fields.append("age")
        if not sex and "sex" not in missing_fields:
            missing_fields.append("sex")

        results.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "decision": llm_out.get("decision", "Uncertain"),
            "confidence": llm_out.get("confidence", 0.0),
            "reason": llm_out.get("reason", ""),
            "missing_fields": missing_fields,
            "evidence_protocol": llm_out.get("evidence_protocol") or protocol_evidence_snips,
            "evidence_patient": llm_out.get("evidence_patient") or c.get("evidence_patient", [])[:3],
        })

    # Final ranking:
    # prioritize Eligible > Uncertain > Not eligible, then confidence
    order = {"Eligible": 2, "Uncertain": 1, "Not eligible": 0}
    results.sort(key=lambda r: (order.get(r["decision"], 1), float(r.get("confidence", 0.0))), reverse=True)

    # cut to top_k
    results = results[:top_k]

    return {"results": results}
