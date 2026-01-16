from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from rag.retriever import RagRetriever
from agent.gemini_judge import judge_with_gemini



# ============================================================
# Paths
# ============================================================

DB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DB_DIR.parent

DATA_DIR = PROJECT_DIR / "rag_data"
PROTOCOLS_DIR = DATA_DIR / "protocols"
PATIENTS_DIR = DATA_DIR / "patients"

DB_STORE = PROJECT_DIR / "db_store"
DB_STORE.mkdir(parents=True, exist_ok=True)

PROTOCOLS_JSON = DB_STORE / "protocols.json"
PATIENTS_JSON = DB_STORE / "patients.json"


# ============================================================
# Utilities
# ============================================================

def _ensure_dirs() -> None:
    DB_STORE.mkdir(parents=True, exist_ok=True)
    PROTOCOLS_DIR.mkdir(parents=True, exist_ok=True)
    PATIENTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_sex(value: Any) -> str:
    if value is None:
        return ""
    v = str(value).strip().lower()
    if v in ["f", "female"]:
        return "Female"
    if v in ["m", "male"]:
        return "Male"
    return str(value).strip()


def _patients_csv_path() -> Path:
    # Your seed file name (as in your screenshot)
    return PATIENTS_DIR / "patients_for_trial_screening.csv"


# ============================================================
# Protocols (used by UI)
# ============================================================

def list_protocols() -> List[Dict[str, Any]]:
    """
    Returns protocols for UI.
    MUST include keys used by streamlit_app:
      - protocol_id
      - title
      - file
    Seeds automatically from rag_data/protocols if JSON is empty.
    """
    _ensure_dirs()
    protocols = _load_json(PROTOCOLS_JSON, [])

    protocol_files = [
        p for p in PROTOCOLS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in [".md", ".txt", ".docx", ".pdf"]
    ]
    by_stem = {p.stem: p.name for p in protocol_files}

    changed = False

    if not protocols:
        # Seed from folder
        protocols = []
        for p in sorted(protocol_files, key=lambda x: x.stem.lower()):
            protocols.append(
                {"protocol_id": p.stem, "title": p.stem, "file": p.name}
            )
        changed = True
    else:
        # Ensure required keys exist
        for pr in protocols:
            pid = str(pr.get("protocol_id", "")).strip()
            if not pid:
                continue

            if "title" not in pr or not pr.get("title"):
                pr["title"] = pid
                changed = True

            if "file" not in pr or not pr.get("file"):
                pr["file"] = by_stem.get(pid, f"{pid}.md")
                changed = True

        protocols.sort(key=lambda x: str(x.get("protocol_id", "")).lower())

    if changed:
        _save_json(PROTOCOLS_JSON, protocols)

    return protocols


def add_protocol(protocol_id: str, title: str, file_name: str) -> None:
    """
    Used by UI upload flow (if you still have it in Protocols tab).
    This only registers in protocols.json. The actual file should be saved by the UI into rag_data/protocols.
    """
    _ensure_dirs()
    protocols = _load_json(PROTOCOLS_JSON, [])

    protocols.append(
        {"protocol_id": protocol_id, "title": title, "file": file_name}
    )
    _save_json(PROTOCOLS_JSON, protocols)


# ============================================================
# Patients (used by UI)
# ============================================================

def list_patients_summary(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Returns a preview table of patients:
      - patient_id
      - age
      - sex
    Seeds from patients_for_trial_screening.csv if patients.json is empty.
    """
    _ensure_dirs()
    patients = _load_json(PATIENTS_JSON, [])

    csv_path = _patients_csv_path()
    if not patients and csv_path.exists():
        df = pd.read_csv(csv_path)

        # Normalize common column variants
        col_map = {}
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ["patient_id", "patientid", "id"]:
                col_map[c] = "patient_id"
            elif lc in ["age", "patient_age"]:
                col_map[c] = "age"
            elif lc in ["sex", "gender"]:
                col_map[c] = "sex"
        if col_map:
            df = df.rename(columns=col_map)

        if "patient_id" not in df.columns:
            df["patient_id"] = [f"ROW_{i}" for i in range(len(df))]

        out = []
        for _, row in df.head(limit).iterrows():
            out.append(
                {
                    "patient_id": str(row.get("patient_id", "")).strip(),
                    "age": row.get("age", ""),
                    "sex": _normalize_sex(row.get("sex", "")),
                }
            )
        return out

    # If patients.json has data, keep preview minimal
    out = []
    for p in patients[:limit]:
        out.append(
            {
                "patient_id": p.get("patient_id", ""),
                "age": p.get("age", ""),
                "sex": _normalize_sex(p.get("sex", "")),
            }
        )
    return out


def add_patient_note(patient_id: str, note_text: str) -> None:
    """
    Used by UI when uploading a single patient note (TXT/PDF/DOCX).
    This function only stores a lightweight registry in patients.json.
    The UI should save the actual file under rag_data/patients and you should re-index.
    """
    _ensure_dirs()
    patients = _load_json(PATIENTS_JSON, [])

    patients.append({"patient_id": patient_id, "note": note_text})
    _save_json(PATIENTS_JSON, patients)


# ============================================================
# RAG Matching (used by Request tab)
# ============================================================

def find_candidates(
    protocol_id: str,
    request_text: str = "",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    LLM Agent flow (LLM + RAG), without changing the UI:
    1) RAG: retrieve protocol evidence (inclusion/exclusion chunks)
    2) RAG: retrieve candidate patient docs using request + protocol evidence
    3) LLM: judge each candidate with Gemini -> decision + confidence (match %)
    4) Return top_k results for the Request tab
    """
    rr = RagRetriever()

    # -------------------------
    # 1) Protocol evidence (RAG)
    # -------------------------
    protocol_hits = rr.get_protocol_chunks(protocol_id, top_k=8)
    protocol_evidence = [h.text.strip().replace("\n", " ")[:300] for h in protocol_hits if h.text]

    if not protocol_hits:
        return {"results": []}

    protocol_context = "\n\n".join([h.text for h in protocol_hits])[:2500]

    # -------------------------
    # 2) Candidate retrieval (RAG)
    # -------------------------
    request_text = (request_text or "").strip()

    query = f"""
Protocol ID: {protocol_id}

Protocol context:
{protocol_context}

Coordinator request:
{request_text if request_text else "(no request provided)"}

Goal: retrieve patient candidates who likely match inclusion/exclusion.
""".strip()

    hits = rr.search(query=query, top_k=max(80, top_k * 20), where=None)

    # Aggregate evidence by patient_id
    patient_best_score: Dict[str, float] = {}
    patient_evidence_map: Dict[str, List[str]] = {}

    for h in hits:
        meta = h.meta or {}
        pid = str(meta.get("patient_id", "")).strip()
        if not pid:
            continue

        patient_best_score[pid] = max(patient_best_score.get(pid, -1.0), float(h.score))

        snippet = (h.text or "").strip().replace("\n", " ")
        if snippet:
            patient_evidence_map.setdefault(pid, []).append(snippet[:300])

    if not patient_best_score:
        return {"results": []}

    # shortlist for LLM (keep it reasonable)
    shortlist = sorted(patient_best_score.items(), key=lambda x: x[1], reverse=True)[: max(25, top_k * 5)]
    shortlist_pids = [pid for pid, _ in shortlist]

    # -------------------------
    # 3) Enrich from CSV (age/sex only for UI)
    # -------------------------
    csv_path = _patients_csv_path()
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    id_col = None
    if not df.empty:
        for c in df.columns:
            if c.strip().lower() in ["patient_id", "patientid", "id"]:
                id_col = c
                break
        if id_col is None:
            id_col = df.columns[0]
        df["_pid"] = df[id_col].astype(str).str.strip()

    age_col = None
    sex_col = None
    if not df.empty:
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ["age", "patient_age"]:
                age_col = c
            if lc in ["sex", "gender"]:
                sex_col = c

    results: List[Dict[str, Any]] = []

    # -------------------------
    # 4) LLM judge (Gemini) per candidate
    # -------------------------
    for pid in shortlist_pids:
        age = ""
        sex = ""

        if not df.empty:
            row = df[df["_pid"] == pid].head(1)
            if len(row) == 1:
                r0 = row.iloc[0]
                age = r0[age_col] if age_col else ""
                sex = _normalize_sex(r0[sex_col]) if sex_col else ""

        missing_fields = []
        if str(age).strip() in ["", "nan", "None"]:
            missing_fields.append("age")
        if str(sex).strip() in ["", "nan", "None"]:
            missing_fields.append("sex")

        patient_summary = {
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "has_note": bool(patient_evidence_map.get(pid)),
        }

        patient_evidence = (patient_evidence_map.get(pid) or [])[:8]

        llm = judge_with_gemini(
            protocol_id=protocol_id,
            protocol_evidence=protocol_evidence,
            patient_id=pid,
            patient_summary=patient_summary,
            patient_evidence=patient_evidence,
        )

        match_percent = int(round(float(llm.confidence) * 100))

        results.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "decision": llm.decision,
            "confidence": float(llm.confidence),
            "match_percent": match_percent,
            "reason": llm.reason_short,
            "missing_fields": list(dict.fromkeys(missing_fields + (llm.missing_info or []))),
            "evidence_protocol": protocol_evidence[:6],
            "evidence_patient": patient_evidence[:6],
        })

    # Rank: by match %, then by retrieval score
    score_lookup = {pid: patient_best_score.get(pid, -1.0) for pid in shortlist_pids}
    results.sort(key=lambda r: (r["match_percent"], score_lookup.get(r["patient_id"], -1.0)), reverse=True)
    results = results[:top_k]

    return {"results": results}
