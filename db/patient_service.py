from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from rag.retriever import RagRetriever


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

def find_candidates(protocol_id: str, request_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Returns list of candidates for the UI table.
    Keys expected by the UI (based on your screenshot):
      - patient_id
      - age
      - sex
      - reason  (evidence-based)
    """
    rr = RagRetriever()

    # 1) build a query that is protocol-aware
    protocol_hits = rr.get_protocol_chunks(protocol_id, top_k=6)
    protocol_context = "\n\n".join([h.text for h in protocol_hits])[:2500]

    query = f"""
Protocol ID: {protocol_id}

Protocol context:
{protocol_context}

Coordinator request:
{request_text}
""".strip()

    # 2) retrieve many and then aggregate by patient_id
    hits = rr.search(query=query, top_k=max(40, top_n), where=None)

    agg: Dict[str, Dict[str, Any]] = {}
    for h in hits:
        meta = h.meta or {}
        pid = str(meta.get("patient_id", "")).strip()
        if not pid:
            continue

        if pid not in agg:
            agg[pid] = {
                "patient_id": pid,
                "score": float(h.score),
                "reason": "",
            }
        else:
            agg[pid]["score"] = max(float(agg[pid]["score"]), float(h.score))

        if not agg[pid]["reason"]:
            snippet = (h.text or "").replace("\n", " ").strip()
            agg[pid]["reason"] = f'Matched based on patient note evidence: "{snippet[:200]}"'

    ranked = sorted(agg.values(), key=lambda x: x["score"], reverse=True)[:top_n]

    # 3) enrich age/sex from CSV if available
    csv_path = _patients_csv_path()
    if csv_path.exists() and ranked:
        df = pd.read_csv(csv_path)

        # find id column
        id_col = None
        for c in df.columns:
            if c.strip().lower() in ["patient_id", "patientid", "id"]:
                id_col = c
                break
        if id_col is None:
            id_col = df.columns[0]

        df["_pid"] = df[id_col].astype(str).str.strip()

        # normalize potential columns
        age_col = None
        sex_col = None
        for c in df.columns:
            lc = c.strip().lower()
            if lc in ["age", "patient_age"]:
                age_col = c
            if lc in ["sex", "gender"]:
                sex_col = c

        for r in ranked:
            pid = r["patient_id"]
            row = df[df["_pid"] == pid].head(1)
            if len(row) == 1:
                row0 = row.iloc[0]
                r["age"] = row0[age_col] if age_col else ""
                r["sex"] = _normalize_sex(row0[sex_col]) if sex_col else ""
            else:
                r["age"] = ""
                r["sex"] = ""
    else:
        for r in ranked:
            r["age"] = ""
            r["sex"] = ""

    # 4) match UI column name from your screenshot ("Reason (evidence-based)")
    for r in ranked:
        r["Reason (evidence-based)"] = r.pop("reason", "")
        # Keep also "reason" if UI uses it somewhere else
        r["reason"] = r["Reason (evidence-based)"]

    return ranked
