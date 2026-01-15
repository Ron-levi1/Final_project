# tools/bootstrap_seed.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import sqlite3

from db.patient_service import add_protocol, add_patient_note, list_protocols, list_patients_summary


PROJECT_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = PROJECT_DIR / "rag_data"
SEED_PROTOCOLS_DIR = SEED_DIR / "protocols"
SEED_PATIENTS_DIR = SEED_DIR / "patients"
SEED_PATIENTS_CSV = SEED_PATIENTS_DIR / "patients_for_trial_screening.csv"


def _safe_count_protocols() -> int:
    try:
        return len(list_protocols())
    except Exception:
        return 0


def _safe_count_patients() -> int:
    try:
        return len(list_patients_summary())
    except Exception:
        return 0


def bootstrap_if_empty() -> dict:
    """
    Seed DB from rag_data/ if the DB is empty.
    This restores the "already full" UI state on startup.
    """
    report = {"imported_protocols": 0, "imported_patients": 0, "skipped": False}

    # If already populated, do nothing
    if _safe_count_protocols() > 0 or _safe_count_patients() > 0:
        report["skipped"] = True
        report["reason"] = "DB already has data."
        return report

    # ---- import protocols (.md)
    if SEED_PROTOCOLS_DIR.exists():
        for md in sorted(SEED_PROTOCOLS_DIR.rglob("*.md")):
            name = md.stem
            text = md.read_text(encoding="utf-8", errors="ignore")
            add_protocol(name, text)
            report["imported_protocols"] += 1

    # ---- import patients from CSV
    if SEED_PATIENTS_CSV.exists():
        df = pd.read_csv(SEED_PATIENTS_CSV)

        # We try to infer columns in a flexible way.
        # You likely have patient_id + note/text or columns that can be concatenated to a note.
        # Prefer an explicit "note"/"text" column if exists.
        lower_cols = {c.lower(): c for c in df.columns}

        pid_col = lower_cols.get("patient_id") or lower_cols.get("id") or df.columns[0]
        note_col = lower_cols.get("note") or lower_cols.get("text") or lower_cols.get("clinical_note")

        for _, row in df.iterrows():
            patient_id = str(row.get(pid_col, "")).strip()
            if not patient_id:
                continue

            if note_col and pd.notna(row.get(note_col, None)):
                note = str(row.get(note_col, "")).strip()
            else:
                # fallback: create a short note from all columns (excluding empty)
                parts = []
                for c in df.columns:
                    v = row.get(c, "")
                    if pd.isna(v) or str(v).strip() == "":
                        continue
                    parts.append(f"{c}: {v}")
                note = "\n".join(parts)

            # Your patient_service signature might differ; support both common cases.
            try:
                add_patient_note(patient_id, note)
            except TypeError:
                add_patient_note(patient_id, None, note)

            report["imported_patients"] += 1

    return report
