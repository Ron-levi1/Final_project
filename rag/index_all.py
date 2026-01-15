# rag/index_all.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import faiss  # faiss-cpu
from sentence_transformers import SentenceTransformer

from docx import Document
from pypdf import PdfReader


# ============================================================
# 0) Paths (fits your requirement: data is under rag_data/)
# ============================================================

RAG_DIR = Path(__file__).resolve().parent               # .../Final_project/rag
PROJECT_DIR = RAG_DIR.parent                            # .../Final_project

# ✅ Fixed structure: data is under rag_data/
DATA_DIR = PROJECT_DIR / "rag_data"                     # .../Final_project/rag_data
PATIENTS_DIR = DATA_DIR / "patients"                    # .../Final_project/rag_data/patients
PROTOCOLS_DIR = DATA_DIR / "protocols"                  # .../Final_project/rag_data/protocols

STORE_DIR = PROJECT_DIR / "rag_store"                   # .../Final_project/rag/rag_store

INDEX_PATH = STORE_DIR / "faiss.index"
DOCS_PATH = STORE_DIR / "docs.jsonl"

# Model for embeddings (small + good baseline)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# 1) Small utilities
# ============================================================

def _clean_text(text: str) -> str:
    """Normalize whitespace, remove weird repeated spaces, keep it readable."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_probably_protocol_file(p: Path) -> bool:
    """Heuristic: protocol files often start with NCT or DECLARE_TIMI etc."""
    name = p.stem.lower()
    if name.startswith("nct"):
        return True
    if "declare" in name:
        return True
    # also treat markdown protocols as protocols by default
    if p.suffix.lower() == ".md" and "patient" not in name and "patients" not in name:
        return True
    return False


def _is_patients_csv(p: Path) -> bool:
    """Heuristic: patients CSV naming."""
    return p.suffix.lower() == ".csv" and "patient" in p.stem.lower()


def _read_md_or_txt(path: Path) -> str:
    return _clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return _clean_text("\n".join(parts))


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return _clean_text("\n".join(pages))


def _read_any_text(path: Path) -> str:
    """Load file content depending on suffix."""
    suf = path.suffix.lower()
    if suf in [".md", ".txt"]:
        return _read_md_or_txt(path)
    if suf == ".docx":
        return _read_docx(path)
    if suf == ".pdf":
        return _read_pdf(path)
    return ""


# ============================================================
# 2) Chunking (very important for RAG)
# ============================================================

def chunk_text(
    text: str,
    max_chars: int = 900,
    overlap: int = 120,
) -> List[str]:
    """
    Split text into chunks.
    Why: embeddings work better on medium-size pieces; huge text reduces relevance.

    Strategy:
    - Split by paragraphs
    - Pack paragraphs until max_chars
    - Add overlap so context isn't lost between chunks
    """
    text = _clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for para in paragraphs:
        if buf_len + len(para) + 2 <= max_chars:
            buf.append(para)
            buf_len += len(para) + 2
        else:
            chunk = _clean_text("\n\n".join(buf))
            if chunk:
                chunks.append(chunk)

            # start new buffer with overlap from previous chunk
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                buf = [tail, para]
                buf_len = len(tail) + len(para) + 2
            else:
                buf = [para]
                buf_len = len(para)

    last = _clean_text("\n\n".join(buf))
    if last:
        chunks.append(last)

    return chunks


# ============================================================
# 3) Define what we store per chunk
# ============================================================

@dataclass
class RagDoc:
    text: str
    meta: Dict[str, str]


# ============================================================
# 4) Build documents from protocols
# ============================================================

def build_protocol_docs(protocol_path: Path) -> List[RagDoc]:
    """
    Convert one protocol file into many chunk-docs.
    meta includes:
    - doc_type=protocol
    - protocol_id derived from filename
    - source_file
    - chunk_id
    """
    raw = _read_any_text(protocol_path)
    if not raw:
        return []

    protocol_id = protocol_path.stem
    chunks = chunk_text(raw)

    docs: List[RagDoc] = []
    for i, ch in enumerate(chunks):
        docs.append(
            RagDoc(
                text=ch,
                meta={
                    "doc_type": "protocol",
                    "protocol_id": protocol_id,
                    "source_file": protocol_path.name,
                    "chunk_id": f"{protocol_id}__chunk_{i}",
                },
            )
        )
    return docs


# ============================================================
# 5) Build documents from patients CSV
# ============================================================

def _normalize_sex(value: str) -> str:
    """Map F/M to Female/Male when possible."""
    if value is None:
        return ""
    v = str(value).strip().lower()
    if v in ["f", "female"]:
        return "Female"
    if v in ["m", "male"]:
        return "Male"
    return str(value).strip()


def build_patient_docs_from_csv(csv_path: Path) -> Tuple[List[RagDoc], pd.DataFrame]:
    """
    Convert a patients CSV into:
    - docs for RAG (one per patient row, text representation)
    - a clean DataFrame for UI/table usage later

    IMPORTANT:
    - We don't assume fixed columns like egfr.
    - We only REQUIRE minimal fields for warning: patient_id, age, sex (if exist).
    - We still index other columns into the patient text.
    """
    df = pd.read_csv(csv_path)

    # Standardize common column names if they appear in variants
    col_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ["patient_id", "patientid", "id"]:
            col_map[c] = "patient_id"
        elif lc in ["sex", "gender"]:
            col_map[c] = "sex"
        elif lc in ["age", "patient_age"]:
            col_map[c] = "age"
    if col_map:
        df = df.rename(columns=col_map)

    # Warnings on missing fields (doesn't block indexing)
    missing_required = [c for c in ["patient_id", "age", "sex"] if c not in df.columns]
    if missing_required:
        print(
            f"⚠️ Patients CSV is missing columns: {missing_required}. "
            f"Indexing will continue, but matching may be weaker."
        )

    docs: List[RagDoc] = []
    for idx, row in df.iterrows():
        patient_id = str(row.get("patient_id", f"ROW_{idx}")).strip()
        age = row.get("age", "")
        sex = _normalize_sex(row.get("sex", ""))

        # Additional warning per row if critical fields missing
        row_warnings = []
        if str(age).strip() in ["", "nan", "None"]:
            row_warnings.append("age")
        if str(sex).strip() in ["", "nan", "None"]:
            row_warnings.append("sex")
        if row_warnings:
            print(f"⚠️ Patient {patient_id}: missing {row_warnings} in CSV row.")

        # Build a flexible text "patient profile" from ALL columns
        parts = []
        parts.append(f"Patient ID: {patient_id}")
        if str(age).strip() not in ["", "nan", "None"]:
            parts.append(f"Age: {age}")
        if str(sex).strip():
            parts.append(f"Sex: {sex}")

        for c in df.columns:
            if c in ["patient_id", "age", "sex"]:
                continue
            val = row.get(c, "")
            if str(val).strip() in ["", "nan", "None"]:
                continue
            parts.append(f"{c}: {val}")

        text = _clean_text("\n".join(parts))

        docs.append(
            RagDoc(
                text=text,
                meta={
                    "doc_type": "patient_structured",
                    "patient_id": patient_id,
                    "source_file": csv_path.name,
                    "chunk_id": f"{patient_id}__structured",
                },
            )
        )

    # For UI: show nicer sex labels if needed
    if "sex" in df.columns:
        df["sex"] = df["sex"].apply(_normalize_sex)

    return docs, df


# ============================================================
# 6) Build documents from patient notes (docx/txt/pdf/md)
# ============================================================

def build_patient_note_docs(note_path: Path) -> List[RagDoc]:
    """
    Index patient notes as chunks.

    We try to extract patient_id from filename:
    - If filename contains something like P001, P002, etc -> use it
    - Otherwise patient_id="UNKNOWN"
    """
    raw = _read_any_text(note_path)
    if not raw:
        return []

    m = re.search(r"(P\d{3,})", note_path.stem.upper())
    patient_id = m.group(1) if m else "UNKNOWN"

    chunks = chunk_text(raw)
    docs: List[RagDoc] = []
    for i, ch in enumerate(chunks):
        docs.append(
            RagDoc(
                text=ch,
                meta={
                    "doc_type": "patient_note",
                    "patient_id": patient_id,
                    "source_file": note_path.name,
                    "chunk_id": f"{patient_id}__note_{note_path.stem}__chunk_{i}",
                },
            )
        )
    return docs


# ============================================================
# 7) Index building (Embeddings -> FAISS)
# ============================================================

def embed_texts(model: SentenceTransformer, texts: List[str]):
    """
    Returns embeddings as float32 vectors for FAISS.
    normalize_embeddings=True helps cosine-like similarity.
    """
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return emb.astype("float32")


def build_faiss_index(vectors) -> faiss.Index:
    """
    We use IndexFlatIP (inner product).
    Because vectors are normalized, inner product ~= cosine similarity.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_docs_jsonl(docs: List[RagDoc], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"text": d.text, "meta": d.meta}, ensure_ascii=False) + "\n")


# ============================================================
# 8) Main: scan rag_data and index everything
# ============================================================

def index_all() -> None:
    # --- 8.0) Validate folders ---
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {DATA_DIR} (expected: Final_project/rag_data/)")

    if not PROTOCOLS_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {PROTOCOLS_DIR} (expected: rag_data/protocols/)")

    if not PATIENTS_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {PATIENTS_DIR} (expected: rag_data/patients/)")

    STORE_DIR.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL_NAME)
    all_docs: List[RagDoc] = []

    # --- 8.1) Protocols: ONLY from rag_data/protocols ---
    protocol_files = [
        p for p in PROTOCOLS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in [".md", ".txt", ".docx", ".pdf"]
    ]

    # (Optional) keep your heuristic, but now it's mostly redundant since we're in /protocols
    # protocol_files = [p for p in protocol_files if _is_probably_protocol_file(p)]

    for p in protocol_files:
        all_docs.extend(build_protocol_docs(p))

    # --- 8.2) Patients CSV (structured): ONLY from rag_data/patients ---
    patients_csv_files = [p for p in PATIENTS_DIR.rglob("*.csv")]

    for csv_path in patients_csv_files:
        if _is_patients_csv(csv_path) or True:
            docs, _df = build_patient_docs_from_csv(csv_path)
            all_docs.extend(docs)

    # --- 8.3) Patient notes (unstructured): ONLY from rag_data/patients ---
    note_files = [
        p for p in PATIENTS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in [".md", ".txt", ".docx", ".pdf"]
    ]

    for p in note_files:
        # If someone accidentally puts a protocol file into patients folder, skip it
        if _is_probably_protocol_file(p):
            continue
        all_docs.extend(build_patient_note_docs(p))

    if not all_docs:
        print("❌ No documents found to index.")
        print("   Put protocol files under: rag_data/protocols/")
        print("   Put patients CSV + notes under: rag_data/patients/")
        return

    # --- 8.4) Embed + build index ---
    texts = [d.text for d in all_docs]
    vectors = embed_texts(model, texts)
    index = build_faiss_index(vectors)

    # --- 8.5) Save ---
    faiss.write_index(index, str(INDEX_PATH))
    save_docs_jsonl(all_docs, DOCS_PATH)

    print("✅ RAG indexing complete!")
    print(f"   - Docs: {len(all_docs)}")
    print(f"   - Index saved to: {INDEX_PATH}")
    print(f"   - Docs saved to:  {DOCS_PATH}")


if __name__ == "__main__":
    index_all()
