# agent/gemini_judge.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError
import google.generativeai as genai


# -------------------------
# Output contract (LLM must follow)
# -------------------------
class CriterionAssessment(BaseModel):
    criterion: str
    status: str = Field(..., description="Met / Not Met / Uncertain")
    evidence: str = Field(..., description="Short quote or structured value that supports the status")


class LLMMatchResult(BaseModel):
    decision: str = Field(..., description="Likely eligible / Likely ineligible / Need more data")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason_short: str = Field(..., description="One short sentence explaining why")
    criteria: List[CriterionAssessment] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)


def _get_api_key() -> str:
    k = os.getenv("GEMINI_API_KEY")
    if not k:
        raise RuntimeError(
            "Missing GEMINI_API_KEY environment variable.\n"
            "Set it in PyCharm: Run -> Edit Configurations -> Environment variables\n"
            "Example: GEMINI_API_KEY=YOUR_KEY"
        )
    return k


# ✅ NEW: make anything JSON-serializable (fixes numpy/pandas int64/float64)
def _to_jsonable(x: Any) -> Any:
    # basic primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # numpy scalars (int64/float64/etc.)
    try:
        import numpy as np  # type: ignore
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass

    # dict / list containers
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # fallback: string
    return str(x)


def _build_prompt(
    protocol_id: str,
    protocol_evidence: List[str],
    patient_id: str,
    patient_summary: Dict[str, Any],
    patient_evidence: List[str],
) -> str:
    proto_block = "\n".join([f"- {str(x)}" for x in protocol_evidence[:8]])
    patient_block = "\n".join([f"- {str(x)}" for x in patient_evidence[:8]])

    patient_min = {
        "patient_id": patient_summary.get("patient_id", patient_id),
        "age": patient_summary.get("age"),
        "sex": patient_summary.get("sex"),
        "has_note": patient_summary.get("has_note", False),
    }

    # ✅ FIX: convert numpy/pandas types to JSON-safe types
    patient_min = _to_jsonable(patient_min)

    return f"""
You are a clinical trial eligibility assistant.
Your job: assess if the patient is a candidate for the given protocol.

IMPORTANT RULES:
- Use ONLY the evidence provided below. Do NOT invent facts.
- If a key criterion cannot be confirmed from evidence, mark it "Uncertain" and add it to missing_info.
- Return STRICT JSON only (no markdown, no extra text).

Protocol ID: {protocol_id}

Protocol evidence (quotes):
{proto_block}

Patient summary (structured):
{json.dumps(patient_min, ensure_ascii=False)}

Patient evidence (quotes):
{patient_block}

Return JSON with this exact schema:
{{
  "decision": "Likely eligible" | "Likely ineligible" | "Need more data",
  "confidence": 0.0-1.0,
  "reason_short": "one short sentence",
  "criteria": [
    {{"criterion": "string", "status": "Met|Not Met|Uncertain", "evidence": "string"}}
  ],
  "missing_info": ["string", ...]
}}

Guidance:
- Prefer "Need more data" when evidence is insufficient.
- confidence should be lower when many items are uncertain.
""".strip()


def judge_with_gemini(
    protocol_id: str,
    protocol_evidence: List[str],
    patient_id: str,
    patient_summary: Dict[str, Any],
    patient_evidence: List[str],
    model_name: str = "gemini-2.5-flash",
) -> LLMMatchResult:
    genai.configure(api_key=_get_api_key())
    model = genai.GenerativeModel(model_name)

    prompt = _build_prompt(
        protocol_id=protocol_id,
        protocol_evidence=protocol_evidence,
        patient_id=patient_id,
        patient_summary=patient_summary,
        patient_evidence=patient_evidence,
    )

    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()

    try:
        data = json.loads(text)
        return LLMMatchResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return LLMMatchResult(
            decision="Need more data",
            confidence=0.2,
            reason_short="LLM output was not valid JSON; please retry.",
            criteria=[],
            missing_info=["LLM_output_parse_error"],
        )
