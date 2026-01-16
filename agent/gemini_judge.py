# agent/gemini_judge.py
from __future__ import annotations

import json
import os
import re
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


def _build_prompt(
    protocol_id: str,
    protocol_evidence: List[str],
    patient_id: str,
    patient_summary: Dict[str, Any],
    patient_evidence: List[str],
) -> str:
    # Keep evidence short to avoid leaking full notes
    proto_block = "\n".join([f"- {x}" for x in protocol_evidence[:8]]) or "- (no protocol evidence provided)"
    patient_block = "\n".join([f"- {x}" for x in patient_evidence[:8]]) or "- (no patient evidence provided)"

    patient_min = {
        "patient_id": patient_summary.get("patient_id", patient_id),
        "age": patient_summary.get("age"),
        "sex": patient_summary.get("sex"),
        "has_note": patient_summary.get("has_note", False),
    }

    # Professional + constrained rubric + explicit calibration
    return f"""
ROLE
You are a clinical trial eligibility assessor supporting recruitment coordinators.

TASK
Given protocol inclusion/exclusion evidence snippets and patient evidence snippets, assess whether the patient is a candidate for the specified protocol.

STRICT RULES (MUST FOLLOW)
1) Use ONLY the evidence provided below. Do NOT invent, assume, or infer missing facts.
2) If evidence is insufficient to decide a criterion, set status="Uncertain" and add the missing item to missing_info.
3) Output MUST be STRICT JSON only: no markdown, no commentary, no backticks, no surrounding text.

PROTOCOL
Protocol ID: {protocol_id}

Protocol evidence (snippets):
{proto_block}

PATIENT
Patient summary (structured):
{json.dumps(patient_min, ensure_ascii=False)}

Patient evidence (snippets):
{patient_block}

OUTPUT JSON SCHEMA (exact keys)
{{
  "decision": "Likely eligible" | "Likely ineligible" | "Need more data",
  "confidence": 0.0-1.0,
  "reason_short": "one short sentence grounded in evidence",
  "criteria": [
    {{"criterion": "string", "status": "Met|Not Met|Uncertain", "evidence": "string"}}
  ],
  "missing_info": ["string", ...]
}}

DECISION LOGIC
- "Likely ineligible": at least one clearly NOT MET exclusion OR a required inclusion clearly NOT MET (based on evidence).
- "Need more data": key criteria cannot be confirmed from evidence (many Uncertain).
- "Likely eligible": no clear exclusion triggered AND key inclusions appear MET, with limited uncertainty.

CONFIDENCE CALIBRATION (important)
Set confidence using this rubric:
- Start at 0.50
- For each key criterion clearly supported by evidence (Met/Not Met with direct snippet), +0.10
- For each Uncertain criterion, -0.10
- If any exclusion is clearly triggered (Not Met / ineligible), cap confidence at 0.85 (still allow uncertainty)
- If evidence is very sparse (no protocol or no patient snippets), cap confidence at 0.35
Finally clamp confidence to [0.0, 1.0].

CRITERIA FORMAT
- Provide 3–8 criteria entries.
- Criteria should reflect inclusion/exclusion items that are most relevant to this protocol and patient.
- Evidence must quote or reference a specific provided snippet or a structured value (age/sex).

Now output ONLY the JSON object.
""".strip()


def _extract_json_object(text: str) -> str:
    """
    If the model wraps JSON with extra text, try to extract the first {...} block.
    Keeps the app resilient without changing UI.
    """
    text = (text or "").strip()
    if not text:
        return ""
    # If it's already JSON, return as-is
    if text.startswith("{") and text.endswith("}"):
        return text
    # Try to find a JSON object block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else text


def judge_with_gemini(
    protocol_id: str,
    protocol_evidence: List[str],
    patient_id: str,
    patient_summary: Dict[str, Any],
    patient_evidence: List[str],
    model_name: str = "gemini-2.5-flash",
) -> LLMMatchResult:
    # Configure once per call (fine for small scale); can be moved to module init if desired.
    genai.configure(api_key=_get_api_key())

    # Try to make outputs deterministic + JSON-friendly
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_output_tokens": 800,
    }

    model = genai.GenerativeModel(model_name, generation_config=generation_config)

    prompt = _build_prompt(
        protocol_id=protocol_id,
        protocol_evidence=protocol_evidence,
        patient_id=patient_id,
        patient_summary=patient_summary,
        patient_evidence=patient_evidence,
    )

    resp = model.generate_content(prompt)
    raw = (resp.text or "").strip()
    text = _extract_json_object(raw)

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
