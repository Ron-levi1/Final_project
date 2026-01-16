import streamlit as st
import pandas as pd

import os
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db.patient_service import (
    list_protocols,
    add_protocol,
    list_patients_summary,
    add_patient_note,
    find_candidates,
)

# --------------------
# Page config (less wide + cleaner)
# --------------------
st.set_page_config(
    page_title="Clinical Trial Recruitment Agent",
    page_icon="🧬",
    layout="centered",
)

# --------------------
# Simple professional CSS
# --------------------
st.markdown(
    """
<style>
.block-container {
    max-width: 980px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 { letter-spacing: -0.2px; }
div[data-testid="stVerticalBlock"] { gap: 0.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

def normalize_sex(value):
    if value is None:
        return ""
    v = str(value).strip().lower()
    if v in ["m", "male"]:
        return "Male"
    if v in ["f", "female"]:
        return "Female"
    return str(value).strip()

st.title("Clinical Trial Recruitment Agent")
st.caption("LLM + RAG: Upload protocols & patient notes, retrieve candidates with evidence-based explanations.")

tab_request, tab_protocols, tab_patients = st.tabs(["Request", "Protocols", "Patients"])

# =========================
# TAB 1: REQUEST
# =========================
with tab_request:
    st.subheader("Coordinator Request")

    protocols = list_protocols()
    if not protocols:
        st.warning("No protocols found yet. Go to the 'Protocols' tab and upload at least one protocol (.md).")
        st.stop()

    protocol_labels = [f"{p['protocol_id']} — {p.get('title','')}" for p in protocols]
    selected_label = st.selectbox("Select protocol", protocol_labels)
    selected_protocol_id = selected_label.split(" — ")[0].strip()

    top_k = st.number_input("How many patients to retrieve?", min_value=1, max_value=50, value=5, step=1)

    # No free-text input (as per requirement): the LLM should infer criteria from the selected protocol
    request_text = (
        "Infer the protocol's inclusion and exclusion criteria from the retrieved protocol evidence. "
        "Then evaluate each patient's retrieved evidence against those criteria and return the best matching candidates."
    )

    # (Optional UI note so the page doesn't feel 'empty')
    st.caption(
        "The system automatically infers inclusion/exclusion criteria from the selected protocol (no free-text needed).")

    if st.button("Find candidates (LLM + RAG)", type="primary"):
        with st.spinner("Running RAG retrieval + Gemini reasoning..."):
            try:
                out = find_candidates(
                    protocol_id=selected_protocol_id,
                    request_text=request_text,
                    top_k=int(top_k),
                )

            except Exception as e:
                st.error(f"Failed to retrieve candidates: {e}")
                st.stop()

        results = out.get("results", [])
        if not results:
            st.info("No candidates found (or protocol not indexed).")
        else:
            # minimal table
            rows = []
            missing_any = False

            for r in results:
                sex = normalize_sex(r.get("sex", ""))
                miss = r.get("missing_fields") or []
                if miss:
                    missing_any = True

                rows.append({
                    "Patient ID": r.get("patient_id", ""),
                    "Age": r.get("age", ""),
                    "Sex": sex,
                    "Decision": r.get("decision", "Uncertain"),
                    "Confidence": round(float(r.get("confidence", 0.0)), 2),
                    "Match %": int(r.get("match_percent", round(float(r.get("confidence", 0.0)) * 100))),
                    "Reason (evidence-based)": r.get("reason", ""),
                })

            if missing_any:
                st.warning("Some retrieved patients are missing key fields (age/sex). Consider uploading richer TXT notes.")

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("### Evidence (why each patient matched)")
            for r in results:
                pid = r.get("patient_id", "UNKNOWN")
                decision = r.get("decision", "Uncertain")
                conf = r.get("confidence", 0.0)

                with st.expander(f"Patient {pid} — {decision} (confidence {conf:.2f})"):
                    if r.get("missing_fields"):
                        st.warning("Missing: " + ", ".join(r["missing_fields"]))

                    if r.get("reason"):
                        st.markdown("**Reason:**")
                        st.write(r["reason"])

                    evp = r.get("evidence_protocol") or []
                    if evp:
                        st.markdown("**Protocol evidence:**")
                        for x in evp[:3]:
                            st.write("• " + str(x))

                    evpt = r.get("evidence_patient") or []
                    if evpt:
                        st.markdown("**Patient evidence:**")
                        for x in evpt[:3]:
                            st.write("• " + str(x))


# =========================
# TAB 2: PROTOCOLS
# =========================
with tab_protocols:
    st.subheader("Protocols")

    protocols = list_protocols()
    if protocols:
        st.dataframe(
            pd.DataFrame(protocols)[["protocol_id", "title", "file"]].rename(
                columns={"protocol_id": "Protocol ID", "title": "Title", "file": "File"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No protocols uploaded yet.")

    st.markdown("---")
    st.markdown("### Upload a new protocol (.md)")

    up = st.file_uploader("Upload protocol file", type=["md"], accept_multiple_files=False)

    if up is not None:
        if st.button("Add protocol + reindex", type="primary"):
            with st.spinner("Saving & indexing protocol..."):
                try:
                    res = add_protocol(up.getvalue(), up.name)
                except Exception as e:
                    st.error(f"Failed to add protocol: {e}")
                else:
                    st.success(f"Protocol added: {res['protocol_id']} (index updated)")


# =========================
# TAB 3: PATIENTS
# =========================
with tab_patients:
    st.subheader("Patients (database)")

    try:
        patients = list_patients_summary()
    except Exception as e:
        st.error(f"Could not load patient index: {e}")
        st.stop()

    if patients:
        dfp = pd.DataFrame(patients).copy()
        if "sex" in dfp.columns:
            dfp["sex"] = dfp["sex"].apply(normalize_sex)
        dfp.rename(columns={"patient_id": "Patient ID", "age": "Age", "sex": "Sex"}, inplace=True)
        st.dataframe(dfp[["Patient ID", "Age", "Sex"]], use_container_width=True, hide_index=True)
    else:
        st.info("No patient records found yet. Upload a note below.")

    st.markdown("---")
    st.markdown("### Add new patient (Auto ID) by uploading a note")

    up_p = st.file_uploader(
        "Upload patient note (TXT / PDF / DOCX)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=False,
        key="patient_uploader",
    )

    if up_p is not None:
        st.caption("The system will extract age/sex automatically (best from TXT). If missing, you will get a warning.")
        if st.button("Create patient + reindex", type="primary"):
            with st.spinner("Saving & indexing patient note..."):
                try:
                    res = add_patient_note(up_p.getvalue(), up_p.name)
                except Exception as e:
                    st.error(f"Failed to add patient: {e}")
                else:
                    st.success(f"Patient created: {res['patient_id']} (index updated)")

                    if res.get("missing_fields"):
                        st.warning(
                            "Missing important fields for classification: "
                            + ", ".join(res["missing_fields"])
                            + ". Consider uploading a richer TXT note that contains these."
                        )

                    preview = res.get("txt_preview")
                    if preview:
                        with st.expander("Patient note preview (TXT only)"):
                            st.write(preview)
