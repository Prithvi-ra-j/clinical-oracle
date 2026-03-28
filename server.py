"""
Clinical Oracle MCP Server
Exposes polypharmacy risk intelligence tools via Model Context Protocol.
Gemini-compatible: manual inputSchema definitions strip additionalProperties.
"""
from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

import faers_client
import fhir_bridge
import risk_engine

# ── Gemini schema sanitizer ──────────────────────────────────────────────────
def _strip(obj):
    """Recursively remove fields Gemini's function-calling spec rejects."""
    if isinstance(obj, dict):
        obj.pop("additionalProperties", None)
        obj.pop("$schema", None)
        obj.pop("title", None)
        obj.pop("$defs", None)
        for v in list(obj.values()):
            _strip(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip(item)
    return obj

mcp = FastMCP(
    name="clinical-oracle",
    version="1.0.0",
    instructions="""Clinical Oracle provides polypharmacy risk intelligence using real FDA FAERS adverse event data.

Tools:
- signal_scan: Query FDA FAERS for drug combination adverse events
- risk_score: Generate patient-specific risk tier using RAG
- alert_draft: Create EHR-pasteable clinical alert
- health_check: Verify server and API health

Privacy: No PHI is sent to external APIs. Only extracted clinical parameters are used.
Compliance: This is an FDA-exempt Clinical Decision Support tool. Clinician review required for all outputs."""
)

# ── Patch tool listing to strip Gemini-incompatible schema fields ─────────────
_orig_list_tools = mcp._mcp_server.list_tools

async def _patched_list_tools():
    tools = await _orig_list_tools()
    for tool in tools:
        if hasattr(tool, "inputSchema") and tool.inputSchema:
            _strip(tool.inputSchema)
        if hasattr(tool, "outputSchema") and tool.outputSchema:
            _strip(tool.outputSchema)
    return tools

mcp._mcp_server.list_tools = _patched_list_tools
# ─────────────────────────────────────────────────────────────────────────────


@mcp.tool()
async def signal_scan(
    medications: list[str],
    patient_age_range: str = "",
    patient_sex: str = "",
    sharp_patient_id: str = "",
    sharp_fhir_base_url: str = ""
) -> dict:
    """
    Query FDA FAERS for adverse event signals in a drug combination.

    This tool provides decision SUPPORT only. Clinician review required.

    Args:
        medications: List of medication names (2+ required)
        patient_age_range: Optional age range e.g. 65-74. Leave empty if not applicable.
        patient_sex: Optional sex filter - male or female. Leave empty if not applicable.
        sharp_patient_id: Optional SHARP patient ID for FHIR context. Leave empty if not applicable.
        sharp_fhir_base_url: Optional SHARP FHIR server URL. Leave empty if not applicable.

    Returns:
        Dict with drug_combination, total_faers_reports, top_adverse_reactions, etc.
    """
    if sharp_patient_id and sharp_fhir_base_url:
        fhir_context = await fhir_bridge.get_patient_context(
            patient_id=sharp_patient_id,
            fhir_base=sharp_fhir_base_url
        )
        fhir_meds = fhir_context.get("medications", [])
        all_meds = list(set(medications + fhir_meds))
        medications = all_meds

    sex_code = None
    if patient_sex:
        if patient_sex.lower() == "male":
            sex_code = "1"
        elif patient_sex.lower() == "female":
            sex_code = "2"

    for med in medications:
        if any(char.isdigit() for char in med) and len(med) < 5:
            return {
                "error": "Potential PHI detected in medications list",
                "drug_combination": [],
                "total_faers_reports": 0,
                "top_adverse_reactions": [],
                "data_source": "FDA FAERS",
                "disclaimer": "Query rejected for privacy reasons"
            }

    result = await faers_client.query_combination(
        drugs=medications,
        sex_code=sex_code,
        limit=20
    )
    return result


@mcp.tool()
async def risk_score(
    signal_scan_output: dict,
    patient_age: int = 0,
    creatinine_mg_dl: float = 0.0,
    inr_value: float = 0.0,
    additional_conditions: list[str] = []
) -> dict:
    """
    Synthesize patient-specific risk tier from FAERS signals using RAG.

    This tool provides decision SUPPORT only. Clinician review required.

    Args:
        signal_scan_output: Output dict from signal_scan tool
        patient_age: Patient age in years. Use 0 if unknown.
        creatinine_mg_dl: Serum creatinine in mg/dL. Use 0.0 if unknown.
        inr_value: International Normalized Ratio. Use 0.0 if unknown.
        additional_conditions: List of medical conditions. Use empty list if none.

    Returns:
        Dict with risk_tier, rationale, top_risks, and _metadata.
        _metadata.phi_in_prompt is always False.
    """
    result = await risk_engine.score_risk(
        signal_data=signal_scan_output,
        patient_age=patient_age if patient_age > 0 else None,
        creatinine_mg_dl=creatinine_mg_dl if creatinine_mg_dl > 0 else None,
        inr_value=inr_value if inr_value > 0 else None,
        conditions=additional_conditions if additional_conditions else None
    )

    if "_metadata" in result:
        assert result["_metadata"]["phi_in_prompt"] == False, "PHI verification failed"

    return result


@mcp.tool()
async def alert_draft(
    risk_score_output: dict,
    clinician_name: str = "",
    patient_identifier: str = ""
) -> dict:
    """
    Generate EHR-pasteable clinical alert with mandatory clinician confirmation.

    This tool provides decision SUPPORT only. Clinician review required.

    Args:
        risk_score_output: Output dict from risk_score tool
        clinician_name: Optional clinician name for note. Leave empty if not applicable.
        patient_identifier: Optional de-identified patient ID e.g. PT-001. Leave empty if not applicable.

    Returns:
        Dict with alert_status, risk_tier, ehr_note, evidence_citations, etc.
    """
     # Unwrap MCP content wrapper if Prompt Opinion passes raw MCP response
    if isinstance(risk_score_output, dict) and "content" in risk_score_output:
        try:
            import json
            risk_score_output = json.loads(risk_score_output["content"][0]["text"])
        except Exception:
            pass
    if "error" in risk_score_output:
        return {
            "error": "Cannot draft alert — risk_score failed",
            "alert_status": "ERROR",
            "risk_tier": "UNKNOWN"
        }

    risk_tier = risk_score_output.get("risk_tier", "UNKNOWN")
    rationale = risk_score_output.get("risk_tier_rationale", "")
    top_risks = risk_score_output.get("top_risks", [])
    confidence = risk_score_output.get("confidence", "UNKNOWN")
    limitations = risk_score_output.get("limitations", "")
    patient_label = patient_identifier if patient_identifier else "[Patient ID]"

    ehr_note = f"""
═══════════════════════════════════════════════════════════════
CLINICAL DECISION SUPPORT ALERT — POLYPHARMACY RISK ASSESSMENT
═══════════════════════════════════════════════════════════════

Patient: {patient_label}
Generated: [Date/Time]
Risk Tier: {risk_tier}
Confidence: {confidence}

RISK ASSESSMENT:
{rationale}

TOP IDENTIFIED RISKS:
"""

    for i, risk in enumerate(top_risks[:5], 1):
        reaction = risk.get("reaction", "Unknown")
        count = risk.get("faers_report_count", 0)
        amplifier = risk.get("patient_specific_amplifier", "")
        ehr_note += f"\n{i}. {reaction} ({count} FAERS reports)\n   {amplifier}\n"

    ehr_note += f"""
LIMITATIONS:
{limitations}

───────────────────────────────────────────────────────────────
DEMOGRAPHIC BIAS NOTE:
FAERS adverse event data may underrepresent non-white and female
patients. Clinical judgment should account for potential reporting
biases in the source data.
───────────────────────────────────────────────────────────────

REGULATORY NOTE:
This is a Clinical Decision Support (CDS) tool under FDA CDS
exemption (21st Century Cures Act). This is NOT a Software as a
Medical Device (SaMD). This tool does NOT diagnose, treat, cure,
or prevent any disease.
───────────────────────────────────────────────────────────────

CLINICIAN ACTION REQUIRED:

☐ CONFIRM — I have reviewed this alert and will take appropriate action
☐ DISMISS — I have reviewed this alert and determined no action needed
☐ ESCALATE — I am escalating this to [specialist/pharmacist]

Clinician Name: _______________________________________________

Date/Time: ____________________________________________________

Signature: ____________________________________________________

═══════════════════════════════════════════════════════════════
"""

    plain_summary = f"Risk tier: {risk_tier}. {rationale[:200]}..."

    evidence_citations = []
    for risk in top_risks:
        reaction = risk.get("reaction", "Unknown")
        count = risk.get("faers_report_count", 0)
        evidence_citations.append(f"{reaction}: {count} FAERS reports")

    return {
        "alert_status": "READY_FOR_REVIEW",
        "risk_tier": risk_tier,
        "ehr_note": ehr_note,
        "plain_summary": plain_summary,
        "evidence_citations": evidence_citations,
        "action_required": True,
        "hipaa_note": "No PHI was transmitted to external LLM APIs. Only extracted clinical parameters were used.",
        "regulatory_note": "This is a CDS tool under FDA CDS exemption, not a SaMD. Clinician review required."
    }


@mcp.tool()
async def health_check() -> dict:
    """
    Verify MCP server health and upstream API reachability.

    Returns:
        Dict with status, faers_api reachability, tools list, version, and phi_handling.
    """
    import time
    start_time = time.time()

    faers_status = "unknown"
    try:
        result = await faers_client.query_combination(["aspirin", "ibuprofen"], limit=1)
        if "error" not in result:
            faers_status = "reachable"
        else:
            faers_status = "degraded"
    except:
        faers_status = "unreachable"

    elapsed = time.time() - start_time
    status = "healthy" if faers_status == "reachable" and elapsed < 6.0 else "degraded"

    return {
        "status": status,
        "faers_api": faers_status,
        "response_time_seconds": round(elapsed, 2),
        "tools": ["signal_scan", "risk_score", "alert_draft", "health_check"],
        "version": "1.0.0",
        "phi_handling": "No PHI transmitted to external APIs. Only extracted clinical parameters used."
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=port
    )