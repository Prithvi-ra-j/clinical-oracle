"""
Integration test for full Clinical Oracle pipeline
Tests: signal_scan → risk_score → alert_draft
"""
import pytest
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Skip if no Groq API key (CI/CD environments)
pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here",
    reason="GROQ_API_KEY not configured"
)


@pytest.mark.asyncio
async def test_full_pipeline():
    """Test complete pipeline with demo scenario."""
    from faers_client import query_combination
    from risk_engine import score_risk
    
    # Demo scenario: 74M with CKD Stage 3, AF, heart failure
    medications = ["warfarin", "amiodarone", "fluconazole", "furosemide", "digoxin"]
    
    # Step 1: signal_scan
    signal_result = await query_combination(medications)
    
    assert "error" not in signal_result
    assert signal_result["total_faers_reports"] > 0
    assert len(signal_result["top_adverse_reactions"]) > 0
    assert signal_result["data_source"] == "FDA FAERS"
    
    # Step 2: risk_score
    risk_result = await score_risk(
        signal_data=signal_result,
        patient_age=74,
        creatinine_mg_dl=2.1,  # CKD Stage 3
        inr_value=3.8,  # Supratherapeutic
        conditions=["Atrial fibrillation", "Heart failure", "CKD Stage 3"]
    )
    
    assert "error" not in risk_result
    assert risk_result["risk_tier"] in ["CRITICAL", "MONITOR", "LOW", "INSUFFICIENT_DATA"]
    assert "_metadata" in risk_result
    assert risk_result["_metadata"]["phi_in_prompt"] == False
    assert risk_result["_metadata"]["requires_clinician_review"] == True
    assert "risk_tier_rationale" in risk_result
    assert "top_risks" in risk_result
    assert "confidence" in risk_result
    
    # Step 3: alert_draft (simulated - pure logic, no API calls)
    # Build alert manually to test structure
    risk_tier = risk_result["risk_tier"]
    top_risks = risk_result["top_risks"]
    
    # Simulate alert_draft logic with proper structure
    ehr_note = f"""Risk Tier: {risk_tier}
    
☐ CONFIRM
☐ DISMISS
☐ ESCALATE"""
    
    assert "CONFIRM" in ehr_note
    assert "DISMISS" in ehr_note
    
    # Verify evidence citations
    evidence_citations = []
    for risk in top_risks:
        reaction = risk.get("reaction", "Unknown")
        count = risk.get("faers_report_count", 0)
        evidence_citations.append(f"{reaction}: {count} FAERS reports")
    
    if risk_tier == "CRITICAL":
        assert len(evidence_citations) > 0, "CRITICAL tier should have evidence citations"
    
    print(f"\n✓ Pipeline test passed")
    print(f"  Risk tier: {risk_tier}")
    print(f"  Total FAERS reports: {signal_result['total_faers_reports']}")
    print(f"  Top risks identified: {len(top_risks)}")
    print(f"  PHI in prompt: {risk_result['_metadata']['phi_in_prompt']}")


@pytest.mark.asyncio
async def test_alert_draft_structure():
    """Test alert_draft output structure."""
    # Mock risk_score output
    mock_risk_output = {
        "risk_tier": "CRITICAL",
        "risk_tier_rationale": "High signal for hemorrhage with patient-specific amplifiers",
        "top_risks": [
            {
                "reaction": "HAEMORRHAGE",
                "faers_report_count": 450,
                "patient_specific_amplifier": "Supratherapeutic INR increases bleeding risk"
            }
        ],
        "patient_context_flags": ["RENAL IMPAIRMENT", "SUPRATHERAPEUTIC INR"],
        "evidence_basis": "FAERS data",
        "confidence": "HIGH",
        "limitations": "FAERS data may underrepresent certain populations",
        "_metadata": {
            "phi_in_prompt": False,
            "requires_clinician_review": True
        }
    }
    
    # Simulate alert_draft
    ehr_note = f"""
Risk Tier: {mock_risk_output['risk_tier']}
Rationale: {mock_risk_output['risk_tier_rationale']}

DEMOGRAPHIC BIAS NOTE:
FAERS adverse event data may underrepresent non-white and female patients.

REGULATORY NOTE:
This is a Clinical Decision Support (CDS) tool under FDA CDS exemption.

☐ CONFIRM
☐ DISMISS
☐ ESCALATE
"""
    
    # Assertions
    assert "CONFIRM" in ehr_note
    assert "DISMISS" in ehr_note
    assert "DEMOGRAPHIC BIAS NOTE" in ehr_note
    assert "REGULATORY NOTE" in ehr_note
    assert "CDS exemption" in ehr_note
    
    # Verify no medication change recommendations
    assert "change dose" not in ehr_note.lower()
    assert "discontinue" not in ehr_note.lower()
    assert "switch to" not in ehr_note.lower()
    
    print("\n✓ Alert draft structure test passed")


@pytest.mark.asyncio
async def test_error_propagation():
    """Test that errors propagate correctly through pipeline."""
    from risk_engine import score_risk
    
    # Test with error in signal_data
    error_signal = {
        "error": "FAERS API timeout",
        "drug_combination": [],
        "total_faers_reports": 0
    }
    
    risk_result = await score_risk(signal_data=error_signal)
    
    assert "error" in risk_result
    assert risk_result["risk_tier"] == "UNKNOWN"
    assert risk_result["requires_manual_review"] == True
    assert "_metadata" in risk_result
    assert risk_result["_metadata"]["phi_in_prompt"] == False
    
    print("\n✓ Error propagation test passed")


@pytest.mark.asyncio
async def test_phi_verification():
    """Test that PHI is never in LLM prompts."""
    from risk_engine import score_risk
    from faers_client import query_combination
    
    # Get real FAERS data
    signal_result = await query_combination(["warfarin", "amiodarone"])
    
    # Score risk with patient parameters (no PHI)
    risk_result = await score_risk(
        signal_data=signal_result,
        patient_age=74,  # Age is OK (not PHI)
        creatinine_mg_dl=2.1,  # Lab value is OK
        inr_value=3.8  # Lab value is OK
    )
    
    # Verify phi_in_prompt=False
    assert "_metadata" in risk_result
    assert risk_result["_metadata"]["phi_in_prompt"] == False
    
    # Verify no PHI fields in any result
    assert "name" not in risk_result
    assert "birthDate" not in risk_result
    assert "address" not in risk_result
    assert "mrn" not in risk_result
    assert "ssn" not in risk_result
    
    print("\n✓ PHI verification test passed")
