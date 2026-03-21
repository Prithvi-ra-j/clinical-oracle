"""
Demo Scenario - Full Pipeline Test
Runs the complete Clinical Oracle pipeline with demo patient data.
"""
import asyncio
import json
from dotenv import load_dotenv
from faers_client import query_combination
from risk_engine import score_risk

# Load environment variables from .env file
load_dotenv()


# Demo scenario from design.md
DEMO_SCENARIO = {
    "patient_identifier": "PT-DEMO-001",
    "description": "74M with CKD Stage 3, AF, heart failure",
    "medications": ["warfarin", "amiodarone", "fluconazole", "furosemide", "digoxin"],
    "age": 74,
    "sex": "male",
    "creatinine_mg_dl": 2.1,   # CKD Stage 3
    "inr_value": 3.8,           # Supratherapeutic
    "conditions": ["Atrial fibrillation", "Heart failure", "CKD Stage 3"],
    "expected_risk_tier": "CRITICAL",
    "expected_top_reactions": ["HAEMORRHAGE", "INTERNATIONAL NORMALISED RATIO INCREASED"]
}


async def run_demo():
    """Run full pipeline with demo scenario."""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("CLINICAL ORACLE - DEMO SCENARIO")
    print("=" * 70)
    print(f"\nPatient: {DEMO_SCENARIO['patient_identifier']}")
    print(f"Description: {DEMO_SCENARIO['description']}")
    print(f"Medications: {', '.join(DEMO_SCENARIO['medications'])}")
    print(f"Age: {DEMO_SCENARIO['age']}")
    print(f"Creatinine: {DEMO_SCENARIO['creatinine_mg_dl']} mg/dL")
    print(f"INR: {DEMO_SCENARIO['inr_value']}")
    print(f"Conditions: {', '.join(DEMO_SCENARIO['conditions'])}")
    
    # Step 1: signal_scan
    print("\n" + "-" * 70)
    print("STEP 1: SIGNAL_SCAN (querying FDA FAERS)")
    print("-" * 70)
    
    signal_result = await query_combination(DEMO_SCENARIO["medications"])
    
    if "error" in signal_result:
        print(f"ERROR: {signal_result['error']}")
        return
    
    print(f"✓ Total FAERS reports: {signal_result['total_faers_reports']}")
    print(f"✓ Top adverse reactions:")
    for i, reaction in enumerate(signal_result['top_adverse_reactions'][:5], 1):
        print(f"  {i}. {reaction['reaction']}: {reaction['report_count']} reports ({reaction['signal_strength']})")
    
    # Step 2: risk_score
    print("\n" + "-" * 70)
    print("STEP 2: RISK_SCORE (RAG-based risk assessment)")
    print("-" * 70)
    
    risk_result = await score_risk(
        signal_data=signal_result,
        patient_age=DEMO_SCENARIO["age"],
        creatinine_mg_dl=DEMO_SCENARIO["creatinine_mg_dl"],
        inr_value=DEMO_SCENARIO["inr_value"],
        conditions=DEMO_SCENARIO["conditions"]
    )
    
    if "error" in risk_result:
        print(f"ERROR: {risk_result['error']}")
        return
    
    print(f"✓ Risk Tier: {risk_result['risk_tier']}")
    print(f"✓ Confidence: {risk_result['confidence']}")
    print(f"✓ Rationale: {risk_result['risk_tier_rationale'][:200]}...")
    print(f"✓ Top Risks: {len(risk_result['top_risks'])} identified")
    print(f"✓ PHI in prompt: {risk_result['_metadata']['phi_in_prompt']}")
    
    # Step 3: alert_draft (simulated)
    print("\n" + "-" * 70)
    print("STEP 3: ALERT_DRAFT (EHR-pasteable note)")
    print("-" * 70)
    
    ehr_note = f"""
═══════════════════════════════════════════════════════════════
CLINICAL DECISION SUPPORT ALERT — POLYPHARMACY RISK ASSESSMENT
═══════════════════════════════════════════════════════════════

Patient: {DEMO_SCENARIO['patient_identifier']}
Risk Tier: {risk_result['risk_tier']}
Confidence: {risk_result['confidence']}

RISK ASSESSMENT:
{risk_result['risk_tier_rationale']}

TOP IDENTIFIED RISKS:
"""
    
    for i, risk in enumerate(risk_result['top_risks'][:3], 1):
        ehr_note += f"\n{i}. {risk['reaction']} ({risk['faers_report_count']} FAERS reports)\n   {risk['patient_specific_amplifier']}\n"
    
    ehr_note += """
───────────────────────────────────────────────────────────────
DEMOGRAPHIC BIAS NOTE:
FAERS adverse event data may underrepresent non-white and female
patients. Clinical judgment should account for potential reporting
biases in the source data.
───────────────────────────────────────────────────────────────

REGULATORY NOTE:
This is a Clinical Decision Support (CDS) tool under FDA CDS
exemption (21st Century Cures Act). This is NOT a Software as a
Medical Device (SaMD).
───────────────────────────────────────────────────────────────

CLINICIAN ACTION REQUIRED:

☐ CONFIRM — I have reviewed this alert and will take appropriate action
☐ DISMISS — I have reviewed this alert and determined no action needed
☐ ESCALATE — I am escalating this to [specialist/pharmacist]

Clinician Name: _______________________________________________

Date/Time: ____________________________________________________
═══════════════════════════════════════════════════════════════
"""
    
    print(ehr_note)
    
    # Performance check
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    print("=" * 70)
    
    if elapsed < 30:
        print("✓ Performance requirement met (< 30 seconds)")
    else:
        print("⚠ Performance requirement not met (> 30 seconds)")
    
    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    checks = []
    checks.append(("Risk tier is CRITICAL", risk_result['risk_tier'] == "CRITICAL"))
    checks.append(("HAEMORRHAGE in top reactions", any("HAEMORRHAGE" in r['reaction'].upper() for r in risk_result['top_risks'])))
    checks.append(("INR INCREASED in signal data", any("INR" in r['reaction'].upper() for r in signal_result['top_adverse_reactions'])))
    checks.append(("PHI not in prompt", risk_result['_metadata']['phi_in_prompt'] == False))
    checks.append(("Clinician review required", risk_result['_metadata']['requires_clinician_review'] == True))
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    # Save output
    output = {
        "demo_scenario": DEMO_SCENARIO,
        "signal_scan_result": signal_result,
        "risk_score_result": risk_result,
        "ehr_note": ehr_note,
        "elapsed_seconds": elapsed,
        "verification_checks": {name: passed for name, passed in checks}
    }
    
    with open("demo_output.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Output saved to demo_output.json")


if __name__ == "__main__":
    asyncio.run(run_demo())
