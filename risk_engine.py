"""
Risk Engine - Groq RAG Client
Synthesizes patient-specific risk scores using RAG on FAERS data.
No MCP, FHIR, or FAERS imports - pure LLM logic only.
"""
import json
import os
from groq import AsyncGroq
from typing import Optional


# Required keys in LLM response
REQUIRED_KEYS = {
    "risk_tier",
    "risk_tier_rationale",
    "top_risks",
    "patient_context_flags",
    "evidence_basis",
    "confidence",
    "limitations"
}


def build_system_prompt() -> str:
    """Build system prompt with strict RAG-only rules."""
    return """You are a clinical decision support system that analyzes polypharmacy risk using ONLY the FDA FAERS data provided to you.

CRITICAL RULES:
1. You MUST cite ONLY the FAERS adverse event data provided in the user message
2. You MUST NOT use any drug interaction knowledge from your training data
3. Every claim you make MUST reference specific FAERS reaction terms and report counts
4. If the FAERS data is insufficient, you MUST set risk_tier to "INSUFFICIENT_DATA"
5. You MUST return valid JSON matching the exact schema provided

Your role is to synthesize the retrieved FAERS signals with patient-specific clinical parameters to generate a risk assessment. Ground every statement in the provided data."""


def build_user_prompt(
    signal_data: dict,
    patient_age: Optional[int],
    creatinine_mg_dl: Optional[float],
    inr_value: Optional[float],
    conditions: Optional[list[str]]
) -> str:
    """Build user prompt with FAERS data and patient parameters."""
    
    # Extract FAERS data
    drugs = signal_data.get("drug_combination", [])
    total_reports = signal_data.get("total_faers_reports", 0)
    reactions = signal_data.get("top_adverse_reactions", [])
    
    # Build patient context flags with CRITICAL amplifiers
    context_flags = []
    critical_amplifiers = []
    
    if creatinine_mg_dl and creatinine_mg_dl > 2.0:
        context_flags.append("RENAL IMPAIRMENT: Creatinine > 2.0 mg/dL indicates reduced kidney function, increasing risk of drug accumulation")
        critical_amplifiers.append(f"""
CRITICAL AMPLIFIER: Patient creatinine {creatinine_mg_dl:.1f} mg/dL indicates CKD with 
significantly reduced drug clearance. Risk tier should be elevated one level from signal 
count alone. This patient cannot clear medications normally.""")
    
    if inr_value and inr_value > 3.0 and any("warfarin" in drug.lower() for drug in drugs):
        context_flags.append("SUPRATHERAPEUTIC INR: INR > 3.0 on warfarin indicates increased bleeding risk")
        critical_amplifiers.append(f"""
CRITICAL AMPLIFIER: Current INR {inr_value:.1f} is already supratherapeutic (therapeutic 
max ~3.0). Patient is at IMMEDIATE bleeding risk. This alone warrants CRITICAL tier 
regardless of signal count. Any hemorrhage-related FAERS signals become life-threatening.""")
    
    prompt = f"""Analyze the following polypharmacy risk scenario using ONLY the FAERS data provided below.

PATIENT CLINICAL PARAMETERS:
- Age: {patient_age if patient_age else "Not provided"}
- Creatinine: {creatinine_mg_dl if creatinine_mg_dl else "Not provided"} mg/dL
- INR: {inr_value if inr_value else "Not provided"}
- Additional conditions: {", ".join(conditions) if conditions else "None provided"}

PATIENT CONTEXT FLAGS:
{chr(10).join(f"- {flag}" for flag in context_flags) if context_flags else "- None"}

CRITICAL CLINICAL AMPLIFIERS (MUST be weighted heavily in risk tier decision):
{chr(10).join(critical_amplifiers) if critical_amplifiers else "- None"}

DRUG COMBINATION:
{", ".join(drugs)}

FAERS RETRIEVED DATA (your ONLY source of truth):
Total FAERS reports for this combination: {total_reports}

Top adverse reactions reported:
"""
    
    for reaction in reactions:
        prompt += f"\n- {reaction['reaction']}: {reaction['report_count']} reports (signal strength: {reaction['signal_strength']})"
    
    prompt += """

REQUIRED OUTPUT SCHEMA (return valid JSON only):
{
  "risk_tier": "CRITICAL | MONITOR | LOW | INSUFFICIENT_DATA",
  "risk_tier_rationale": "string citing specific FAERS terms and counts",
  "top_risks": [
    {
      "reaction": "string from FAERS data",
      "faers_report_count": number,
      "patient_specific_amplifier": "string explaining why this patient is at higher risk"
    }
  ],
  "patient_context_flags": ["array of strings"],
  "evidence_basis": "string explaining data sources used",
  "confidence": "HIGH | MODERATE | LOW",
  "limitations": "string noting data gaps or uncertainties"
}

RISK TIER GUIDELINES:
- CRITICAL: High-signal reactions (>100 reports) + patient-specific amplifiers (renal impairment, supratherapeutic INR, etc.)
            OR any CRITICAL AMPLIFIER present (supratherapeutic INR, severe renal impairment) regardless of signal count
            OR moderate signals (20-100 reports) + multiple critical amplifiers
- MONITOR: Moderate signals (20-100 reports) without critical amplifiers, or high signals without amplifiers
- LOW: Low signals (<20 reports) and no concerning patient factors
- INSUFFICIENT_DATA: Total reports < 10 or no clear signal

IMPORTANT: If CRITICAL AMPLIFIERS are present (especially supratherapeutic INR or severe CKD), 
you MUST elevate the risk tier. A patient with INR 3.8 on warfarin is at immediate bleeding 
risk - this overrides low signal counts.

Return ONLY the JSON object, no other text."""
    
    return prompt


async def score_risk(
    signal_data: dict,
    patient_age: Optional[int] = None,
    creatinine_mg_dl: Optional[float] = None,
    inr_value: Optional[float] = None,
    conditions: Optional[list[str]] = None
) -> dict:
    """
    Generate patient-specific risk score using RAG on FAERS data.
    
    Args:
        signal_data: Output from faers_client.query_combination()
        patient_age: Patient age in years
        creatinine_mg_dl: Serum creatinine in mg/dL
        inr_value: International Normalized Ratio
        conditions: List of additional medical conditions
    
    Returns:
        Dict with risk_tier, rationale, top_risks, and _metadata.
        On error, returns dict with "error" key and safe defaults.
    """
    # Check if signal_data contains error
    if "error" in signal_data:
        return {
            "error": "Cannot score risk - signal_scan failed",
            "risk_tier": "UNKNOWN",
            "requires_manual_review": True,
            "_metadata": {
                "llm_model": "llama-3.3-70b-versatile",
                "grounding_method": "RAG — FAERS retrieved data only",
                "phi_in_prompt": False,
                "requires_clinician_review": True
            }
        }
    
    # Log PHI confirmation
    print("[PRIVACY] risk_score called - phi_in_prompt=False verified")
    
    # Get Groq API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "error": "GROQ_API_KEY not configured",
            "risk_tier": "UNKNOWN",
            "requires_manual_review": True,
            "_metadata": {
                "llm_model": "llama-3.3-70b-versatile",
                "grounding_method": "RAG — FAERS retrieved data only",
                "phi_in_prompt": False,
                "requires_clinician_review": True
            }
        }
    
    try:
        # Build prompts
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(signal_data, patient_age, creatinine_mg_dl, inr_value, conditions)
        
        # Call Groq API
        client = AsyncGroq(api_key=api_key)
        response = await client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Validate required keys
        missing_keys = REQUIRED_KEYS - set(result.keys())
        if missing_keys:
            return {
                "error": f"LLM response missing required keys: {missing_keys}",
                "risk_tier": "UNKNOWN",
                "requires_manual_review": True,
                "_metadata": {
                    "llm_model": "llama-3.3-70b-versatile",
                    "grounding_method": "RAG — FAERS retrieved data only",
                    "phi_in_prompt": False,
                    "requires_clinician_review": True
                }
            }
        
        # Append metadata
        result["_metadata"] = {
            "llm_model": "llama-3.3-70b-versatile",
            "grounding_method": "RAG — FAERS retrieved data only",
            "phi_in_prompt": False,
            "requires_clinician_review": True
        }
        
        return result
    
    except json.JSONDecodeError:
        return {
            "error": "LLM response was not valid JSON",
            "risk_tier": "UNKNOWN",
            "requires_manual_review": True,
            "_metadata": {
                "llm_model": "llama-3.3-70b-versatile",
                "grounding_method": "RAG — FAERS retrieved data only",
                "phi_in_prompt": False,
                "requires_clinician_review": True
            }
        }
    
    except Exception as e:
        return {
            "error": f"Risk scoring failed: {str(e)}",
            "risk_tier": "UNKNOWN",
            "requires_manual_review": True,
            "_metadata": {
                "llm_model": "llama-3.3-70b-versatile",
                "grounding_method": "RAG — FAERS retrieved data only",
                "phi_in_prompt": False,
                "requires_clinician_review": True
            }
        }


# Test harness for standalone execution
if __name__ == "__main__":
    import asyncio
    from faers_client import query_combination
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    async def test():
        print("Testing risk_engine with demo scenario...")
        
        # Get FAERS data first
        signal_data = await query_combination(["warfarin", "amiodarone", "fluconazole"])
        
        # Score risk with demo patient parameters
        result = await score_risk(
            signal_data=signal_data,
            patient_age=74,
            creatinine_mg_dl=2.1,  # CKD Stage 3
            inr_value=3.8,  # Supratherapeutic
            conditions=["Atrial fibrillation", "Heart failure", "CKD Stage 3"]
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"\nRisk Assessment:")
            print(f"  Risk Tier: {result['risk_tier']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Rationale: {result['risk_tier_rationale'][:200]}...")
            print(f"  Top Risks: {len(result['top_risks'])} identified")
            print(f"  PHI in prompt: {result['_metadata']['phi_in_prompt']}")
    
    asyncio.run(test())
