"""
FHIR R4 Bridge
Fetches patient context from FHIR server and strips PHI.
No MCP, Groq, or FAERS imports - pure FHIR logic only.
"""
import httpx
import os
from datetime import datetime
from typing import Optional


DEFAULT_FHIR_BASE = "https://hapi.fhir.org/baseR4"


def calculate_age(birth_date_str: str) -> Optional[int]:
    """Calculate age from FHIR birthDate string (YYYY-MM-DD)."""
    try:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return age
    except:
        return None


async def get_patient_context(
    patient_id: str,
    fhir_base: Optional[str] = None
) -> dict:
    """
    Fetch patient clinical context from FHIR server.
    
    PHI STRIPPING: This function ONLY returns extracted clinical parameters.
    NO patient name, address, MRN, telecom, or raw birthDate is returned.
    
    Args:
        patient_id: FHIR Patient resource ID
        fhir_base: FHIR server base URL (defaults to HAPI public server)
    
    Returns:
        Dict with: medications (list[str]), age (int), sex (str),
        creatinine_mg_dl (float|None), inr_value (float|None), fhir_server (str)
    """
    if fhir_base is None:
        fhir_base = os.getenv("FHIR_BASE_URL", DEFAULT_FHIR_BASE)
    
    # Initialize empty context (returned on error)
    context = {
        "medications": [],
        "age": None,
        "sex": None,
        "creatinine_mg_dl": None,
        "inr_value": None,
        "fhir_server": fhir_base
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Fetch Patient resource
            try:
                patient_response = await client.get(f"{fhir_base}/Patient/{patient_id}")
                if patient_response.status_code == 200:
                    patient_data = patient_response.json()
                    
                    # Extract age from birthDate (PHI-safe: only age, not raw date)
                    birth_date = patient_data.get("birthDate")
                    if birth_date:
                        context["age"] = calculate_age(birth_date)
                    
                    # Extract sex from gender
                    gender = patient_data.get("gender")
                    if gender in ["male", "female"]:
                        context["sex"] = gender
            except:
                pass  # Continue even if Patient fetch fails
            
            # Fetch active MedicationRequest resources
            try:
                med_response = await client.get(
                    f"{fhir_base}/MedicationRequest",
                    params={"patient": patient_id, "status": "active", "_count": 100}
                )
                if med_response.status_code == 200:
                    med_data = med_response.json()
                    entries = med_data.get("entry", [])
                    
                    medications = []
                    for entry in entries:
                        resource = entry.get("resource", {})
                        
                        # Try medicationCodeableConcept first
                        med_concept = resource.get("medicationCodeableConcept", {})
                        if med_concept:
                            codings = med_concept.get("coding", [])
                            for coding in codings:
                                display = coding.get("display")
                                if display:
                                    medications.append(display.lower())
                                    break
                            if not codings:
                                # Try text field
                                text = med_concept.get("text")
                                if text:
                                    medications.append(text.lower())
                    
                    context["medications"] = list(set(medications))  # Deduplicate
            except:
                pass  # Continue even if MedicationRequest fetch fails
            
            # Fetch laboratory Observation resources (sorted by date descending)
            try:
                obs_response = await client.get(
                    f"{fhir_base}/Observation",
                    params={
                        "patient": patient_id,
                        "category": "laboratory",
                        "_sort": "-date",
                        "_count": 100
                    }
                )
                if obs_response.status_code == 200:
                    obs_data = obs_response.json()
                    entries = obs_data.get("entry", [])
                    
                    for entry in entries:
                        resource = entry.get("resource", {})
                        code = resource.get("code", {})
                        
                        # Get code text
                        code_text = code.get("text", "").lower()
                        
                        # Also check coding displays
                        codings = code.get("coding", [])
                        for coding in codings:
                            display = coding.get("display", "").lower()
                            code_text += " " + display
                        
                        # Extract creatinine (first match only - most recent)
                        if context["creatinine_mg_dl"] is None and "creatinine" in code_text:
                            value_quantity = resource.get("valueQuantity", {})
                            value = value_quantity.get("value")
                            if value is not None:
                                context["creatinine_mg_dl"] = float(value)
                        
                        # Extract INR (first match only - most recent)
                        if context["inr_value"] is None:
                            if "inr" in code_text or "international normalised ratio" in code_text or "international normalized ratio" in code_text:
                                value_quantity = resource.get("valueQuantity", {})
                                value = value_quantity.get("value")
                                if value is not None:
                                    context["inr_value"] = float(value)
            except:
                pass  # Continue even if Observation fetch fails
    
    except httpx.TimeoutException:
        # Return empty context on timeout
        return context
    except Exception:
        # Return empty context on any other error
        return context
    
    return context


# Test harness for standalone execution
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing FHIR bridge with HAPI public server...")
        print("Note: Using synthetic patient data from HAPI FHIR")
        
        # Try a few patient IDs (HAPI has synthetic data)
        test_patient_id = "example"  # HAPI has an 'example' patient
        
        result = await get_patient_context(test_patient_id)
        
        print(f"\nPatient context (PHI stripped):")
        print(f"  Medications: {result['medications']}")
        print(f"  Age: {result['age']}")
        print(f"  Sex: {result['sex']}")
        print(f"  Creatinine: {result['creatinine_mg_dl']} mg/dL")
        print(f"  INR: {result['inr_value']}")
        print(f"  FHIR server: {result['fhir_server']}")
        
        # Verify no PHI fields present
        assert "name" not in result
        assert "address" not in result
        assert "birthDate" not in result
        assert "telecom" not in result
        print("\n✓ PHI stripping verified - no name, address, birthDate, or telecom in output")
    
    asyncio.run(test())
