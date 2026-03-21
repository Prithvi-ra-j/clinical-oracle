"""
Unit tests for FHIR bridge
"""
import pytest
from fhir_bridge import get_patient_context, calculate_age


def test_calculate_age():
    """Test age calculation from birthDate."""
    # Test with a known date (will need adjustment based on current year)
    age = calculate_age("1950-01-01")
    assert age is not None
    assert age > 70  # Should be 70+ years old
    
    # Test with invalid date
    age = calculate_age("invalid")
    assert age is None


@pytest.mark.asyncio
async def test_get_patient_context_structure():
    """Test that patient context has correct structure and PHI is stripped."""
    result = await get_patient_context("example")
    
    # Required fields
    assert "medications" in result
    assert "age" in result
    assert "sex" in result
    assert "creatinine_mg_dl" in result
    assert "inr_value" in result
    assert "fhir_server" in result
    
    # PHI fields must NOT be present
    assert "name" not in result
    assert "address" not in result
    assert "birthDate" not in result  # Only age, not raw birthDate
    assert "telecom" not in result
    assert "identifier" not in result
    
    # Medications should be lowercase
    for med in result["medications"]:
        assert med == med.lower()


@pytest.mark.asyncio
async def test_get_patient_context_invalid_patient():
    """Test with non-existent patient ID."""
    result = await get_patient_context("nonexistent-patient-12345")
    
    # Should return empty context, not raise exception
    assert result["medications"] == []
    assert result["age"] is None
    assert result["sex"] is None
    assert result["creatinine_mg_dl"] is None
    assert result["inr_value"] is None


@pytest.mark.asyncio
async def test_get_patient_context_custom_fhir_base():
    """Test with custom FHIR base URL."""
    result = await get_patient_context("example", fhir_base="https://hapi.fhir.org/baseR4")
    
    assert result["fhir_server"] == "https://hapi.fhir.org/baseR4"
