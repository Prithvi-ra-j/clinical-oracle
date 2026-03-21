"""
Unit tests for FAERS client
"""
import pytest
from faers_client import query_combination, classify_signal_strength


def test_signal_strength_classification():
    """Test signal strength classification logic."""
    assert classify_signal_strength(150) == "high"
    assert classify_signal_strength(101) == "high"
    assert classify_signal_strength(100) == "moderate"
    assert classify_signal_strength(50) == "moderate"
    assert classify_signal_strength(20) == "moderate"
    assert classify_signal_strength(19) == "low"
    assert classify_signal_strength(5) == "low"
    assert classify_signal_strength(0) == "low"


@pytest.mark.asyncio
async def test_query_combination_insufficient_drugs():
    """Test that single drug returns validation error."""
    result = await query_combination(["warfarin"])
    assert "error" in result
    assert "At least 2 medications required" in result["error"]
    assert result["total_faers_reports"] == 0


@pytest.mark.asyncio
async def test_query_combination_success():
    """Test successful query with real FAERS data."""
    result = await query_combination(["warfarin", "amiodarone"])
    
    # Should not have error key
    assert "error" not in result
    
    # Should have required fields
    assert "drug_combination" in result
    assert "total_faers_reports" in result
    assert "top_adverse_reactions" in result
    assert "data_source" in result
    assert "disclaimer" in result
    
    # Should have actual data
    assert result["total_faers_reports"] > 0
    assert len(result["top_adverse_reactions"]) > 0
    
    # Check reaction structure
    first_reaction = result["top_adverse_reactions"][0]
    assert "reaction" in first_reaction
    assert "report_count" in first_reaction
    assert "signal_strength" in first_reaction
    assert first_reaction["signal_strength"] in ["high", "moderate", "low"]


@pytest.mark.asyncio
async def test_query_combination_with_sex_filter():
    """Test query with sex filter."""
    result = await query_combination(["warfarin", "amiodarone"], sex_code="1")
    
    # Should succeed
    assert "error" not in result
    assert result["total_faers_reports"] >= 0


@pytest.mark.asyncio
async def test_query_combination_no_results():
    """Test query that returns no results (404)."""
    # Use nonsense drug names that won't have reports
    result = await query_combination(["xyzabc123", "qwerty999"])
    
    # Should not have error key (404 is not an error)
    assert "error" not in result
    assert result["total_faers_reports"] == 0
    assert len(result["top_adverse_reactions"]) == 0
    assert "No FAERS reports found" in result["disclaimer"]
