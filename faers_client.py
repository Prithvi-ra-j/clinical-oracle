"""
FAERS API Client
Queries FDA Adverse Event Reporting System for drug combination signals.
No MCP, Groq, or FHIR imports - pure API logic only.
"""
import httpx
import os
from typing import Optional


FAERS_BASE_URL = "https://api.fda.gov/drug/event.json"


def classify_signal_strength(count: int) -> str:
    """Classify signal strength based on report count."""
    if count > 100:
        return "high"
    elif count >= 20:
        return "moderate"
    else:
        return "low"


async def query_combination(
    drugs: list[str],
    sex_code: Optional[str] = None,
    limit: int = 20
) -> dict:
    """
    Query FAERS API for adverse event reports involving a drug combination.
    
    Args:
        drugs: List of medication names (2+ required)
        sex_code: Optional FAERS sex code ("1" for male, "2" for female)
        limit: Max results to return (default 20)
    
    Returns:
        Dict with drug_combination, total_faers_reports, top_adverse_reactions, etc.
        On error, returns dict with "error" key and default values.
    """
    if len(drugs) < 2:
        return {
            "error": "At least 2 medications required for combination query",
            "drug_combination": drugs,
            "total_faers_reports": 0,
            "top_adverse_reactions": [],
            "data_source": "FDA FAERS",
            "disclaimer": "Error occurred during query"
        }
    
    # Build search query: AND-join all drug names
    search_parts = [f'patient.drug.medicinalproduct:"{drug.lower()}"' for drug in drugs]
    search_query = "+AND+".join(search_parts)
    
    # Add sex filter if provided
    if sex_code in ["1", "2"]:
        search_query += f"+AND+patient.patientsex:{sex_code}"
    
    # Build full URL
    url = f"{FAERS_BASE_URL}?search={search_query}&count=patient.reaction.reactionmeddrapt.exact&limit={limit}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            # Handle 404 - no reports found (not an error)
            if response.status_code == 404:
                return {
                    "drug_combination": drugs,
                    "total_faers_reports": 0,
                    "top_adverse_reactions": [],
                    "data_source": "FDA FAERS",
                    "disclaimer": "No FAERS reports found for this drug combination"
                }
            
            # Handle other non-200 status codes
            if response.status_code != 200:
                return {
                    "error": f"FAERS returned {response.status_code}",
                    "drug_combination": drugs,
                    "total_faers_reports": 0,
                    "top_adverse_reactions": [],
                    "data_source": "FDA FAERS",
                    "disclaimer": "Error occurred during query"
                }
            
            # Parse successful response
            data = response.json()
            results = data.get("results", [])
            
            # Build adverse reactions list with signal strength
            adverse_reactions = []
            for item in results[:10]:  # Max 10 reactions
                reaction_name = item.get("term", "Unknown")
                report_count = item.get("count", 0)
                signal_strength = classify_signal_strength(report_count)
                
                adverse_reactions.append({
                    "reaction": reaction_name,
                    "report_count": report_count,
                    "signal_strength": signal_strength
                })
            
            total_reports = sum(item.get("count", 0) for item in results)
            
            return {
                "drug_combination": drugs,
                "total_faers_reports": total_reports,
                "top_adverse_reactions": adverse_reactions,
                "data_source": "FDA FAERS",
                "disclaimer": "FAERS data represents reported adverse events and does not establish causation. Clinical judgment required."
            }
    
    except httpx.TimeoutException:
        return {
            "error": "FAERS API timeout",
            "drug_combination": drugs,
            "total_faers_reports": 0,
            "top_adverse_reactions": [],
            "data_source": "FDA FAERS",
            "disclaimer": "Error occurred during query"
        }
    
    except Exception as e:
        return {
            "error": f"FAERS query failed: {str(e)}",
            "drug_combination": drugs,
            "total_faers_reports": 0,
            "top_adverse_reactions": [],
            "data_source": "FDA FAERS",
            "disclaimer": "Error occurred during query"
        }


# Test harness for standalone execution
if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing FAERS client with warfarin + amiodarone...")
        result = await query_combination(["warfarin", "amiodarone"])
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Total reports: {result['total_faers_reports']}")
            print(f"Top reactions:")
            for reaction in result['top_adverse_reactions'][:5]:
                print(f"  - {reaction['reaction']}: {reaction['report_count']} ({reaction['signal_strength']})")
    
    asyncio.run(test())
