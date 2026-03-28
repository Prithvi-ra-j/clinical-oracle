# Clinical Oracle

> Polypharmacy risk intelligence — real FDA signals, zero PHI, clinician always in the loop.

An MCP superpower built for the **Agents Assemble: The Healthcare AI Endgame** hackathon by Prompt Opinion.

[![Live on Prompt Opinion](https://img.shields.io/badge/Prompt%20Opinion-Published-teal)](https://app.promptopinion.ai/api/workspaces/019d125a-5b69-700c-b7a4-5fe6fbd6e8eb/ai-agents/019d34a7-80aa-71df-b1f4-cedae308595d/.well-known/agent-card.json)
[![MCP Server](https://img.shields.io/badge/MCP-Streamable%20HTTP-blue)](https://clinical-oracle.onrender.com/mcp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

Clinical Oracle queries real FDA FAERS adverse event data for dangerous drug combinations, synthesizes patient-specific risk tiers using RAG, and delivers EHR-pasteable clinical alerts with mandatory clinician confirmation.

**Three composable MCP tools:**

| Tool | What it does |
|------|-------------|
| `signal_scan` | Queries FDA FAERS API live for a drug combination — real reported adverse events, not LLM memory |
| `risk_score` | RAG-based risk tier (CRITICAL/MONITOR/LOW) grounded exclusively in retrieved FAERS data + patient amplifiers |
| `alert_draft` | EHR-pasteable clinical note with Confirm/Dismiss/Escalate — clinician always in the loop |
| `health_check` | Verifies FAERS API reachability and server health |

---

## Demo

**Demo scenario:** 74M, CKD Stage 3, warfarin + amiodarone + fluconazole + furosemide + digoxin, Creatinine 2.1, INR 3.8

**Result:** CRITICAL risk tier — supratherapeutic INR + renal impairment flagged as critical amplifiers, HYPOTENSION and RENAL FAILURE ACUTE surfaced from real FAERS reports.

[Watch the demo video](https://youtu.be/uoA9vFH-w7Q)

---

## Architecture
```
Clinician prompt
      │
      ▼
[A2A Agent — Prompt Opinion]
      │
      ├── signal_scan(medications) ──────► FDA FAERS API (real data)
      │         │
      │         ▼
      ├── risk_score(faers_output, age, creatinine, inr)
      │         │
      │         └──► Groq (llama-3.3-70b) ◄── RAG prompt, FAERS data only
      │                    │                    phi_in_prompt=False enforced
      │                    ▼
      └── alert_draft(risk_score_output)
                │
                ▼
      EHR-pasteable note → Clinician confirms or dismisses
```

PHI never reaches the LLM. The FHIR bridge strips all patient identifiers — only extracted clinical parameters (age, creatinine value, INR value, medication names) are used in prompts.

---

## Key differentiators

- **Real FDA data** — queries `api.fda.gov/drug/event.json` live on every call, not a static database
- **RAG, not memory** — LLM system prompt explicitly prohibits using training knowledge about drug interactions
- **`phi_in_prompt=False`** — enforced, verified, and logged on every `risk_score` call
- **SHARP/FHIR native** — accepts `sharp_patient_id` and pulls medications automatically from FHIR R4
- **Demographic bias disclosure** — every alert flags that FAERS may underrepresent non-white and female patients
- **FDA CDS exemption compliant** — regulatory note baked into every alert, never recommends medication changes
- **Full A2A composability** — published on Prompt Opinion marketplace, any agent can call it as a skill

---

## Quick Start

**Prerequisites:** Python 3.11+, Groq API key (free at console.groq.com)
```bash
# 1. Clone and setup
git clone https://github.com/Prithvi-ra-j/clinical-oracle.git
cd clinical-oracle
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 3. Run locally
python server.py --transport streamable-http --port 8000
```

Server will be available at `http://localhost:8000/mcp`

---

## Environment Variables
```bash
GROQ_API_KEY=        # Required — get from console.groq.com
FHIR_BASE_URL=       # Optional — default: https://hapi.fhir.org/baseR4
FAERS_BASE_URL=      # Optional — default: https://api.fda.gov/drug/event.json
PORT=                # Set automatically by Render
```

---

## Live Deployment

- **MCP Server:** `https://clinical-oracle.onrender.com/mcp`
- **Prompt Opinion Marketplace:** [A2A Agent Card](https://app.promptopinion.ai/api/workspaces/019d125a-5b69-700c-b7a4-5fe6fbd6e8eb/ai-agents/019d34a7-80aa-71df-b1f4-cedae308595d/.well-known/agent-card.json)

---

## Tools

- **signal_scan**: Query FDA FAERS for drug combination adverse events
- **risk_score**: Generate patient-specific risk tier using RAG
- **alert_draft**: Create EHR-pasteable clinical alert
- **health_check**: Verify server and API health

---

## Privacy & Compliance

- No PHI (name, DOB, address, MRN) is ever sent to any external LLM API
- Only extracted clinical parameters are used in prompts
- `phi_in_prompt=False` is logged on every risk_score call
- FDA CDS exemption compliant — this is NOT a Software as a Medical Device
- Clinician review and confirmation required for every alert

---

## Tech Stack

- **MCP Framework:** FastMCP 2.3.3
- **LLM:** Groq API (llama-3.3-70b-versatile)
- **Adverse Event Data:** FDA FAERS public API
- **Patient Context:** HAPI FHIR R4
- **Deployment:** Render
- **Platform:** Prompt Opinion (MCP + A2A)

---

## License

MIT
