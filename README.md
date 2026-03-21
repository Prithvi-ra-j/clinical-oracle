# Clinical Oracle

MCP server for polypharmacy risk intelligence using real FDA FAERS data.

## Quick Start

1. Clone and setup:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. Run locally:
   ```bash
   python server.py --transport streamable-http --port 8000
   ```

## Tools

- **signal_scan**: Query FDA FAERS for drug combination adverse events
- **risk_score**: Generate patient-specific risk tier using RAG
- **alert_draft**: Create EHR-pasteable clinical alert
- **health_check**: Verify server and API health

## Deployment

See Procfile for Railway deployment configuration.

## Privacy

No PHI is sent to external APIs. Only extracted clinical parameters are used.

## License

MIT
