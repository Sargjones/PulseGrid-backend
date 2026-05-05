# PulseGrid Backend

FastAPI backend server for the PulseGrid civic infrastructure awareness app — Brantford, Ontario.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check |
| `GET /api/weather` | Live Environment Canada weather + alerts |
| `GET /api/roads` | Road events near Brantford (Ontario 511) |
| `GET /api/briefing` | AI civic briefing (Claude) |
| `GET /api/all` | All three in one call (used by PWA) |

## Deploy to Railway

1. Push this repo to GitHub
2. Connect repo in Railway → New Project → Deploy from GitHub
3. Add environment variable: `ANTHROPIC_API_KEY=your_key_here`
4. Railway auto-detects Python and deploys

## Local development

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
uvicorn main:app --reload
```

Then open http://localhost:8000/docs for the interactive API docs.

## Data sources

- **Weather**: Environment Canada RSS (weather.gc.ca) — no key required
- **Roads**: Ontario 511 REST API — no key required  
- **AI Briefing**: Anthropic Claude — API key required
- **Cache**: In-memory, 5-minute TTL per endpoint
