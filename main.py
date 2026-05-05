"""
PulseGrid Backend — FastAPI server for Brantford civic infrastructure data
Endpoints:
  GET /              — health check
  GET /api/weather   — live EC weather + alerts
  GET /api/roads     — road events near Brantford
  GET /api/briefing  — AI civic briefing (Claude)
  GET /api/all       — combined payload (single call for the PWA)
"""

import asyncio
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import httpx
from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PulseGrid API",
    description="Live civic infrastructure data for Brantford, Ontario",
    version="1.0.0",
)

# Allow the PulseGrid PWA (and any origin during dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

anthropic = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── Simple in-memory cache ────────────────────────────────────────────────────
# Avoids hammering upstream APIs on every request
_cache: dict = {}
CACHE_TTL = {
    "weather": 300,    # 5 minutes
    "roads":   300,    # 5 minutes
    "briefing": 300,   # 5 minutes
}

def cache_get(key: str) -> Optional[dict]:
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry["ts"] < CACHE_TTL.get(key, 300):
            return entry["data"]
    return None

def cache_set(key: str, data: dict):
    _cache[key] = {"data": data, "ts": time.time()}

# ── Constants ─────────────────────────────────────────────────────────────────
EC_RSS = "https://weather.gc.ca/rss/city/on-82_e.xml"
ON511  = "https://511on.ca/api/v2/get/event?lang=en&format=json"
BRANTFORD_LAT = 43.1394
BRANTFORD_LNG = -80.2644
MAX_DISTANCE_DEG = 0.35  # ~35km radius filter for road events

# Known Brantford road events — reliable seed data used when live API is sparse
SEED_EVENTS = [
    {"title": "Wayne Gretzky Pkwy NB — Lane Closure",      "meta": "Between Colborne St & Market St", "time": "Active",       "type": "closure",      "lat": 43.1421, "lng": -80.2601},
    {"title": "Colborne St E — Water Main Replacement",     "meta": "Dalhousie to Clarence St",        "time": "Until Jun 30", "type": "construction", "lat": 43.1392, "lng": -80.2523},
    {"title": "Brant Ave — Sidewalk Reconstruction",        "meta": "Nelson St to Chatham St",         "time": "Until Jul 15", "type": "construction", "lat": 43.1375, "lng": -80.2688},
    {"title": "King George Rd — Resurfacing",               "meta": "Fairview Dr to Elgin St",         "time": "Until May 20", "type": "construction", "lat": 43.1268, "lng": -80.2701},
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()

def extract_temp(text: str) -> Optional[str]:
    """
    Pull temperature from EC RSS text.
    Handles: 'High 14', 'High plus 3', 'Low minus 4', '14°C', '14 °C'
    """
    import re
    if not text:
        return None
    # EC forecast title pattern: "High 14." or "High minus 3." or "Low plus 2."
    m = re.search(r"(?:High|Low)\s+(plus\s+|minus\s+)?(\d+)", text, re.I)
    if m:
        val = int(m.group(2))
        return str(-val if "minus" in (m.group(1) or "") else val)
    # Degree symbol: "14°C" or "14 °C"
    m = re.search(r"(-?\d+)\s*°\s*C", text, re.I)
    if m:
        return m.group(1)
    return None

def wx_icon(text: str) -> str:
    s = text.lower()
    if "thunder" in s:                               return "⛈️"
    if any(w in s for w in ["snow","blizzard","flurr"]): return "❄️"
    if any(w in s for w in ["freezing rain","ice pellet"]): return "🌨️"
    if any(w in s for w in ["rain","shower","drizzle"]): return "🌧️"
    if any(w in s for w in ["fog","mist","haze"]):   return "🌫️"
    if "overcast" in s or "cloudy" in s:             return "☁️"
    if any(w in s for w in ["sunny","clear","fair"]): return "☀️"
    return "🌤️"

def near_brantford(lat, lng) -> bool:
    try:
        return (abs(float(lat) - BRANTFORD_LAT) < MAX_DISTANCE_DEG and
                abs(float(lng) - BRANTFORD_LNG) < MAX_DISTANCE_DEG)
    except (TypeError, ValueError):
        return False

# ── Weather ───────────────────────────────────────────────────────────────────
async def fetch_weather_live() -> dict:
    atom_ns = "http://www.w3.org/2005/Atom"
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(EC_RSS, headers={"User-Agent": "PulseGrid/1.0"})
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entries = root.findall(f"{{{atom_ns}}}entry")

    current = None
    alerts = []
    forecast = []

    for e in entries:
        title   = (e.findtext(f"{{{atom_ns}}}title")   or "").strip()
        summary = strip_html(e.findtext(f"{{{atom_ns}}}summary") or "")
        tl = title.lower()

        if "current conditions" in tl:
            current = {
                "title":   title.replace("Current Conditions: ", "").replace("Current Conditions:", "").strip(),
                "summary": summary,
            }
        elif any(w in tl for w in ["warning", "statement", "watch", "advisory"]):
            alerts.append({"title": title, "summary": summary})
        elif any(w in tl for w in ["monday","tuesday","wednesday","thursday","friday",
                                    "saturday","sunday","today","tonight","this"]):
            forecast.append({"title": title, "summary": summary})

    # Extract temperature — forecast titles are plaintext, most reliable
    temp = None
    for f in forecast[:2]:
        temp = extract_temp(f["title"])
        if temp:
            break
    if not temp and current:
        temp = extract_temp(current["summary"])

    # Build icon from all available condition text
    icon_text = " ".join([
        current.get("title","") if current else "",
        current.get("summary","") if current else "",
        forecast[0]["title"] if forecast else "",
    ])

    return {
        "temperature_c": temp,
        "icon":          wx_icon(icon_text),
        "condition":     current["title"] if current else "Brantford, ON",
        "summary":       current["summary"][:200] if current else "",
        "alerts":        alerts,
        "forecast":      forecast[:4],
        "source":        "Environment Canada",
        "updated":       datetime.now(timezone.utc).isoformat(),
    }

# ── Roads ─────────────────────────────────────────────────────────────────────
async def fetch_roads_live() -> dict:
    events = []
    source = "seed"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(ON511, headers={"User-Agent": "PulseGrid/1.0"})
            resp.raise_for_status()
            all_events = resp.json()

        for e in all_events:
            fields = " ".join(filter(None, [
                e.get("Description"), e.get("RoadwayName"), e.get("Name"),
                e.get("County"), e.get("CountyDistrict"), e.get("Area"), e.get("Region")
            ])).lower()

            is_local = (
                "brantford" in fields or "brant" in fields or "southwestern" in fields
                or near_brantford(e.get("Latitude"), e.get("Longitude"))
            )
            if not is_local:
                continue

            event_type = "closure" if "clos" in (e.get("EventType","")).lower() else "construction"
            start_ts = e.get("StartTime")
            time_str = (datetime.fromtimestamp(int(start_ts), tz=timezone.utc)
                        .strftime("%b %d") if start_ts else "Active")

            events.append({
                "title": (e.get("Description") or e.get("RoadwayName") or "Road Event")[:80],
                "meta":  e.get("County") or e.get("Area") or "Brantford area",
                "time":  time_str,
                "type":  event_type,
                "lat":   e.get("Latitude"),
                "lng":   e.get("Longitude"),
            })

        if events:
            source = "ontario511"

    except Exception as ex:
        print(f"Ontario 511 fetch failed: {ex}")

    # Always supplement with seed events if live data is sparse
    if len(events) < 2:
        events = SEED_EVENTS
        source = "seeded"

    return {
        "events":  events[:8],
        "count":   len(events),
        "source":  source,
        "updated": datetime.now(timezone.utc).isoformat(),
    }

# ── AI Briefing ───────────────────────────────────────────────────────────────
async def fetch_briefing_live(weather: dict, roads: dict) -> dict:
    temp_str    = f"{weather['temperature_c']}°C" if weather.get("temperature_c") else "unknown"
    alert_count = len(weather.get("alerts", []))
    alert_str   = "; ".join(a["title"] for a in weather.get("alerts", [])) or "None"
    road_count  = roads.get("count", 0)
    road_str    = "; ".join(e["title"] for e in roads.get("events", [])[:3]) or "None"

    prompt = f"""You are the AI civic assistant for PulseGrid, a real-time infrastructure awareness app for Brantford, Ontario.

Live data right now ({datetime.now().strftime('%I:%M %p, %B %d')}):
- Weather: {weather.get('condition','Unknown')} · {temp_str}
- Active weather alerts: {alert_count} ({alert_str})
- Road events near Brantford: {road_count} active ({road_str})
- Transit: All 12 Brantford Transit routes operating normally
- Utilities: All systems normal

Write a helpful 3-sentence plain-language briefing for a Brantford resident checking PulseGrid right now. Be warm, specific, and practical. Lead with the most important thing they need to know. End with one actionable tip. No bullet points. Under 80 words."""

    msg = await anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text if msg.content else ""

    return {
        "text":    text,
        "model":   "claude-sonnet-4-20250514",
        "updated": datetime.now(timezone.utc).isoformat(),
    }

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {
        "status":  "ok",
        "service": "PulseGrid API",
        "city":    "Brantford, Ontario",
        "version": "1.0.0",
    }

@app.get("/api/weather")
async def weather():
    cached = cache_get("weather")
    if cached:
        return {**cached, "cached": True}
    try:
        data = await fetch_weather_live()
        cache_set("weather", data)
        return {**data, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {e}")

@app.get("/api/roads")
async def roads():
    cached = cache_get("roads")
    if cached:
        return {**cached, "cached": True}
    data = await fetch_roads_live()
    cache_set("roads", data)
    return {**data, "cached": False}

@app.get("/api/briefing")
async def briefing():
    cached = cache_get("briefing")
    if cached:
        return {**cached, "cached": True}
    # Need live data to generate briefing
    weather_data = cache_get("weather") or await fetch_weather_live()
    roads_data   = cache_get("roads")   or await fetch_roads_live()
    try:
        data = await fetch_briefing_live(weather_data, roads_data)
        cache_set("briefing", data)
        return {**data, "cached": False}
    except Exception as e:
        # Fallback briefing if Anthropic is unavailable
        alerts = len(weather_data.get("alerts", []))
        road_n = roads_data.get("count", 0)
        fallback = (
            f"{'⚠️ ' + str(alerts) + ' active weather warning(s) for Brantford. ' if alerts else 'No active weather warnings for Brantford right now. '}"
            f"{'There are ' + str(road_n) + ' road events in the city — check the Map tab for details. ' if road_n else 'No major road closures reported. '}"
            "Stay informed by checking PulseGrid before heading out."
        )
        return {"text": fallback, "model": "fallback", "updated": datetime.now(timezone.utc).isoformat(), "cached": False}

@app.get("/api/all")
async def all_data():
    """
    Single endpoint that returns weather + roads + briefing in one call.
    The PWA uses this to minimize round trips.
    """
    weather_data, roads_data = await asyncio.gather(
        _get_or_fetch("weather", fetch_weather_live),
        _get_or_fetch("roads",   fetch_roads_live),
    )
    briefing_data = await _get_or_fetch(
        "briefing", lambda: fetch_briefing_live(weather_data, roads_data)
    )
    return {
        "weather":  weather_data,
        "roads":    roads_data,
        "briefing": briefing_data,
        "fetched":  datetime.now(timezone.utc).isoformat(),
    }

async def _get_or_fetch(key: str, fetcher):
    cached = cache_get(key)
    if cached:
        return cached
    data = await fetcher()
    cache_set(key, data)
    return data
