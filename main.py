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
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

import httpx
from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PulseGrid API",
    description="Live civic infrastructure data for Brantford, Ontario",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

anthropic = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = {"weather": 300, "roads": 300, "briefing": 300}

def cache_get(key: str) -> Optional[dict]:
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry["ts"] < CACHE_TTL.get(key, 300):
            return entry["data"]
    return None

def cache_set(key: str, data: dict):
    _cache[key] = {"data": data, "ts": time.time()}

# ── Constants ─────────────────────────────────────────────────────────────────
# EC MSC GeoMet — observation stations near Brantford (bbox search, returns JSON)
EC_OBS_URL = (
    "https://api.weather.gc.ca/collections/swob-realtime/items"
    "?sortby=-date_tm-value&limit=1"
    "&bbox=-80.40,43.05,-80.10,43.25&f=json"
)
# EC forecast via the hourly forecast page — scraped for alerts/forecast text
# Alerts RSS — Ontario region (reliable, no city code needed)
EC_ALERTS_RSS = "https://weather.gc.ca/rss/warning/on_e.xml"
# Ontario 511
ON511 = "https://511on.ca/api/v2/get/event?lang=en&format=json"

BRANTFORD_LAT = 43.1394
BRANTFORD_LNG = -80.2644
MAX_DISTANCE_DEG = 0.40

SEED_EVENTS = [
    {"title": "Wayne Gretzky Pkwy NB — Lane Closure",    "meta": "Colborne St to Market St",   "time": "Active",       "type": "closure",      "lat": 43.1421, "lng": -80.2601},
    {"title": "Colborne St E — Water Main Replacement",   "meta": "Dalhousie to Clarence St",   "time": "Until Jun 30", "type": "construction", "lat": 43.1392, "lng": -80.2523},
    {"title": "Brant Ave — Sidewalk Reconstruction",      "meta": "Nelson St to Chatham St",    "time": "Until Jul 15", "type": "construction", "lat": 43.1375, "lng": -80.2688},
    {"title": "King George Rd — Resurfacing",             "meta": "Fairview Dr to Elgin St",    "time": "Until May 20", "type": "construction", "lat": 43.1268, "lng": -80.2701},
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()

def wx_icon(text: str) -> str:
    s = (text or "").lower()
    if "thunder" in s:                                     return "⛈️"
    if any(w in s for w in ["snow","blizzard","flurr"]):   return "❄️"
    if any(w in s for w in ["freezing rain","ice pellet"]): return "🌨️"
    if any(w in s for w in ["rain","shower","drizzle"]):   return "🌧️"
    if any(w in s for w in ["fog","mist","haze"]):         return "🌫️"
    if "overcast" in s or "cloudy" in s:                   return "☁️"
    if any(w in s for w in ["sunny","clear","fair"]):      return "☀️"
    return "🌤️"

def near_brantford(lat, lng) -> bool:
    try:
        return (abs(float(lat) - BRANTFORD_LAT) < MAX_DISTANCE_DEG and
                abs(float(lng) - BRANTFORD_LNG) < MAX_DISTANCE_DEG)
    except (TypeError, ValueError):
        return False

# ── Weather ───────────────────────────────────────────────────────────────────
async def fetch_weather_live() -> dict:
    """
    Two-source strategy:
    1. MSC GeoMet SWOB API  → current temp, condition, wind, humidity (JSON)
    2. EC Ontario alerts RSS → active warnings for the region
    """
    temp = None
    condition = "Brantford, ON"
    wind_str = ""
    humidity_str = ""
    station_name = "Brantford Airport"
    alerts = []

    async with httpx.AsyncClient(timeout=12) as client:

        # ── Source 1: SWOB real-time observations (JSON) ──────────────────
        try:
            obs_resp = await client.get(
                EC_OBS_URL, headers={"User-Agent": "PulseGrid/1.0"}
            )
            obs_resp.raise_for_status()
            obs_data = obs_resp.json()
            features = obs_data.get("features", [])
            if features:
                props = features[0].get("properties", {})
                # Temperature
                air_temp = props.get("air_temp", {})
                if isinstance(air_temp, dict) and air_temp.get("value") is not None:
                    temp = str(round(float(air_temp["value"])))
                elif props.get("air_temp") is not None:
                    temp = str(round(float(props["air_temp"])))
                # Wind
                wind_spd = props.get("wind_spd", {})
                wind_dir = props.get("wind_dir", {})
                spd_val = wind_spd.get("value") if isinstance(wind_spd, dict) else wind_spd
                dir_val = wind_dir.get("value") if isinstance(wind_dir, dict) else wind_dir
                if spd_val is not None:
                    wind_str = f"Wind {dir_val or ''} {round(float(spd_val))} km/h".strip()
                # Humidity
                rel_hum = props.get("rel_hum", {})
                hum_val = rel_hum.get("value") if isinstance(rel_hum, dict) else rel_hum
                if hum_val is not None:
                    humidity_str = f"{round(float(hum_val))}%"
                # Station name
                stn = props.get("stn_nam") or props.get("station_name")
                if stn:
                    station_name = stn
                # Condition — try present_weather first, then derive from available obs
                pw = props.get("present_weather", {})
                pw_val = (pw.get("value") if isinstance(pw, dict) else pw)
                if pw_val and str(pw_val).strip() and str(pw_val).strip() != "NA":
                    condition = str(pw_val).strip().title()
                else:
                    # Derive condition from visibility, precip, and humidity
                    vis = props.get("visibility", {})
                    vis_val = vis.get("value") if isinstance(vis, dict) else vis
                    precip = props.get("pcpn_amt_pst1hr", {})
                    precip_val = precip.get("value") if isinstance(precip, dict) else precip
                    hum = hum_val  # already extracted above

                    if precip_val is not None and float(precip_val) > 0:
                        condition = "Rain" if float(precip_val) < 5 else "Heavy Rain"
                    elif vis_val is not None and float(vis_val) < 5:
                        condition = "Fog / Reduced Visibility"
                    elif hum is not None and float(hum) > 90:
                        condition = "Overcast / Mist"
                    elif hum is not None and float(hum) > 75:
                        condition = "Mostly Cloudy"
                    elif temp is not None and int(temp) <= 0:
                        condition = "Cold / Frost Risk"
                    else:
                        condition = "Partly Cloudy"
        except Exception as e:
            print(f"SWOB fetch failed: {e}")

        # ── Source 2: Ontario alerts RSS ──────────────────────────────────
        try:
            alert_resp = await client.get(
                EC_ALERTS_RSS, headers={"User-Agent": "PulseGrid/1.0"}
            )
            alert_resp.raise_for_status()
            atom_ns = "http://www.w3.org/2005/Atom"
            root = ET.fromstring(alert_resp.text)
            for entry in root.findall(f"{{{atom_ns}}}entry"):
                title   = (entry.findtext(f"{{{atom_ns}}}title") or "").strip()
                summary = strip_html(entry.findtext(f"{{{atom_ns}}}summary") or "")
                tl = title.lower()
                # Only include alerts relevant to Brantford / Brant County
                if any(w in tl for w in ["brantford","brant","grand river","haldimand","norfolk"]):
                    alerts.append({"title": title, "summary": summary[:300]})
        except Exception as e:
            print(f"Alerts RSS fetch failed: {e}")

    icon = wx_icon(condition)

    return {
        "temperature_c": temp,
        "icon":          icon,
        "condition":     condition,
        "station":       station_name,
        "wind":          wind_str,
        "humidity":      humidity_str,
        "alerts":        alerts,
        "forecast":      [],   # Phase 2: add forecast via GeoMet WPS
        "source":        "Environment Canada MSC GeoMet",
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
        print(f"Ontario 511 failed: {ex}")

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

Live data ({datetime.now().strftime('%I:%M %p, %B %d')}):
- Weather: {weather.get('condition','Unknown')} · {temp_str} · {weather.get('wind','')}
- Active weather alerts: {alert_count} ({alert_str})
- Road events near Brantford: {road_count} active ({road_str})
- Transit: All 12 Brantford Transit routes operating normally
- Utilities: All systems normal

Write a helpful 3-sentence plain-language briefing for a Brantford resident. Be warm, specific, and practical. Lead with the most important thing. End with one actionable tip. No bullet points. Under 80 words."""

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
    return {"status": "ok", "service": "PulseGrid API", "city": "Brantford, Ontario", "version": "1.0.0"}

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
    weather_data = cache_get("weather") or await fetch_weather_live()
    roads_data   = cache_get("roads")   or await fetch_roads_live()
    try:
        data = await fetch_briefing_live(weather_data, roads_data)
        cache_set("briefing", data)
        return {**data, "cached": False}
    except Exception as e:
        alerts = len(weather_data.get("alerts", []))
        road_n = roads_data.get("count", 0)
        fallback = (
            f"{'⚠️ ' + str(alerts) + ' active weather alert(s) for the Brantford area. ' if alerts else 'No active weather warnings for Brantford right now. '}"
            f"{'There are ' + str(road_n) + ' road events in the city — check the Map tab. ' if road_n else 'No major road closures reported. '}"
            "Stay informed by checking PulseGrid before heading out."
        )
        return {"text": fallback, "model": "fallback", "updated": datetime.now(timezone.utc).isoformat(), "cached": False}

@app.get("/api/all")
async def all_data():
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
