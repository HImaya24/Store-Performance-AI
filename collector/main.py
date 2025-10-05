#collector/main.py
import os, re
from fastapi import FastAPI, HTTPException
from typing import List
from common.models import StoreEvent
import httpx
import uvicorn
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = FastAPI(title="Collector Agent")
EVENT_STORE = []

COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8110/orchestrate")
API_KEY = os.environ.get("API_KEY", "demo-key")

#Redact sensitive data
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")

def redact(payload: dict):
    s = str(payload)
    s = EMAIL_RE.sub("[REDACTED_EMAIL]", s)
    s = PHONE_RE.sub("[REDACTED_PHONE]", s)
    return {"redacted": s}

@app.post("/collect/batch")
async def collect_batch(events: List[StoreEvent]):
    sanitized = []
    for e in events:
        if not e.store_id or not e.event_type:
            raise HTTPException(status_code=400, detail="Malformed event")
        ev = e.dict()
        ev["payload_redacted"] = redact(ev.get("payload", {}))
        
        #Convert datetime to ISO string for JSON serialization
        if isinstance(ev.get("ts"), datetime):
            ev["ts"] = ev["ts"].isoformat()
        
        EVENT_STORE.append(ev)
        sanitized.append(ev)

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            await client.post(
                COORDINATOR_URL,
                json={"events": sanitized},
                headers={"X-API-KEY": API_KEY},
            )
        except Exception as ex:
            print("Coordinator forward failed:", ex)
    return {"status": "ok", "received": len(sanitized)}

@app.get("/events")
def list_events():
    return EVENT_STORE

@app.get("/health")
def health_check():
    return {"status": "healthy", "events_count": len(EVENT_STORE)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)