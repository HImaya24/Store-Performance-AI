#coordinator/main.py
import os
import uuid
import datetime  # ← import the module
from fastapi import FastAPI, HTTPException, Request
import httpx
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Coordinator Agent")

ANALYZER_URL = os.environ.get("ANALYZER_URL", "http://localhost:8101/analyze")
KPI_URL = os.environ.get("KPI_URL", "http://localhost:8102/kpis")
API_KEY = os.environ.get("API_KEY", "demo-key")
REPORT_URL = os.environ.get("REPORT_URL", "http://localhost:8103")

AUDIT_STORE = {}

@app.get("/health")
def health_check():
    return {"status": "healthy", "batches_processed": len(AUDIT_STORE)}

@app.post("/orchestrate")
async def orchestrate(payload: dict, request: Request):
    if request.headers.get("X-API-KEY") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    events = payload.get("events", [])
    batch_id = str(uuid.uuid4())
    # FIX: Use datetime.datetime (module.class)
    ts = datetime.datetime.utcnow().isoformat()

    AUDIT_STORE[batch_id] = {
        "batch_id": batch_id,
        "ts": ts,
        "status": "received",
        "events_count": len(events),
        "events_preview": events[:3],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        # Analyzer
        try:
            a = await client.post(ANALYZER_URL, json=events)
            a.raise_for_status()
            analyzer_json = a.json()
            AUDIT_STORE[batch_id]["status"] = "analyzed"
            AUDIT_STORE[batch_id]["analyzer"] = {
                "count": analyzer_json.get("insights", 0),
                "insights_preview": analyzer_json.get("insights_list", [])[:3],
            }
        except Exception as ex:
            AUDIT_STORE[batch_id]["status"] = "analyzer_failed"
            AUDIT_STORE[batch_id]["error"] = f"{ex}"
            return {"batch_id": batch_id, "status": "analyzer_failed"}

        # KPI
        insights = analyzer_json.get("insights_list", [])
        try:
            k = await client.get(KPI_URL)
            k.raise_for_status()
            kpi_data = k.json()
            AUDIT_STORE[batch_id]["status"] = "kpi_updated"
            AUDIT_STORE[batch_id]["kpi_results"] = kpi_data
        except Exception as ex:
            AUDIT_STORE[batch_id]["status"] = "kpi_failed"
            AUDIT_STORE[batch_id]["error_kpi"] = f"{ex}"

        # Report - SIMPLIFIED
        try:
            # Just note that report would be generated via dashboard
            AUDIT_STORE[batch_id]["status"] = "processing_complete"
            AUDIT_STORE[batch_id]["report"] = {
                "batch_id": batch_id,
                "message": "Use dashboard to generate reports",
                "insights_count": len(insights)
            }
        except Exception as ex:
            AUDIT_STORE[batch_id]["status"] = "report_note_failed"
            AUDIT_STORE[batch_id]["error_report"] = f"{ex}"

    return {"batch_id": batch_id, "status": AUDIT_STORE[batch_id]["status"]}

@app.get("/audit/{batch_id}")
def audit(batch_id: str):
    return AUDIT_STORE.get(batch_id, {})

@app.get("/audits")
def audits():
    return sorted(AUDIT_STORE.values(), key=lambda x: x["ts"], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)