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
    events = payload.get("events", [])
    if len(events) > 20:
        return {
            "batch_id": "rejected",
            "status": "rejected", 
            "message": f"Too many events ({len(events)}). Maximum allowed: 20 events per batch."
        }
    if request.headers.get("X-API-KEY") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    events = payload.get("events", [])
    batch_size = int(os.environ.get("BATCH_SIZE", 10))  # Get batch size from env
    
    print(f"📦 Processing {len(events)} events in batches of {batch_size}")
    
    # Split events into batches
    batches = []
    for i in range(0, len(events), batch_size):
        batch = events[i:i + batch_size]
        batches.append(batch)
    
    batch_id = str(uuid.uuid4())
    ts = datetime.datetime.utcnow().isoformat()

    AUDIT_STORE[batch_id] = {
        "batch_id": batch_id,
        "ts": ts,
        "status": "received",
        "events_count": len(events),
        "batches_count": len(batches),
        "batch_size": batch_size,
    }

    all_insights = []
    
    # Process each batch separately
    for i, batch_events in enumerate(batches):
        print(f"🔍 Processing batch {i+1}/{len(batches)} with {len(batch_events)} events")
        
        async with httpx.AsyncClient(timeout=180) as client:
            # Analyzer - send only batch_events (not all events)
            try:
                a = await client.post(ANALYZER_URL, json=batch_events)
                a.raise_for_status()
                analyzer_json = a.json()
                batch_insights = analyzer_json.get("insights_list", [])
                all_insights.extend(batch_insights)
                
                print(f"✅ Batch {i+1} analyzed: {len(batch_insights)} insights")
                
            except Exception as ex:
                AUDIT_STORE[batch_id]["status"] = f"analyzer_failed_batch_{i}"
                AUDIT_STORE[batch_id]["error"] = f"Batch {i}: {ex}"
                return {"batch_id": batch_id, "status": "analyzer_failed"}

    # Update status after all batches processed
    AUDIT_STORE[batch_id]["status"] = "analyzed"
    AUDIT_STORE[batch_id]["analyzer"] = {
        "count": len(all_insights),
        "insights_preview": all_insights[:3],
        "batches_processed": len(batches)
    }

    # KPI - after all analysis complete
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            k = await client.get(KPI_URL)
            k.raise_for_status()
            kpi_data = k.json()
            AUDIT_STORE[batch_id]["status"] = "kpi_updated"
            AUDIT_STORE[batch_id]["kpi_results"] = kpi_data
        except Exception as ex:
            AUDIT_STORE[batch_id]["status"] = "kpi_failed"
            AUDIT_STORE[batch_id]["error_kpi"] = f"{ex}"

    # Final status
    AUDIT_STORE[batch_id]["status"] = "processing_complete"
    AUDIT_STORE[batch_id]["report"] = {
        "batch_id": batch_id,
        "insights_count": len(all_insights),
        "message": f"Processed {len(batches)} batches successfully"
    }

    return {"batch_id": batch_id, "status": "processing_complete", "insights_count": len(all_insights)}

@app.get("/audit/{batch_id}")
def audit(batch_id: str):
    return AUDIT_STORE.get(batch_id, {})

@app.get("/audits")
def audits():
    return sorted(AUDIT_STORE.values(), key=lambda x: x["ts"], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)