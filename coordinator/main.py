# coordinator/main.py
import os
import uuid
import datetime
from fastapi import FastAPI, HTTPException
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Coordinator Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALYZER_URL = os.environ.get("ANALYZER_URL", "http://localhost:8101/analyze")
print(f"🔧 Coordinator configured with ANALYZER_URL: {ANALYZER_URL}")
KPI_URL = os.environ.get("KPI_URL", "http://localhost:8102/kpis")
API_KEY = os.environ.get("API_KEY", "demo-key")
REPORT_URL = os.environ.get("REPORT_URL", "http://localhost:8103")

AUDIT_STORE = {}

@app.get("/health")
def health_check():
    return {"status": "healthy", "batches_processed": len(AUDIT_STORE)}

@app.post("/orchestrate")
async def orchestrate(payload: dict):
    print(f"📦 Received orchestration request with {len(payload.get('events', []))} events")
    
    events = payload.get("events", [])
    
    # Validate event count
    if len(events) > 20:
        return {
            "batch_id": "rejected",
            "status": "rejected", 
            "message": f"Too many events ({len(events)}). Maximum allowed: 20 events per batch."
        }
    
    if not events:
        return {
            "batch_id": "rejected",
            "status": "rejected",
            "message": "No events provided"
        }
    
    # Use smaller batch size
    batch_size = int(os.environ.get("BATCH_SIZE", 3))
    
    print(f"\n{'='*60}")
    print(f"📦 NEW BATCH: Processing {len(events)} events")
    print(f"📦 Batch size: {batch_size} events per sub-batch")
    print(f"{'='*60}\n")
    
    # Split events into smaller batches
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
        "errors": []
    }

    all_insights = []
    failed_batches = 0
    
    # Process each batch separately with detailed error tracking
    for i, batch_events in enumerate(batches):
        print(f"\n🔍 Processing sub-batch {i+1}/{len(batches)} ({len(batch_events)} events)")
        
        try:
            # Create client with very long timeout
            async with httpx.AsyncClient(timeout=300.0) as client:
                print(f"   ⏳ Sending to analyzer at {ANALYZER_URL}...")
                
                # Send batch to analyzer - NO AUTH HEADERS
                a = await client.post(ANALYZER_URL, json=batch_events)
                
                print(f"   📡 Analyzer response status: {a.status_code}")
                
                if a.status_code != 200:
                    error_msg = f"Batch {i+1}: Analyzer returned status {a.status_code} - {a.text}"
                    print(f"   ❌ {error_msg}")
                    AUDIT_STORE[batch_id]["errors"].append(error_msg)
                    failed_batches += 1
                    continue
                
                analyzer_json = a.json()
                batch_insights = analyzer_json.get("insights_list", [])
                all_insights.extend(batch_insights)
                
                print(f"   ✅ Got {len(batch_insights)} insights from batch {i+1}")
                
        except httpx.TimeoutException:
            error_msg = f"Batch {i+1}: Timeout after 300 seconds"
            print(f"   ⏰ {error_msg}")
            AUDIT_STORE[batch_id]["errors"].append(error_msg)
            failed_batches += 1
            
        except httpx.HTTPStatusError as ex:
            error_msg = f"Batch {i+1}: HTTP error {ex.response.status_code} - {ex.response.text[:200]}"
            print(f"   ❌ {error_msg}")
            AUDIT_STORE[batch_id]["errors"].append(error_msg)
            failed_batches += 1
            
        except httpx.ConnectError:
            error_msg = f"Batch {i+1}: Cannot connect to analyzer (is it running on port 8101?)"
            print(f"   🔌 {error_msg}")
            AUDIT_STORE[batch_id]["errors"].append(error_msg)
            failed_batches += 1
            
        except Exception as ex:
            error_msg = f"Batch {i+1}: Unexpected error - {type(ex).__name__}: {str(ex)}"
            print(f"   ❌ {error_msg}")
            AUDIT_STORE[batch_id]["errors"].append(error_msg)
            failed_batches += 1

    # Check results
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"   Total sub-batches: {len(batches)}")
    print(f"   Successful: {len(batches) - failed_batches}")
    print(f"   Failed: {failed_batches}")
    print(f"   Total insights: {len(all_insights)}")
    print(f"{'='*60}\n")

    if not all_insights:
        AUDIT_STORE[batch_id]["status"] = "analyzer_failed"
        return {
            "batch_id": batch_id, 
            "status": "analyzer_failed",
            "message": f"Failed to generate any insights. {failed_batches}/{len(batches)} batches failed.",
            "errors": AUDIT_STORE[batch_id]["errors"]
        }

    # Update status after successful analysis
    AUDIT_STORE[batch_id]["status"] = "analyzed"
    AUDIT_STORE[batch_id]["analyzer"] = {
        "count": len(all_insights),
        "insights_preview": all_insights[:3],
        "batches_processed": len(batches),
        "batches_failed": failed_batches
    }

    # KPI calculation
    print(f"📈 Requesting KPI calculation from {KPI_URL}...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            k = await client.get(KPI_URL)
            
            if k.status_code == 200:
                kpi_data = k.json()
                AUDIT_STORE[batch_id]["status"] = "kpi_updated"
                AUDIT_STORE[batch_id]["kpi_results"] = kpi_data
                print(f"   ✅ KPI calculation successful")
            else:
                print(f"   ⚠️ KPI returned status {k.status_code}")
                AUDIT_STORE[batch_id]["status"] = "kpi_failed"
                
    except Exception as ex:
        print(f"   ⚠️ KPI calculation failed: {ex}")
        AUDIT_STORE[batch_id]["status"] = "kpi_warning"
        AUDIT_STORE[batch_id]["error_kpi"] = str(ex)

    # Final status
    AUDIT_STORE[batch_id]["status"] = "processing_complete"
    AUDIT_STORE[batch_id]["report"] = {
        "batch_id": batch_id,
        "insights_count": len(all_insights),
        "message": f"Successfully processed {len(batches) - failed_batches}/{len(batches)} batches"
    }

    print(f"\n✅ Batch {batch_id} completed successfully!\n")

    return {
        "batch_id": batch_id, 
        "status": "processing_complete", 
        "insights_count": len(all_insights),
        "batches_processed": len(batches) - failed_batches,
        "batches_failed": failed_batches
    }

@app.get("/audit/{batch_id}")
def audit(batch_id: str):
    return AUDIT_STORE.get(batch_id, {})

@app.get("/audits")
def audits():
    return sorted(AUDIT_STORE.values(), key=lambda x: x["ts"], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8110)