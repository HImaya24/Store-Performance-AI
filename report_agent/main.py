import os, datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import httpx, uvicorn
from dotenv import load_dotenv

load_dotenv()
KPI_URL = os.environ.get("KPI_URL", "http://localhost:8102/kpis")

app = FastAPI(title="Report Agent")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# FIXED: Only one /generate endpoint
@app.post("/generate")
async def generate_report(payload: dict):
    """Endpoint that coordinator calls to generate reports"""
    try:
        batch_id = payload.get("batch_id", "unknown")
        insights = payload.get("insights", [])
        kpi_results = payload.get("kpi_results", {})
        
        # Create a simple report response
        return {
            "status": "success",
            "batch_id": batch_id,
            "insights_count": len(insights),
            "report_url": f"http://localhost:8103/report/{batch_id}",
            "generated_at": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/report/{store_id}", response_class=HTMLResponse)
async def report(store_id: str, confirm: bool = Query(False)):
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{KPI_URL}/{store_id}")
    if r.status_code != 200:
        raise HTTPException(status_code=404, detail="KPI not found")
    kpi = r.json()
    metrics = kpi.get("metrics", {})
    requires_confirm = metrics.get("aov",0) > 1000
    note = ("<p style='color:orange'>Needs human confirmation (?confirm=true).</p>"
            if (requires_confirm and not confirm) else
            "<p style='color:green'>Auto-approved.</p>")

    html = f"""
    <html><head><title>Report {store_id}</title></head><body>
      <h1>Store {store_id} Report</h1>
      <p>Generated: {datetime.datetime.utcnow().isoformat()}</p>
      <h2>KPIs</h2><ul>
    """
    for k,v in metrics.items():
        html += f"<li>{k}: {v}</li>"
    html += f"</ul>{note}<p>Insights & explanations are logged via Coordinator audit.</p></body></html>"
    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8103)