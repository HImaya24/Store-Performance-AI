from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import httpx
import datetime
import json
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="Report Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KPI_URL = "http://localhost:8102"

def generate_ai_summary(metrics: dict) -> str:
    """
    Generate natural language insights from store performance metrics.
    Uses Groq's LLaMA 3 model for analysis.
    """
    prompt = f"""
    You are an expert retail data analyst. Analyze this store's performance data
    and give 3 clear, concise business insights.
    Data:
    {json.dumps(metrics, indent=2)}
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI summary generation failed: {str(e)}"


@app.get("/report/{store_id}", response_class=HTMLResponse)
async def report(store_id: str, confirm: bool = False):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Try store-specific KPI first
            r = await client.get(f"{KPI_URL}/kpis/{store_id}")
            if r.status_code == 200:
                kpi = r.json()
            elif r.status_code == 404:
                # Fallback to overall KPIs if store-specific not found
                fallback_r = await client.get(f"{KPI_URL}/kpis")
                if fallback_r.status_code == 200:
                    kpis = fallback_r.json()
                    kpi = next((kp for kp in kpis if kp.get('store_id') == store_id), None) or kpis[0] if kpis else {}
                else:
                    raise HTTPException(status_code=500, detail="KPI service unavailable")
            else:
                raise HTTPException(status_code=r.status_code, detail=f"KPI fetch failed: {r.text}")

        if not kpi:
            raise HTTPException(status_code=404, detail="No KPI data available for this store")

        metrics = kpi.get("metrics", {})
        by_category = kpi.get("by_customer_category", {})
        by_payment = kpi.get("by_payment_method", {})
        by_promotion = kpi.get("by_promotion", {})

        ai_summary = generate_ai_summary(metrics)

        requires_confirm = metrics.get("average_order_value", 0) > 1000
        note = ("<p style='color:orange'>Needs human confirmation (?confirm=true).</p>"
                if (requires_confirm and not confirm) else
                "<p style='color:green'>Auto-approved.</p>")

        # Generate HTML with tables and charts
        html = f"""
        <html>
        <head>
            <title>Report for {store_id}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>table {{border-collapse: collapse; width: 50%;}} th, td {{border: 1px solid #ddd; padding: 8px; text-align: left;}}</style>
        </head>
        <body>
            <h1>Store {store_id} Report</h1>
            <p>Generated: {datetime.datetime.utcnow().isoformat()}</p>
            {note}
            
            <h2>Key Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        for k, v in metrics.items():
            value = f"${v:,.2f}" if isinstance(v, (int, float)) and ("sales" in k or "value" in k) else str(v)
            html += f"<tr><td>{k.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        html += "</table>"

        # Breakdown sections with charts
        breakdowns = [
            ("Sales by Customer Category", by_category),
            ("Sales by Payment Method", by_payment),
            ("Sales by Promotion", by_promotion)
        ]
        for title, data in breakdowns:
            if data:
                labels = list(data.keys())
                values = list(data.values())
                chart_id = title.replace(' ', '_').lower()
                html += f"""
                <h2>{title}</h2>
                <canvas id="{chart_id}" width="400" height="200"></canvas>
                <script>
                    new Chart(document.getElementById('{chart_id}'), {{
                        type: 'bar',
                        data: {{ labels: {json.dumps(labels)}, datasets: [{{ label: 'Sales ($)', data: {json.dumps(values)}, backgroundColor: 'rgba(31, 119, 180, 0.7)' }}] }},
                        options: {{ scales: {{ y: {{ beginAtZero: true }} }} }}
                    }});
                </script>
                """
        html += f"""
            <h2> AI Insights</h2>
            <div style="background-color:#f4f4f4; padding:10px; border-radius:8px; margin-top:10px;">
                <p>{ai_summary}</p>
            </div>
        """

        html += "<p>Insights & explanations are logged via Coordinator audit.</p></body></html>"
        return HTMLResponse(content=html)

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8103)