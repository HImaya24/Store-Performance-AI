<<<<<<< HEAD
<<<<<<< HEAD
from __future__ import annotations
import argparse, asyncio
import httpx
from .clients.analysis_client import fetch_insights
from .clients.collector_client import fetch_metrics
from .reporting.generator import generate_reports

def parse_args():
    p = argparse.ArgumentParser(description="Report Agent CLI")
    p.add_argument("--store-id", required=True)
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--formats", nargs="+", default=["pdf","html","md"])
    return p.parse_args()

async def _run(store_id: str, from_date: str, to_date: str, formats):
    async with httpx.AsyncClient(timeout=30.0) as client:
        trends = await fetch_metrics(store_id, from_date, to_date, client=client)
        insights = await fetch_insights(store_id, from_date, to_date, client=client)

    # naïve KPIs
    if trends:
        total_sales = sum(p.get("sales", 0) for p in trends)
        total_cust = sum(p.get("customers", 0) for p in trends)
        total_orders = max(total_cust, 1)
        avg_order_value = total_sales / total_orders
        returns = sum(p.get("returns", 0) for p in trends)
        conversion_rate = (total_cust / max(total_orders, 1)) if total_orders else 0.0
    else:
        total_sales = avg_order_value = conversion_rate = 0.0
        total_cust = returns = 0

    kpis = {
        "sales": total_sales,
        "customers": total_cust,
        "avg_order_value": avg_order_value,
        "returns": returns,
        "conversion_rate": conversion_rate,
    }

    outputs = generate_reports(
        store_id=store_id,
        kpis=kpis,
        trends=trends,
        anomalies=[a.model_dump() for a in (insights.anomalies or [])],
        insights_text=insights.insights_text or "",
        recommendations=insights.recommendations or [],
        formats=formats
    )
    print("Generated files:", outputs)

def main():
    args = parse_args()
    asyncio.run(_run(args.store_id, args.from_date, args.to_date, args.formats))

if __name__ == "__main__":
    main()
=======
=======

>>>>>>> 3de7d8a86fa90aba1dc162378672b8e898fd7241
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import httpx
import datetime
import json
import uvicorn

app = FastAPI(title="Report Agent")
KPI_URL = "http://localhost:8102"

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
>>>>>>> 46db45df09567d4ab46114196a8092fea0280438
