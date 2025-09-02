from __future__ import annotations
from fastapi import FastAPI, Depends
from typing import Dict, List
import httpx

from .schemas import (
    GenerateReportRequest,
    KPIBlock,
    TrendPoint,
    InsightsResponse
)
from .clients.analysis_client import fetch_insights
from .clients.collector_client import fetch_metrics
from .reporting.generator import generate_reports
from .utils.security import require_api_key
from .config import OUTPUT_DIR


app = FastAPI(title="Report Agent", version="1.0.0")

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/report/generate")
async def generate_report(req: GenerateReportRequest, _: None = Depends(require_api_key)):
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1) Collector -> raw metrics (trend points)
        trends = await fetch_metrics(req.store_id, str(req.from_date), str(req.to_date), client=client)

        # 2) Analysis -> insights + anomalies + recommendations
        insights = await fetch_insights(req.store_id, str(req.from_date), str(req.to_date), client=client)
        anomalies = [a.model_dump() for a in (insights.anomalies or [])]

        # 3) Compute KPIs baseline (simple demo: derive from trends)
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

        files = generate_reports(
            store_id=req.store_id,
            kpis=kpis,
            trends=trends,
            anomalies=anomalies,
            insights_text=insights.insights_text or "",
            recommendations=insights.recommendations or [],
            formats=req.formats,
        )
    return {"output_dir": OUTPUT_DIR, "files": files}
