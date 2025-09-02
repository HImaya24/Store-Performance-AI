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
