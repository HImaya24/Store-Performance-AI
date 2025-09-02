from __future__ import annotations
from typing import Optional
# analysis_client.py
from report_agent.schemas import InsightsResponse


# Fake fetch function with dummy data
async def fetch_insights(
    store_id: str,
    from_date: str,
    to_date: str,
    client: Optional[object] = None
) -> InsightsResponse:
    fake_data = {
        "store_id": store_id,
        "from": from_date,
        "to": to_date,
        "total_sales": 12345.67,
        "total_orders": 98,
        "top_products": [
            {"name": "Product A", "sales": 4567.89},
            {"name": "Product B", "sales": 2345.50},
            {"name": "Product C", "sales": 1234.28},
        ],
        "insight_summary": "Sales increased compared to last period."
    }
    return InsightsResponse(**fake_data)
