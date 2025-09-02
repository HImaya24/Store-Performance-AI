from fastapi import FastAPI

# Fake Collector service
collector_app = FastAPI()

@collector_app.get("/metrics")
async def get_metrics(store_id: str, from_date: str, to_date: str):
    return {
        "store_id": store_id,
        "from_date": from_date,
        "to_date": to_date,
        "sales": [120, 200, 180],
        "customers": [10, 15, 20]
    }

# Fake Analysis service
analysis_app = FastAPI()

@analysis_app.get("/insights")
async def get_insights(store_id: str, from_date: str, to_date: str):
    return {
        "store_id": store_id,
        "from_date": from_date,
        "to_date": to_date,
        "insights": [
            "Sales are increasing compared to last week",
            "Customer visits are stable"
        ]
    }
