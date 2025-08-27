# kpi_agent/main.py - FIXED VERSION
import datetime
from fastapi import FastAPI
from typing import List, Dict
import uvicorn
import requests  # Add this import

app = FastAPI(title="KPI Agent")
KPI_STORE = []



@app.get("/kpis/{store_id}")
def get_kpi_for_store(store_id: str):
    """Get KPI data for a specific store"""
    try:
        # Get all KPIs first
        all_kpis = []
        response = requests.get("http://localhost:8102/kpis", timeout=5)
        if response.status_code == 200:
            all_kpis = response.json()
        
        # Find the specific store
        for kpi in all_kpis:
            if isinstance(kpi, dict) and kpi.get('store_id') == store_id:
                return kpi
        
        # If store not found, return 404
        return {"error": f"Store {store_id} not found"}, 404
        
    except Exception as e:
        return {"error": f"Error: {str(e)}"}, 500

@app.get("/kpis")
def get_kpis():
    """Calculate and return KPIs on demand - ALWAYS returns array"""
    try:
        # First try to get insights from analyzer
        analyzer_url = "http://localhost:8101/analyze"
        insights = []
        
        try:
            # Get data from collector
            collector_response = requests.get("http://localhost:8100/events", timeout=5)
            if collector_response.status_code == 200:
                events = collector_response.json()
                # Send to analyzer
                analyzer_response = requests.post(analyzer_url, json=events, timeout=10)
                if analyzer_response.status_code == 200:
                    insights = analyzer_response.json().get("insights_list", [])
        except:
            insights = []
        
        # Calculate KPIs from insights
        if insights:
            return calculate_kpis_from_insights(insights)  # Returns array ✅
        else:
            return KPI_STORE  # Returns array ✅ - JUST RETURN THE ARRAY!
            
    except Exception as e:
        return []  # Returns empty array ✅ - DON'T return error object
    
def calculate_kpis_from_insights(insights: List[dict]):
    """Calculate KPIs from insights"""
    by_store = {}
    
    for insight in insights:
        store_id = insight.get("store_id", "unknown")
        
        if store_id not in by_store:
            by_store[store_id] = {
                "sales_count": 0, "sale_amount": 0.0, "total_items": 0
            }

        # Extract amount from payload if available
        amount = insight.get("payload", {}).get("amount", 0)
        quantity = insight.get("payload", {}).get("qty", 1)
        
        by_store[store_id]["sales_count"] += 1
        by_store[store_id]["sale_amount"] += amount
        by_store[store_id]["total_items"] += quantity

    # Create KPI records
    kpis = []
    for store_id, metrics in by_store.items():
        kpi = {
            "store_id": store_id,
            "ts": datetime.datetime.now().isoformat(),
            "metrics": {
                "sales_count": metrics["sales_count"],
                "total_sales": round(metrics["sale_amount"], 2),
                "average_order_value": round(metrics["sale_amount"] / metrics["sales_count"], 2) if metrics["sales_count"] > 0 else 0,
                "total_items_sold": metrics["total_items"]
            }
        }
        kpis.append(kpi)
        KPI_STORE.append(kpi)

    return kpis

@app.get("/health")
def health_check():
    return {"status": "healthy", "kpis_count": len(KPI_STORE)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8102)