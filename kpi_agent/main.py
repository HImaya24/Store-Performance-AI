# kpi_agent/main.py
import datetime
from fastapi import FastAPI
from typing import List, Dict
import uvicorn
import requests
import re  # Added missing import

app = FastAPI(title="KPI Agent")
KPI_STORE = []  # Note: This remains empty unless populated externally

@app.get("/kpis/{store_id}")
def get_kpi_for_store(store_id: str):
    """Get KPI data for a specific store"""
    try:
        # Avoid recursive call—use internal KPI_STORE or recalculate if needed
        if not KPI_STORE:
            kpis = get_kpis().json()  # Trigger recalculation
        else:
            kpis = KPI_STORE
        
        # Find the specific store
        for kpi in kpis:
            if isinstance(kpi, dict) and kpi.get('store_id') == store_id:
                return kpi
        
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
        except Exception as e:
            print(f"Warning: Failed to get insights from analyzer - {str(e)}")
            insights = []  # Fallback to empty
        
        # Calculate KPIs from insights
        if insights:
            kpis = calculate_kpis_from_insights(insights)
            global KPI_STORE  # Update global KPI_STORE with latest
            KPI_STORE = kpis
            return kpis
        else:
            return KPI_STORE  # Return cached KPIs if available, else empty list
    except Exception as e:
        print(f"Error in get_kpis: {str(e)}")
        return []  # Return empty array on failure

def calculate_kpis_from_insights(insights: List[dict]):
    by_store: Dict[str, Dict] = {}
    
    for insight in insights:
        store_id = insight.get("store_id", "unknown")
        
        if store_id not in by_store:
            by_store[store_id] = {
                "sales_count": 0,
                "sale_amount": 0.0,
                "total_items": 0,
                "by_customer_category": {},
                "by_payment_method": {},
                "by_promotion": {}
            }
        
        # Parse the insight text (e.g., "Homemak customer: 3 items totaling $71.65 via Mobile with BOGO (Buy One Get One) promotion")
        text = insight.get("text", "").lower()  # Case-insensitive parsing
        match = re.search(
            r"([\w ]+) customer: (\d+) items totaling \$([\d.]+) via ([\w ]+)( with ([\w ()]+) promotion)?",
            text
        )
        
        if match:
            customer_category = match.group(1).strip()
            item_count = int(match.group(2))
            amount = float(match.group(3))
            payment = match.group(4).strip()
            promotion = match.group(6).strip() if match.group(6) else "None"
        else:
            # Fallback if parsing fails
            customer_category = "Unknown"
            item_count = 0
            amount = 0.0
            payment = "Unknown"
            promotion = "None"
        
        # Accumulate metrics
        by_store[store_id]["sales_count"] += 1 if "sale" in insight.get("tags", []) else 0
        by_store[store_id]["sale_amount"] += amount
        by_store[store_id]["total_items"] += item_count
        
        # Breakdowns (accumulate sales amount)
        by_store[store_id]["by_customer_category"][customer_category] = (
            by_store[store_id]["by_customer_category"].get(customer_category, 0) + amount
        )
        by_store[store_id]["by_payment_method"][payment] = (
            by_store[store_id]["by_payment_method"].get(payment, 0) + amount
        )
        by_store[store_id]["by_promotion"][promotion] = (
            by_store[store_id]["by_promotion"].get(promotion, 0) + amount
        )
    
    # Create KPI records
    kpis = []
    for store_id, metrics in by_store.items():
        sales_count = metrics["sales_count"]
        total_sales = round(metrics["sale_amount"], 2)
        aov = round(total_sales / sales_count, 2) if sales_count > 0 else 0.0
        kpi = {
            "store_id": store_id,
            "ts": datetime.datetime.now().isoformat(),
            "metrics": {
                "sales_count": sales_count,
                "total_sales": total_sales,
                "average_order_value": aov,
                "total_items_sold": metrics["total_items"]
            },
            "by_customer_category": {k: round(v, 2) for k, v in metrics["by_customer_category"].items()},
            "by_payment_method": {k: round(v, 2) for k, v in metrics["by_payment_method"].items()},
            "by_promotion": {k: round(v, 2) for k, v in metrics["by_promotion"].items()}
        }
        kpis.append(kpi)
    
    return kpis

@app.get("/health")
def health_check():
    return {"status": "healthy", "kpis_count": len(KPI_STORE)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8102)