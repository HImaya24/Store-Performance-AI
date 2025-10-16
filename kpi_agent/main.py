# kpi/main.py
import datetime
from fastapi import FastAPI
from typing import List, Dict
import uvicorn
import requests
import re
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="KPI Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KPI_STORE = []

@app.get("/kpis/{store_id}")
def get_kpi_for_store(store_id: str):
    """Get KPI data for a specific store"""
    try:
        if not KPI_STORE:
            kpis = get_kpis()
        else:
            kpis = KPI_STORE
        
        for kpi in kpis:
            if isinstance(kpi, dict) and kpi.get('store_id') == store_id:
                return kpi
        
        return {"error": f"Store {store_id} not found"}, 404
    except Exception as e:
        return {"error": f"Error: {str(e)}"}, 500

@app.get("/kpis")
def get_kpis():
    """Calculate and return KPIs on demand - ALWAYS returns array"""
    global KPI_STORE
    
    try:
        # Get events from collector
        print("📊 KPI Calculation Started")
        collector_response = requests.get("http://localhost:8100/events", timeout=10)
        
        if collector_response.status_code != 200:
            print(f"⚠️ Collector returned status {collector_response.status_code}")
            return KPI_STORE if KPI_STORE else []
        
        events = collector_response.json()
        print(f"📦 Got {len(events)} events from collector")
        
        if not events:
            print("⚠️ No events available from collector")
            return KPI_STORE if KPI_STORE else []
        
        # ALWAYS calculate from events directly (most reliable)
        print("💡 Calculating KPIs directly from events...")
        kpis = calculate_kpis_from_events(events)
        
        # Cache the results
        KPI_STORE = kpis
        print(f"✅ Generated {len(kpis)} KPI records")
        
        # Print summary
        for kpi in kpis:
            metrics = kpi.get('metrics', {})
            print(f"   Store {kpi['store_id']}: Sales=${metrics.get('total_sales', 0)}, Count={metrics.get('sales_count', 0)}")
        
        return kpis
        
    except Exception as e:
        print(f"❌ Error in get_kpis: {str(e)}")
        import traceback
        traceback.print_exc()
        return KPI_STORE if KPI_STORE else []

def calculate_kpis_from_events(events: List[dict]):
    """Calculate KPIs directly from events (most reliable method)"""
    by_store: Dict[str, Dict] = {}
    
    print(f"\n🔍 Processing {len(events)} events...")
    
    for idx, event in enumerate(events):
        event_type = event.get("event_type", "unknown")
        
        if event_type != "sale":
            print(f"   Event {idx}: Skipping (type={event_type})")
            continue
            
        store_id = event.get("store_id", "unknown")
        payload = event.get("payload", {})
        
        # Extract amount - try multiple ways
        amount = 0.0
        if isinstance(payload, dict):
            amount = float(payload.get("amount", 0))
        
        # Extract items
        items = payload.get("items", [])
        if not isinstance(items, list):
            items = [items] if items else []
        item_count = len(items)
        
        # Extract metadata
        customer_category = payload.get("customer_category", "Unknown")
        payment = payload.get("payment_method", "Unknown")
        promotion = payload.get("promotion", "None")
        
        print(f"   Event {idx}: Store={store_id}, Amount=${amount}, Items={item_count}, Customer={customer_category}")
        
        # Initialize store data if needed
        if store_id not in by_store:
            by_store[store_id] = {
                "sales_count": 0,
                "sale_amount": 0.0,
                "total_items": 0,
                "by_customer_category": {},
                "by_payment_method": {},
                "by_promotion": {}
            }
        
        # Aggregate data
        by_store[store_id]["sales_count"] += 1
        by_store[store_id]["sale_amount"] += amount
        by_store[store_id]["total_items"] += item_count
        
        # Category breakdowns
        by_store[store_id]["by_customer_category"][customer_category] = (
            by_store[store_id]["by_customer_category"].get(customer_category, 0) + amount
        )
        by_store[store_id]["by_payment_method"][payment] = (
            by_store[store_id]["by_payment_method"].get(payment, 0) + amount
        )
        by_store[store_id]["by_promotion"][promotion] = (
            by_store[store_id]["by_promotion"].get(promotion, 0) + amount
        )
    
    # Build KPI results
    kpis = []
    print(f"\n📈 Building KPIs for {len(by_store)} stores...")
    
    for store_id, metrics in by_store.items():
        sales_count = metrics["sales_count"]
        total_sales = round(metrics["sale_amount"], 2)
        aov = round(total_sales / sales_count, 2) if sales_count > 0 else 0.0
        
        print(f"   {store_id}: Sales={sales_count}, Total=${total_sales}, AOV=${aov}")
        
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