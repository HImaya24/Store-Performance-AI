from __future__ import annotations
from typing import Dict, Any

def format_kpis(kpis: Dict[str, Any]) -> Dict[str, str]:
    return {
        "Total Sales": f"${kpis.get('sales', 0):,.2f}",
        "Customers": f"{int(kpis.get('customers', 0)):,}",
        "Avg Order Value": f"${kpis.get('avg_order_value', 0):,.2f}",
        "Returns": f"{int(kpis.get('returns', 0)):,}",
        "Conversion Rate": f"{(kpis.get('conversion_rate', 0) * 100):.2f}%",
    }
