from __future__ import annotations
from typing import List, Dict, Any

BIAS_PHRASES = [
    "always performs poorly",
    "never buy from",
    "all customers in",
]

def bias_flags(text: str) -> List[str]:
    flags = []
    lower = (text or "").lower()
    for p in BIAS_PHRASES:
        if p in lower:
            flags.append(p)
    return flags

def explainability_context(kpis: Dict[str, Any], recs: List[str]) -> List[str]:
    explanations = []
    for r in recs:
        if "price" in r.lower():
            explanations.append("Linked to avg_order_value and conversion_rate trends.")
        elif "campaign" in r.lower() or "marketing" in r.lower():
            explanations.append("Linked to customers and conversion_rate changes.")
        elif "inventory" in r.lower() or "stock" in r.lower():
            explanations.append("Linked to returns and sales volatility.")
        else:
            explanations.append("Grounded on combined KPI deltas and anomalies.")
    return explanations
