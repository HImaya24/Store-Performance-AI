from __future__ import annotations
from typing import Optional, List, Dict, Any

# Temporary quick fix: fake response instead of real API call
async def fetch_metrics(
    store_id: str,
    from_date: str,
    to_date: str,
    client: Optional[object] = None
) -> List[Dict[str, Any]]:
    # Fake static data (mocked)
    return [
        {"date": from_date, "sales": 120, "visits": 300},
        {"date": to_date, "sales": 150, "visits": 350},
    ]
