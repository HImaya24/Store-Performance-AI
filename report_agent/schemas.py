from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import datetime as dt

class TrendPoint(BaseModel):
    date: dt.date
    sales: float
    customers: int
    returns: int

class Anomaly(BaseModel):
    date: dt.date
    metric: str
    severity: str
    note: Optional[str] = None

class KPIBlock(BaseModel):
    sales: float
    customers: int
    avg_order_value: float
    returns: int
    conversion_rate: float

class InsightsResponse(BaseModel):
    store_id: str
    insights_text: Optional[str] = ""
    recommendations: List[str] = []
    anomalies: List[Anomaly] = []

class GenerateReportRequest(BaseModel):
    store_id: str = Field(..., min_length=1, max_length=64)
    from_date: dt.date
    to_date: dt.date
    formats: List[str] = ["pdf", "html", "md"]

    @field_validator("formats")
    @classmethod
    def ensure_formats(cls, v):
        allowed = {"pdf", "html", "md"}
        bad = [f for f in v if f not in allowed]
        if bad:
            raise ValueError(f"Unsupported formats: {bad}. Allowed: {sorted(allowed)}")
        return v
