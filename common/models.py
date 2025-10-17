# common/models.py
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class StoreEvent(BaseModel):
    event_id: str
    store_id: str
    ts: datetime
    event_type: str   # "sale", "inventory", "visit", etc.
    payload: Dict[str, Any]  # Changed to accept any payload structure

class AnalyzedInsight(BaseModel):
    insight_id: str
    store_id: str
    ts: datetime
    text: str
    explanation: Optional[str] = None
    tags: List[str] = []
    confidence: float = 0.0
    # Add fields for your rich data
    customer_category: Optional[str] = None
    payment_method: Optional[str] = None
    promotion: Optional[str] = None

class KPIResult(BaseModel):
    store_id: str
    ts: datetime
    metrics: Dict[str, float]
    # Add breakdowns for your rich data
    by_customer_category: Dict[str, float] = {}
    by_payment_method: Dict[str, float] = {}
    by_promotion: Dict[str, float] = {}

class User(BaseModel):
    username: str
    password: str  # In real app, this would be hashed
    role: str = "user"

class LoginRequest(BaseModel):
    username: str
    password: str