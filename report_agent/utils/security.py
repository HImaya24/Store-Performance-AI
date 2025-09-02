from __future__ import annotations
from fastapi import Header, HTTPException, status
from ..config import REPORT_AGENT_API_KEY
import re, html
from typing import List

def require_api_key(x_api_key: str = Header(..., alias="x-api-key")) -> None:
    if x_api_key != REPORT_AGENT_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

def sanitize_text(text: str) -> str:
    text = html.escape(text or "", quote=True)
    text = re.sub(r"[\u0000-\u001F\u007F]", "", text)
    return text

def sanitize_recommendations(recs: List[str]) -> List[str]:
    return [sanitize_text(r) for r in (recs or [])]
