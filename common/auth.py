# common/auth.py
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, Header

# Simple in-memory token storage (use Redis in production)
ACTIVE_TOKENS: Dict[str, dict] = {}

def create_token(username: str, role: str = "user") -> str:
    """Create a simple JWT-like token"""
    token = secrets.token_hex(32)
    expires = datetime.utcnow() + timedelta(hours=24)
    
    ACTIVE_TOKENS[token] = {
        "username": username,
        "role": role,
        "expires": expires
    }
    return token

def verify_token(token: str) -> Optional[dict]:
    """Verify token and return user data"""
    if token not in ACTIVE_TOKENS:
        return None
    
    user_data = ACTIVE_TOKENS[token]
    if datetime.utcnow() > user_data["expires"]:
        del ACTIVE_TOKENS[token]
        return None
    
    return user_data

def get_current_user(authorization: str = Header(...)) -> dict:
    """Dependency to get current user from token"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.replace("Bearer ", "")
    user = verify_token(token)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return user