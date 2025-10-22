# collector/main.py 
import os, re
from fastapi import FastAPI, HTTPException, Depends, Header
from typing import List, Optional
from common.models import StoreEvent
import httpx
import uvicorn
from dotenv import load_dotenv
from datetime import datetime  
from fastapi.middleware.cors import CORSMiddleware
import secrets
from datetime import timedelta

load_dotenv()

app = FastAPI(title="Collector Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVENT_STORE = []

COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "http://localhost:8110/orchestrate")
API_KEY = os.environ.get("API_KEY", "demo-key")

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")

# === SIMPLE AUTH SYSTEM ===
active_tokens = {}
users = {
    "admin": "admin123",
    "user": "user123"
}

def create_token(username: str) -> str:
    token = secrets.token_hex(32)
    expires = datetime.utcnow() + timedelta(hours=24)
    
    active_tokens[token] = {
        "username": username,
        "expires": expires
    }
    return token

def verify_token(token: str) -> Optional[dict]:
    if token not in active_tokens:
        return None
    
    user_data = active_tokens[token]
    if datetime.utcnow() > user_data["expires"]:
        del active_tokens[token]
        return None
    
    return user_data

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization.replace("Bearer ", "")
    user = verify_token(token)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return user

# === PUBLIC ENDPOINTS (No auth required) ===
@app.post("/login")
async def login(credentials: dict):
    username = credentials.get("username")
    password = credentials.get("password")
    
    if username not in users or users[username] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_token(username)
    return {
        "token": token, 
        "message": "Login successful", 
        "user": username,
        "role": "admin" if username == "admin" else "user"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "events_count": len(EVENT_STORE)}

# PROTECTED ENDPOINTS (Authentication required) ===
@app.post("/collect/batch")
async def collect_batch(events: List[StoreEvent], user: dict = Depends(get_current_user)):
    print(f"🔐 User {user['username']} is collecting {len(events)} events")
    
    sanitized = []
    for e in events:
        if not e.store_id or not e.event_type:
            raise HTTPException(status_code=400, detail="Malformed event")
        ev = e.dict()
        ev["payload_redacted"] = redact(ev.get("payload", {}))
        
        if isinstance(ev.get("ts"), datetime):
            ev["ts"] = ev["ts"].isoformat()
        
        EVENT_STORE.append(ev)
        sanitized.append(ev)

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            await client.post(
                COORDINATOR_URL,
                json={"events": sanitized},
                headers={"X-API-KEY": API_KEY},
            )
        except Exception as ex:
            print("Coordinator forward failed:", ex)
    return {"status": "ok", "received": len(sanitized), "user": user['username']}

@app.get("/events")
async def list_events():
    
    return EVENT_STORE

def redact(payload: dict):
    s = str(payload)
    s = EMAIL_RE.sub("[REDACTED_EMAIL]", s)
    s = PHONE_RE.sub("[REDACTED_PHONE]", s)
    return {"redacted": s}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)