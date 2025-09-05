from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
from modules.security import create_token, verify_token
from agents.retriever_agent import RetrieverAgent
from agents.analyst_agent import AnalystAgent
from modules.security import sanitize_text

app = FastAPI(title='Store Performance AI - Demo')

# Simple in-memory 'users' for demo
USERS = {'student1': 'password123'}

# instantiate agents (heavy work at startup)
retriever = RetrieverAgent()
analyst = AnalystAgent()

class LoginIn(BaseModel):
    username: str
    password: str

class QueryIn(BaseModel):
    query: str
    top_k: int = 5

# Dependency to check JWT
async def get_current_user(request: Request):
    auth = request.headers.get('Authorization')
    if not auth or not auth.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing auth')
    token = auth.split(' ', 1)[1]
    res = verify_token(token)
    if not res.get('ok'):
        raise HTTPException(status_code=401, detail='Invalid token')
    return res['sub']

@app.post('/auth/login')
async def login(payload: LoginIn):
    u = payload.username
    p = payload.password
    if USERS.get(u) != p:
        raise HTTPException(status_code=401, detail='Invalid credentials')
    token = create_token(u)
    return {'access_token': token}

@app.post('/agents/retrieve')
async def agents_retrieve(payload: QueryIn, user: str = Depends(get_current_user)):
    q = sanitize_text(payload.query)
    results = retriever.handle_query(q, top_k=payload.top_k)
    return {'results': results}

@app.post('/agents/analyze')
async def agents_analyze(payload: QueryIn, user: str = Depends(get_current_user)):
    q = sanitize_text(payload.query)
    results = retriever.handle_query(q, top_k=payload.top_k)
    analysis = analyst.analyze_documents(results)
    return {'analysis': analysis}
