# Store Performance AI

Multi-agent AI system (student assignment) — delivered as a runnable Python project.

## Overview
- Two agents: Retriever Agent (IR) and Analyst Agent (LLM + NLP)
- FastAPI-based microservice exposing endpoints for agent communication and for a simple client
- Uses TF-IDF for retrieval (scikit-learn), spaCy for NER, and a placeholder LLM wrapper
- Security: JWT auth, input sanitization, and optional encryption for stored sensitive fields

## Structure
- app/main.py            : FastAPI app and routes
- agents/retriever_agent.py : IR agent (indexing + retrieval)
- agents/analyst_agent.py   : Analyst agent (summarize, explain, uses LLM wrapper)
- modules/ir.py          : TF-IDF indexer and search
- modules/nlp.py         : NER and summarization helpers
- modules/security.py    : Authentication (JWT), sanitization, encryption
- utils/config.py        : Configuration points (API keys, secrets)
- data/store_data.csv    : Example dataset (replace with your provided dataset)
- requirements.txt       : Python dependencies
- .env.example           : Example env file

## Quick start
1. Create virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. (Optional) Install spaCy model: `python -m spacy download en_core_web_sm`
4. Place your dataset at `./data/store_data.csv` (or modify utils/config.py)
5. Start the app: `uvicorn app.main:app --reload --port 8000`

## Demo credentials
- username: `student1`
- password: `password123`

## Notes
- The LLM client is a placeholder; if you want full LLM integration, set `OPENAI_API_KEY` in `.env` and I can add working code.
- This project is intentionally simple and well-commented for educational use and the assignment brief.
