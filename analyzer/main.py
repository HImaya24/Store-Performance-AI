# analyzer/main.py
import os, uuid, datetime, json
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn
import requests  # for local LLM call
from dotenv import load_dotenv

# Load environment variables (if any)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

print(f"✅ Environment loaded for local mode")

app = FastAPI(title="Analyzer Agent")

# Use Local LLM (Ollama) instead of cloud
USE_LLM = os.environ.get("USE_LLM", "true").lower() == "true"
LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "mistral")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")

# ---------- Helper: regex-free simple feature extraction ----------
def simple_ner_and_summary(ev: dict):
    etype = ev.get("event_type")
    payload = ev.get("payload", {})

    if etype == "sale":
        amt = payload.get("amount", 0)
        items = payload.get("items", [])
        customer = payload.get("customer_name", "Unknown")
        payment = payload.get("payment_method", "Unknown")
        promotion = payload.get("promotion", "None")
        customer_category = payload.get("customer_category", "Unknown")
        store_type = payload.get("store_type", "Unknown")
        season = payload.get("season", "Unknown")

        text = f"{customer_category} customer: {len(items)} items totaling ${amt:.2f} via {payment}"
        if promotion != "None":
            text += f" with {promotion} promotion"

        tags = [
            "sale",
            payment.lower().replace(" ", "_"),
            promotion.lower().replace(" ", "_"),
            customer_category.lower().replace(" ", "_"),
            store_type.lower().replace(" ", "_"),
            season.lower()
        ]

        if payload.get("discount_applied"):
            tags.append("discount")

        explanation = f"Store: {store_type}, Season: {season}, Customer: {customer_category}"

    else:
        text = f"Event {etype}"
        tags = [etype]
        explanation = f"Event type: {etype}"

    return text, tags, explanation


# ---------- Local LLM wrapper (Ollama) ----------
def llm_insight_text(event: dict) -> Dict[str, Any]:
    """
    Local LLM (Mistral/Phi-3) through Ollama API
    """
    try:
        print(f"🔍 Local AI analyzing event: {event.get('event_id')}")

        prompt = f"""
        Analyze this retail transaction data and provide insights. 
        Use this exact format:

        INSIGHT: [1 sentence insight about customer behavior or business impact]
        ANALYSIS: [2-3 sentences of detailed analysis about patterns or implications]
        TAGS: [3-5 comma-separated relevant tags like customer_type,payment_method,season,behavior]

        Transaction Data:
        {json.dumps(event, indent=2)}

        Important: Start directly with "INSIGHT:" — no markdown or extra headers.
        """

        payload = {
            "model": LOCAL_LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        content = response.json().get("response", "").strip()
        print(f"✅ Local AI Response: {content}")

        # Parse structured sections
        insight, analysis, tags_str = "", "", ""
        lines = content.splitlines()

        for line in lines:
            line = line.strip()
            if line.startswith("INSIGHT:") and not insight:
                insight = line.replace("INSIGHT:", "").strip()
            elif line.startswith("ANALYSIS:") and not analysis:
                analysis = line.replace("ANALYSIS:", "").strip()
            elif line.startswith("TAGS:") and not tags_str:
                tags_str = line.replace("TAGS:", "").strip()

        if not insight:
            insight = "AI-generated insight"
        if not analysis:
            analysis = "Detailed analysis of transaction patterns"
        if not tags_str:
            tags_str = "ai_analysis,retail"

        tags = [t.strip() for t in tags_str.split(",")]

        return {
            "text": insight,
            "explanation": analysis,
            "tags": tags,
            "prompt": prompt[:120] + "...",
        }

    except Exception as ex:
        print(f"❌ Local AI failed: {ex}")
        # fallback
        text, tags, explanation = simple_ner_and_summary(event)
        return {
            "text": f"AI: {text}",
            "explanation": f"Fallback analysis: {explanation}",
            "tags": tags,
            "prompt": "fallback"
        }


@app.post("/analyze")
def analyze(events: List[dict]):
    insights = []
    llm_traces = []

    for ev in events:

        text, tags, explanation = simple_ner_and_summary(ev)


        if USE_LLM:
            try:
                out = llm_insight_text(ev)
                text = out["text"] or text
                explanation = out["explanation"] or explanation

                llm_traces.append({
                    "event_id": ev.get("event_id"),
                    "store_id": ev.get("store_id"),
                    "prompt_preview": out["prompt"][:400],
                    "response_preview": f"{text} | {explanation}"[:400],

                })

            except Exception as ex:
                
                explanation += f" | LLM fallback: {ex}"

        insights.append({
            "insight_id": str(uuid.uuid4()),
            "store_id": ev.get("store_id"),
            "ts": datetime.datetime.utcnow().isoformat(),
            "text": text,
            "explanation": explanation,
            "tags": tags,
            "confidence": 0.9 if "sale" in tags else 0.8
        })

    return {
        "status": "ok",
        "insights": len(insights),
        "insights_list": insights,
        "llm_traces": llm_traces,
        "mode": "LOCAL LLM" if USE_LLM else "STUB"
    }


@app.get("/health")
def health():
    return {"status": "analyzer up", "mode": "LOCAL LLM" if USE_LLM else "STUB"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101)
