# analyzer/main.py
import os, uuid, datetime
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Analyzer Agent")

USE_LLM = os.environ.get("USE_LLM", "false").lower() == "true"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT", "15"))

# ---------- Helper: regex-free simple feature extraction ----------
def simple_ner_and_summary(ev: dict):
    etype = ev.get("event_type")
    payload = ev.get("payload", {})
    if etype == "sale":
        amt = payload.get("amount", 0)
        items = payload.get("items", [])
        text = f"Sale of {len(items)} items totaling {amt}."
        tags = ["sale"]
    elif etype == "inventory":
        text = f"Inventory update: {payload.get('sku')} qty {payload.get('qty')}"
        tags = ["inventory"]
    else:
        text = f"Event {etype}"
        tags = [etype]
    explanation = f"Derived facts: type={etype}, fields={list(payload.keys())}"
    return text, tags, explanation

# ---------- LLM wrapper (OpenAI v1 SDK) ----------
def llm_insight_text(event: dict) -> Dict[str, Any]:
    """
    Returns {"text": ..., "explanation": ..., "prompt": ..., "response_token_count": ...}
    If LLM unavailable, raises Exception.
    """
    from openai import OpenAI
    client = OpenAI(timeout=OPENAI_TIMEOUT)

    prompt = (
        "You are a retail analytics assistant.\n"
        "Given a single store event JSON, extract a concise insight sentence and a brief explanation.\n"
        "Return two lines ONLY:\n"
        "INSIGHT: <one short sentence>\n"
        "EXPLAIN: <one short sentence>\n\n"
        f"EVENT_JSON:\n{event}"
    )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120,
    )

    content = completion.choices[0].message.content.strip()
    # Basic parse (robust enough for demo)
    insight_line, explain_line = "", ""
    for line in content.splitlines():
        if line.upper().startswith("INSIGHT:"):
            insight_line = line.split(":", 1)[1].strip()
        elif line.upper().startswith("EXPLAIN:"):
            explain_line = line.split(":", 1)[1].strip()

    return {
        "text": insight_line or "Insight unavailable",
        "explanation": explain_line or "Explanation unavailable",
        "prompt": prompt,
        "response_token_count": completion.usage and completion.usage.completion_tokens or None,
    }

"""def simple_ner_and_summary(ev: dict):
    etype = ev.get("event_type")
    payload = ev.get("payload", {})
    
    if etype == "sale":
        amt = payload.get("amount", 0)
        items = payload.get("items", [])
        customer = payload.get("customer_name", "Unknown")
        payment = payload.get("payment_method", "Unknown")
        promotion = payload.get("promotion", "None")
        
        text = f"Sale to {customer}: {len(items)} items totaling ${amt:.2f} via {payment}"
        if promotion != "None":
            text += f" with promotion: {promotion}"
            
        tags = ["sale", payment.lower(), promotion.lower()]
        if payload.get("discount_applied"):
            tags.append("discount")
            
        explanation = f"Customer category: {payload.get('customer_category', 'Unknown')}, Store type: {payload.get('store_type', 'Unknown')}"
        
    else:
        text = f"Event {etype}"
        tags = [etype]
        explanation = f"Derived facts: type={etype}, fields={list(payload.keys())}"
        
    return text, tags, explanation """

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
            
        tags = ["sale", payment.lower().replace(" ", "_"), promotion.lower().replace(" ", "_"),
                customer_category.lower().replace(" ", "_"), store_type.lower().replace(" ", "_"),
                season.lower()]
        
        if payload.get("discount_applied"):
            tags.append("discount")
            
        explanation = f"Store: {store_type}, Season: {season}, Customer: {customer_category}"
        
    else:
        text = f"Event {etype}"
        tags = [etype]
        explanation = f"Event type: {etype}"
        
    return text, tags, explanation

@app.post("/analyze")
def analyze(events: List[dict]):
    insights = []
    llm_traces = []

    for ev in events:
        # Always compute a deterministic base insight (stub)
        text, tags, explanation = simple_ner_and_summary(ev)

        # If LLM enabled, replace text/explanation with model output, but keep tags from stub
        if USE_LLM:
            try:
                out = llm_insight_text(ev)
                text = out["text"] or text
                explanation = out["explanation"] or explanation
                # keep a compact trace for audit
                llm_traces.append({
                    "event_id": ev.get("event_id"),
                    "store_id": ev.get("store_id"),
                    "prompt_preview": out["prompt"][:400],
                    "response_preview": f"{text} | {explanation}"[:400],
                    "response_token_count": out.get("response_token_count"),
                })
            except Exception as ex:
                # Fallback to stub on any LLM error (offline demo safety)
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
        "status":"ok",
        "insights": len(insights),
        "insights_list": insights,
        "llm_traces": llm_traces,
        "mode": "LLM" if USE_LLM else "STUB"
    }

@app.get("/health")
def health():
    return {"status":"analyzer up", "mode": "LLM" if USE_LLM else "STUB"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101)
