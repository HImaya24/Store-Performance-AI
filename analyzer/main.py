# analyzer/main.py
import os, uuid, datetime
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn
import json
from dotenv import load_dotenv

load_dotenv()


# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

print(f"✅ Environment loaded - GROQ_API_KEY: {'GROQ_API_KEY' in os.environ}")

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

# ---------- LLM wrapper (Groq) ----------
def llm_insight_text(event: dict) -> Dict[str, Any]:
    """
    Groq LLM Insights with STRICT formatting
    """
    try:
        from groq import Groq
        
        print(f"🔍 Advanced AI analyzing event: {event.get('event_id')}")
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # STRICTER PROMPT - be very explicit about format
        prompt = f"""
        Analyze this retail transaction data and provide insights. You MUST follow this EXACT format without any additional headers, titles, or markdown:

        INSIGHT: [Write a concise 1-sentence insight about customer behavior or business impact]
        ANALYSIS: [Write 2-3 sentences of detailed analysis about patterns, significance, or implications]
        TAGS: [Provide 3-5 comma-separated relevant tags like customer_type,payment_method,season,behavior]

        Transaction Data:
        {json.dumps(event, indent=2)}

        Important: Start directly with "INSIGHT:" without any headers, titles, or introductory text. Do not use markdown formatting like **bold** or headings.
        """
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent formatting
            max_tokens=250,
        )
        
        content = completion.choices[0].message.content.strip()
        print(f"✅ Advanced AI Response: {content}")
        
        # Clean the response - remove any markdown or headers
        content = content.replace("**", "").replace("CUSTOMER BEHAVIOR:", "").strip()
        
        # Parse the structured response
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
        
        # Fallback if parsing failed
        if not insight and lines:
            # Try to extract insight from first meaningful line
            for line in lines:
                if line and not line.startswith(("INSIGHT:", "ANALYSIS:", "TAGS:")) and len(line) > 10:
                    insight = line.strip()
                    break
        
        # Ensure we have at least basic content
        if not insight:
            insight = "AI-generated insight"
        if not analysis:
            analysis = "Detailed analysis of transaction patterns"
        if not tags_str:
            tags_str = "ai_analysis,retail"
        
        tags = [tag.strip() for tag in tags_str.split(",")] if tags_str else ["ai_generated"]
        
        return {
            "text": insight,
            "explanation": analysis,
            "tags": tags,
            "prompt": prompt[:100] + "...",
            "response_token_count": completion.usage.completion_tokens if completion.usage else None,
        }
        
    except Exception as ex:
        print(f"❌ Advanced AI failed: {ex}")
        # Fallback to basic analysis
        text, tags, explanation = simple_ner_and_summary(event)
        return {
            "text": f"AI: {text}",
            "explanation": f"Analysis: {explanation}",
            "tags": tags,
            "prompt": "fallback",
            "response_token_count": 0
        }
    
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