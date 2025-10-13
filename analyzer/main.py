<<<<<<< HEAD
# analyzer/main.py
import os, uuid, datetime, json
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn
import requests  # for local LLM call
=======
# analyzer/main.py 
import os, uuid, json
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn
>>>>>>> 57137ae35c363315aa07d833fcad6191e6406ce9
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi import FastAPI, HTTPException  
from .retrieval_system import SemanticSearchEngine  
import requests

<<<<<<< HEAD
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

=======
load_dotenv()

app = FastAPI(title="Analyzer Agent")

search_engine = SemanticSearchEngine()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug environment variables
USE_LLM = os.environ.get("USE_LLM", "true").lower() == "true"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

print(f"🔧 DEBUG - Environment Variables:")
print(f"   USE_LLM: {USE_LLM}")
print(f"   GROQ_API_KEY: {'***' + GROQ_API_KEY[-4:] if GROQ_API_KEY else 'NOT SET'}")

# ThreadPool for parallel LLM calls
executor = ThreadPoolExecutor(max_workers=5)

def test_groq_connection():
    """Test if we can connect to Groq API"""
    try:
        if not GROQ_API_KEY:
            return False, "GROQ_API_KEY not set in environment"
            
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        # Test with a simple completion
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'Hello World'"}],
            max_tokens=10,
            timeout=10
        )
        
        return True, "Groq API connection successful"
        
    except ImportError:
        return False, "Groq package not installed. Run: pip install groq"
    except Exception as e:
        return False, f"Groq API error: {str(e)}"

# Test connection on startup
print("🧪 Testing Groq API connection...")
groq_working, groq_message = test_groq_connection()
print(f"   Groq Status: {groq_working} - {groq_message}")

def extract_event_features(ev: dict):
    """Extract features from event for LLM context"""
    etype = ev.get("event_type", "unknown")
    payload = ev.get("payload", {})
    
    features = {
        "event_type": etype,
        "store_id": ev.get("store_id", "unknown"),
        "timestamp": ev.get("ts", "unknown")
    }
    
    if etype == "sale":
        items = payload.get("items", [])
        if not isinstance(items, list):
            items = [items]
            
        features.update({
            "amount": payload.get("amount", 0),
            "items": items,
            "customer_name": payload.get("customer_name", "Unknown"),
            "payment_method": payload.get("payment_method", "Unknown"),
            "promotion": payload.get("promotion", "None"),
            "customer_category": payload.get("customer_category", "Unknown"),
            "store_type": payload.get("store_type", "Unknown"),
            "season": payload.get("season", "Unknown"),
            "discount_applied": payload.get("discount_applied", False),
            "total_items": len(items)
        })
    
    return features

def llm_insight_text(event: dict) -> Dict[str, Any]:
    print(f"🤖 Attempting LLM analysis...")
    
    try:
        if not GROQ_API_KEY:
            raise Exception("❌ GROQ_API_KEY environment variable not set")
            
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        features = extract_event_features(event)
        
        prompt = f"""
        Analyze this retail transaction and provide business insights:

        STORE: {features.get('store_id')}
        CUSTOMER: {features.get('customer_category')} 
        AMOUNT: ${features.get('amount', 0):.2f}
        PAYMENT: {features.get('payment_method')}
        ITEMS: {features.get('items', [])}
        SEASON: {features.get('season')}
        STORE TYPE: {features.get('store_type')}
        PROMOTION: {features.get('promotion')}
        DISCOUNT: {features.get('discount_applied')}

        Provide response in format:
        INSIGHT: [key insight]
        ANALYSIS: [detailed analysis] 
        TAGS: [tag1,tag2,tag3]
        """

        print(f"   Sending request to Groq API...")
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            timeout=30
        )

        content = completion.choices[0].message.content.strip()
        print(f"   ✅ LLM Response received: {content[:100]}...")

        # Parse response
        insight, analysis, tags_str = "", "", ""
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("INSIGHT:") and not insight:
                insight = line[8:].strip()
            elif line.startswith("ANALYSIS:") and not analysis:
                analysis = line[9:].strip()
            elif line.startswith("TAGS:") and not tags_str:
                tags_str = line[5:].strip()

        # Fallback if parsing fails
        if not insight:
            insight = content.split('\n')[0] if content else "AI-generated business insight"
        if not analysis:
            analysis = "Detailed analysis of customer purchasing behavior and opportunities."
        if not tags_str:
            tags_str = "ai_analysis,business_insight,customer_behavior"

        tags = [tag.strip() for tag in tags_str.split(",")]
        
        print(f"   ✅ Parsed - Insight: {insight[:50]}...")
        
>>>>>>> 57137ae35c363315aa07d833fcad6191e6406ce9
        return {
            "text": insight,
            "explanation": analysis,
            "tags": tags,
<<<<<<< HEAD
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
=======
            "llm_used": True,
            "tokens": completion.usage.total_tokens if completion.usage else 0
        }

    except Exception as ex:
        print(f"   ❌ LLM Failed: {str(ex)}")
        # Fallback analysis
        features = extract_event_features(event)
        
        if features["event_type"] == "sale":
            text = f"Sale: ${features.get('amount', 0):.2f} via {features.get('payment_method')}"
            explanation = f"{features.get('customer_category')} customer at {features.get('store_type')}"
            tags = ["sale", features.get('customer_category', '').lower(), "fallback"]
        else:
            text = f"{features['event_type']} event"
            explanation = "Event recorded"
            tags = [features['event_type'], "fallback"]
        
        return {
>>>>>>> 57137ae35c363315aa07d833fcad6191e6406ce9
            "text": text,
            "explanation": explanation,
            "tags": tags,
            "llm_used": False,
            "error": str(ex)
        }

async def llm_insight_async(event):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, llm_insight_text, event)


def load_events_from_collector() -> List[dict]:
    """
    Load events from the Collector service
    """
    try:
        collector_url = "http://localhost:8100/events"  # Collector endpoint
        
        print(f"📥 Loading events from Collector: {collector_url}")
        response = requests.get(collector_url, timeout=30)
        response.raise_for_status()
        
        events = response.json()
        print(f"✅ Loaded {len(events)} events from Collector")
        return events
        
    except Exception as e:
        print(f"❌ Failed to load events from Collector: {e}")
        return []
    
@app.post("/analyze")
async def analyze(events: List[dict]):
    print(f"\n{'='*60}")
    print(f"🎯 ANALYZER: Processing {len(events)} events")
    print(f"   USE_LLM: {USE_LLM}")
    print(f"   GROQ_API_KEY: {'SET' if GROQ_API_KEY else 'NOT SET'}")
    print(f"{'='*60}")
    
    insights = []
    llm_traces = []

    if not events:
        return {"status": "error", "message": "No events provided"}

    # Show sample event for debugging
    sample_event = events[0]
    print(f"📋 Sample event structure:")
    print(json.dumps(sample_event, indent=2, default=str)[:500] + "...")

    if USE_LLM and GROQ_API_KEY:
        print(f"🤖 Using LLM for analysis...")
        tasks = [llm_insight_async(ev) for ev in events]
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    print(f"   ❌ Event {i+1}: LLM error - {str(res)}")
                    # Fallback
                    features = extract_event_features(events[i])
                    insights.append({
                        "event": events[i],
                        "text": f"Error: {str(res)}",
                        "explanation": "LLM processing failed",
                        "tags": ["error", "fallback"],
                        "llm_used": False
                    })
                else:
                    print(f"   ✅ Event {i+1}: {res['text'][:60]}...")
                    insights.append({
                        "event": events[i],
                        "text": res["text"],
                        "explanation": res["explanation"],
                        "tags": res["tags"],
                        "llm_used": res.get("llm_used", True)
                    })
                    
        except Exception as e:
            print(f"❌ LLM processing failed: {str(e)}")
            # Fallback for all events
            for ev in events:
                features = extract_event_features(ev)
                insights.append({
                    "event": ev,
                    "text": f"System error - Basic analysis",
                    "explanation": f"LLM system failed: {str(e)}",
                    "tags": [features.get('event_type', 'event'), "system_error"],
                    "llm_used": False
                })
    else:
        print(f"🔧 Using simple analysis (LLM disabled)")
        # Simple analysis without LLM
        for ev in events:
            features = extract_event_features(ev)
            insights.append({
                "event": ev,
                "text": f"Simple: {features.get('event_type', 'event')} at {features.get('store_id')}",
                "explanation": f"Basic event analysis",
                "tags": [features.get('event_type', 'event'), "simple"],
                "llm_used": False
            })

    # Final output
    output = []
    llm_count = sum(1 for item in insights if item.get("llm_used"))
    
    for item in insights:
        ev = item["event"]
        output.append({
            "insight_id": str(uuid.uuid4()),
            "store_id": ev.get("store_id", "unknown"),
            "ts": datetime.utcnow().isoformat(),
            "text": item["text"],
            "explanation": item["explanation"],
            "tags": item["tags"],
            "confidence": 0.9 if item.get("llm_used") else 0.6,
            "llm_used": item.get("llm_used", False)
        })

    print(f"\n📊 RESULTS:")
    print(f"   Total insights: {len(output)}")
    print(f"   LLM insights: {llm_count}")
    print(f"   Simple insights: {len(output) - llm_count}")
    print(f"{'='*60}\n")

    return {
        "status": "ok",
<<<<<<< HEAD
        "insights": len(insights),
        "insights_list": insights,
        "llm_traces": llm_traces,
        "mode": "LOCAL LLM" if USE_LLM else "STUB"
=======
        "insights": len(output),
        "insights_list": output,
        "llm_traces": llm_traces,
        "mode": "LLM" if USE_LLM and GROQ_API_KEY else "SIMPLE",
        "llm_insights_count": llm_count
>>>>>>> 57137ae35c363315aa07d833fcad6191e6406ce9
    }
@app.post("/semantic-search")
async def semantic_search(query: str):
    """
    Handle semantic search queries from the frontend
    """
    try:
        if not query.strip():
            return {"error": "Query cannot be empty"}
        
        # Load data from Collector if search engine is empty
        if not search_engine.fitted or len(search_engine.documents) == 0:
            print("📥 Loading events from Collector for semantic search...")
            events = load_events_from_collector()
            
            if events:
                search_engine.index_events(events)
                print(f"✅ Indexed {len(events)} events for semantic search")
            else:
                print("❌ No events available from Collector")
                return {
                    "query": query,
                    "results": [],
                    "total_matches": 0,
                    "message": "No data available for search. Please load data first."
                }
        
        # Perform semantic search
        results = search_engine.search_similar_patterns(query, top_k=10)
        
        print(f"🔍 Search results for '{query}': {len(results)} matches")
        
        return {
            "query": query,
            "results": results,
            "total_matches": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Load data when analyzer starts
@app.on_event("startup")
async def startup_event():
    print("🚀 Analyzer starting up - loading data from Collector...")
    events = load_events_from_collector()
    if events:
        search_engine.index_events(events)
        print(f"✅ Pre-loaded {len(events)} events for semantic search")
    else:
        print("⚠️ No events loaded on startup - semantic search will load on demand")


@app.get("/health")
def health():
<<<<<<< HEAD
    return {"status": "analyzer up", "mode": "LOCAL LLM" if USE_LLM else "STUB"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8101)
=======
    return {
        "status": "analyzer up", 
        "llm_enabled": USE_LLM,
        "groq_api_available": bool(GROQ_API_KEY),
        "groq_working": groq_working
    }

@app.get("/debug/env")
def debug_env():
    """Show all relevant environment variables"""
    return {
        "USE_LLM": USE_LLM,
        "GROQ_API_KEY_set": bool(GROQ_API_KEY),
        "GROQ_API_KEY_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0,
        "groq_test_result": groq_message,
        "python_path": os.environ.get("PYTHONPATH", "Not set"),
        "all_env_keys": [k for k in os.environ.keys() if "GROQ" in k or "LLM" in k or "API" in k]
    }

@app.get("/debug/data-status")
async def debug_data_status():
    """Check what data we have access to"""
    try:
        # Check Collector
        collector_events = load_events_from_collector()
        
        return {
            "collector_events_available": len(collector_events),
            "search_engine_indexed": len(search_engine.documents),
            "search_engine_ready": search_engine.fitted,
            "status": "READY" if search_engine.fitted else "NEEDS_DATA"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/debug/collector-events")
async def debug_collector_events():
    """Check what events are available from Collector"""
    events = load_events_from_collector()
    return {
        "total_events": len(events),
        "sample_events": events[:2] if events else []
    }

@app.get("/debug/search-terms")
async def debug_search_terms():
    """See what terms are available for searching"""
    if not search_engine.fitted:
        return {"error": "Search engine not fitted"}
    
    # Get the feature names (words) from the vectorizer
    feature_names = search_engine.vectorizer.get_feature_names_out()
    
    # Show sample documents
    sample_docs = search_engine.documents[:5] if search_engine.documents else []
    
    return {
        "total_documents": len(search_engine.documents),
        "available_search_terms": feature_names.tolist()[:50],  # First 50 terms
        "sample_documents": sample_docs
    }

@app.get("/debug/search-status")
async def debug_search_status():
    """Check search engine status"""
    return {
        "search_engine_ready": search_engine.fitted,
        "documents_indexed": len(search_engine.documents),
        "status": "READY" if search_engine.fitted and search_engine.documents else "NO DATA"
    }

@app.get("/debug/sample-search/{query}")
async def debug_sample_search(query: str):
    """Test search directly"""
    results = search_engine.search_similar_patterns(query, top_k=5)
    return {
        "query": query,
        "results": results,
        "total_matches": len(results)
    }

if __name__ == "__main__":
    print("🚀 Starting Analyzer Agent (DEBUG MODE)...")
    uvicorn.run(app, host="0.0.0.0", port=8101)
>>>>>>> 57137ae35c363315aa07d833fcad6191e6406ce9
