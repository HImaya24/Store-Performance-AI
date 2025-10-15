# analyzer/main.py 
import os, uuid, json
from typing import List, Dict, Any
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi import FastAPI, HTTPException  
from .retrieval_system import SemanticSearchEngine  
import requests

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
        
        return {
            "text": insight,
            "explanation": analysis,
            "tags": tags,
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
        "insights": len(output),
        "insights_list": output,
        "llm_traces": llm_traces,
        "mode": "LLM" if USE_LLM and GROQ_API_KEY else "SIMPLE",
        "llm_insights_count": llm_count
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
