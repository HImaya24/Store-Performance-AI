# analyzer/main.py
import os, uuid, json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
import uvicorn
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

app = FastAPI(title="Analyzer Agent")

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

# Simple Semantic Search Engine
class SemanticSearchEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.documents = []
        self.document_metadata = []
        self.fitted = False
    
    def index_events(self, events: List[dict]):
        """Index events for semantic search"""
        documents = []
        metadata = []
        
        for event in events:
            if event.get("event_type") == "sale":
                payload = event.get("payload", {})
                
                # Create searchable document
                doc_parts = [
                    f"store_{event.get('store_id', '')}",
                    f"season_{payload.get('season', '')}",
                    f"customer_{payload.get('customer_category', '')}",
                    f"payment_{payload.get('payment_method', '')}",
                    f"promotion_{payload.get('promotion', '')}",
                ]
                
                # Add products
                products = payload.get("items", [])
                if isinstance(products, list):
                    doc_parts.extend([f"product_{p}" for p in products])
                else:
                    doc_parts.append(f"product_{products}")
                
                document_text = " ".join(doc_parts).lower()
                documents.append(document_text)
                metadata.append({
                    "event_id": event.get("event_id"),
                    "store_id": event.get("store_id"),
                    "amount": payload.get("amount", 0),
                    "products": products,
                    "timestamp": event.get("ts")
                })
        
        if documents:
            self.documents = documents
            self.document_metadata = metadata
            self.vectorizer.fit(documents)
            self.fitted = True
    
    def search_similar_patterns(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar patterns using semantic search"""
        if not self.fitted or not self.documents:
            print("❌ Search engine not ready - no data indexed")
            return []
        
        # Transform query and documents
        query_vec = self.vectorizer.transform([query.lower()])
        doc_vecs = self.vectorizer.transform(self.documents)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for more results
                results.append({
                    "metadata": self.document_metadata[idx],
                    "similarity_score": float(similarities[idx]),
                    "document_preview": self.documents[idx][:100] + "..."
                })
        
        print(f"📊 Found {len(results)} results for query: '{query}'")
        return results

search_engine = SemanticSearchEngine()

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


# Add this to your analyzer/main.py

@app.post("/chat")
async def chat_with_data(query: str):
    """
    Handle natural language queries about the retail data
    """
    try:
        # First, try semantic search for data-specific queries
        search_results = search_engine.search_similar_patterns(query, top_k=5)
        
        # Generate a natural language response
        if search_results:
            # Create summary from search results
            total_sales = sum(r["metadata"]["amount"] for r in search_results)
            stores_involved = list(set(r["metadata"]["store_id"] for r in search_results))
            
            response = {
                "answer": f"I found {len(search_results)} relevant transactions. " +
                         f"Total sales: ${total_sales:.2f} across {len(stores_involved)} stores. " +
                         f"The most similar transaction was at {search_results[0]['metadata']['store_id']} " +
                         f"with ${search_results[0]['metadata']['amount']:.2f} in sales.",
                "results": search_results,
                "type": "data_analysis"
            }
        else:
            # Fallback for general questions
            response = {
                "answer": "I can help you analyze your retail data. Try asking about sales, customers, products, or store performance.",
                "results": [],
                "type": "general"
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/chat/query")
async def chat_query(chat_request: dict):
    """
    AI-powered chatbot that understands natural language questions about retail data
    """
    try:
        user_question = chat_request.get("question", "")
        conversation_history = chat_request.get("history", [])
        
        print(f"🤖 AI Chat Query: '{user_question}'")
        
        if not user_question.strip():
            return {"response": "Please ask a question about your retail data!"}
        
        # First, analyze the user's intent and extract key information
        intent_analysis = await analyze_user_intent(user_question)
        print(f"🔍 Intent Analysis: {intent_analysis}")
        
        # Get relevant data based on the intent
        context_data = await get_relevant_data(intent_analysis)
        
        # Generate AI response using Groq
        ai_response = await generate_ai_response(
            user_question=user_question,
            intent_analysis=intent_analysis,
            context_data=context_data,
            history=conversation_history
        )
        
        return {
            "response": ai_response,
            "intent": intent_analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Chat query error: {e}")
        return {
            "response": "I apologize, but I'm having trouble processing your question right now. Please try again in a moment.",
            "error": str(e)
        }

async def analyze_user_intent(question: str) -> dict:
    """Use AI to understand what the user is asking about"""
    
    prompt = f"""
    Analyze this user question about retail data and extract key information:
    
    USER QUESTION: "{question}"
    
    Extract the following:
    1. Primary intent (sales_analysis, product_info, store_performance, customer_behavior, comparison, trends)
    2. Store names mentioned (if any)
    3. Product categories/names mentioned (if any)  
    4. Time period mentioned (if any)
    5. Specific metrics requested (revenue, products, customers, etc.)
    6. Question type (ranking, comparison, details, trends)
    
    Return as JSON format.
    """
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        intent_data = json.loads(completion.choices[0].message.content)
        return intent_data
        
    except Exception as e:
        print(f"❌ Intent analysis failed: {e}")
        # Fallback simple analysis
        question_lower = question.lower()
        return {
            "primary_intent": "general_query",
            "stores": [word for word in ["new york", "los angeles", "chicago", "miami"] if word in question_lower],
            "products": [],
            "time_period": "recent",
            "metrics": ["general"],
            "question_type": "information"
        }

async def get_relevant_data(intent_analysis: dict) -> dict:
    """Get relevant data based on user intent"""
    
    # Load events from collector
    events = load_events_from_collector()
    if not events:
        return {"error": "No data available"}
    
    relevant_data = {
        "total_events": len(events),
        "stores": list(set(e.get("store_id") for e in events if e.get("store_id"))),
        "analysis_summary": {}
    }
    
    # Analyze sales data
    sales_events = [e for e in events if e.get("event_type") == "sale"]
    if sales_events:
        # Store performance
        store_sales = {}
        for event in sales_events:
            store = event.get("store_id")
            amount = event.get("payload", {}).get("amount", 0)
            store_sales[store] = store_sales.get(store, 0) + amount
        
        relevant_data["store_performance"] = {
            "top_stores": dict(sorted(store_sales.items(), key=lambda x: x[1], reverse=True)[:5]),
            "total_sales": sum(store_sales.values()),
            "store_count": len(store_sales)
        }
        
        # Product analysis
        product_sales = {}
        for event in sales_events:
            items = event.get("payload", {}).get("items", [])
            amount = event.get("payload", {}).get("amount", 0)
            if isinstance(items, list):
                for item in items:
                    product_sales[item] = product_sales.get(item, 0) + (amount / len(items))
            elif items:
                product_sales[items] = product_sales.get(items, 0) + amount
        
        relevant_data["product_analysis"] = {
            "top_products": dict(sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:10]),
            "total_products": len(product_sales)
        }
        
        # Time-based analysis (last 30 days)
        recent_events = [e for e in sales_events if is_recent(e.get("ts"))]
        relevant_data["recent_trends"] = {
            "recent_sales": sum(e.get("payload", {}).get("amount", 0) for e in recent_events),
            "recent_transactions": len(recent_events)
        }
    
    return relevant_data

async def generate_ai_response(user_question: str, intent_analysis: dict, context_data: dict, history: list) -> str:
    """Generate intelligent AI response using context and data"""
    
    # Build context prompt with actual data
    data_context = ""
    if "store_performance" in context_data:
        top_stores = context_data["store_performance"]["top_stores"]
        data_context += f"TOP PERFORMING STORES:\n"
        for store, sales in list(top_stores.items())[:3]:
            data_context += f"- {store}: ${sales:,.2f}\n"
    
    if "product_analysis" in context_data:
        top_products = context_data["product_analysis"]["top_products"]
        data_context += f"\nTOP SELLING PRODUCTS:\n"
        for product, revenue in list(top_products.items())[:5]:
            data_context += f"- {product}: ${revenue:,.2f}\n"
    
    # Build conversation context
    history_context = ""
    if history:
        history_context = "PREVIOUS CONVERSATION:\n"
        for msg in history[-3:]:  # Last 3 messages
            role = "User" if msg.get("isUser") else "Assistant"
            history_context += f"{role}: {msg.get('text')}\n"
    
    prompt = f"""
    You are an intelligent retail data assistant. Answer the user's question naturally and helpfully using the available data.
    
    {history_context}
    
    AVAILABLE RETAIL DATA:
    {data_context}
    
    USER'S QUESTION: "{user_question}"
    
    USER'S INTENT: {intent_analysis}
    
    Please provide:
    1. A direct, natural answer to their question
    2. Relevant insights from the data (if available)
    3. Specific numbers and facts when possible
    4. Follow-up questions or suggestions for deeper analysis
    5. Keep it conversational but professional
    
    If the data doesn't have exactly what they're asking for, be honest but helpful - suggest what you CAN tell them.
    """
    
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"❌ AI response generation failed: {e}")
        return f"I understand you're asking about: {user_question}. Based on our data, I can provide insights about store performance, product sales, and customer trends. Could you be more specific about what you'd like to know?"

def is_recent(timestamp) -> bool:
    """Check if event is from last 30 days"""
    if not timestamp:
        return False
    try:
        from datetime import datetime, timedelta
        event_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return datetime.now() - event_time < timedelta(days=30)
    except:
        return False    
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

if __name__ == "__main__":
    print("🚀 Starting Analyzer Agent...")
    uvicorn.run(app, host="0.0.0.0", port=8101)