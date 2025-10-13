# analyzer/retrieval_system.py
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

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
    
    def expand_query(self, query: str) -> str:
        """Expand query with related terms for better matching"""
        query_lower = query.lower()
        
        # Add related terms based on common retail concepts
        related_terms = {
            "winter": ["cold", "snow", "jacket", "coat", "scarf", "gloves", "sweater"],
            "holiday": ["christmas", "gift", "celebration", "festive", "seasonal", "vacation"],
            "sales": ["purchase", "transaction", "buy", "revenue", "income", "order"],
            "trends": ["pattern", "behavior", "popular", "frequent", "common", "popularity"],
            "shopping": ["retail", "purchase", "buying", "consumer", "customer", "mall"],
            "patterns": ["behavior", "habit", "trend", "frequency", "routine", "tendency"],
            "butter": ["dairy", "food", "grocery", "spread", "cooking", "kitchen"],
            "summer": ["hot", "beach", "swim", "sun", "vacation", "warm"],
            "spring": ["flower", "fresh", "rain", "garden", "renewal"],
            "fall": ["autumn", "leaf", "harvest", "cool", "pumpkin"],
            "discount": ["sale", "offer", "promotion", "deal", "bargain", "reduced"],
            "premium": ["luxury", "expensive", "high_end", "exclusive", "deluxe"],
            "customer": ["client", "buyer", "shopper", "consumer", "patron"]
        }
        
        expanded = query_lower
        for term, related in related_terms.items():
            if term in query_lower:
                expanded += " " + " ".join(related)
        
        print(f"🔍 Expanded query: '{query}' -> '{expanded}'")
        return expanded
    
    def search_similar_patterns(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar patterns using semantic search"""
        if not self.fitted or not self.documents:
            print("❌ Search engine not ready - no data indexed")
            return []
        
        # Expand query with related terms
        query_expanded = self.expand_query(query)
        
        # Transform query and documents
        query_vec = self.vectorizer.transform([query_expanded.lower()])
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
        
        # If no results with expanded query, try original query
        if len(results) == 0:
            print("🔄 No results with expanded query, trying original query...")
            query_vec_original = self.vectorizer.transform([query.lower()])
            similarities_original = cosine_similarity(query_vec_original, doc_vecs).flatten()
            
            top_indices_original = np.argsort(similarities_original)[-top_k:][::-1]
            
            for idx in top_indices_original:
                if similarities_original[idx] > 0.01:  # Even lower threshold
                    results.append({
                        "metadata": self.document_metadata[idx],
                        "similarity_score": float(similarities_original[idx]),
                        "document_preview": self.documents[idx][:100] + "..."
                    })
            
            print(f"📊 Found {len(results)} results with original query")
        
        return results
    
    def find_cross_selling_opportunities(self, events: List[dict]) -> List[Dict[str, Any]]:
        """Find product bundling opportunities"""
        product_cooccurrence = {}
        
        for event in events:
            if event.get("event_type") == "sale":
                products = event.get("payload", {}).get("items", [])
                if isinstance(products, list) and len(products) >= 2:
                    # Track product pairs
                    for i, product1 in enumerate(products):
                        for j, product2 in enumerate(products):
                            if i != j:
                                pair = tuple(sorted([str(product1), str(product2)]))
                                if pair not in product_cooccurrence:
                                    product_cooccurrence[pair] = 0
                                product_cooccurrence[pair] += 1
        
        # Convert to recommendations
        opportunities = []
        for (product1, product2), count in sorted(product_cooccurrence.items(), 
                                                key=lambda x: x[1], reverse=True)[:10]:
            if count >= 2:  # Minimum co-occurrence threshold
                opportunities.append({
                    "product_a": product1,
                    "product_b": product2,
                    "co_occurrence_count": count,
                    "recommendation": f"Bundle {product1} with {product2} - purchased together {count} times"
                })
        
        return opportunities

    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the search engine"""
        if not self.fitted:
            return {"status": "not_fitted"}
        
        feature_names = self.vectorizer.get_feature_names_out()
        return {
            "status": "ready",
            "documents_indexed": len(self.documents),
            "unique_terms": len(feature_names),
            "sample_terms": feature_names.tolist()[:20]
        }