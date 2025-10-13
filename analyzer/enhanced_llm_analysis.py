# analyzer/enhanced_llm_analysis.py
import pandas as pd
from datetime import datetime
import re
from typing import List, Dict, Any

class AdvancedPatternAnalyzer:
    def __init__(self):
        self.seasonal_patterns = {
            'winter': ['hot chocolate', 'soup', 'blanket', 'heater'],
            'summer': ['ice cream', 'sunscreen', 'fan', 'lemonade'],
            'spring': ['gardening', 'cleaning', 'allergy', 'flowers'],
            'fall': ['pumpkin', 'sweater', 'hot drink', 'candle']
        }
    
    def detect_product_seasonality(self, events: List[dict]) -> Dict[str, Any]:
        """Advanced seasonality detection with product categorization"""
        seasonal_insights = {}
        
        for event in events:
            if event.get("event_type") == "sale":
                payload = event.get("payload", {})
                products = payload.get("items", [])
                season = payload.get("season", "").lower()
                amount = payload.get("amount", 0)
                timestamp = event.get("ts", "")
                
                if not isinstance(products, list):
                    products = [products]
                
                for product in products:
                    product_lower = str(product).lower()
                    
                    # Detect seasonal patterns
                    detected_season = self._predict_season(product_lower, season, timestamp)
                    
                    if detected_season not in seasonal_insights:
                        seasonal_insights[detected_season] = {
                            "products": {},
                            "total_revenue": 0,
                            "peak_months": []
                        }
                    
                    if product not in seasonal_insights[detected_season]["products"]:
                        seasonal_insights[detected_season]["products"][product] = {
                            "revenue": 0,
                            "transactions": 0,
                            "monthly_trend": {}
                        }
                    
                    # Extract month from timestamp for trend analysis
                    month = self._extract_month(timestamp)
                    seasonal_insights[detected_season]["products"][product]["revenue"] += amount / len(products)
                    seasonal_insights[detected_season]["products"][product]["transactions"] += 1
                    
                    # Track monthly trends
                    if month not in seasonal_insights[detected_season]["products"][product]["monthly_trend"]:
                        seasonal_insights[detected_season]["products"][product]["monthly_trend"][month] = 0
                    seasonal_insights[detected_season]["products"][product]["monthly_trend"][month] += amount / len(products)
        
        return self._generate_seasonal_recommendations(seasonal_insights)
    
    def _predict_season(self, product: str, actual_season: str, timestamp: str) -> str:
        """Predict the most likely season for a product"""
        # If season is provided, use it
        if actual_season and actual_season != "unknown":
            return actual_season
        
        # Otherwise predict based on product name and timestamp
        for season, keywords in self.seasonal_patterns.items():
            if any(keyword in product for keyword in keywords):
                return season
        
        # Fallback to timestamp-based season detection
        return self._get_season_from_timestamp(timestamp)
    
    def _extract_month(self, timestamp: str) -> str:
        """Extract month from timestamp"""
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime("%B")
        except:
            pass
        return "Unknown"
    
    def _get_season_from_timestamp(self, timestamp: str) -> str:
        """Determine season from timestamp"""
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                month = dt.month
                if month in [12, 1, 2]:
                    return "winter"
                elif month in [3, 4, 5]:
                    return "spring"
                elif month in [6, 7, 8]:
                    return "summer"
                elif month in [9, 10, 11]:
                    return "fall"
        except:
            pass
        return "unknown"
    
    def _generate_seasonal_recommendations(self, seasonal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable seasonal recommendations"""
        recommendations = []
        insights = []
        
        for season, data in seasonal_data.items():
            products = data.get("products", {})
            if products:
                # Find top performing product for the season
                top_product = max(products.items(), key=lambda x: x[1]["revenue"], default=None)
                if top_product:
                    product_name, stats = top_product
                    
                    # Find peak month
                    monthly_trend = stats.get("monthly_trend", {})
                    peak_month = max(monthly_trend.items(), key=lambda x: x[1], default=None)
                    
                    insight = {
                        "type": "seasonal_insight",
                        "season": season,
                        "product": product_name,
                        "revenue": stats["revenue"],
                        "transactions": stats["transactions"],
                        "peak_performance": peak_month[0] if peak_month else "Unknown",
                        "confidence": min(stats["revenue"] / 1000, 1.0)  # Simple confidence score
                    }
                    insights.append(insight)
                    
                    recommendations.append(
                        f"🚀 **Seasonal Opportunity**: {product_name} performs best in {season} "
                        f"(generated ${stats['revenue']:.2f} revenue). "
                        f"Stock up inventory before {peak_month[0] if peak_month else season}."
                    )
        
        return {
            "insights": insights,
            "recommendations": recommendations[:10],
            "summary": f"Found {len(insights)} seasonal patterns across {len(seasonal_data)} seasons"
        }