# analyzer/advanced_analysis.py
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import re

class AdvancedPatternAnalyzer:
    def __init__(self):
        self.seasonal_keywords = {
            'winter': ['soup', 'hot chocolate', 'coat', 'gloves', 'scarf', 'heater', 'blanket', 'thermos'],
            'summer': ['ice cream', 'sunscreen', 'fan', 'lemonade', 'swimsuit', 'sunglasses', 'cooler', 'shorts'],
            'spring': ['gardening', 'flowers', 'seeds', 'cleaning', 'allergy', 'raincoat', 'umbrella', 'plant'],
            'fall': ['pumpkin', 'sweater', 'candle', 'jacket', 'boots', 'hot drink', 'spice', 'harvest']
        }
    
    def detect_product_seasonality(self, events: List[dict]) -> Dict[str, Any]:
        """Detect seasonal patterns from real transaction data"""
        seasonal_data = {
            'winter': {'products': {}, 'total_revenue': 0, 'transaction_count': 0},
            'spring': {'products': {}, 'total_revenue': 0, 'transaction_count': 0},
            'summer': {'products': {}, 'total_revenue': 0, 'transaction_count': 0},
            'fall': {'products': {}, 'total_revenue': 0, 'transaction_count': 0}
        }
        
        monthly_data = {}
        
        for event in events:
            if event.get("event_type") == "sale":
                payload = event.get("payload", {})
                products = payload.get("items", [])
                amount = payload.get("amount", 0)
                timestamp = event.get("ts", "")
                season = payload.get("season", "").lower()
                
                if not isinstance(products, list):
                    products = [products]
                
                # Extract month for trend analysis
                month = self._extract_month(timestamp)
                if month:
                    if month not in monthly_data:
                        monthly_data[month] = {'revenue': 0, 'transactions': 0}
                    monthly_data[month]['revenue'] += amount
                    monthly_data[month]['transactions'] += 1
                
                for product in products:
                    product_str = str(product).lower()
                    
                    # Determine season for this product
                    detected_season = self._predict_season_for_product(product_str, season)
                    
                    if detected_season in seasonal_data:
                        if product not in seasonal_data[detected_season]['products']:
                            seasonal_data[detected_season]['products'][product] = {
                                'revenue': 0,
                                'transactions': 0,
                                'monthly_trend': {}
                            }
                        
                        seasonal_data[detected_season]['products'][product]['revenue'] += amount / len(products)
                        seasonal_data[detected_season]['products'][product]['transactions'] += 1
                        seasonal_data[detected_season]['total_revenue'] += amount / len(products)
                        seasonal_data[detected_season]['transaction_count'] += 1
                        
                        # Track monthly trend
                        if month:
                            if month not in seasonal_data[detected_season]['products'][product]['monthly_trend']:
                                seasonal_data[detected_season]['products'][product]['monthly_trend'][month] = 0
                            seasonal_data[detected_season]['products'][product]['monthly_trend'][month] += amount / len(products)
        
        return self._generate_seasonal_insights(seasonal_data, monthly_data)
    
    def _predict_season_for_product(self, product: str, actual_season: str) -> str:
        """Predict the most likely season for a product"""
        # If season is provided in data, use it
        if actual_season and actual_season != "unknown":
            return actual_season
        
        # Otherwise predict based on product name keywords
        product_lower = product.lower()
        for season, keywords in self.seasonal_keywords.items():
            if any(keyword in product_lower for keyword in keywords):
                return season
        
        # Fallback: distribute evenly or use other logic
        seasons = ['winter', 'spring', 'summer', 'fall']
        return seasons[hash(product) % len(seasons)]  # Simple hash-based distribution
    
    def _extract_month(self, timestamp: str) -> str:
        """Extract month name from timestamp"""
        try:
            if isinstance(timestamp, str):
                # Handle different timestamp formats
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                return dt.strftime("%B")  # Full month name
        except:
            try:
                # Try other common formats
                dt = datetime.strptime(timestamp, '%Y-%m-%d')
                return dt.strftime("%B")
            except:
                pass
        return ""
    
    def _generate_seasonal_insights(self, seasonal_data: Dict, monthly_data: Dict) -> Dict[str, Any]:
        """Generate insights from analyzed seasonal data"""
        insights = []
        recommendations = []
        
        for season, data in seasonal_data.items():
            products = data.get('products', {})
            if products:
                # Find top 2 products for each season
                top_products = sorted(
                    products.items(), 
                    key=lambda x: x[1]['revenue'], 
                    reverse=True
                )[:2]
                
                for product_name, stats in top_products:
                    if stats['revenue'] > 0:  # Only include products with actual revenue
                        # Find peak month
                        monthly_trend = stats.get('monthly_trend', {})
                        peak_month = max(monthly_trend.items(), key=lambda x: x[1], default=('Unknown', 0))
                        
                        insight = {
                            'season': season,
                            'product': product_name,
                            'revenue': round(stats['revenue'], 2),
                            'transactions': stats['transactions'],
                            'peak_performance': peak_month[0],
                            'confidence': min(stats['revenue'] / 1000, 1.0)  # Simple confidence score
                        }
                        insights.append(insight)
                        
                        recommendations.append(
                            f"Focus on {product_name} during {season} season "
                            f"(generated ${stats['revenue']:.2f} revenue)"
                        )
        
        # If no seasonal insights found, create some from monthly data
        if not insights and monthly_data:
            insights = self._generate_fallback_insights(monthly_data)
        
        return {
            "insights": insights,
            "recommendations": recommendations[:8],
            "summary": {
                "total_seasons_analyzed": len([s for s in seasonal_data.values() if s['products']]),
                "total_insights": len(insights),
                "monthly_trends": monthly_data
            }
        }
    
    def _generate_fallback_insights(self, monthly_data: Dict) -> List[Dict]:
        """Generate insights from monthly data when seasonal data is sparse"""
        insights = []
        month_to_season = {
            'December': 'winter', 'January': 'winter', 'February': 'winter',
            'March': 'spring', 'April': 'spring', 'May': 'spring',
            'June': 'summer', 'July': 'summer', 'August': 'summer', 
            'September': 'fall', 'October': 'fall', 'November': 'fall'
        }
        
        for month, data in monthly_data.items():
            season = month_to_season.get(month, 'unknown')
            if data['revenue'] > 0:
                insights.append({
                    'season': season,
                    'product': f"General {season} products",
                    'revenue': data['revenue'],
                    'transactions': data['transactions'],
                    'peak_performance': month,
                    'confidence': 0.5
                })
        
        return insights


class SemanticSearchEngine:
    def find_cross_selling_opportunities(self, events: List[dict]) -> List[Dict[str, Any]]:
        """Find product bundling opportunities from real data"""
        product_pairs = {}
        
        for event in events:
            if event.get("event_type") == "sale":
                products = event.get("payload", {}).get("items", [])
                if isinstance(products, list) and len(products) >= 2:
                    # Track all product pairs in this transaction
                    for i in range(len(products)):
                        for j in range(i + 1, len(products)):
                            pair = tuple(sorted([str(products[i]), str(products[j])]))
                            if pair not in product_pairs:
                                product_pairs[pair] = 0
                            product_pairs[pair] += 1
        
        # Convert to recommendations
        opportunities = []
        for (product1, product2), count in sorted(product_pairs.items(), 
                                                key=lambda x: x[1], reverse=True)[:10]:
            if count >= 2:  # Minimum co-occurrence threshold
                opportunities.append({
                    "product_a": product1,
                    "product_b": product2,
                    "co_occurrence_count": count,
                    "recommendation": f"Bundle {product1} with {product2} - purchased together {count} times"
                })
        
        return opportunities