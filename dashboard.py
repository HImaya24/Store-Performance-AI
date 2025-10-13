import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Store Performance AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with new search styles
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #1f77b4;
    }
    .agent-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .status-online {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-offline {
        color: #F44336;
        font-weight: bold;
    }
    .port-label {
        background-color: #e0e0e0;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-left: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .chart-container {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .viz-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 15px;
        font-size: 1.5rem;
    }
    .search-highlight {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
    .product-insight {
        background-color: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .seasonal-tag {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    .insight-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .ir-search-result {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
    }
    .similarity-score {
        background-color: #4caf50;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Agent endpoints with your specific ports
AGENT_ENDPOINTS = {
    "collector": "http://localhost:8100",
    "coordinator": "http://localhost:8110", 
    "analyzer": "http://localhost:8101",
    "kpi": "http://localhost:8102",
    "report": "http://localhost:8103"
}

# Check agent status
def check_agent_status(agent_name):
    try:
        response = requests.get(f"{AGENT_ENDPOINTS[agent_name]}/health", timeout=2)
        return response.status_code == 200, "Online" if response.status_code == 200 else "Offline"
    except:
        return False, "Offline"

# Load data from collector agent
@st.cache_data
def load_data_from_collector():
    try:
        response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=30)
        if response.status_code == 200:
            return pd.DataFrame(response.json()), True
    except:
        pass

    # Fallback: generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    products = ['coffee maker', 'blender', 'toaster', 'microwave', 'air fryer', 
                'rice cooker', 'juicer', 'food processor', 'electric kettle', 'mixer']
    data = {
        'event_id': [f"event_{i}" for i in range(len(dates))],
        'store_id': np.random.choice(['Los Angeles', 'New York', 'Chicago', 'Miami', 'Seattle'], len(dates)),
        'ts': dates,
        'event_type': np.random.choice(['sale', 'inventory', 'visit', 'return', 'restock'], len(dates), p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'payload': [{
            'amount': np.random.uniform(10, 500), 
            'items': np.random.choice(products, np.random.randint(1, 4)),
            'customer_category': np.random.choice(['VIP', 'Regular', 'New'], p=[0.2, 0.6, 0.2]),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'Cash', 'Digital Wallet']),
            'season': np.random.choice(['winter', 'spring', 'summer', 'fall'])
        } for _ in range(len(dates))]
    }
    return pd.DataFrame(data), False

# Function to extract unique products from events
def extract_unique_products(df):
    products = set()
    for payload in df['payload']:
        if isinstance(payload, dict) and 'items' in payload:
            items = payload['items']
            if isinstance(items, list):
                products.update(items)
            else:
                products.add(items)
    return sorted(list(products))

# Function to search for products in events
def search_products(df, search_term):
    search_term = search_term.lower()
    results = []
    
    for idx, row in df.iterrows():
        payload = row['payload']
        if isinstance(payload, dict) and 'items' in payload:
            items = payload['items']
            if isinstance(items, list):
                for item in items:
                    if search_term in str(item).lower():
                        results.append({
                            'event_id': row['event_id'],
                            'store_id': row['store_id'],
                            'timestamp': row['ts'],
                            'product': item,
                            'amount': payload.get('amount', 0),
                            'customer_category': payload.get('customer_category', 'Unknown'),
                            'payment_method': payload.get('payment_method', 'Unknown'),
                            'season': payload.get('season', 'Unknown')
                        })
            else:
                if search_term in str(items).lower():
                    results.append({
                        'event_id': row['event_id'],
                        'store_id': row['store_id'],
                        'timestamp': row['ts'],
                        'product': items,
                        'amount': payload.get('amount', 0),
                        'customer_category': payload.get('customer_category', 'Unknown'),
                        'payment_method': payload.get('payment_method', 'Unknown'),
                        'season': payload.get('season', 'Unknown')
                    })
    
    return pd.DataFrame(results)

# Function to get product analysis from analyzer
def get_product_analysis(product_name, events):
    try:
        # Filter events for this product
        product_events = []
        for event in events:
            if event.get("event_type") == "sale":
                payload = event.get("payload", {})
                items = payload.get("items", [])
                if not isinstance(items, list):
                    items = [items]
                
                if any(product_name.lower() in str(item).lower() for item in items):
                    product_events.append(event)
        
        if not product_events:
            return {"error": f"No sales data found for '{product_name}'"}, False
        
        # Get seasonal analysis
        try:
            response = requests.post(
                f"{AGENT_ENDPOINTS['analyzer']}/advanced-analysis/seasonal",
                json=product_events,
                timeout=30
            )
            if response.status_code == 200:
                seasonal_data = response.json()
            else:
                seasonal_data = {"insights": [], "recommendations": []}
        except:
            seasonal_data = {"insights": [], "recommendations": []}
        
        # Get cross-selling opportunities
        try:
            response = requests.post(
                f"{AGENT_ENDPOINTS['analyzer']}/advanced-analysis/cross-selling",
                json=product_events,
                timeout=30
            )
            if response.status_code == 200:
                cross_selling_data = response.json()
            else:
                cross_selling_data = {"opportunities": []}
        except:
            cross_selling_data = {"opportunities": []}
        
        # Calculate basic product metrics
        total_revenue = 0
        total_transactions = len(product_events)
        stores = set()
        seasons = {}
        
        for event in product_events:
            payload = event.get("payload", {})
            total_revenue += payload.get("amount", 0)
            stores.add(event.get("store_id", "Unknown"))
            season = payload.get("season", "Unknown")
            seasons[season] = seasons.get(season, 0) + 1
        
        avg_revenue = total_revenue / total_transactions if total_transactions > 0 else 0
        
        return {
            "product_name": product_name,
            "total_revenue": round(total_revenue, 2),
            "total_transactions": total_transactions,
            "average_revenue_per_sale": round(avg_revenue, 2),
            "stores_sold_in": list(stores),
            "seasonal_distribution": seasons,
            "seasonal_insights": seasonal_data.get("insights", []),
            "recommendations": seasonal_data.get("recommendations", []),
            "cross_selling_opportunities": cross_selling_data.get("opportunities", [])
        }, True
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}, False

# Function to perform IR Search
def perform_ir_search(query: str, events: List[dict]):
    """Perform semantic IR search using the analyzer"""
    try:
        response = requests.post(
            f"{AGENT_ENDPOINTS['analyzer']}/semantic-search",
            json=events,
            params={"query": query},
            timeout=30
        )
        if response.status_code == 200:
            return response.json(), True
        else:
            return {"error": f"Analyzer returned status {response.status_code}"}, False
    except Exception as e:
        return {"error": str(e)}, False

# Function to create product performance charts
def create_product_charts(product_df, product_name):
    charts = {}
    
    if product_df.empty:
        return charts
    
    # 1. Sales Over Time
    product_df['timestamp'] = pd.to_datetime(product_df['timestamp'])
    product_df['month'] = product_df['timestamp'].dt.to_period('M').astype(str)
    monthly_sales = product_df.groupby('month')['amount'].sum().reset_index()
    
    if not monthly_sales.empty:
        fig_trend = px.line(monthly_sales, x='month', y='amount',
                           title=f"Monthly Sales Trend for {product_name}",
                           labels={'amount': 'Sales Amount ($)', 'month': 'Month'})
        charts['sales_trend'] = fig_trend
    
    # 2. Sales by Store
    store_sales = product_df.groupby('store_id')['amount'].sum().reset_index()
    if not store_sales.empty:
        fig_stores = px.bar(store_sales, x='store_id', y='amount',
                           title=f"Sales by Store for {product_name}",
                           labels={'amount': 'Total Sales ($)', 'store_id': 'Store'})
        charts['store_sales'] = fig_stores
    
    # 3. Sales by Season
    if 'season' in product_df.columns:
        season_sales = product_df.groupby('season')['amount'].sum().reset_index()
        if not season_sales.empty:
            fig_season = px.pie(season_sales, values='amount', names='season',
                               title=f"Sales Distribution by Season for {product_name}")
            charts['season_sales'] = fig_season
    
    # 4. Customer Category Analysis
    if 'customer_category' in product_df.columns:
        customer_sales = product_df.groupby('customer_category')['amount'].sum().reset_index()
        if not customer_sales.empty:
            fig_customer = px.bar(customer_sales, x='customer_category', y='amount',
                                 title=f"Sales by Customer Category for {product_name}",
                                 labels={'amount': 'Total Sales ($)', 'customer_category': 'Customer Category'})
            charts['customer_sales'] = fig_customer
    
    return charts

# Function to call coordinator agent
def trigger_data_processing(process_type):
    try:
        # First get data from collector
        response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=30)
        if response.status_code != 200:
            return False, "Could not get data from collector"

        events = response.json()

        # ✅ FIX: Limit events to 20 as per coordinator requirement
        if len(events) > 20:
            events = events[:20]
            st.info(f"📊 Limited to 20 events (coordinator limit)")

        # Send to coordinator for processing
        response = requests.post(
            f"{AGENT_ENDPOINTS['coordinator']}/orchestrate",
            json={"events": events},
            headers={"X-API-KEY": "demo-key"},
            timeout=300
        )
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

# Function to get KPIs from KPI agent
def get_kpis():
    try:
        response = requests.get(f"{AGENT_ENDPOINTS['kpi']}/kpis", timeout=30)
        if response.status_code == 200:
            kpi_data = response.json()
            return kpi_data, True
        else:
            st.error(f"❌ KPI agent returned status {response.status_code}")
            return None, False
    except Exception as e:
        st.error(f"💥 KPI error: {str(e)}")
        return None, False

# Function to get analysis from analyzer agent
def get_analysis(analysis_type):
    try:
        # First get data from collector
        response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=30)
        if response.status_code != 200:
            st.error(f"❌ Collector returned status {response.status_code}")
            return None, False
        
        events = response.json()
        
        # Limit to reasonable number for testing
        if len(events) > 10:
            events = events[:10]
            st.info(f"📊 Limited to 10 events for faster analysis")
        
        # Send to analyzer
        response = requests.post(
            f"{AGENT_ENDPOINTS['analyzer']}/analyze", 
            json=events,
            timeout=60
        )
        
        if response.status_code == 200:
            analysis_data = response.json()
            return analysis_data, True
        else:
            st.error(f"❌ Analyzer returned status {response.status_code}: {response.text}")
            return None, False
            
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to Analyzer agent. Make sure it's running on port 8101.")
        return None, False
    except requests.exceptions.Timeout:
        st.error("⏰ Analyzer request timed out. Try with fewer events.")
        return None, False
    except Exception as e:
        st.error(f"💥 Unexpected error: {str(e)}")
        return None, False

# Function to generate report from report agent
def generate_report(store_id=None):
    """
    Calls the Report Agent to generate and return a report for a store.
    If no store_id provided, tries to find available stores.
    """
    # If no store specified, try to get the first available store
    if store_id is None:
        try:
            # Try to get available stores from KPI agent
            response = requests.get(f"{AGENT_ENDPOINTS['kpi']}/kpis", timeout=5)
            if response.status_code == 200:
                kpis = response.json()
                if kpis and len(kpis) > 0:
                    store_id = kpis[0].get('store_id', 'Los Angeles')
                else:
                    return {"status": "error", "message": "No KPI data found. Please run analysis first."}, False
            else:
                return {"status": "error", "message": "Could not connect to KPI agent to find available stores."}, False
        except:
            return {"status": "error", "message": "Error finding available stores. Try specifying a store ID."}, False

    try:
        # Call the correct endpoint to get the HTML report
        report_url = f"{AGENT_ENDPOINTS['report']}/report/{store_id}"
        response = requests.get(report_url, timeout=60)

        if response.status_code == 200:
            return {"status": "success", "report_html": response.text, "store_id": store_id}, True
        else:
            return {"status": "error", "message": f"Report agent returned status {response.status_code} for store '{store_id}'. Have you run analysis first?"}, False

    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Could not connect to Report Agent (is it running on 8103?)"}, False
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}, False

# Function to create charts for visualization tab
def create_visualization_charts(df):
    charts = {}

    # Extract amount from payload if available
    if 'payload' in df.columns:
        df['amount'] = df['payload'].apply(lambda x: x.get('amount', 0) if isinstance(x, dict) else 0)
        df['items'] = df['payload'].apply(lambda x: x.get('items', 1) if isinstance(x, dict) else 1)

    # 1. Sales Over Time by Store
    if 'store_id' in df.columns and 'ts' in df.columns and 'amount' in df.columns:
        sales_by_store = df[df['event_type'] == 'sale'].groupby(['ts', 'store_id'])['amount'].sum().reset_index()
        fig_sales = px.line(sales_by_store, x='ts', y='amount', color='store_id',
                           title="Sales Over Time by Store",
                           labels={'amount': 'Sales Amount ($)', 'ts': 'Date', 'store_id': 'Store'})
        charts['sales_over_time'] = fig_sales

    # 2. Event Type Distribution
    if 'event_type' in df.columns:
        event_counts = df['event_type'].value_counts().reset_index()
        event_counts.columns = ['event_type', 'count']
        fig_events = px.pie(event_counts, values='count', names='event_type',
                           title="Distribution of Event Types")
        charts['event_distribution'] = fig_events

    # 3. Monthly Sales Trend
    if 'ts' in df.columns and 'amount' in df.columns:
        df['month'] = df['ts'].dt.to_period('M').astype(str)
        monthly_sales = df[df['event_type'] == 'sale'].groupby('month')['amount'].sum().reset_index()
        fig_monthly = px.bar(monthly_sales, x='month', y='amount',
                            title="Monthly Sales Trend",
                            labels={'amount': 'Total Sales ($)', 'month': 'Month'})
        charts['monthly_sales'] = fig_monthly

    # 4. Store Performance Comparison
    if 'store_id' in df.columns and 'amount' in df.columns:
        store_performance = df[df['event_type'] == 'sale'].groupby('store_id')['amount'].agg(['sum', 'count', 'mean']).reset_index()
        store_performance.columns = ['store_id', 'total_sales', 'transaction_count', 'avg_transaction']

        fig_performance = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Total Sales by Store', 'Number of Transactions', 'Average Transaction Value')
        )

        fig_performance.add_trace(
            go.Bar(x=store_performance['store_id'], y=store_performance['total_sales'],
                  name='Total Sales', marker_color='#1f77b4'),
            row=1, col=1
        )

        fig_performance.add_trace(
            go.Bar(x=store_performance['store_id'], y=store_performance['transaction_count'],
                  name='Transaction Count', marker_color='#ff7f0e'),
            row=1, col=2
        )

        fig_performance.add_trace(
            go.Bar(x=store_performance['store_id'], y=store_performance['avg_transaction'],
                  name='Avg Transaction', marker_color='#2ca02c'),
            row=1, col=3
        )

        fig_performance.update_layout(height=400, showlegend=False, title_text="Store Performance Comparison")
        charts['store_performance'] = fig_performance

    return charts

# Main dashboard
def main():
    st.markdown('<h1 class="main-header">Store Performance AI Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar with agent status and controls
    with st.sidebar:
        st.header("🛠️ Agent Controls")

        # Agent status
        st.subheader("Agent Status")
        for agent in AGENT_ENDPOINTS:
            is_online, status = check_agent_status(agent)
            status_class = "status-online" if is_online else "status-offline"
            port = AGENT_ENDPOINTS[agent].split(":")[-1]
            st.markdown(f"**{agent.capitalize()}** <span class='port-label'>:{port}</span>: <span class='{status_class}'>{status}</span>",
                       unsafe_allow_html=True)

        st.divider()

        # Quick actions
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh All Data"):
            st.rerun()

        if st.button("📊 Generate Full Report"):
            with st.spinner("Generating comprehensive report..."):
                report, success = generate_report("Los Angeles")
                if success:
                    st.success("Report generated successfully!")
                    with st.expander("View Report"):
                        st.components.v1.html(report["report_html"], height=400, scrolling=True)
                else:
                    st.error(f"Failed to generate report: {report.get('message', 'Unknown error')}")

        st.divider()
        st.subheader("Agent Ports Configuration")
        st.info("""
        - Collector: :8100
        - Coordinator: :8110  
        - Analyzer: :8101
        - KPI: :8102
        - Report: :8103
        """)

    # Load data
    df, from_collector = load_data_from_collector()

    # Convert date column if needed
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date
        df['month'] = df['ts'].dt.to_period('M')

    # Create tabs with IR Search added
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📥 Collector",
        "🔄 Coordinator", 
        "🔍 Analyzer",
        "🔎 Product Search",
        "🎯 IR Search",
        "📈 KPI",
        "📋 Report"
    ])

    # 1. Collector Agent Tab
    with tab1:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">📥 Collector Agent <span class="port-label">:8100</span></h2>', unsafe_allow_html=True)

        # Display data source info
        if from_collector:
            st.success("✅ Data loaded successfully from Collector agent (port 8100)")
        else:
            st.warning("⚠️ Using sample data (Collector agent on port 8100 unavailable)")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("Data Summary")
            st.metric("Total Events", len(df))
            if 'store_id' in df.columns:
                st.metric("Unique Stores", df['store_id'].nunique())
            if 'event_type' in df.columns:
                st.metric("Event Types", df['event_type'].nunique())

        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Coordinator Agent Tab
    with tab2:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">🔄 Coordinator Agent <span class="port-label">:8110</span></h2>', unsafe_allow_html=True)

        # Display available events count
        try:
            response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=10)
            if response.status_code == 200:
                events = response.json()
                st.info(f"📊 Collector has {len(events)} events available")

                # Let user select how many events to process
                max_events = min(len(events), 20)
                event_count = st.slider(
                    "Number of events to process",
                    min_value=1,
                    max_value=max_events,
                    value=min(10, max_events),
                    help=f"Coordinator limit: 20 events maximum"
                )
            else:
                st.warning("Cannot get event count from collector")
                event_count = 10
        except:
            st.warning("Cannot connect to collector to get event count")
            event_count = 10

        process_type = st.selectbox("Select Process Type", ["full_pipeline", "quick_scan"], key="process_select")

        if st.button("Process Data", key="process_btn"):
            with st.spinner("Processing data through coordinator..."):
                success, result = trigger_data_processing(process_type)
                if success:
                    if result.get('status') != 'rejected':
                        st.success(f"✅ Processing complete! Batch ID: {result.get('batch_id')}, Status: {result.get('status')}")
                        st.json(result)
                    else:
                        st.error(f"❌ Processing rejected: {result.get('message')}")
                else:
                    st.error(f"❌ Processing failed: {result}")

        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Analyzer Agent Tab
    with tab3:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">🔍 Analyzer Agent <span class="port-label">:8101</span></h2>', unsafe_allow_html=True)

        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["sales_analysis", "inventory_analysis", "customer_analysis"],
            key="analysis_select"
        )

        if st.button("Run Analysis", key="analyze_btn"):
            with st.spinner("Running analysis..."):
                analysis, success = get_analysis(analysis_type)
                if success:
                    st.success("✅ Analysis completed successfully!")
                    
                    # ✅ FIXED: Actually display the insights
                    if analysis and 'insights_list' in analysis:
                        insights = analysis['insights_list']
                        st.subheader(f"📊 Generated Insights ({len(insights)} total)")
                        
                        # Display all insights in a nice format
                        for i, insight in enumerate(insights):
                            with st.container():
                                st.markdown('<div class="insight-card">', unsafe_allow_html=True)
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Insight #{i+1}:** {insight.get('text', 'No text')}")
                                    st.markdown(f"*Explanation:* {insight.get('explanation', 'No explanation')}")
                                    st.markdown(f"**Store:** {insight.get('store_id', 'Unknown')}")
                                    st.markdown(f"**Tags:** {', '.join(insight.get('tags', []))}")
                                    
                                with col2:
                                    confidence = insight.get('confidence', 0)
                                    st.metric("Confidence", f"{confidence:.0%}")
                                    
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.divider()
                        
                        # Show analysis metadata in expander
                        with st.expander("View Raw Analysis Data"):
                            st.json(analysis)
                            
                    else:
                        st.warning("No insights were generated. Check if there's data to analyze.")
                        
                else:
                    st.error("❌ Failed to run analysis (Analyzer agent on port 8101 unavailable)")

        st.markdown('</div>', unsafe_allow_html=True)

    # 4. Product Search Tab
    with tab4:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">🔎 Product Search & Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Search input
            search_query = st.text_input(
                "🔍 Search for products",
                placeholder="Enter product name (e.g., coffee maker, blender, toaster...)",
                help="Search for specific products in your sales data"
            )
        
        with col2:
            st.markdown("###")
            analyze_btn = st.button("Analyze Product", type="primary")
        
        # Extract unique products for suggestions
        if from_collector and not df.empty:
            unique_products = extract_unique_products(df)
            if unique_products:
                st.markdown(f"**Available products:** {', '.join(unique_products[:10])}{'...' if len(unique_products) > 10 else ''}")
        
        if search_query and analyze_btn:
            with st.spinner(f"Searching for '{search_query}' and analyzing..."):
                # Search for products
                search_results = search_products(df, search_query)
                
                if not search_results.empty:
                    st.success(f"✅ Found {len(search_results)} transactions for products matching '{search_query}'")
                    
                    # Display search results summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_revenue = search_results['amount'].sum()
                        st.metric("Total Revenue", f"${total_revenue:,.2f}")
                    
                    with col2:
                        st.metric("Transactions", len(search_results))
                    
                    with col3:
                        unique_stores = search_results['store_id'].nunique()
                        st.metric("Stores", unique_stores)
                    
                    with col4:
                        avg_sale = search_results['amount'].mean()
                        st.metric("Avg Sale", f"${avg_sale:,.2f}")
                    
                    # Get detailed product analysis
                    all_events = [row.to_dict() for _, row in df.iterrows() if isinstance(df.loc[_, 'payload'], dict)]
                    analysis_result, success = get_product_analysis(search_query, all_events)
                    
                    if success:
                        # Display product insights
                        st.markdown("### 📈 Product Insights")
                        
                        # Key metrics
                        st.markdown('<div class="product-insight">', unsafe_allow_html=True)
                        st.markdown(f"**Product:** {analysis_result['product_name']}")
                        st.markdown(f"**Total Revenue:** ${analysis_result['total_revenue']:,.2f}")
                        st.markdown(f"**Total Transactions:** {analysis_result['total_transactions']}")
                        st.markdown(f"**Average Revenue per Sale:** ${analysis_result['average_revenue_per_sale']:,.2f}")
                        st.markdown(f"**Stores:** {', '.join(analysis_result['stores_sold_in'])}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Seasonal insights
                        if analysis_result['seasonal_insights']:
                            st.markdown("### 🌸 Seasonal Patterns")
                            for insight in analysis_result['seasonal_insights']:
                                if search_query.lower() in str(insight.get('product', '')).lower():
                                    st.markdown(f"""
                                    <div class="search-highlight">
                                        <strong>Season:</strong> <span class="seasonal-tag">{insight.get('season', 'N/A')}</span><br>
                                        <strong>Revenue:</strong> ${insight.get('revenue', 0):,.2f}<br>
                                        <strong>Peak Performance:</strong> {insight.get('peak_performance', 'N/A')}<br>
                                        <strong>Confidence:</strong> {insight.get('confidence', 0):.1%}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Recommendations
                        if analysis_result['recommendations']:
                            st.markdown("### 💡 Recommendations")
                            for rec in analysis_result['recommendations'][:3]:
                                st.info(rec)
                        
                        # Cross-selling opportunities
                        if analysis_result['cross_selling_opportunities']:
                            st.markdown("### 🛍️ Cross-Selling Opportunities")
                            for opportunity in analysis_result['cross_selling_opportunities'][:3]:
                                st.success(f"**Bundle:** {opportunity['product_a']} + {opportunity['product_b']} "
                                          f"({opportunity['co_occurrence_count']} times together)")
                        
                        # Create and display product charts
                        st.markdown("### 📊 Product Performance Charts")
                        product_charts = create_product_charts(search_results, search_query)
                        
                        if product_charts:
                            cols = st.columns(2)
                            chart_keys = list(product_charts.keys())
                            for i, chart_key in enumerate(chart_keys):
                                with cols[i % 2]:
                                    st.plotly_chart(product_charts[chart_key], use_container_width=True)
                        else:
                            st.info("Not enough data to generate detailed charts for this product.")
                    
                    else:
                        st.error(f"❌ Failed to analyze product: {analysis_result.get('error', 'Unknown error')}")
                    
                    # Show raw search results in expander
                    with st.expander("View Raw Search Results"):
                        st.dataframe(search_results, use_container_width=True)
                        
                else:
                    st.warning(f"❌ No transactions found for products matching '{search_query}'")
                    st.info("Try searching for common products like: coffee maker, blender, toaster, microwave, etc.")
        
        elif analyze_btn and not search_query:
            st.warning("Please enter a product name to search for")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 5. NEW: IR Search Tab
    with tab5:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">🎯 IR Search (Semantic Search)</h2>', unsafe_allow_html=True)
        
        st.info("""
        **IR Search** uses semantic understanding to find patterns in your data. 
        Instead of just matching keywords, it understands the meaning of your queries.
        """)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ir_query = st.text_input(
                "🔍 Enter your semantic search query",
                placeholder="e.g., high value customers, winter sales trends, mobile payment users...",
                help="Use natural language to find patterns in your data"
            )
        
        with col2:
            st.markdown("###")
            ir_search_btn = st.button("Run IR Search", type="primary", key="ir_search_btn")
        
        # Example queries
        with st.expander("💡 Example IR Search Queries"):
            st.markdown("""
            **Customer Behavior:**
            - `loyal customers with repeat purchases`
            - `high value transactions`
            - `new customer acquisition patterns`
            
            **Seasonal Trends:**
            - `winter season best sellers` 
            - `holiday shopping patterns`
            - `seasonal product preferences`
            
            **Payment & Store Patterns:**
            - `mobile payment adoption`
            - `premium store customers`
            - `discount-driven purchases`
            
            **Product Analysis:**
            - `frequently bundled products`
            - `premium product buyers`
            - `electronic appliance trends`
            """)
        
        if ir_query and ir_search_btn:
            with st.spinner(f"Performing semantic search for '{ir_query}'..."):
                # Get events from collector
                try:
                    response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=30)
                    if response.status_code == 200:
                        events = response.json()
                        
                        # Perform IR search
                        search_results, success = perform_ir_search(ir_query, events)
                        
                        if success:
                            if 'results' in search_results and search_results['results']:
                                st.success(f"✅ Found {len(search_results['results'])} semantic matches for '{ir_query}'")
                                
                                # Display results
                                for i, result in enumerate(search_results['results']):
                                    metadata = result.get('metadata', {})
                                    similarity = result.get('similarity_score', 0)
                                    
                                    st.markdown('<div class="ir-search-result">', unsafe_allow_html=True)
                                    
                                    col1, col2 = st.columns([4, 1])
                                    
                                    with col1:
                                        st.markdown(f"**Match #{i+1}**")
                                        st.markdown(f"**Store:** {metadata.get('store_id', 'Unknown')}")
                                        st.markdown(f"**Amount:** ${metadata.get('amount', 0):.2f}")
                                        st.markdown(f"**Products:** {', '.join([str(p) for p in metadata.get('products', [])])}")
                                        st.markdown(f"**Preview:** {result.get('document_preview', 'No preview')}")
                                        
                                    with col2:
                                        st.markdown(f"<div class='similarity-score'>Score: {similarity:.2f}</div>", 
                                                   unsafe_allow_html=True)
                                    
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                            else:
                                st.warning(f"❌ No semantic matches found for '{ir_query}'")
                                st.info("Try using different terms or check the example queries above.")
                                
                        else:
                            st.error(f"❌ IR Search failed: {search_results.get('error', 'Unknown error')}")
                            
                    else:
                        st.error("❌ Could not fetch events from collector")
                        
                except Exception as e:
                    st.error(f"❌ IR Search error: {str(e)}")
        
        elif ir_search_btn and not ir_query:
            st.warning("Please enter a search query")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # 6. KPI Agent Tab (moved to tab6)
    with tab6:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">📈 KPI Agent <span class="port-label">:8102</span></h2>', unsafe_allow_html=True)

        if st.button("Calculate KPIs", key="kpi_btn"):
            with st.spinner("Calculating KPIs..."):
                kpis, success = get_kpis()
                if success:
                    st.success("✅ KPIs calculated successfully!")

                    if kpis and len(kpis) > 0:
                        # ✅ FIXED: Display KPIs properly
                        st.subheader("📈 Store Performance KPIs")
                        
                        # Create metrics for each store
                        for kpi in kpis:
                            if isinstance(kpi, dict):
                                store_id = kpi.get('store_id', 'Unknown Store')
                                metrics = kpi.get('metrics', {})
                                
                                st.markdown(f"### 🏪 {store_id}")
                                
                                # Display main metrics in columns
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    total_sales = metrics.get('total_sales', 0)
                                    st.metric("Total Sales", f"${total_sales:,.2f}")
                                    
                                with col2:
                                    sales_count = metrics.get('sales_count', 0)
                                    st.metric("Transactions", f"{sales_count:,}")
                                    
                                with col3:
                                    aov = metrics.get('average_order_value', 0)
                                    st.metric("Avg Order Value", f"${aov:,.2f}")
                                    
                                with col4:
                                    total_items = metrics.get('total_items_sold', 0)
                                    st.metric("Items Sold", f"{total_items:,}")
                                
                                # Display breakdowns if available
                                if kpi.get('by_customer_category'):
                                    st.markdown("**Customer Category Breakdown:**")
                                    cat_data = kpi['by_customer_category']
                                    if cat_data:
                                        cat_cols = st.columns(len(cat_data))
                                        for idx, (category, amount) in enumerate(cat_data.items()):
                                            with cat_cols[idx % len(cat_cols)]:
                                                st.metric(category, f"${amount:,.2f}")
                                
                                if kpi.get('by_payment_method'):
                                    st.markdown("**Payment Method Breakdown:**")
                                    pay_data = kpi['by_payment_method']
                                    if pay_data:
                                        pay_cols = st.columns(len(pay_data))
                                        for idx, (method, amount) in enumerate(pay_data.items()):
                                            with pay_cols[idx % len(pay_cols)]:
                                                st.metric(method, f"${amount:,.2f}")
                                
                                st.divider()
                            else:
                                st.error(f"Invalid KPI data format: {kpi}")
                        
                        # Show raw KPI data in expander
                        with st.expander("View Raw KPI Data"):
                            st.json(kpis)
                            
                    else:
                        st.info("No KPI data available. Process some data first.")
                else:
                    st.error("❌ Failed to calculate KPIs (KPI agent on port 8102 unavailable)")

        st.markdown('</div>', unsafe_allow_html=True)

    # 7. Report Agent Tab (moved to tab7)
    with tab7:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">📋 Report Agent <span class="port-label">:8103</span></h2>', unsafe_allow_html=True)

        # Get available stores from data
        available_stores = []
        if 'store_id' in df.columns:
            available_stores = df['store_id'].unique().tolist()

        if available_stores:
            selected_store = st.selectbox("Select Store", available_stores, key="report_store")
        else:
            selected_store = "Los Angeles"
            st.info("No store data available. Using default store.")

        if st.button("Generate Report", key="report_btn"):
            with st.spinner("Generating report..."):
                report, success = generate_report(selected_store)
                if success:
                    st.success(f"✅ Report for {selected_store} generated successfully!")

                    # Display the HTML report
                    st.components.v1.html(
                        report["report_html"], height=600, scrolling=True
                    )

                    # Add download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report["report_html"],
                        file_name=f"store_report_{selected_store}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                else:
                    st.error(f"❌ Failed to generate report: {report.get('message', 'Unknown error')}")

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()