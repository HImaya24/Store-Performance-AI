# dashboard.py
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

# Page configuration
st.set_page_config(
    page_title="Store Performance AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=5)
        if response.status_code == 200:
            return pd.DataFrame(response.json()), True
    except:
        pass
    
    # Fallback: generate sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = {
        'event_id': [f"event_{i}" for i in range(len(dates))],
        'store_id': np.random.choice(['Los Angeles', 'New York', 'Chicago', 'Miami', 'Seattle'], len(dates)),
        'ts': dates,
        'event_type': np.random.choice(['sale', 'inventory', 'visit', 'return', 'restock'], len(dates), p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'payload': [{'amount': np.random.uniform(10, 500), 'items': np.random.randint(1, 10)} for _ in range(len(dates))]
    }
    return pd.DataFrame(data), False

# Function to call coordinator agent
def trigger_data_processing(process_type):
    try:
        # First get data from collector
        response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=5)
        if response.status_code != 200:
            return False, "Could not get data from collector"
        
        events = response.json()
        
        # Send to coordinator for processing
        response = requests.post(
            f"{AGENT_ENDPOINTS['coordinator']}/orchestrate", 
            json={"events": events},
            headers={"X-API-KEY": "demo-key"},
            timeout=30
        )
        return response.status_code == 200, response.json() if response.status_code == 200 else response.text
    except Exception as e:
        return False, str(e)

# Function to get KPIs from KPI agent
def get_kpis():
    try:
        response = requests.get(f"{AGENT_ENDPOINTS['kpi']}/kpis", timeout=5)
        if response.status_code == 200:
            return response.json(), True
    except:
        pass
    return None, False

# Function to get analysis from analyzer agent
def get_analysis(analysis_type):
    try:
        # First get data from collector
        response = requests.get(f"{AGENT_ENDPOINTS['collector']}/events", timeout=5)
        if response.status_code != 200:
            return None, False
        
        events = response.json()
        
        # Send to analyzer
        response = requests.post(
            f"{AGENT_ENDPOINTS['analyzer']}/analyze", 
            json=events,
            timeout=10
        )
        if response.status_code == 200:
            return response.json(), True
    except:
        pass
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
                    # Get the first store ID from available KPIs
                    store_id = kpis[0].get('store_id', 'Los Angeles')  # Fallback to Los Angeles
                else:
                    return {"status": "error", "message": "No KPI data found. Please run analysis first."}, False
            else:
                return {"status": "error", "message": "Could not connect to KPI agent to find available stores."}, False
        except:
            return {"status": "error", "message": "Error finding available stores. Try specifying a store ID."}, False
    
    try:
        # Call the correct endpoint to get the HTML report
        report_url = f"{AGENT_ENDPOINTS['report']}/report/{store_id}"
        response = requests.get(report_url, timeout=15)
        
        if response.status_code == 200:
            # Return the HTML content wrapped in the expected format
            return {"status": "success", "report_html": response.text, "store_id": store_id}, True
        else:
            # Handle cases where KPI data might not exist yet
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
    
    # 5. Daily Sales Heatmap
    if 'ts' in df.columns and 'amount' in df.columns:
        df['day_of_week'] = df['ts'].dt.day_name()
        df['hour'] = df['ts'].dt.hour
        daily_sales = df[df['event_type'] == 'sale'].groupby(['day_of_week', 'hour'])['amount'].sum().reset_index()
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_sales['day_of_week'] = pd.Categorical(daily_sales['day_of_week'], categories=days_order, ordered=True)
        daily_sales = daily_sales.sort_values('day_of_week')
        
        fig_heatmap = px.density_heatmap(daily_sales, x='hour', y='day_of_week', z='amount',
                                        title="Sales Heatmap by Day and Hour",
                                        labels={'hour': 'Hour of Day', 'day_of_week': 'Day of Week', 'amount': 'Total Sales ($)'})
        charts['sales_heatmap'] = fig_heatmap
    
    # 6. Sales by Weekday
    if 'ts' in df.columns and 'amount' in df.columns:
        df['weekday'] = df['ts'].dt.day_name()
        weekday_sales = df[df['event_type'] == 'sale'].groupby('weekday')['amount'].sum().reset_index()
        
        # Order by day of week
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sales['weekday'] = pd.Categorical(weekday_sales['weekday'], categories=days_order, ordered=True)
        weekday_sales = weekday_sales.sort_values('weekday')
        
        fig_weekday = px.bar(weekday_sales, x='weekday', y='amount', 
                            title="Sales by Day of Week",
                            labels={'amount': 'Total Sales ($)', 'weekday': 'Day of Week'})
        charts['weekday_sales'] = fig_weekday
    
    # 7. Top Selling Stores
    if 'store_id' in df.columns and 'amount' in df.columns:
        store_sales = df[df['event_type'] == 'sale'].groupby('store_id')['amount'].sum().reset_index()
        store_sales = store_sales.sort_values('amount', ascending=False)
        
        fig_stores = px.bar(store_sales, x='store_id', y='amount', 
                           title="Total Sales by Store",
                           labels={'amount': 'Total Sales ($)', 'store_id': 'Store'})
        charts['store_sales'] = fig_stores
    
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
    
    # Create tabs for each agent
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📥 Collector", 
        "🔄 Coordinator", 
        "📊 Visualization", 
        "📈 KPI", 
        "🔍 Analyzer", 
        "📋 Report"
    ])
    
    # Collector Agent Tab
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
    
    # Coordinator Agent Tab
    with tab2:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">🔄 Coordinator Agent <span class="port-label">:8110</span></h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Process Data", key="process_btn", use_container_width=True):
                with st.spinner("Processing data through all agents..."):
                    success, message = trigger_data_processing("full")
                    if success:
                        st.success("Data processed successfully!")
                        with st.expander("View Processing Details"):
                            st.json(message)
                    else:
                        st.error(f"Error: {message}")
            
            st.info("Click the button to process data through all agents and generate insights.")
        
        with col2:
            # Quick stats
            if 'amount' in df.columns:
                total_sales = df[df['event_type'] == 'sale']['amount'].sum()
                avg_sale = df[df['event_type'] == 'sale']['amount'].mean()
                st.metric("Total Sales", f"${total_sales:,.2f}")
                st.metric("Average Sale", f"${avg_sale:,.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization Tab
    with tab3:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">📊 Data Visualization</h2>', unsafe_allow_html=True)
        
        # Create and display charts
        charts = create_visualization_charts(df)
        
        if charts:
            # Sales Over Time
            if 'sales_over_time' in charts:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="viz-header">Sales Over Time by Store</h3>', unsafe_allow_html=True)
                st.plotly_chart(charts['sales_over_time'], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Store Performance Comparison
            if 'store_performance' in charts:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="viz-header">Store Performance Comparison</h3>', unsafe_allow_html=True)
                st.plotly_chart(charts['store_performance'], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Event Distribution
                if 'event_distribution' in charts:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 class="viz-header">Event Type Distribution</h3>', unsafe_allow_html=True)
                    st.plotly_chart(charts['event_distribution'], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Monthly Sales
                if 'monthly_sales' in charts:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 class="viz-header">Monthly Sales Trend</h3>', unsafe_allow_html=True)
                    st.plotly_chart(charts['monthly_sales'], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Sales by Weekday
                if 'weekday_sales' in charts:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 class="viz-header">Sales by Day of Week</h3>', unsafe_allow_html=True)
                    st.plotly_chart(charts['weekday_sales'], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                # Top Selling Stores
                if 'store_sales' in charts:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown('<h3 class="viz-header">Total Sales by Store</h3>', unsafe_allow_html=True)
                    st.plotly_chart(charts['store_sales'], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Sales Heatmap
            if 'sales_heatmap' in charts:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="viz-header">Sales Heatmap by Day and Hour</h3>', unsafe_allow_html=True)
                st.plotly_chart(charts['sales_heatmap'], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No data available for visualization. Please process data first.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # KPI Agent Tab
    with tab4:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">📈 KPI Agent <span class="port-label">:8102</span></h2>', unsafe_allow_html=True)
        
        if st.button("Calculate KPIs", key="kpi_btn"):
            with st.spinner("Calculating KPIs..."):
                kpis, success = get_kpis()
                if success:
                    st.success("KPIs calculated successfully!")
                    
                    if kpis and len(kpis) > 0:
                        # Display KPIs in columns
                        kpi_cols = st.columns(2)
                        
                        for i, kpi in enumerate(kpis):
                            if i < 2:  # Show first 2 stores
                                with kpi_cols[i]:
                                    metrics = kpi.get('metrics', {})
                                    st.subheader(f"Store {kpi.get('store_id')}")
                                    for metric_name, metric_value in metrics.items():
                                        if isinstance(metric_value, (int, float)):
                                            if any(word in metric_name for word in ['amount', 'aov']):
                                                st.metric(metric_name.replace('_', ' ').title(), f"${metric_value:,.2f}")
                                            else:
                                                st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:,.0f}")
                    
                        # Show remaining KPIs in expander
                        if len(kpis) > 2:
                            with st.expander("Show All Store KPIs"):
                                for kpi in kpis[2:]:
                                    if isinstance(kpi, dict):
                                        metrics = kpi.get('metrics', {})
                                    else:
                                        st.error(f"KPI data format error: {kpi}")
                                        metrics = {}
                                    st.subheader(f"Store {kpi.get('store_id')}")
                                    for metric_name, metric_value in metrics.items():
                                        if isinstance(metric_value, (int, float)):
                                            if any(word in metric_name for word in ['amount', 'aov']):
                                                st.metric(metric_name.replace('_', ' ').title(), f"${metric_value:,.2f}")
                                            else:
                                                st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:,.0f}")
                    else:
                        st.info("No KPIs available yet. Process data first.")
                else:
                    st.error("Failed to calculate KPIs (KPI agent on port 8102 unavailable)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyzer Agent Tab
    with tab5:
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
                    st.success("Analysis completed successfully!")
                    
                    if analysis and 'insights_list' in analysis:
                        insights = analysis['insights_list']
                        st.subheader(f"Top Insights ({len(insights)} total)")
                        
                        # Show first 5 insights
                        for i, insight in enumerate(insights[:5]):
                            with st.expander(f"Insight {i+1}: {insight.get('text', 'No text')}"):
                                st.write(f"**Store:** {insight.get('store_id')}")
                                st.write(f"**Explanation:** {insight.get('explanation')}")
                                st.write(f"**Tags:** {', '.join(insight.get('tags', []))}")
                                st.write(f"**Confidence:** {insight.get('confidence')}")
                    
                    # Show analysis metadata
                    with st.expander("Analysis Details"):
                        st.json(analysis)
                else:
                    st.error("Failed to run analysis (Analyzer agent on port 8101 unavailable)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Report Agent Tab
    with tab6:
        st.markdown('<div class="agent-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="agent-header">📋 Report Agent <span class="port-label">:8103</span></h2>', unsafe_allow_html=True)
        
        # Get available stores from data
        available_stores = []
        if 'store_id' in df.columns:
            available_stores = df['store_id'].unique().tolist()
        
        if available_stores:
            selected_store = st.selectbox("Select Store", available_stores, key="report_store")
        else:
            selected_store = "Los Angeles"  # Default fallback
            st.info("No store data available. Using default store.")
        
        if st.button("Generate Report", key="report_btn"):
            with st.spinner("Generating report..."):
                report, success = generate_report(selected_store)
                if success:
                    st.success(f"Report for {selected_store} generated successfully!")
                    
                    # Display the HTML report
                    st.components.v1.html(report["report_html"], height=600, scrolling=True)
                    
                    # Add download button
                    st.download_button(
                        label="📥 Download Report",
                        data=report["report_html"],
                        file_name=f"store_report_{selected_store}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                else:
                    st.error(f"Failed to generate report: {report.get('message', 'Unknown error')}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()