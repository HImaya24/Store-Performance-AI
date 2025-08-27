import streamlit as st
import pandas as pd
import plotly.express as px
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
        'store_id': np.random.choice(['S1', 'S2', 'S3'], len(dates)),
        'ts': dates,
        'event_type': np.random.choice(['sale', 'inventory', 'visit'], len(dates)),
        'payload': [{'amount': np.random.uniform(10, 500)} for _ in range(len(dates))]
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

# DEBUG: Function to check what stores have KPI data
def debug_check_kpi_data():
    try:
        response = requests.get(f"{AGENT_ENDPOINTS['kpi']}/kpis", timeout=5)
        if response.status_code == 200:
            kpis = response.json()
            store_ids = [kpi.get('store_id') for kpi in kpis if isinstance(kpi, dict)]
            return store_ids, True
        return [], False
    except:
        return [], False

# FIXED: Function to generate report from report agent
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
                report, success = generate_report("Los Angeles")  # FIXED: Use actual store name
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
    
    # Display data source info
    if from_collector:
        st.success("✅ Data loaded successfully from Collector agent (port 8100)")
    else:
        st.warning("⚠️ Using sample data (Collector agent on port 8100 unavailable)")
    
    # Collector Agent Section
    st.markdown('<div class="agent-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="agent-header">📥 Collector Agent <span class="port-label">:8100</span></h2>', unsafe_allow_html=True)
    
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
    
    # Coordinator Agent Section
    st.markdown('<div class="agent-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="agent-header">🔄 Coordinator Agent <span class="port-label">:8110</span></h2>', unsafe_allow_html=True)
    
    if st.button("Process Data", key="process_btn"):
        with st.spinner("Processing data through all agents..."):
            success, message = trigger_data_processing("full")
            if success:
                st.success("Data processed successfully!")
                st.json(message)
            else:
                st.error(f"Error: {message}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # KPI Agent Section
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
    
    # Analyzer Agent Section
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
    
    # Report Agent Section
    st.markdown('<div class="agent-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="agent-header">📋 Report Agent <span class="port-label">:8103</span></h2>', unsafe_allow_html=True)
    
    if st.button("Generate Sample Report", key="report_btn"):
        with st.spinner("Generating report..."):
            report, success = generate_report("Los Angeles")  # FIXED: Use actual store name
            if success:
                st.success("Report generated successfully!")
                with st.expander("View Report"):
                    st.components.v1.html(report["report_html"], height=400, scrolling=True)
            else:
                st.error(f"Failed to generate report: {report.get('message', 'Unknown error')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()