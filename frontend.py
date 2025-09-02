import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import asyncio

# Relative imports from package
from report_agent.clients.collector_client import fetch_metrics
from report_agent.clients.analysis_client import fetch_insights

st.set_page_config(page_title="Store Performance Dashboard", layout="wide")
st.title("📊 Store Performance Dashboard")

# Sidebar input section
with st.sidebar:
    st.header("Report Filters")
    store_id = st.text_input("Enter Store ID", "Los Angeles")
    from_date = st.date_input("From Date")
    to_date = st.date_input("To Date")
    fetch_button = st.button("Fetch Report")

# Fetch and display report
if fetch_button:
    try:
        with st.spinner("Fetching data..."):
            # Call fake async functions
            metrics = asyncio.run(fetch_metrics(
                store_id,
                from_date.strftime("%Y-%m-%d"),
                to_date.strftime("%Y-%m-%d")
            ))

            insights = asyncio.run(fetch_insights(
                store_id,
                from_date.strftime("%Y-%m-%d"),
                to_date.strftime("%Y-%m-%d")
            ))

        # Convert metrics to DataFrame
        df = pd.DataFrame(metrics)

        # Display raw data
        st.subheader("📄 Raw Data")
        st.dataframe(df)

        # Sales Trend chart
        if not df.empty:
            st.subheader("📈 Sales Trend")
            fig, ax = plt.subplots()
            df.plot(x="date", y="sales", ax=ax, marker="o", linestyle='-', color='blue')
            ax.set_ylabel("Sales")
            ax.set_xlabel("Date")
            ax.set_title("Sales Over Time")
            st.pyplot(fig)

        # Visits Trend chart
        if not df.empty:
            st.subheader("👥 Visits Trend")
            fig, ax = plt.subplots()
            df.plot(x="date", y="visits", ax=ax, marker="o", linestyle='-', color='green')
            ax.set_ylabel("Visits")
            ax.set_xlabel("Date")
            ax.set_title("Visits Over Time")
            st.pyplot(fig)

        # Pie chart: Sales vs Visits
        if not df.empty:
            st.subheader("🥧 Sales vs Visits Distribution")
            totals = pd.Series({
                "Total Sales": df["sales"].sum(),
                "Total Visits": df["visits"].sum()
            })
            fig, ax = plt.subplots()
            ax.pie(totals, labels=totals.index, autopct='%1.1f%%', startangle=140, colors=['#4e79a7', '#f28e2b'])
            ax.axis('equal')
            st.pyplot(fig)

        # Insights summary
        st.subheader("💡 Insights Summary")
        st.json(insights.dict())

        # Top products table
        st.subheader("🏆 Top Products Table")
        top_products = pd.DataFrame(insights.top_products)
        st.table(top_products)

        # Top products bar chart
        if not top_products.empty:
            st.subheader("📊 Top Products Sales")
            fig, ax = plt.subplots()
            ax.bar(top_products["name"], top_products["sales"], color="#2ca02c")
            ax.set_ylabel("Sales")
            ax.set_xlabel("Product Name")
            ax.set_title("Top Products by Sales")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Metrics summary
        st.subheader("📊 Metrics Summary")
        st.markdown(f"- **Total Sales:** ${insights.total_sales:,.2f}")
        st.markdown(f"- **Total Orders:** {insights.total_orders}")

    except Exception as e:
        st.error(f"Error fetching report: {e}")
