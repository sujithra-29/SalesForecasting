
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from xgboost import XGBRegressor

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── LOAD DATA ────────────────────────────────
@st.cache_data
def load_data():
    monthly = pd.read_csv("monthly_sales.csv", parse_dates=["Month"])
    weekly = pd.read_csv("weekly_sales.csv", parse_dates=["Week"])
    clusters = pd.read_csv("cluster_features.csv")
    with open("forecast_results.json") as f:
        forecasts = json.load(f)
    df = pd.read_csv("train.csv", encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month_num"] = df["Order Date"].dt.month
    return monthly, weekly, clusters, forecasts, df

monthly, weekly, clusters, forecasts, df = load_data()

# ── SIDEBAR ──────────────────────────────────
st.sidebar.title("📊 Sales Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to:", [
    "📊 Sales Overview",
    "🔮 Forecast Explorer",
    "⚠️ Anomaly Report",
    "🎯 Product Segments"
])

# ════════════════════════════════════════════
# PAGE 1 — SALES OVERVIEW
# ════════════════════════════════════════════
if page == "📊 Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    st.markdown("---")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_sales = df["Sales"].sum()
    avg_monthly = monthly["Sales"].mean()
    best_month = monthly.loc[monthly["Sales"].idxmax(), "Month"]
    top_category = df.groupby("Category")["Sales"].sum().idxmax()

    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Avg Monthly Sales", f"${avg_monthly:,.0f}")
    col3.metric("Best Month", best_month.strftime("%b %Y"))
    col4.metric("Top Category", top_category)

    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    selected_region = col1.multiselect(
        "Filter by Region:",
        options=df["Region"].unique().tolist(),
        default=df["Region"].unique().tolist()
    )
    selected_category = col2.multiselect(
        "Filter by Category:",
        options=df["Category"].unique().tolist(),
        default=df["Category"].unique().tolist()
    )

    filtered = df[
        (df["Region"].isin(selected_region)) &
        (df["Category"].isin(selected_category))
    ]

    st.markdown("---")

    # Chart 1: Sales by Year
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total Sales by Year")
        yearly = filtered.groupby("Year")["Sales"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(yearly["Year"], yearly["Sales"],
                     color="steelblue", edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 1000,
                   f"${bar.get_height():,.0f}",
                   ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Year")
        ax.set_ylabel("Sales ($)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Sales by Category")
        cat_sales = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["steelblue", "coral", "green"]
        ax.bar(cat_sales["Category"], cat_sales["Sales"],
               color=colors, edgecolor="white")
        ax.set_xlabel("Category")
        ax.set_ylabel("Sales ($)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    # Chart 2: Monthly Trend
    st.subheader("Monthly Sales Trend")
    monthly_filtered = filtered.groupby(
        filtered["Order Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly_filtered["Order Date"] = monthly_filtered["Order Date"].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly_filtered["Order Date"], monthly_filtered["Sales"],
            color="steelblue", linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales ($)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Sales by Region
    st.subheader("Sales by Region")
    region_sales = filtered.groupby("Region")["Sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(region_sales["Region"], region_sales["Sales"],
            color="steelblue")
    ax.set_xlabel("Sales ($)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ════════════════════════════════════════════
# PAGE 2 — FORECAST EXPLORER
# ════════════════════════════════════════════
elif page == "🔮 Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.markdown("---")

    segment = st.selectbox(
        "Select Segment to Forecast:",
        options=list(forecasts.keys())
    )

    months_ahead = st.slider(
        "Forecast Horizon (months):",
        min_value=1, max_value=3, value=3
    )

    st.markdown("---")

    forecast_values = forecasts[segment]["forecast"][:months_ahead]
    months = [f"Month {i+1}" for i in range(months_ahead)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Forecast — {segment}")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(months, forecast_values,
               color="steelblue", edgecolor="white")
        for i, val in enumerate(forecast_values):
            ax.text(i, val + 100, f"${val:,.0f}",
                   ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Forecasted Sales ($)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Forecast Values")
        forecast_df = pd.DataFrame({
            "Period": months,
            "Forecasted Sales ($)": [f"${v:,.2f}" for v in forecast_values]
        })
        st.dataframe(forecast_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Performance")
        st.metric("Model Used", "XGBoost")
        st.metric("MAE", "$11,551")
        st.metric("RMSE", "$14,494")
        st.metric("MAPE", "14.18%")

# ════════════════════════════════════════════
# PAGE 3 — ANOMALY REPORT
# ════════════════════════════════════════════
elif page == "⚠️ Anomaly Report":
    st.title("⚠️ Anomaly Report")
    st.markdown("---")

    method = st.radio(
        "Select Detection Method:",
        ["Isolation Forest", "Z-Score", "Both"]
    )

    st.subheader("Anomaly Detection Chart")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(weekly["Week"], weekly["Sales"],
            color="steelblue", linewidth=1.5, label="Weekly Sales")

    if method in ["Isolation Forest", "Both"]:
        anomalies_if = weekly[weekly["Anomaly_IF"] == 1]
        ax.scatter(anomalies_if["Week"], anomalies_if["Sales"],
                  color="red", s=100, zorder=5,
                  label=f"Isolation Forest ({len(anomalies_if)} anomalies)")

    if method in ["Z-Score", "Both"]:
        anomalies_zs = weekly[weekly["Anomaly_ZS"] == 1]
        ax.scatter(anomalies_zs["Week"], anomalies_zs["Sales"],
                  color="orange", s=100, zorder=5,
                  label=f"Z-Score ({len(anomalies_zs)} anomalies)")

    ax.set_xlabel("Week")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Detected Anomaly Dates")

    if method == "Isolation Forest":
        display_anomalies = weekly[weekly["Anomaly_IF"] == 1][
            ["Week", "Sales"]].reset_index(drop=True)
    elif method == "Z-Score":
        display_anomalies = weekly[weekly["Anomaly_ZS"] == 1][
            ["Week", "Sales"]].reset_index(drop=True)
    else:
        display_anomalies = weekly[
            (weekly["Anomaly_IF"] == 1) |
            (weekly["Anomaly_ZS"] == 1)][
            ["Week", "Sales"]].reset_index(drop=True)

    display_anomalies["Sales"] = display_anomalies["Sales"].apply(
        lambda x: f"${x:,.2f}")
    display_anomalies["Possible Cause"] = display_anomalies["Week"].apply(
        lambda x: "Black Friday/Christmas Season" 
        if pd.to_datetime(x).month in [11, 12]
        else "Flash Sale or Promotion" 
        if pd.to_datetime(x).month in [6, 7]
        else "Post-Holiday Slowdown"
        if pd.to_datetime(x).month in [1, 2]
        else "Unusual Sales Event"
    )
    st.dataframe(display_anomalies, use_container_width=True)

# ════════════════════════════════════════════
# PAGE 4 — PRODUCT SEGMENTS
# ════════════════════════════════════════════
elif page == "🎯 Product Segments":
    st.title("🎯 Product Demand Segments")
    st.markdown("---")

    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["steelblue", "coral", "green"]
    for i, label in enumerate(clusters["Cluster_Label"].unique()):
        mask = clusters["Cluster_Label"] == label
        ax.scatter(
            clusters[mask]["PCA1"],
            clusters[mask]["PCA2"],
            c=colors[i], s=150, label=label,
            zorder=5, edgecolors="black", linewidth=0.5
        )
        for _, row in clusters[mask].iterrows():
            ax.annotate(row["Sub-Category"],
                       (row["PCA1"], row["PCA2"]),
                       textcoords="offset points",
                       xytext=(5, 5), fontsize=8)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Products by Demand Cluster")

    for label in clusters["Cluster_Label"].unique():
        products = clusters[clusters["Cluster_Label"] == label][
            "Sub-Category"].tolist()
        with st.expander(f"{label} ({len(products)} products)"):
            for p in products:
                st.write(f"→ {p}")

    st.markdown("---")
    st.subheader("Recommended Stocking Strategy")
    strategies = {
        "High Volume, Stable Demand": 
            "Maintain high safety stock. Negotiate bulk contracts. Never stockout!",
        "Low Volume, High Growth Demand": 
            "Gradually increase stock. Monitor monthly. Invest in marketing.",
        "High Volatility, Explosive Growth": 
            "Just-in-time ordering. Flexible supplier agreements. Never overstock!"
    }
    for label, strategy in strategies.items():
        st.info(f"**{label}:** {strategy}")
