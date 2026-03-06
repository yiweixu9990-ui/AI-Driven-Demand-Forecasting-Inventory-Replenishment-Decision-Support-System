import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional style for cleaner charts
plt.style.use("seaborn-v0_8")

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="AI Supply Chain DSS",
    page_icon="📦",
    layout="wide"
)

# =========================
# Title and intro
# =========================
st.title("📦 AI Supply Chain Decision Support System")

st.markdown(
    """
This dashboard demonstrates a simple **AI-driven supply chain workflow**:

1. Generate synthetic supply chain data  
2. Train a demand forecasting model  
3. Predict future demand  
4. Calculate inventory policies  
5. Generate replenishment recommendations  
"""
)

# =========================
# File paths
# =========================
forecast_path = "outputs/forecasts/test_forecast.csv"
reorder_path = "outputs/replenishment/reorder_plan.csv"

# =========================
# File existence check
# =========================
if not os.path.exists(forecast_path):
    st.error(f"Forecast file not found: {forecast_path}")
    st.info("Please run the forecasting script first.")
    st.stop()

if not os.path.exists(reorder_path):
    st.error(f"Replenishment file not found: {reorder_path}")
    st.info("Please run the inventory policy script first.")
    st.stop()

# =========================
# Load data
# =========================
forecast_df = pd.read_csv(forecast_path)
reorder_df = pd.read_csv(reorder_path)

forecast_df["date"] = pd.to_datetime(forecast_df["date"])

# =========================
# Sidebar
# =========================
st.sidebar.header("Filters")

sku_list = sorted(forecast_df["sku"].unique().tolist())
selected_sku = st.sidebar.selectbox("Select SKU", sku_list)

top_n = st.sidebar.slider(
    "Top N SKUs for reorder chart",
    min_value=5,
    max_value=20,
    value=10
)

show_only_reorder = st.sidebar.checkbox("Show only SKUs requiring reorder", value=False)

# =========================
# KPI metrics
# =========================
st.subheader("Key Metrics")

total_skus = reorder_df["sku"].nunique()
skus_to_reorder = int(reorder_df["reorder_flag"].sum())
total_recommended_qty = reorder_df["recommended_order_qty"].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs", f"{total_skus}")
col2.metric("SKUs Needing Reorder", f"{skus_to_reorder}")
col3.metric("Total Recommended Order Qty", f"{total_recommended_qty:.2f}")

# =========================
# Selected SKU forecast chart
# =========================
st.subheader(f"Demand Forecast for {selected_sku}")

sku_forecast = forecast_df[forecast_df["sku"] == selected_sku].sort_values("date")

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(sku_forecast["date"], sku_forecast["sales_qty"], label="Actual Sales", linewidth=2)
ax1.plot(sku_forecast["date"], sku_forecast["predicted_sales_qty"], label="Predicted Sales", linewidth=2)
ax1.set_title(f"Actual vs Predicted Sales - {selected_sku}")
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales Quantity")
ax1.legend()
plt.xticks(rotation=30)
plt.tight_layout()
st.pyplot(fig1)

# =========================
# Selected SKU inventory policy chart
# =========================
st.subheader(f"Inventory Policy Overview for {selected_sku}")

sku_reorder = reorder_df[reorder_df["sku"] == selected_sku]

if not sku_reorder.empty:
    row = sku_reorder.iloc[0]

    labels = [
        "Current Inventory",
        "Reorder Point",
        "Target Stock",
        "Recommended Order Qty"
    ]
    values = [
        row["current_inventory"],
        row["reorder_point"],
        row["target_stock"],
        row["recommended_order_qty"]
    ]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(labels, values)
    ax2.set_title(f"Inventory Policy - {selected_sku}")
    ax2.set_ylabel("Quantity")
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig2)

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Current Inventory", f"{row['current_inventory']:.2f}")
    info_col2.metric("Reorder Point", f"{row['reorder_point']:.2f}")
    info_col3.metric("Recommended Order Qty", f"{row['recommended_order_qty']:.2f}")

    if row["reorder_flag"]:
        st.warning(f"{selected_sku} requires replenishment.")
    else:
        st.success(f"{selected_sku} does not currently require replenishment.")
else:
    st.warning("No inventory policy data found for the selected SKU.")

# =========================
# Top reorder chart
# =========================
st.subheader(f"Top {top_n} SKUs by Recommended Order Quantity")

top_reorder_df = reorder_df.sort_values("recommended_order_qty", ascending=False).head(top_n)

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.bar(top_reorder_df["sku"], top_reorder_df["recommended_order_qty"])
ax3.set_title(f"Top {top_n} Recommended Order Quantities")
ax3.set_xlabel("SKU")
ax3.set_ylabel("Recommended Order Quantity")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig3)

# =========================
# Reorder alert table
# =========================
st.subheader("⚠ SKUs Requiring Replenishment")

reorder_alert_df = reorder_df[reorder_df["reorder_flag"] == True].sort_values(
    "recommended_order_qty", ascending=False
)

if reorder_alert_df.empty:
    st.info("No SKU currently requires replenishment.")
else:
    st.dataframe(reorder_alert_df, use_container_width=True)

# =========================
# Forecast results table
# =========================
st.subheader("Forecast Results Table")
st.dataframe(sku_forecast, use_container_width=True)

# =========================
# Replenishment plan table
# =========================
st.subheader("Replenishment Plan Table")

display_reorder_df = reorder_df.copy()
if show_only_reorder:
    display_reorder_df = display_reorder_df[display_reorder_df["reorder_flag"] == True]

display_reorder_df = display_reorder_df.sort_values("recommended_order_qty", ascending=False)

st.dataframe(display_reorder_df, use_container_width=True)

# =========================
# Download section
# =========================
st.subheader("Download Results")

col4, col5 = st.columns(2)

with col4:
    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="test_forecast.csv",
        mime="text/csv"
    )

with col5:
    st.download_button(
        label="Download Replenishment Plan CSV",
        data=reorder_df.to_csv(index=False).encode("utf-8"),
        file_name="reorder_plan.csv",
        mime="text/csv"
    )
