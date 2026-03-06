import os
import math
import pandas as pd


def main():
    # =========================
    # 1. Read input files
    # =========================
    forecast_path = "outputs/forecasts/test_forecast.csv"
    inventory_path = "data/raw/inventory.csv"
    leadtime_path = "data/raw/leadtime.csv"

    forecast_df = pd.read_csv(forecast_path)
    inventory_df = pd.read_csv(inventory_path)
    leadtime_df = pd.read_csv(leadtime_path)

    # Convert date columns
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    inventory_df["date"] = pd.to_datetime(inventory_df["date"])

    # =========================
    # 2. Aggregate forecast demand by SKU
    # =========================
    # Sum predicted demand over the forecast file horizon
    demand_summary = (
        forecast_df.groupby("sku", as_index=False)
        .agg(
            predicted_demand=("predicted_sales_qty", "sum"),
            demand_std=("predicted_sales_qty", "std")
        )
    )

    # Fill NaN std for edge cases
    demand_summary["demand_std"] = demand_summary["demand_std"].fillna(0)

    # =========================
    # 3. Get latest inventory by SKU
    # =========================
    latest_inventory = (
        inventory_df.sort_values(["sku", "date"])
        .groupby("sku", as_index=False)
        .tail(1)[["sku", "on_hand"]]
        .rename(columns={"on_hand": "current_inventory"})
    )

    # =========================
    # 4. Merge all inputs
    # =========================
    merged = demand_summary.merge(latest_inventory, on="sku", how="left")
    merged = merged.merge(leadtime_df, on="sku", how="left")

    # Fill missing inventory if any
    merged["current_inventory"] = merged["current_inventory"].fillna(0)
    merged["lead_time_days"] = merged["lead_time_days"].fillna(7)

    # =========================
    # 5. Inventory policy calculation
    # =========================
    # Assume service level = 95%, Z ≈ 1.65
    z_value = 1.65

    # We need average daily demand over forecast horizon
    forecast_horizon_days = forecast_df["date"].nunique()
    if forecast_horizon_days == 0:
        raise ValueError("Forecast file contains no dates.")

    merged["avg_daily_demand"] = merged["predicted_demand"] / forecast_horizon_days

    # Safety Stock = Z * sigma * sqrt(L)
    merged["safety_stock"] = (
        z_value * merged["demand_std"] * merged["lead_time_days"].apply(math.sqrt)
    )

    # Reorder Point = avg daily demand * lead time + safety stock
    merged["reorder_point"] = (
        merged["avg_daily_demand"] * merged["lead_time_days"] + merged["safety_stock"]
    )

    # Target stock level:
    # lead time demand + additional 7 days cover + safety stock
    merged["target_stock"] = (
        merged["avg_daily_demand"] * (merged["lead_time_days"] + 7) + merged["safety_stock"]
    )

    # Recommended order quantity
    merged["recommended_order_qty"] = (
        merged["target_stock"] - merged["current_inventory"]
    ).clip(lower=0)

    # Reorder flag
    merged["reorder_flag"] = merged["current_inventory"] < merged["reorder_point"]

    # Round numeric columns
    numeric_cols = [
        "predicted_demand",
        "demand_std",
        "current_inventory",
        "avg_daily_demand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order_qty",
    ]
    merged[numeric_cols] = merged[numeric_cols].round(2)

    # =========================
    # 6. Save output
    # =========================
    os.makedirs("outputs/replenishment", exist_ok=True)

    output_cols = [
        "sku",
        "predicted_demand",
        "current_inventory",
        "lead_time_days",
        "avg_daily_demand",
        "safety_stock",
        "reorder_point",
        "target_stock",
        "recommended_order_qty",
        "reorder_flag",
    ]

    output_path = "outputs/replenishment/reorder_plan.csv"
    merged[output_cols].to_csv(output_path, index=False)

    print("Inventory policy calculation completed")
    print(f"Replenishment plan saved to {output_path}")


if __name__ == "__main__":
    main()
