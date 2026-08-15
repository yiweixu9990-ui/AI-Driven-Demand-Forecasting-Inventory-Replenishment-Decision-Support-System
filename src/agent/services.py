"""Deterministic supply-chain services used by the AI agent tools.

The language model never calculates inventory recommendations itself. It calls
these functions and explains their structured results.
"""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORECAST_PATH = PROJECT_ROOT / "outputs" / "forecasts" / "test_forecast.csv"
REORDER_PATH = PROJECT_ROOT / "outputs" / "replenishment" / "reorder_plan.csv"
INVENTORY_PATH = PROJECT_ROOT / "data" / "raw" / "inventory.csv"
LEAD_TIME_PATH = PROJECT_ROOT / "data" / "raw" / "leadtime.csv"
SKU_MASTER_PATH = PROJECT_ROOT / "data" / "raw" / "sku_master.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required data file is missing: {path.relative_to(PROJECT_ROOT)}"
        )
    return pd.read_csv(path)


def _normalize_sku(sku: str) -> str:
    normalized = sku.strip().upper()
    if not normalized:
        raise ValueError("SKU must not be empty.")
    return normalized


def _sku_forecast(sku: str) -> tuple[str, pd.DataFrame]:
    normalized = _normalize_sku(sku)
    forecast = _read_csv(FORECAST_PATH)
    forecast["date"] = pd.to_datetime(forecast["date"])
    rows = forecast.loc[forecast["sku"] == normalized].sort_values("date")
    if rows.empty:
        raise ValueError(f"Unknown SKU: {normalized}")
    return normalized, rows


def _latest_inventory(sku: str) -> float:
    inventory = _read_csv(INVENTORY_PATH)
    inventory["date"] = pd.to_datetime(inventory["date"])
    rows = inventory.loc[inventory["sku"] == sku].sort_values("date")
    if rows.empty:
        raise ValueError(f"No inventory history found for {sku}")
    return float(rows.iloc[-1]["on_hand"])


def _lead_time(sku: str) -> int:
    lead_times = _read_csv(LEAD_TIME_PATH)
    rows = lead_times.loc[lead_times["sku"] == sku]
    if rows.empty:
        raise ValueError(f"No lead time found for {sku}")
    return int(rows.iloc[0]["lead_time_days"])


def _round_numbers(payload: dict[str, Any]) -> dict[str, Any]:
    rounded: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            rounded[key] = round(value, 2)
        else:
            rounded[key] = value
    return rounded


def get_demand_forecast(sku: str, horizon_days: int = 14) -> dict[str, Any]:
    """Return the latest available backtest forecast window for one SKU."""
    if not 1 <= horizon_days <= 90:
        raise ValueError("horizon_days must be between 1 and 90.")

    normalized, rows = _sku_forecast(sku)
    window = rows.tail(horizon_days)
    predicted = window["predicted_sales_qty"]
    actual = window["sales_qty"]

    return _round_numbers(
        {
            "sku": normalized,
            "forecast_type": "historical_backtest",
            "warning": (
                "This repository currently contains test-period backtest predictions, "
                "not a live future forecast."
            ),
            "requested_horizon_days": horizon_days,
            "available_days": int(len(window)),
            "window_start": window["date"].min().date().isoformat(),
            "window_end": window["date"].max().date().isoformat(),
            "predicted_demand": float(predicted.sum()),
            "average_daily_predicted_demand": float(predicted.mean()),
            "peak_daily_predicted_demand": float(predicted.max()),
            "actual_demand": float(actual.sum()),
            "mean_absolute_error_in_window": float((actual - predicted).abs().mean()),
        }
    )

def get_inventory_status(sku: str) -> dict[str, Any]:
    """Return the current inventory-policy snapshot for one SKU."""
    normalized, _ = _sku_forecast(sku)
    reorder = _read_csv(REORDER_PATH)
    rows = reorder.loc[reorder["sku"] == normalized]
    if rows.empty:
        raise ValueError(f"No replenishment policy found for {normalized}")

    row = rows.iloc[0]
    sku_master = _read_csv(SKU_MASTER_PATH)
    master_rows = sku_master.loc[sku_master["sku"] == normalized]
    category = None if master_rows.empty else str(master_rows.iloc[0]["category"])

    return _round_numbers(
        {
            "sku": normalized,
            "category": category,
            "current_inventory": float(row["current_inventory"]),
            "on_order_inventory": None,
            "on_order_note": "The current dataset does not provide an open-order snapshot.",
            "lead_time_days": int(row["lead_time_days"]),
            "reorder_point": float(row["reorder_point"]),
            "target_stock": float(row["target_stock"]),
            "recommended_order_qty": float(row["recommended_order_qty"]),
            "reorder_required": bool(row["reorder_flag"]),
        }
    )


def calculate_replenishment(
    sku: str,
    service_level: float = 0.95,
    review_period_days: int = 7,
) -> dict[str, Any]:
    """Calculate a deterministic replenishment recommendation for one SKU."""
    normalized, rows = _sku_forecast(sku)
    return _calculate_policy(
        normalized,
        rows,
        current_inventory=_latest_inventory(normalized),
        lead_time_days=_lead_time(normalized),
        demand_multiplier=1.0,
        service_level=service_level,
        review_period_days=review_period_days,
        scenario_name="current_policy",
    )


def run_replenishment_scenario(
    sku: str,
    lead_time_days: int,
    demand_multiplier: float = 1.0,
    service_level: float = 0.95,
    review_period_days: int = 7,
) -> dict[str, Any]:
    """Run a what-if replenishment scenario without changing any source data."""
    normalized, rows = _sku_forecast(sku)
    return _calculate_policy(
        normalized,
        rows,
        current_inventory=_latest_inventory(normalized),
        lead_time_days=lead_time_days,
        demand_multiplier=demand_multiplier,
        service_level=service_level,
        review_period_days=review_period_days,
        scenario_name="what_if",
    )


def _calculate_policy(
    sku: str,
    forecast_rows: pd.DataFrame,
    *,
    current_inventory: float,
    lead_time_days: int,
    demand_multiplier: float,
    service_level: float,
    review_period_days: int,
    scenario_name: str,
) -> dict[str, Any]:
    if not 1 <= lead_time_days <= 365:
        raise ValueError("lead_time_days must be between 1 and 365.")
    if not 0.5 < service_level < 0.999:
        raise ValueError("service_level must be between 0.5 and 0.999.")
    if not 1 <= review_period_days <= 90:
        raise ValueError("review_period_days must be between 1 and 90.")
    if not 0.1 <= demand_multiplier <= 5.0:
        raise ValueError("demand_multiplier must be between 0.1 and 5.0.")

    predicted = forecast_rows["predicted_sales_qty"].astype(float)
    average_daily_demand = float(predicted.mean()) * demand_multiplier
    daily_demand_std = float(predicted.std(ddof=1)) * demand_multiplier
    if math.isnan(daily_demand_std):
        daily_demand_std = 0.0

    z_value = NormalDist().inv_cdf(service_level)
    safety_stock = z_value * daily_demand_std * math.sqrt(lead_time_days)
    reorder_point = average_daily_demand * lead_time_days + safety_stock
    target_stock = (
        average_daily_demand * (lead_time_days + review_period_days) + safety_stock
    )
    recommended_order_qty = max(0.0, target_stock - current_inventory)

    return _round_numbers(
        {
            "scenario": scenario_name,
            "sku": sku,
            "service_level": service_level,
            "lead_time_days": lead_time_days,
            "review_period_days": review_period_days,
            "demand_multiplier": demand_multiplier,
            "average_daily_demand": average_daily_demand,
            "daily_demand_std": daily_demand_std,
            "current_inventory": current_inventory,
            "on_order_inventory": None,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "target_stock": target_stock,
            "recommended_order_qty": recommended_order_qty,
            "reorder_required": current_inventory < reorder_point,
            "method_note": (
                "Safety stock uses variability in point predictions because the current "
                "project does not yet store forecast-error uncertainty. Open orders, MOQ, "
                "case-pack and capacity constraints are not included."
            ),
        }
    )
