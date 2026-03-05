#!/usr/bin/env python3
"""
Generate enterprise-like synthetic supply chain data:
- Daily latent demand with seasonality + promo + price elasticity + noise
- Inventory simulation with review-period ordering and SKU-level lead times
- Stockout effect: realized sales capped by on-hand inventory

Usage:
  python src/data/make_dataset.py --config configs/params.yaml --outdir data/raw
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def date_range(start_date: str, num_days: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, periods=num_days, freq="D")


def sample_categories(rng: np.random.Generator, categories: List[str], mix: List[float], n: int) -> List[str]:
    mix = np.array(mix, dtype=float)
    mix = mix / mix.sum()
    return list(rng.choice(categories, size=n, replace=True, p=mix))


# -----------------------------
# Core generation
# -----------------------------
@dataclass
class GenParams:
    seed: int
    start_date: str
    num_days: int
    num_skus: int

    categories: List[str]
    category_mix: List[float]

    weekly_strength: float
    yearly_strength: float
    noise_cv: float

    base_price_range: Tuple[float, float]
    price_volatility: float
    promo_probability: float
    promo_lift_range: Tuple[float, float]
    price_elasticity_range: Tuple[float, float]

    initial_on_hand_range: Tuple[int, int]
    lead_time_days_range: Tuple[int, int]
    review_period_days: int
    target_days_of_cover_range: Tuple[int, int]
    stockout_penalty_enabled: bool


def build_params(cfg: dict) -> GenParams:
    seed = int(cfg["project"]["random_seed"])
    dg = cfg["data_generation"]
    return GenParams(
        seed=seed,
        start_date=str(dg["start_date"]),
        num_days=int(dg["num_days"]),
        num_skus=int(dg["num_skus"]),
        categories=list(dg["categories"]),
        category_mix=list(dg["category_mix"]),
        weekly_strength=float(dg["weekly_seasonality_strength"]),
        yearly_strength=float(dg["yearly_seasonality_strength"]),
        noise_cv=float(dg["noise_cv"]),
        base_price_range=tuple(dg["base_price_range"]),
        price_volatility=float(dg["price_volatility"]),
        promo_probability=float(dg["promo_probability"]),
        promo_lift_range=tuple(dg["promo_lift_range"]),
        price_elasticity_range=tuple(dg["price_elasticity_range"]),
        initial_on_hand_range=tuple(dg["initial_on_hand_range"]),
        lead_time_days_range=tuple(dg["lead_time_days_range"]),
        review_period_days=int(dg["review_period_days"]),
        target_days_of_cover_range=tuple(dg["target_days_of_cover_range"]),
        stockout_penalty_enabled=bool(dg["stockout_penalty_enabled"]),
    )


def generate_sku_master(p: GenParams, rng: np.random.Generator) -> pd.DataFrame:
    skus = [f"SKU_{i:04d}" for i in range(1, p.num_skus + 1)]
    categories = sample_categories(rng, p.categories, p.category_mix, p.num_skus)

    # Base demand: category A higher, C lower (simple but plausible)
    cat_multiplier = {"A": 1.3, "B": 1.0, "C": 0.7}
    base_demand_raw = rng.lognormal(mean=3.0, sigma=0.35, size=p.num_skus)  # around ~20-40
    base_demand = np.array([base_demand_raw[i] * cat_multiplier[categories[i]] for i in range(p.num_skus)])

    price_base = rng.uniform(p.base_price_range[0], p.base_price_range[1], size=p.num_skus)

    df = pd.DataFrame(
        {
            "sku": skus,
            "category": categories,
            "base_demand": np.round(base_demand, 2),
            "price_base": np.round(price_base, 2),
        }
    )
    return df


def generate_lead_times(p: GenParams, sku_master: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    lead_times = rng.integers(p.lead_time_days_range[0], p.lead_time_days_range[1] + 1, size=len(sku_master))
    return pd.DataFrame({"sku": sku_master["sku"].values, "lead_time_days": lead_times})


def simulate_prices_and_promos(
    p: GenParams,
    sku_master: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate daily price and promo per SKU.
    Price: lognormal random walk around base price (mild volatility).
    Promo: Bernoulli per day with sku-specific promo lift.
    """
    records = []
    for _, row in sku_master.iterrows():
        sku = row["sku"]
        base_price = float(row["price_base"])

        # price path (lognormal random walk)
        log_price = np.log(base_price) + np.cumsum(rng.normal(0, p.price_volatility, size=len(dates)))
        price = np.exp(log_price)

        promo = rng.binomial(1, p.promo_probability, size=len(dates))
        promo_lift = rng.uniform(p.promo_lift_range[0], p.promo_lift_range[1])

        for t, d in enumerate(dates):
            records.append(
                {
                    "date": d,
                    "sku": sku,
                    "price": float(price[t]),
                    "promo": int(promo[t]),
                    "promo_lift": float(promo_lift),
                }
            )
    return pd.DataFrame(records)


def generate_true_demand(
    p: GenParams,
    sku_master: pd.DataFrame,
    price_promo: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Latent demand = base_demand * seasonality * (1 + promo*lift) * price_effect * noise
    - weekly seasonality: day-of-week pattern
    - yearly seasonality: sinusoid over day-of-year
    - price effect: (price / price_base) ^ elasticity
    """
    # weekly pattern: e.g., weekends slightly higher
    dow = dates.dayofweek.values  # Mon=0
    weekly = 1.0 + p.weekly_strength * np.sin(2 * np.pi * (dow / 7.0))

    day_of_year = dates.dayofyear.values
    yearly = 1.0 + p.yearly_strength * np.sin(2 * np.pi * (day_of_year / 365.0))

    seasonal = weekly * yearly  # length = num_days

    # Merge base info
    df = price_promo.merge(sku_master[["sku", "base_demand", "price_base"]], on="sku", how="left")

    # Elasticity per SKU (fixed)
    elasticities = {}
    for sku in sku_master["sku"].values:
        elasticities[sku] = rng.uniform(p.price_elasticity_range[0], p.price_elasticity_range[1])

    # Map seasonal factor by date
    seasonal_map = pd.Series(seasonal, index=dates)

    # Compute demand
    true_demand = []
    for _, r in df.iterrows():
        sku = r["sku"]
        base_demand = float(r["base_demand"])
        price_base = float(r["price_base"])
        price = float(r["price"])
        promo = int(r["promo"])
        lift = float(r["promo_lift"])
        e = float(elasticities[sku])

        s = float(seasonal_map.loc[r["date"]])
        promo_mult = 1.0 + promo * lift
        price_mult = (price / price_base) ** e  # elasticity negative -> higher price reduces demand

        mean_demand = base_demand * s * promo_mult * price_mult

        # multiplicative noise with CV
        noise = rng.lognormal(mean=0.0, sigma=p.noise_cv)
        dmd = max(0.0, mean_demand * noise)

        true_demand.append(dmd)

    df_out = df[["date", "sku", "price", "promo"]].copy()
    df_out["true_demand_qty"] = np.round(true_demand, 2)
    return df_out


def simulate_inventory(
    p: GenParams,
    demand_df: pd.DataFrame,
    sku_master: pd.DataFrame,
    lead_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple periodic review inventory simulation:
    - Every review_period_days, place an order to reach target_days_of_cover * avg_recent_demand
    - Orders arrive after lead_time_days
    - Realized sales = min(true_demand, on_hand) if stockout_penalty_enabled else true_demand
    """
    demand_df = demand_df.sort_values(["sku", "date"]).reset_index(drop=True)
    lead_map = dict(zip(lead_df["sku"].values, lead_df["lead_time_days"].values))

    # Initial inventory
    init_on_hand = {}
    for sku in sku_master["sku"].values:
        init_on_hand[sku] = int(rng.integers(p.initial_on_hand_range[0], p.initial_on_hand_range[1] + 1))

    # Target days of cover per SKU (fixed)
    target_cover = {}
    for sku in sku_master["sku"].values:
        target_cover[sku] = int(rng.integers(p.target_days_of_cover_range[0], p.target_days_of_cover_range[1] + 1))

    # Track open orders: (arrival_date -> qty) per SKU
    open_orders: Dict[str, Dict[pd.Timestamp, float]] = {sku: {} for sku in sku_master["sku"].values}

    inv_records = []
    sales_records = []

    # Pre-compute rolling average demand per sku (for ordering heuristic)
    # We'll compute on the fly using last 28 days of true demand
    history_window = 28

    for sku in sku_master["sku"].values:
        on_hand = init_on_hand[sku]
        lt = int(lead_map[sku])
        sku_dem = demand_df[demand_df["sku"] == sku].reset_index(drop=True)

        for t, d in enumerate(dates):
            # Receive shipments arriving today
            received_qty = 0.0
            if d in open_orders[sku]:
                received_qty = float(open_orders[sku].pop(d))
                on_hand += received_qty

            # True demand today
            td = float(sku_dem.loc[t, "true_demand_qty"])

            # Realized sales affected by stockout
            if p.stockout_penalty_enabled:
                sales = min(td, on_hand)
            else:
                sales = td

            on_hand = max(0.0, on_hand - sales)

            # Ordering decision every review_period_days
            ordered_qty = 0.0
            if (t % p.review_period_days) == 0:
                # recent avg demand
                start_idx = max(0, t - history_window)
                recent_avg = float(sku_dem.loc[start_idx:t, "true_demand_qty"].mean()) if t > 0 else float(sku_dem.loc[0:7, "true_demand_qty"].mean())
                target_level = target_cover[sku] * recent_avg

                # order-up-to: target_level - on_hand (ignore pipeline for simplicity, but add a mild correction)
                pipeline = sum(open_orders[sku].values()) if open_orders[sku] else 0.0
                need = max(0.0, target_level - (on_hand + 0.5 * pipeline))
                ordered_qty = float(np.round(need, 2))

                if ordered_qty > 0:
                    arrival = d + pd.Timedelta(days=lt)
                    open_orders[sku][arrival] = open_orders[sku].get(arrival, 0.0) + ordered_qty

            inv_records.append(
                {
                    "date": d,
                    "sku": sku,
                    "on_hand": float(np.round(on_hand, 2)),
                    "received_qty": float(np.round(received_qty, 2)),
                    "ordered_qty": float(np.round(ordered_qty, 2)),
                    "sales_qty": float(np.round(sales, 2)),
                }
            )

            sales_records.append(
                {
                    "date": d,
                    "sku": sku,
                    "sales_qty": float(np.round(sales, 2)),
                }
            )

    inv_df = pd.DataFrame(inv_records)
    sales_df = pd.DataFrame(sales_records)

    # Merge sales into demand_df for final sales table
    sales_table = demand_df.merge(sales_df, on=["date", "sku"], how="left")
    sales_table["sales_qty"] = sales_table["sales_qty"].fillna(0.0)

    return sales_table, inv_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config, e.g., configs/params.yaml")
    ap.add_argument("--outdir", required=True, help="Output directory, e.g., data/raw")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    p = build_params(cfg)
    outdir = args.outdir

    ensure_dir(outdir)

    rng = np.random.default_rng(p.seed)
    dates = date_range(p.start_date, p.num_days)

    sku_master = generate_sku_master(p, rng)
    lead_df = generate_lead_times(p, sku_master, rng)
    price_promo = simulate_prices_and_promos(p, sku_master, dates, rng)
    demand_df = generate_true_demand(p, sku_master, price_promo, dates, rng)

    sales_table, inventory_table = simulate_inventory(p, demand_df, sku_master, lead_df, dates, rng)

    # Output file names
    out_cfg = cfg.get("outputs", {})
    sales_file = out_cfg.get("sales_file", "sales.csv")
    inventory_file = out_cfg.get("inventory_file", "inventory.csv")
    leadtime_file = out_cfg.get("leadtime_file", "leadtime.csv")
    sku_master_file = out_cfg.get("sku_master_file", "sku_master.csv")

    # Save
    sales_path = os.path.join(outdir, sales_file)
    inv_path = os.path.join(outdir, inventory_file)
    lead_path = os.path.join(outdir, leadtime_file)
    sku_path = os.path.join(outdir, sku_master_file)

    # Ensure date formatted
    for df in (sales_table, inventory_table):
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    sales_table.to_csv(sales_path, index=False)
    inventory_table.to_csv(inv_path, index=False)
    lead_df.to_csv(lead_path, index=False)
    sku_master.to_csv(sku_path, index=False)

    print("✅ Data generated successfully")
    print(f" - {sales_path}")
    print(f" - {inv_path}")
    print(f" - {lead_path}")
    print(f" - {sku_path}")


if __name__ == "__main__":
    main()
