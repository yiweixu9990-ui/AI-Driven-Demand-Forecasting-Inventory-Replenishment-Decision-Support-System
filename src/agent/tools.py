"""OpenAI Agents SDK function-tool wrappers."""

from agents.decorators import tool

from .services import (
    calculate_replenishment,
    get_demand_forecast,
    get_inventory_status,
    run_replenishment_scenario,
)


@tool
def get_demand_forecast_tool(sku: str, horizon_days: int = 14) -> dict:
    """Read the latest available demand backtest for a SKU.

    Args:
        sku: SKU identifier such as SKU_0011.
        horizon_days: Number of latest daily rows to summarize, from 1 to 90.
    """
    return get_demand_forecast(sku, horizon_days)


@tool
def get_inventory_status_tool(sku: str) -> dict:
    """Read inventory, lead time, reorder point and current recommendation for a SKU.

    Args:
        sku: SKU identifier such as SKU_0011.
    """
    return get_inventory_status(sku)


@tool
def calculate_replenishment_tool(
    sku: str,
    service_level: float = 0.95,
    review_period_days: int = 7,
) -> dict:
    """Calculate a replenishment recommendation without changing any data.

    Args:
        sku: SKU identifier such as SKU_0011.
        service_level: Target service probability, greater than 0.5 and below 0.999.
        review_period_days: Days between inventory reviews, from 1 to 90.
    """
    return calculate_replenishment(sku, service_level, review_period_days)


@tool
def run_replenishment_scenario_tool(
    sku: str,
    lead_time_days: int,
    demand_multiplier: float = 1.0,
    service_level: float = 0.95,
    review_period_days: int = 7,
) -> dict:
    """Run a read-only what-if inventory scenario for a SKU.

    Args:
        sku: SKU identifier such as SKU_0011.
        lead_time_days: Scenario supplier lead time, from 1 to 365 days.
        demand_multiplier: Demand adjustment where 1.2 means 20 percent higher demand.
        service_level: Target service probability, greater than 0.5 and below 0.999.
        review_period_days: Days between inventory reviews, from 1 to 90.
    """
    return run_replenishment_scenario(
        sku,
        lead_time_days,
        demand_multiplier,
        service_level,
        review_period_days,
    )


SUPPLY_CHAIN_TOOLS = [
    get_demand_forecast_tool,
    get_inventory_status_tool,
    calculate_replenishment_tool,
    run_replenishment_scenario_tool,
]
