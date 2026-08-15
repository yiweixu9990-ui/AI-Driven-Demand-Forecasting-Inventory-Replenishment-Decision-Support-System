"""Tests for deterministic agent services; no API key or model calls required."""

import unittest

from src.agent.services import (
    calculate_replenishment,
    get_demand_forecast,
    get_inventory_status,
    run_replenishment_scenario,
)


class AgentServiceTests(unittest.TestCase):
    def test_forecast_is_explicitly_labeled_backtest(self) -> None:
        result = get_demand_forecast("sku_0001", horizon_days=14)
        self.assertEqual(result["sku"], "SKU_0001")
        self.assertEqual(result["forecast_type"], "historical_backtest")
        self.assertEqual(result["available_days"], 14)
        self.assertGreater(result["predicted_demand"], 0)

    def test_inventory_status_matches_available_policy(self) -> None:
        result = get_inventory_status("SKU_0002")
        self.assertEqual(result["sku"], "SKU_0002")
        self.assertIn("reorder_point", result)
        self.assertIsNone(result["on_order_inventory"])

    def test_replenishment_is_non_negative(self) -> None:
        result = calculate_replenishment("SKU_0011")
        self.assertGreaterEqual(result["recommended_order_qty"], 0)

    def test_higher_demand_does_not_reduce_order(self) -> None:
        baseline = run_replenishment_scenario("SKU_0011", 10, demand_multiplier=1.0)
        increased = run_replenishment_scenario("SKU_0011", 10, demand_multiplier=1.2)
        self.assertGreaterEqual(
            increased["recommended_order_qty"],
            baseline["recommended_order_qty"],
        )

    def test_invalid_sku_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown SKU"):
            get_demand_forecast("SKU_DOES_NOT_EXIST")


if __name__ == "__main__":
    unittest.main()
