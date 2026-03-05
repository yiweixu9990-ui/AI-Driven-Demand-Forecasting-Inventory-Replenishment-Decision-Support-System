# AI Supply Chain Decision Support System (DSS)
An enterprise-style end-to-end project that forecasts demand and generates inventory replenishment recommendations.

## What this system does
This project mimics an internal supply chain analytics tool:

1. **Demand Forecasting**
   - Forecast daily demand for each SKU for the next N days.
   - Supports seasonality, promotion, and price effects through engineered features (future module).

2. **Inventory Replenishment Decision Support**
   - Computes classic inventory policy components:
     - Safety Stock
     - Reorder Point (ROP)
     - Recommended Reorder Quantity
   - Exports a replenishment plan for business users (future module).

3. **Dashboard (future module)**
   - A lightweight internal dashboard (Streamlit) to visualize demand, forecasts, and replenishment suggestions.

## Repo structure (MVP)
```text
ai-supplychain-dss/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── configs/
│   └── params.yaml
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   └── ...
└── outputs/
