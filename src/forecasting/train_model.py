import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import prepare_features


def main():
    # 1. 读取数据
    file_path = "data/raw/sales.csv"
    df = pd.read_csv(file_path)

    # 2. 特征工程
    df = prepare_features(df)

    # 3. 删除因 lag 产生的缺失值
    df = df.dropna().reset_index(drop=True)

    # 4. 选择特征列
    feature_cols = [
        "sku_code",
        "price",
        "promo",
        "day_of_week",
        "month",
        "day_of_month",
        "week_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
    ]

    target_col = "sales_qty"

    # 5. 按时间切分训练/测试集
    df = df.sort_values("date")
    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 6. 训练模型
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 7. 预测
    y_pred = model.predict(X_test)

    # 8. 评估
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print("Model training completed")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # 9. 保存预测结果
    results = test_df[["date", "sku", "sales_qty"]].copy()
    results["predicted_sales_qty"] = y_pred

    os.makedirs("outputs/forecasts", exist_ok=True)
    results.to_csv("outputs/forecasts/test_forecast.csv", index=False)

    print("Forecast results saved to outputs/forecasts/test_forecast.csv")


if __name__ == "__main__":
    main()

