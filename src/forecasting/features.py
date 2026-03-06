import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    return df


def create_lag_features(df: pd.DataFrame, target_col: str = "sales_qty") -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["sku", "date"])

    df["lag_1"] = df.groupby("sku")[target_col].shift(1)
    df["lag_7"] = df.groupby("sku")[target_col].shift(7)
    df["lag_14"] = df.groupby("sku")[target_col].shift(14)

    df["rolling_mean_7"] = (
        df.groupby("sku")[target_col]
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_14"] = (
        df.groupby("sku")[target_col]
        .shift(1)
        .rolling(14)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = create_time_features(df)
    df = create_lag_features(df, target_col="sales_qty")

    # 把 sku 编码成类别编号，方便模型处理
    df["sku_code"] = df["sku"].astype("category").cat.codes

    return df
