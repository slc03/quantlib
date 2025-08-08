import numpy as np
import pandas as pd
from typing import List


def split_stock_data(all_dfs: List[pd.DataFrame],
                     train_start_date: str, train_end_date: str,
                     valid_start_date: str, valid_end_date: str,
                     test_start_date: str, test_end_date: str,
                     target_field: str, 
                     look_back_days: int=1):
    X_train, X_valid, X_test = [], [], []
    y_train, y_valid, y_test = [], [], []

    for i, df in enumerate(all_dfs):
        df = df.copy()
        
        # 构造滞后特征列
        feature_cols = df.columns
        for lag in range(1, look_back_days):
            for col in feature_cols:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # 构造 y
        df["target_next_day"] = df[target_field].shift(-1)

        # 删除缺失值（因为 shift 会造成前几行/最后一行为空）
        df.dropna(inplace=True)

        # 构造 X 和 y
        X = df.drop(columns=["target_next_day"])
        y = df["target_next_day"]

        # 分段
        X_train_part = X.loc[train_start_date:train_end_date].values
        X_valid_part = X.loc[valid_start_date:valid_end_date].values
        X_test_part = X.loc[test_start_date:test_end_date].values

        y_train_part = y.loc[train_start_date:train_end_date].values
        y_valid_part = y.loc[valid_start_date:valid_end_date].values
        y_test_part = y.loc[test_start_date:test_end_date].values

        # 收集
        X_train.append(X_train_part)
        X_valid.append(X_valid_part)
        X_test.append(X_test_part)

        y_train.append(y_train_part)
        y_valid.append(y_valid_part)
        y_test.append(y_test_part)

        # 打印维度
        print(f"[Stock {i}]")
        print(f"  X_train: {X_train_part.shape}, y_train: {y_train_part.shape}")
        print(f"  X_valid: {X_valid_part.shape}, y_valid: {y_valid_part.shape}")
        print(f"  X_test:  {X_test_part.shape}, y_test:  {y_test_part.shape}")

    return X_train, X_valid, X_test, y_train, y_valid, y_test
