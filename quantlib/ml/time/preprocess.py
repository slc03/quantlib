import numpy as np
import pandas as pd
from typing import List, Literal, Union, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def apply_log_transform(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    对指定列进行 log1p 变换，避免对 0 或负数取对数的问题
    """
    if isinstance(columns, str):
        columns = [columns]
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def scale_dataframe(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: Literal["minmax", "zscore"] = "minmax",
    train_end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    使用 sklearn 对指定列进行归一化或标准化。
    若未指定 columns，则默认处理所有数值型列。
    
    参数：
    - df: 输入 DataFrame，index 为 datetime 类型
    - columns: 要处理的列名列表，若为 None，则处理所有数值列
    - method: "minmax" 或 "zscore"
    - train_end_date: 若指定，则仅在此日期（含）之前的数据上 fit
    """
    df = df.copy()

    # 自动选择所有数值型列
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()

    # 选择 scaler
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "zscore":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported method: {method}")

    # 拟合训练数据（按时间过滤）
    if train_end_date is not None:
        train_df = df.loc[:train_end_date]
    else:
        train_df = df

    fit_data = train_df[columns].dropna()
    scaler.fit(fit_data)

    # transform 整个数据（忽略 NaN，sklearn 支持）
    df[columns] = scaler.transform(df[columns])

    return df
