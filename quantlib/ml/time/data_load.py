import pandas as pd
from typing import List
from tqdm import tqdm


def load_datasets(
    datasets: List[str],
    remain_field: List[str],
    start_date: str,
    end_date: str
) -> List[pd.DataFrame]:
    """
    加载多个股票数据集

    参数：
    - datasets: 每个元素是 CSV 文件路径
    - remain_field: 要提取的字段列（如 ['Open', 'Close', ...]）
    - start_date: 数据集的起止日期（格式 'YYYY-MM-DD'）
    - end_date: 数据集的截止日期

    返回：
    - 一个包含每个数据集的pd.DataFrame列表
    """
    all_dfs = []

    for i, dataset in tqdm(enumerate(datasets), desc="Loading datasets"):
        try:
            # 加载原始文件
            df = pd.read_csv(dataset)
            df.columns = [
                "Date", "StockCode", "Open", "Close", "High", "Low",
                "Volume", "Turnover", "Amplitude", "ChangePct", "ChangeAmt", "TurnoverRate"
            ]
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

            # 检查字段合法性
            missing_fields = set(remain_field) - set(df.columns)
            if missing_fields:
                raise ValueError(f"Missing fields {missing_fields} in dataset: {dataset}")
            
            df = df[remain_field].loc[start_date: end_date]

            print(f"[Stock {i}] {dataset} → {df.shape}")

            all_dfs.append(df)

        except Exception as e:
            print(f"Error processing {dataset}: {e}")
    
    return all_dfs
