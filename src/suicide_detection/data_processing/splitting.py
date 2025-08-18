from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, timestamp_col: str, test_size: float = 0.15, val_size: float = 0.15, ascending: bool = True):
    df_sorted = df.sort_values(timestamp_col, ascending=ascending).reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_df = df_sorted.iloc[-n_test:]
    trainval_df = df_sorted.iloc[: n - n_test]
    val_df = trainval_df.iloc[-n_val:]
    train_df = trainval_df.iloc[: len(trainval_df) - n_val]
    return train_df, val_df, test_df

