import json
from pathlib import Path

import numpy as np
import pandas as pd
from suicide_detection.data_processing.load import load_dataset_secure


def test_load_dataset_secure(tmp_path: Path):
    p = tmp_path / "data.csv"
    df = pd.DataFrame({"text": ["hello @user"], "label": [1]})
    df.to_csv(p, index=False)

    loaded = load_dataset_secure(p)
    assert set(["text", "label"]).issubset(loaded.columns)
    assert "<USER>" in loaded.loc[0, "text"]

