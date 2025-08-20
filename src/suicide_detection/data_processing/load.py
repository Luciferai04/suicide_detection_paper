from pathlib import Path

import pandas as pd

from .anonymize import Anonymizer

REQUIRED_COLUMNS = {"text", "label"}


def load_dataset_secure(path: Path, anonymize: bool = True) -> pd.DataFrame:
    """Load CSV/TSV securely with optional anonymization.

    The loader enforces presence of 'text' and 'label' columns and does not
    perform any network calls. It does not attempt to infer demographics.

    Args:
        path: Local path to CSV or TSV.
        anonymize: If True, PII placeholders will be applied to 'text'.

    Returns:
        DataFrame with at least columns: text, label
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError("Unsupported file type. Use .csv or .tsv")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if anonymize:
        anon = Anonymizer()
        df["text"] = df["text"].astype(str).map(anon.transform)

    return df
