from pathlib import Path

import csv
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

    # Robust CSV/TSV loading with fallbacks
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception:
            try:
                df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
            except Exception:
                df = pd.read_csv(
                    path,
                    encoding="latin-1",
                    engine="python",
                    on_bad_lines="skip",
                    quoting=csv.QUOTE_MINIMAL,
                )
    elif path.suffix.lower() in {".tsv", ".txt"}:
        try:
            df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path, sep="\t", encoding="utf-8", engine="python", on_bad_lines="skip")
    else:
        raise ValueError("Unsupported file type. Use .csv or .tsv")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if anonymize:
        anon = Anonymizer()
        df["text"] = df["text"].astype(str).map(anon.transform)

    return df
