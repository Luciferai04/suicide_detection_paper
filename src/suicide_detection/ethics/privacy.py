from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class KAnonReport:
    k: int
    quasi_identifiers: List[str]
    risky_groups: List[Dict[str, object]]  # each has {"values": tuple, "count": int}

    def to_dict(self) -> Dict[str, object]:
        return {
            "k": self.k,
            "quasi_identifiers": self.quasi_identifiers,
            "risky_groups": self.risky_groups,
        }


def k_anonymity_report(df: pd.DataFrame, quasi_identifiers: List[str], k: int = 5) -> KAnonReport:
    cols = [c for c in quasi_identifiers if c in df.columns]
    if not cols:
        return KAnonReport(k=k, quasi_identifiers=[], risky_groups=[])
    # Normalize values to strings to avoid nan issues
    key = df[cols].astype(str)
    key["__count"] = 1
    grp = key.groupby(cols)["__count"].sum().reset_index()
    risky = grp[grp["__count"] < k]
    risky_groups = [
        {"values": tuple(row[c] for c in cols), "count": int(row["__count"])}
        for _, row in risky.iterrows()
    ]
    return KAnonReport(k=k, quasi_identifiers=cols, risky_groups=risky_groups)


def write_privacy_report(report: KAnonReport, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
