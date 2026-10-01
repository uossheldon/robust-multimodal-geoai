from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DATES = {
    "train": {
        "2025-01-31",
        "2025-03-12",
        "2025-04-08",
        "2025-04-09",
        "2025-05-16",
        "2025-05-18",
        "2025-05-21",
        "2025-08-12",
    },
    "validation": {"2025-01-01", "2025-06-20"},
    "test": {"2025-09-08", "2025-09-21"},
}


def split_for_date(date: str) -> str:
    matches = [split for split, dates in SPLIT_DATES.items() if date in dates]
    if len(matches) != 1:
        raise ValueError(f"Date {date} matched {matches}")
    return matches[0]


def main() -> None:
    manifest_path = PROJECT_ROOT / "results" / "tile_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest["split"] = manifest["date"].map(split_for_date)
    if manifest["split"].isna().any():
        raise ValueError("Some manifest rows did not receive a split.")
    overlap = set(SPLIT_DATES["train"]) & set(SPLIT_DATES["validation"]) | set(SPLIT_DATES["train"]) & set(SPLIT_DATES["test"]) | set(SPLIT_DATES["validation"]) & set(SPLIT_DATES["test"])
    if overlap:
        raise ValueError(f"Date leakage across splits: {overlap}")
    manifest.to_csv(manifest_path, index=False)
    counts = manifest.groupby("split").agg(tiles=("tile_id", "count"), dates=("date", "nunique"))
    print(counts.to_string())


if __name__ == "__main__":
    main()

