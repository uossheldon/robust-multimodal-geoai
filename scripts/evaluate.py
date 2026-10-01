from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "results" / "tile_manifest.csv")
    summary = (
        manifest.groupby("split")
        .agg(
            tiles=("tile_id", "count"),
            dates=("date", "nunique"),
            valid_pixels=("valid_pixel_count", "sum"),
            class_0=("class_0_count", "sum"),
            class_1=("class_1_count", "sum"),
            class_2=("class_2_count", "sum"),
            class_3=("class_3_count", "sum"),
        )
        .reset_index()
    )
    summary["algae_fraction"] = (summary["class_1"] + summary["class_2"] + summary["class_3"]) / summary["valid_pixels"]
    payload = summary.to_dict(orient="records")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

