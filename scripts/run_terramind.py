from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_terramind import train_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen TerraMind RGB+S1RTC benchmark.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weights", action="store_true")
    args = parser.parse_args()
    train_full(batch_size=args.batch_size, epochs=args.epochs, seed=args.seed, use_class_weights=not args.no_class_weights)


if __name__ == "__main__":
    main()
