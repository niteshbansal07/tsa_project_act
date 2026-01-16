"""Generate synthetic monthly loss data with trend + seasonality + noise.

This keeps the project fully shareable (no proprietary datasets) while still
demonstrating time-series reasoning relevant to insurance.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=72)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data/monthly_losses.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t = np.arange(args.months)

    # Smooth upward trend
    trend = 100000 + 1200 * t

    # Seasonality (annual)
    season = 15000 * np.sin(2 * np.pi * t / 12) + 8000 * np.cos(2 * np.pi * t / 12)

    # Noise + occasional spikes (cat-like shocks)
    noise = rng.normal(0, 9000, size=args.months)
    spikes = (rng.random(args.months) < 0.06) * rng.normal(45000, 18000, size=args.months)

    losses = np.maximum(0, trend + season + noise + spikes)

    dates = pd.date_range("2019-01-01", periods=args.months, freq="MS")
    df = pd.DataFrame({"date": dates, "loss": losses})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
