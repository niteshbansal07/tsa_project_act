"""Basic time-series analysis for monthly insurance losses.

Focus:
  - visualize trend/seasonality
  - rolling average smoothing
  - simple decomposition via moving-average seasonal index
  - baseline forecast (last smoothed level)

Run:
  python -m src.generate_data
  python -m src.analyze
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def seasonal_index(series: pd.Series, period: int = 12) -> pd.Series:
    # Compute average by month-of-year, normalize to mean 1.0
    df = series.to_frame("y")
    df["month"] = df.index.month
    idx = df.groupby("month")["y"].mean()
    idx = idx / idx.mean()
    return idx

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/monthly_losses.csv")
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--window", type=int, default=6)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, parse_dates=["date"])
    df = df.sort_values("date")
    df["loss"] = df["loss"].astype(float)
    df = df.set_index("date")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Raw series
    plt.figure()
    plt.plot(df.index, df["loss"])
    plt.title("Monthly Losses (Synthetic)")
    plt.xlabel("Date")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(outdir/"monthly_losses.png", dpi=200)
    plt.close()

    # Rolling mean
    df["roll_mean"] = df["loss"].rolling(args.window).mean()
    plt.figure()
    plt.plot(df.index, df["loss"], label="Loss")
    plt.plot(df.index, df["roll_mean"], label=f"{args.window}-month rolling mean")
    plt.title("Smoothing with Rolling Mean")
    plt.xlabel("Date")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"rolling_mean.png", dpi=200)
    plt.close()

    # Seasonal index
    sidx = seasonal_index(df["loss"])
    plt.figure()
    plt.plot(sidx.index, sidx.values)
    plt.title("Seasonal Index by Month (Normalized)")
    plt.xlabel("Month")
    plt.ylabel("Index")
    plt.xticks(range(1,13))
    plt.tight_layout()
    plt.savefig(outdir/"seasonal_index.png", dpi=200)
    plt.close()

    # Baseline forecast: last rolling mean
    last_level = float(df["roll_mean"].dropna().iloc[-1])
    forecast_horizon = 6
    future = pd.date_range(df.index.max() + pd.offsets.MonthBegin(1), periods=forecast_horizon, freq="MS")
    fc = pd.Series(last_level, index=future, name="forecast")

    plt.figure()
    plt.plot(df.index, df["loss"], label="History")
    plt.plot(fc.index, fc.values, label="Baseline forecast")
    plt.title("Baseline Forecast (Level)")
    plt.xlabel("Date")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"forecast.png", dpi=200)
    plt.close()

    # Save a small summary
    summary = {
        "window_months": args.window,
        "last_smoothed_level": last_level,
        "forecast_horizon_months": forecast_horizon,
    }
    (Path("data")/"summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    print("Saved plots to", outdir)

if __name__ == "__main__":
    main()
