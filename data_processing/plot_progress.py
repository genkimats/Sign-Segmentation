#!/usr/bin/env python3
"""
Plot progress_log.csv: scatter of `keypoints_count` vs `timestamp` with
a fitted linear regression line. Sets y-axis max to `videos_total`.

Saves output to `data_processing/progress_plot.png`.
"""
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
except Exception as e:
    print("Missing dependency:", e)
    print("Install with: pip install pandas matplotlib scikit-learn")
    sys.exit(1)


def main():
    current_dir = Path(__file__).parent
    log_file = current_dir / "progress_log.csv"
    out_file = current_dir / "progress_plot.png"

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return

    df = pd.read_csv(log_file)
    if df.empty:
        print("Log file is empty")
        return

    # Ensure numeric and convert timestamps to datetimes
    df = df.dropna(subset=["timestamp", "keypoints_count"]).copy()
    df["timestamp"] = df["timestamp"].astype(float)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["keypoints_count"] = df["keypoints_count"].astype(float)

    if len(df) < 2:
        print("Need at least 2 data points for regression")
        return

    # Target max (videos_total) - prefer last value, fallback to max seen
    if "videos_total" in df.columns:
        try:
            videos_total = int(df["videos_total"].iloc[-1])
        except Exception:
            videos_total = int(df["videos_total"].max())
    else:
        videos_total = int(max(df["keypoints_count"].max(), 1))

    # Fit regression using numeric timestamps
    X = df[["timestamp"]].values.reshape(-1, 1)
    y = df["keypoints_count"].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = float(r2_score(y, y_pred))

    # Regression line over full timestamp range (as datetimes for plotting)
    ts_min, ts_max = float(df["timestamp"].min()), float(df["timestamp"].max())
    ts_line = np.linspace(ts_min, ts_max, 200).reshape(-1, 1)
    y_line = model.predict(ts_line)
    ts_line_dt = pd.to_datetime(ts_line.flatten(), unit="s")

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["datetime"], df["keypoints_count"], color="tab:blue", label="keypoints_count")
    ax.plot(ts_line_dt, y_line, color="tab:orange", label="linear fit")

    ax.set_xlabel("time")
    ax.set_ylabel("keypoints_count")
    # Set y-axis min to 400 as requested and top to videos_total
    ax.set_ylim(bottom=400, top=videos_total)
    ax.set_title(f"Keypoints progress (max={videos_total}) — R²={r2:.3f}")
    ax.legend()

    # Format x-axis for readable dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()

    plt.tight_layout()
    # Show plot in an interactive window instead of saving
    plt.show()


if __name__ == "__main__":
    main()
