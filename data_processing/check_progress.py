import os
import time
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression


def count_files(directory):
    """Count the number of files in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def main():
    # Get the current directory (data_processing)
    current_dir = Path(__file__).parent
    
    # Define paths relative to the current file
    bio_tags_dir = current_dir / "../processed_data/BIO_tags"
    keypoints_dir = current_dir / "../processed_data/keypoints"
    annotations_dir = current_dir / "../raw_data/annotations"
    videos_dir = current_dir / "../raw_data/videos"
    
    # Convert to absolute paths
    bio_tags_dir = bio_tags_dir.resolve()
    keypoints_dir = keypoints_dir.resolve()
    annotations_dir = annotations_dir.resolve()
    videos_dir = videos_dir.resolve()
    
    # Count files
    bio_tags_count = count_files(bio_tags_dir)
    keypoints_count = count_files(keypoints_dir)
    annotations_count = count_files(annotations_dir)
    videos_count = count_files(videos_dir)

    # Prepare logging
    log_file = current_dir / "progress_log.csv"

    # Do NOT write to the CSV log here. `record_progress.py` is responsible for
    # monitoring the `keypoints` directory and appending rows when progress changes.
    # Here we only read existing log data (if present) to train/predict.
    df = None
    if pd is not None and log_file.exists():
        try:
            df = pd.read_csv(log_file).sort_values('timestamp')
        except Exception:
            df = None

    # Function to format seconds human-readably
    def format_seconds(s):
        if s is None:
            return "unknown"
        if s <= 0:
            return "0s"
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        parts = []
        if h:
            parts.append(f"{h}h")
        if m:
            parts.append(f"{m}m")
        parts.append(f"{sec}s")
        return " ".join(parts)

    # Prediction helper using linear regression on timestamp -> count
    def predict_remaining(df, count_col, total_count):
        if total_count <= 0:
            return None, None
        try:
            if pd is None or LinearRegression is None:
                raise RuntimeError("pandas/sklearn missing")

            df2 = df[["timestamp", count_col]].dropna()
            if len(df2) < 2:
                raise RuntimeError("not enough data")

            X = df2[["timestamp"]].values.reshape(-1, 1)
            y = df2[count_col].values
            model = LinearRegression()
            model.fit(X, y)
            coef = float(model.coef_[0])
            intercept = float(model.intercept_)

            if coef <= 0:
                return None, None

            # solve for timestamp when predicted count == total_count
            t_pred = (total_count - intercept) / coef
            remaining_seconds = t_pred - time.time()
            remaining_seconds = max(0.0, remaining_seconds)
            return remaining_seconds, {"coef": coef, "intercept": intercept}
        except Exception:
            return None, None

    # Load log for modeling and predict remaining time using keypoints -> videos
    key_remaining = None

    if df is not None:
        try:
            key_remaining, _ = predict_remaining(df, 'keypoints_count', videos_count)
        except Exception:
            key_remaining = None
    
    # Calculate progress percentages
    bio_tags_progress = (bio_tags_count / annotations_count * 100) if annotations_count > 0 else 0
    keypoints_progress = (keypoints_count / videos_count * 100) if videos_count > 0 else 0
    
    # Display results
    print("=" * 60)
    print("DATA PREPROCESSING PROGRESS CHECK")
    print("=" * 60)
    
    print(f"\n📁 PROCESSED DATA:")
    print(f"  BIO_tags files:     {bio_tags_count}")
    print(f"  Keypoints files:    {keypoints_count}")
    
    print(f"\n📁 RAW DATA:")
    print(f"  Annotations files:  {annotations_count}")
    print(f"  Videos files:       {videos_count}")
    
    print(f"\n📊 PROGRESS:")
    print(f"  BIO_tags / Annotations:  {bio_tags_progress:.2f}%")
    print(f"  Keypoints / Videos:      {keypoints_progress:.2f}%")
    # Prediction output: only report video preprocessing remaining
    try:
        remaining_text = format_seconds(key_remaining)
    except Exception:
        remaining_text = "unknown"
    print(f"\n⏳ Estimated remaining (video preprocessing): {remaining_text}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
