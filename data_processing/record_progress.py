#!/usr/bin/env python3
"""
Monitor `processed_data/keypoints` and append a row to `data_processing/progress_log.csv`
whenever the `keypoints_count` changes. Intended to run in a tmux session.

Usage:
    python data_processing/record_progress.py --interval 60

Default interval is 60 seconds. Ctrl-C stops the monitor.
"""
import os
import time
import argparse
import csv
from pathlib import Path


def count_files(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def read_last_keypoints(log_file):
    if not log_file.exists():
        return None
    try:
        with open(log_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) <= 1:
            return None
        hdr = lines[0].split(',')
        vals = lines[-1].split(',')
        if 'keypoints_count' in hdr:
            idx = hdr.index('keypoints_count')
            try:
                return int(vals[idx])
            except Exception:
                return None
        return None
    except Exception:
        return None


def append_row(log_file, row):
    header = not log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(['timestamp', 'annotations_total', 'videos_total', 'bio_tags_count', 'keypoints_count'])
        writer.writerow([row['timestamp'], row['annotations_total'], row['videos_total'], row['bio_tags_count'], row['keypoints_count']])
        f.flush()
        os.fsync(f.fileno())


def main(interval):
    current_dir = Path(__file__).parent
    keypoints_dir = (current_dir / "../processed_data/keypoints").resolve()
    bio_tags_dir = (current_dir / "../processed_data/BIO_tags").resolve()
    annotations_dir = (current_dir / "../raw_data/annotations").resolve()
    videos_dir = (current_dir / "../raw_data/videos").resolve()
    log_file = current_dir / 'progress_log.csv'

    print(f"Monitoring {keypoints_dir} every {interval}s. Logging to {log_file}.")

    last_k = read_last_keypoints(log_file)
    try:
        while True:
            bio_tags_count = count_files(bio_tags_dir)
            keypoints_count = count_files(keypoints_dir)
            annotations_count = count_files(annotations_dir)
            videos_count = count_files(videos_dir)

            if last_k is None or keypoints_count != last_k:
                ts = time.time()
                row = {
                    'timestamp': ts,
                    'annotations_total': int(annotations_count),
                    'videos_total': int(videos_count),
                    'bio_tags_count': int(bio_tags_count),
                    'keypoints_count': int(keypoints_count),
                }
                append_row(log_file, row)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Progress changed: keypoints {last_k} -> {keypoints_count}; appended to log.")
                last_k = keypoints_count
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\nStopped monitoring.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', '-i', type=int, default=60, help='Polling interval in seconds')
    args = parser.parse_args()
    main(args.interval)
