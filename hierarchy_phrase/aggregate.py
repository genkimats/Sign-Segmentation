"""
hierarchy_phrase/aggregate.py -- mean +- std over seeds from the JSON files evaluate.py writes.

    python aggregate.py results/sa_s*__main_s*.json            # the method
    python aggregate.py results/sa_s*__main_s*.json --split val
Prints one table per row type (flat_tuned, hier_tuned, ...) with the headline columns; std is over the files given
(use 3 seeds to match the 2023 paper's reporting).
"""
import argparse
import json
import sys

import numpy as np

COLS = ["frame_f1", "frame_f1_B", "mask_iou", "ratio", "start_f1@2", "start_f1@5", "start_f1@10", "seg_f1@0.5", "mF1S(0.1-0.5)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    runs = [json.load(open(f)) for f in a.files]
    print(f"{len(runs)} runs, split = {a.split}\n{'row':<22}" + "".join(f"{c:>20}" for c in COLS))
    for row in runs[0][a.split]:
        line = f"{row:<22}"
        for c in COLS:
            v = np.array([r[a.split][row][c] for r in runs], dtype=float)
            line += f"{v.mean():>12.3f}+-{v.std(ddof=1) if len(v) > 1 else 0:<6.3f}"
        print(line)


if __name__ == "__main__":
    main()