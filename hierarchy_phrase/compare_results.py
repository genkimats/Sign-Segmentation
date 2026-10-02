"""
hierarchy_phrase/compare_results.py -- one table comparing CONFIGURATIONS side by side (aggregate.py averages SEEDS of one configuration).

    python compare_results.py results/al_sa_s42__main_s42.json results/al_sa_stgcnlstm_s42__main_s42.json
    python compare_results.py results/al_*__main_s42.json --split val       # choose on VAL, look at test only afterwards

Per file it prints the three rows that answer the main questions, using the tuned thresholds (validation-tuned, so test numbers are fair):
  flat_tuned          the flat frame-level phrase head of the same Stage-A encoder (the E1s-style baseline)
  hier_tuned          sign head -> predicted signs -> Stage C                      (the method)
  hier_oracle_tuned   GOLD signs -> Stage C                                          (ceiling: the gap to hier_tuned is the cascade loss)
"""
import argparse
import json
import os

COLS = ["frame_f1", "ratio", "start_f1@5", "seg_f1@0.5", "mask_iou"]
ROWS = ["flat_tuned", "hier_tuned", "hier_oracle_tuned"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    print(f"split = {a.split}\n{'run':<44}{'row':<20}" + "".join(f"{c:>12}" for c in COLS))
    for f in a.files:
        r = json.load(open(f))
        name = os.path.basename(f).replace(".json", "")
        for i, row in enumerate(ROWS):
            m = r[a.split].get(row)
            cells = "".join(f"{m[c]:>12.3f}" for c in COLS) if m else "".join(f"{'n/a':>12}" for _ in COLS)
            print(f"{name if i == 0 else '':<44}{row:<20}" + cells)
        sg = r[a.split].get("sign_stage_tuned")
        if sg:
            print(f"{'':<44}{'(sign stage, tuned)':<20}" + f"{sg['frame_f1']:>12.3f}{sg['ratio']:>12.3f}{sg['start_f1@5']:>12.3f}{sg['seg_f1@0.5']:>12.3f}{sg['mask_iou']:>12.3f}")
        print()


if __name__ == "__main__":
    main()