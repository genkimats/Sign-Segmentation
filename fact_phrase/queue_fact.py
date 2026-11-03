"""
queue_fact.py -- writes fact_phrase/train_queue_fact.json, the job list train_fact.py works through.

    python fact_phrase/queue_fact.py               # write the queue (refuses to overwrite a non-empty one)
    python fact_phrase/queue_fact.py --overwrite
    python fact_phrase/queue_fact.py --stage 1     # only stage 1 (the go/no-go pilot)

Design (every comparison changes ONE thing; 3 seeds each; seeds vary only the new encoder -- the
frozen front end is one fixed run):

  stage 1 -- go/no-go, features of R0 seed 42 (multistream_bilstm-310)
    P0  frozen BiLSTM, 512-frame windows      reference: R0's encoder retrained on R0's own features.
                                              Should land near R0 (310-312); if it does not, the
                                              export or the data pipeline is wrong -- stop and check.
    P1  FACT, 1500-frame windows (30 s)       the pilot.
    P2  frozen BiLSTM, 1500-frame windows     same window as FACT: separates "more context" from
                                              "FACT's architecture".
  stage 2 -- robustness (only if FACT wins stage 1), features of the CE reference (multistream_bilstm-208)
    P3  FACT on CE features                    does the result depend on the front end being trained
    P4  frozen BiLSTM on CE features           with the count loss?

Run order: one seed of P0 and P1 first (fast feedback), then the remaining seeds.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, "train_queue_fact.json")
SEEDS = (42, 123, 2024)

R0 = "multistream_bilstm-310"        # R0: B+H+Fc + count loss w=2, epoch by bnd5, seed 42
CE = "multistream_bilstm-208"        # CE reference: B+H+Fc, epoch by bnd5, seed 42

ARMS = {
    "P0": {"model": "frozen_bilstm", "source_run": R0, "window_frames": 512,
           "description": "P0: frozen BiLSTM (R0 recipe) on R0-310 features, 512-frame windows"},
    "P1": {"model": "fact", "source_run": R0,
           "description": "P1: FACT (iuUU, BIO) on R0-310 features, 1500-frame windows"},
    "P2": {"model": "frozen_bilstm", "source_run": R0, "window_frames": 1500,
           "description": "P2: frozen BiLSTM (R0 recipe) on R0-310 features, 1500-frame windows"},
    "P3": {"model": "fact", "source_run": CE,
           "description": "P3: FACT (iuUU, BIO) on CE-208 features, 1500-frame windows"},
    "P4": {"model": "frozen_bilstm", "source_run": CE, "window_frames": 512,
           "description": "P4: frozen BiLSTM (R0 recipe) on CE-208 features, 512-frame windows"},
}
STAGES = {1: ["P0", "P1", "P2"], 2: ["P3", "P4"]}


def jobs_for(arms):
    first = [dict(ARMS[a], seed=SEEDS[0]) for a in arms if a in ("P0", "P1", "P3", "P4")]
    rest = [dict(ARMS[a], seed=s) for a in arms for s in SEEDS if not (s == SEEDS[0] and a in ("P0", "P1", "P3", "P4"))]
    return first + rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None, help="default: both stages")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    arms = STAGES[args.stage] if args.stage else STAGES[1] + STAGES[2]
    jobs = jobs_for(arms)
    if os.path.exists(QUEUE) and not args.overwrite:
        with open(QUEUE) as f:
            if json.load(f):
                raise SystemExit(f"{QUEUE} is not empty -- use --overwrite")
    with open(QUEUE, "w") as f:
        json.dump(jobs, f, indent=1)
    print(f"{len(jobs)} jobs -> {QUEUE}")
    for j in jobs:
        print(f"  {j['description']}  [seed={j['seed']}]")


if __name__ == "__main__":
    main()
