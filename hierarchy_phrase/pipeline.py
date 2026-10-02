"""
hierarchy_phrase/pipeline.py -- prints (default) or runs (--run) the full command sequence, including every ablation.

    python pipeline.py                         # dry run: just print the commands
    python pipeline.py --seeds 42 123 2024 --run
    python pipeline.py --only main flat_vs_hier --seeds 42 --run

Stage A is trained ONCE per seed and shared by every Stage-C variant of that seed. Out-of-fold Stage-A models are
trained only for the 'oof' ablation (K folds per seed -> K extra BiLSTM trainings; the expensive step).
"""
import argparse
import subprocess
import sys

PY = sys.executable
END_RULE = "last_sign_end"
TAG_RULE = "first_end_after"      # set by --tag-rule; applied to every Stage-C training command


def sa(seed, *extra):
    return [PY, "stage_a.py", "train", "--name", f"sa_s{seed}", "--seed", str(seed), *extra]


def ex(name, *extra):
    return [PY, "stage_a.py", "export", "--name", name, *extra]


def sc(seed, tag, *extra):
    return [PY, "stage_c.py", "train", "--stage-a", f"sa_s{seed}", "--tag", tag, "--seed", str(seed), "--tag-rule", TAG_RULE, "--end-rule", END_RULE, *extra]


def ev(seed, tag):
    return [PY, "evaluate.py", "--stage-a", f"sa_s{seed}", "--tag", tag, "--seed", str(seed), "--end-rule", END_RULE]


def plan(seeds, folds):
    P = {"audit": [[PY, "data_audit.py"]]}
    P["stage_a"] = [c for s in seeds for c in (sa(s), ex(f"sa_s{s}"))]
    # main method + ablations 1, 2, 3, 4, 5  (ablation 2 = the hier_oracle rows that every evaluation already contains)
    P["main"] = [c for s in seeds for c in (sc(s, "main", "--source", "mix"), ev(s, "main"))]
    P["flat_vs_hier"] = []      # ablation 1: the 'flat_*' rows of every evaluate.py run (same Stage-A encoder, phrase head)
    P["oracle_signs"] = []      # ablation 2: the 'hier_oracle_*' rows of every evaluate.py run (ceiling + cascade loss)
    P["features"] = [c for s in seeds for t, g in (("pooled_only", ["h_pool"]), ("prosody_only", ["prosody"]),
                                                  ("pooled_prosody", ["h_pool", "prosody"]))
                     for c in (sc(s, t, "--source", "mix", "--groups", *g), ev(s, t))]               # ablation 3 (main = all groups)
    P["training_source"] = [c for s in seeds for t in ("gold", "jitter", "schedule", "pred")
                            for c in (sc(s, f"src_{t}", "--source", t), ev(s, f"src_{t}"))]            # ablation 4 (+ 'main' = mix)
    P["oof"] = []
    for s in seeds:                                    # ablation 4(b): out-of-fold Stage-A predictions, model-agnostic tokens
        fold_runs = [f"sa_s{s}_f{f}" for f in range(folds)]
        for f, name in enumerate(fold_runs):
            P["oof"] += [[PY, "stage_a.py", "train", "--name", name, "--seed", str(s), "--fold", str(f), "--n-folds", str(folds)],
                         ex(name)]
        P["oof"] += [sc(s, "oof", "--source", "pred", "--groups", "sign_probs", "prosody", "--oof-runs", *fold_runs), ev(s, "oof")]
    P["bilstm_control"] = [c for s in seeds for c in (sc(s, "bilstm", "--arch", "bilstm", "--source", "mix"), ev(s, "bilstm"))]   # ablation 5
    P["sign_only_encoder"] = [c for s in seeds for c in (
        [PY, "stage_a.py", "train", "--name", f"sa_s{s}_signonly", "--seed", str(s), "--no-phrase-head"], ex(f"sa_s{s}_signonly"),
        [PY, "stage_c.py", "train", "--stage-a", f"sa_s{s}_signonly", "--tag", "main", "--seed", str(s), "--source", "mix", "--tag-rule", TAG_RULE, "--end-rule", END_RULE],
        [PY, "evaluate.py", "--stage-a", f"sa_s{s}_signonly", "--tag", "main", "--seed", str(s), "--end-rule", END_RULE])]  # ablation 6 (encoder without phrase supervision)
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="subset of: " + ", ".join(plan([0], 2)))
    ap.add_argument("--tag-rule", default="first_end_after", choices=["first_end_after", "nearest_start", "contain_or_next"],
                    help="phrase-start -> sign assignment; pick it from the ORACLE_CEILING block of data_audit.py")
    ap.add_argument("--end-rule", default="last_sign_end", choices=["last_sign_end", "next_start"])
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    global TAG_RULE, END_RULE
    TAG_RULE, END_RULE = a.tag_rule, a.end_rule
    P = plan(a.seeds, a.folds)
    keys = a.only or list(P)
    for k in keys:
        print(f"\n# ---- {k}")
        for cmd in P[k]:
            print(" ".join(cmd))
            if a.run:
                subprocess.run(cmd, check=True)
    print("\n# aggregate:  python aggregate.py results/sa_s*__main_s*.json   (repeat per tag)")


if __name__ == "__main__":
    main()