"""
Paired comparison of two ENCODER runs on the same split -- the tool for reading
the novelty_crf ablation ladder (is arm 3 really better than arm 2?).

    python compare_runs.py --a nov_bilstm_sim_crf-4 --b nov_bilstm_sim_ce-3 \
        [--decoder-a "crf viterbi (learned transitions)"] [--decoder-b "viterbi+collapse"] [--split test]

Each run is decoded with its chosen decoder (default: the CRF decoder if the run
has one, else 'viterbi+collapse'), then the difference A - B in each metric gets a
95% bootstrap CI that resamples the SAME recordings (A and B of one recording
together) for both runs in every draw. Pairing removes the shared per-video
difficulty that makes two marginal CIs overlap even when one run is reliably
ahead, exactly as in run_decoder_eval's paired section -- but across encoders.

Both runs must have been exported for the same split (same videos, same gold).
NOTE: runs differing by seed are different encoders; compare seed-averaged
results, or run each arm with several seeds and compare the ones you intend.
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import load_export
import study_metrics as M
import study_decoders as D
from rescore_protocol import standard_decoders, CRF_DECODER

KEY_METRICS = ["segF1@0.5", "frame_macro_f1", "frame_f1_B", "startF1@2", "segment_ratio"]


def doc_id(vid):
    return vid.rsplit("_", 1)[0]


def default_decoder(crf):
    return CRF_DECODER if crf is not None else "viterbi+collapse"


def compare(records_a, records_b, dec_a, dec_b, crf_a=None, crf_b=None, n_boot=2000, seed=0):
    """Returns {metric: {"a": .., "b": .., "diff": .., "ci": (lo, hi), "p_a_better": ..}}."""
    vids = sorted(records_a)
    if vids != sorted(records_b):
        raise ValueError(f"the two runs cover different videos ({len(vids)} vs {len(records_b)}); "
                         f"export both for the same split")
    for v in vids:
        if not np.array_equal(records_a[v]["labels"], records_b[v]["labels"]):
            raise ValueError(f"gold labels differ for {v} -- these exports are not from the same corpus/split")
    fa, fb = standard_decoders(crf_a)[dec_a], standard_decoders(crf_b)[dec_b]
    gold = {v: records_a[v]["labels"].astype(np.int8) for v in vids}
    pa = {v: fa(D.to_logp(records_a[v]["logits"])) for v in vids}
    pb = {v: fb(D.to_logp(records_b[v]["logits"])) for v in vids}
    sa, stats_a, _ = M.evaluate(pa, gold)
    sb, stats_b, _ = M.evaluate(pb, gold)
    paired = M.paired_bootstrap_diff(stats_a, stats_b, groups=[doc_id(v) for v in vids], n_boot=n_boot, seed=seed)
    out = {}
    for k in KEY_METRICS:
        lo, hi, p = paired[k]
        out[k] = {"a": sa[k], "b": sb[k], "diff": sa[k] - sb[k], "ci": (lo, hi), "p_a_better": p}
    out["_n_documents"] = len({doc_id(v) for v in vids})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--decoder-a", default=None); ap.add_argument("--decoder-b", default=None)
    ap.add_argument("--split", default="test"); ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    ra, ma = load_export(args.a, args.split)
    rb, mb = load_export(args.b, args.split)
    da = args.decoder_a or default_decoder(ma.get("crf"))
    db = args.decoder_b or default_decoder(mb.get("crf"))
    res = compare(ra, rb, da, db, ma.get("crf"), mb.get("crf"), args.boot)
    print(f"\nA = {args.a}  [{da}]\nB = {args.b}  [{db}]\nsplit '{args.split}', {res['_n_documents']} independent recordings "
          f"(CIs resample whole recordings)\n")
    print(f"{'metric':16s}{'A':>9s}{'B':>9s}{'A - B':>10s}   {'95% CI of A - B':>20s}   verdict")
    for k in KEY_METRICS:
        r = res[k]; lo, hi = r["ci"]
        better = "A better" if lo > 0 else ("B better" if hi < 0 else "not resolved")
        if k == "segment_ratio":
            better += "  (ideal ratio is 1.0: judge by distance from 1, not by sign)"
        print(f"{k:16s}{r['a']:9.4f}{r['b']:9.4f}{r['diff']:+10.4f}   [{lo:+8.4f}, {hi:+8.4f}]   {better}")


if __name__ == "__main__":
    main()