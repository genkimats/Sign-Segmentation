"""
Decoder study runner: greedy baselines vs constrained Viterbi vs semi-Markov,
all on the SAME frozen-encoder logits, scored against RAW gold.

    python run_decoder_eval.py --run stgcn_bilstm-08

Protocol (per the brief, with two deliberate deviations explained in the output):
  * Label-derived quantities (duration prior, duration-bucket edges) are fit on
    TRAIN gold labels -- no encoder outputs involved.
  * Everything that touches encoder logits (temperature, thresholds, semi-Markov
    weight/penalty) is fit on the SELECTION split (default val), never on train
    logits, because the encoder saw train and its train logits are overconfident.
    Val is tiny (~10 videos), so tuned values are noisy -- read the CIs.
  * Baselines get a Begin-run collapse variant, because the encoder was trained
    on dilated Begin targets (see study_decoders.py). Without it the comparison
    would credit structured decoders for undoing a training artifact.
  * An ORACLE section checks the pipeline itself: gold logits must decode back to
    gold, and a perfect model of the DILATED training targets shows how much each
    decoder loses to dilation alone, before any real encoder error.
"""
import os
import sys
import csv
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import (RESULTS_DIR, load_export, load_gold_for_split, training_targets)
import study_metrics as M
import study_decoders as D
import study_calibration as C

SELECT_METRIC = "segF1@0.5"
DEFAULT_GRIDS = {
    "t_b": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), "t_o": (0.3, 0.4, 0.5, 0.6, 0.7),
    "hyst": (0.5, 0.6, 0.7, 0.8),
    "sm_w": (0.25, 0.5, 1.0, 2.0), "sm_pen": (-2.0, 0.0, 2.0, 4.0),
}


def _collapse(fn):
    return lambda lp: D.collapse_b_runs(fn(lp), lp)


def build_families(prior, grids):
    fam = {}
    fam["argmax"] = [({}, D.decode_argmax)]
    fam["argmax+collapse"] = [({}, _collapse(D.decode_argmax))]
    th = [({"t_b": tb, "t_o": to}, (lambda lp, tb=tb, to=to: D.decode_threshold(lp, tb, to)))
          for tb in grids["t_b"] for to in grids["t_o"]]
    fam["threshold (tuned)"] = th
    fam["threshold+collapse (tuned)"] = [(p, _collapse(f)) for p, f in th]
    fam["hysteresis (project 'linguistic', tuned)"] = [
        ({"thr": t}, (lambda lp, t=t: D.decode_hysteresis(lp, t))) for t in grids["hyst"]]
    fam["viterbi (constrained)"] = [({}, D.decode_viterbi)]
    fam["viterbi+collapse"] = [({}, _collapse(D.decode_viterbi))]
    fam["semi-Markov (tuned)"] = [
        ({"dur_w": w, "seg_pen": pen}, (lambda lp, w=w, pen=pen: D.decode_semi_markov(lp, prior["dur_logp"], w, pen)))
        for w in grids["sm_w"] for pen in grids["sm_pen"]]
    return fam


def _score(summ):
    return (summ[SELECT_METRIC], -abs(summ["segment_ratio"] - 1.0))


def tune(cands, logp_sel, gold_sel):
    best = None
    for params, fn in cands:
        preds = {v: fn(lp) for v, lp in logp_sel.items()}
        summ, _, _ = M.evaluate(preds, gold_sel)
        if best is None or _score(summ) > best[0]:
            best = (_score(summ), params, fn, summ)
    return best[1], best[2], best[3]


def run_evaluation(select_records, eval_records, train_gold, tolerance_window=5,
                   temperature="auto", n_boot=1000, targets_fn=None, grids=None, log=print):
    targets_fn = targets_fn or training_targets
    grids = {**DEFAULT_GRIDS, **(grids or {})}
    gold_sel = {v: r["labels"] for v, r in select_records.items()}
    gold_ev = {v: r["labels"] for v, r in eval_records.items()}
    out = {"n_select_videos": len(gold_sel), "n_eval_videos": len(gold_ev)}

    # ---- calibration (selection split only) -------------------------------
    sel_logits = [r["logits"] for r in select_records.values()]
    dil = [targets_fn(g, tolerance_window)[1] for g in gold_sel.values()]
    raw = [g for g in gold_sel.values()]
    T_fit = C.fit_temperature(sel_logits, dil)
    T = T_fit if temperature == "auto" else float(temperature)
    log("\n=== CALIBRATION (selection split; T fit against the DILATED training targets) ===")
    for tag, tg in (("vs dilated training targets", dil), ("vs raw gold", raw)):
        for t_use, lab in ((1.0, "T=1.00"), (T, f"T={T:.2f}")):
            r = C.calibration_report(sel_logits, tg, t_use)
            log(f"  {tag:28s} {lab:8s} NLL={r['nll']:.4f}  ECE_top={r['ece_top']:.4f}  ECE_B={r['ece_B']:.4f}  "
                f"mean p_B={r['mean_p_B']:.4f} vs freq_B={r['freq_B']:.4f}")
    log(f"  -> using temperature T = {T:.3f}")
    out["temperature"] = T

    # ---- label-derived quantities (train GOLD only) -------------------------
    prior = D.fit_duration_prior(list(train_gold))
    durs = np.array([e - s for g in train_gold for s, e in M.bio_to_segments(g)])
    q1, q2 = np.quantile(durs, [1 / 3, 2 / 3])
    bucket_edges = np.array([0, max(int(round(q1)), 2), max(int(round(q2)), 3), 1e9])
    out["prior"] = {k: v for k, v in prior.items() if k != "dur_logp"}
    log(f"\n=== DURATION PRIOR (train gold, {prior['n_segments']} signs) ===\n  median={prior['median']:.0f} frames, "
        f"5-95%=[{prior['p05']:.0f},{prior['p95']:.0f}], dmax={prior['dmax']}, "
        f"beyond dmax={prior['frac_beyond_dmax']*100:.2f}%\n  duration buckets (frames): "
        f"short<{bucket_edges[1]:.0f}  medium<{bucket_edges[2]:.0f}  long>=")

    fams = build_families(prior, grids)

    # ---- oracle checks ------------------------------------------------------
    canon = {"argmax": D.decode_argmax, "threshold": lambda lp: D.decode_threshold(lp, .5, .5),
             "hysteresis": D.decode_hysteresis, "viterbi": D.decode_viterbi,
             "semi-Markov": lambda lp: D.decode_semi_markov(lp, prior["dur_logp"])}
    log("\n=== ORACLE 1: gold logits must decode back to gold (pipeline sanity) ===")
    oracle_ok = {}
    for name, fn in canon.items():
        preds = {v: fn(D.oracle_logp(g)) for v, g in gold_ev.items()}
        s, _, _ = M.evaluate(preds, gold_ev)
        oracle_ok[name] = s["segF1@0.5"] > 1 - 1e-9 and s["frame_macro_f1"] > 1 - 1e-9
        log(f"  {name:12s} segF1@0.5={s['segF1@0.5']:.4f} frameF1={s['frame_macro_f1']:.4f}  {'ok' if oracle_ok[name] else '*** LOSES INFORMATION ***'}")
    out["oracle_gold_ok"] = oracle_ok
    log("\n=== ORACLE 2: PERFECT model of the dilated training targets, scored vs RAW gold ===")
    log("  (isolates what each decoder loses to Begin-dilation alone, before any encoder error)")
    dil_fns = {"argmax": D.decode_argmax, "argmax+collapse": _collapse(D.decode_argmax),
               "viterbi": D.decode_viterbi, "viterbi+collapse": _collapse(D.decode_viterbi),
               "semi-Markov": lambda lp: D.decode_semi_markov(lp, prior["dur_logp"])}
    out["oracle_dilated"] = {}
    for name, fn in dil_fns.items():
        preds = {v: fn(D.oracle_soft_logp(targets_fn(g, tolerance_window)[0])) for v, g in gold_ev.items()}
        s, _, _ = M.evaluate(preds, gold_ev)
        out["oracle_dilated"][name] = {k: s[k] for k in (SELECT_METRIC, "segment_ratio", "frame_macro_f1")}
        log(f"  {name:18s} segF1@0.5={s[SELECT_METRIC]:.3f}  ratio={s['segment_ratio']:.2f}  frameF1={s['frame_macro_f1']:.3f}")

    # ---- main comparison ----------------------------------------------------
    logp_sel = {v: D.to_logp(r["logits"], T) for v, r in select_records.items()}
    logp_ev = {v: D.to_logp(r["logits"], T) for v, r in eval_records.items()}
    rows, out["decoders"] = [], {}
    log(f"\n=== TUNING on selection split ({len(gold_sel)} videos), EVALUATING on eval split ({len(gold_ev)} videos) ===")
    for name, cands in fams.items():
        params, fn, sel_summ = tune(cands, logp_sel, gold_sel) if len(cands) > 1 else (cands[0][0], cands[0][1], None)
        preds = {v: fn(lp) for v, lp in logp_ev.items()}
        summ, _, ci = M.evaluate(preds, gold_ev, bucket_edges=bucket_edges, n_boot=n_boot)
        out["decoders"][name] = {"params": params, "eval": summ, "ci": ci, "selection": sel_summ}
        rows.append((name, params, summ, ci))
        log(f"  done: {name}  {params if params else ''}")

    def fmt(ci, k, s):
        return f"{s[k]:.3f}" + (f" [{ci[k][0]:.3f},{ci[k][1]:.3f}]" if ci else "")
    log("\n" + "=" * 118)
    log(f"{'decoder':42s} {'segF1@0.5 [95% CI]':24s} {'seg ratio [95% CI]':22s} {'startF1@2':9s} {'frameF1':8s} {'legacyF1':8s}")
    log("-" * 118)
    for name, params, s, ci in rows:
        log(f"{name:42s} {fmt(ci, 'segF1@0.5', s):24s} {fmt(ci, 'segment_ratio', s):22s} "
            f"{s['startF1@2']:<9.3f} {s['frame_macro_f1']:<8.3f} {s['legacy_segF1']:<8.3f}")
    log("=" * 118)
    log("  segment ratio = predicted/true segments (ideal 1.0). legacyF1 = src/metrics.py's many-to-one Segment F1")
    log("  (can exceed 1; shown only to connect to older numbers). startF1@2 is in FRAMES -- its meaning depends on fps.")
    log("\n--- stratified by gold sign duration: recall of gold signs matched at IoU>=0.5 ---")
    log(f"{'decoder':42s} {'short':8s} {'medium':8s} {'long':8s}   frag_gold_rate  merge_pred_rate")
    for name, params, s, ci in rows:
        log(f"{name:42s} {s['recall_bucket0']:<8.3f} {s['recall_bucket1']:<8.3f} {s['recall_bucket2']:<8.3f}   "
            f"{s['frag_gold_rate']:<14.3f}  {s['merge_pred_rate']:.3f}")
    log("  frag_gold_rate = share of gold signs split across >=2 predictions (over-segmentation);")
    log("  merge_pred_rate = share of predictions spanning >=2 gold signs (under-segmentation).")
    return out


def _jsonable(o):
    if isinstance(o, dict): return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--select-split", default="val")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--temperature", default="auto", help="'auto' (fit on selection split) or a number")
    args = ap.parse_args()

    sel, meta = load_export(args.run, args.select_split)
    ev, _ = load_export(args.run, args.eval_split)
    if set(sel) & set(ev):
        raise SystemExit("selection and eval splits share videos -- that would leak tuning into evaluation")
    train_gold = list(load_gold_for_split("train").values())
    fps = sorted({round(v, 2) for v in (meta.get("fps_by_vid") or {}).values() if v})
    print(f"Run {args.run}: window={meta.get('window')}, fps seen: {fps or 'unknown'}")
    if len(fps) > 1:
        print("  WARNING: multiple frame rates present -- durations in frames are not comparable across them; "
              "fit one duration prior per fps group before trusting the semi-Markov numbers.")
    out = run_evaluation(sel, ev, train_gold, meta.get("tolerance_window", 5), args.temperature, args.boot)

    rd = os.path.join(RESULTS_DIR, args.run); os.makedirs(rd, exist_ok=True)
    json.dump(_jsonable(out), open(os.path.join(rd, "results.json"), "w"), indent=2)
    with open(os.path.join(rd, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); keys = list(next(iter(out["decoders"].values()))["eval"].keys())
        w.writerow(["decoder", "params"] + keys)
        for name, d in out["decoders"].items():
            w.writerow([name, json.dumps(_jsonable(d["params"]))] + [d["eval"][k] for k in keys])
    print(f"\nSaved {rd}/results.json and results.csv")


if __name__ == "__main__":
    main()