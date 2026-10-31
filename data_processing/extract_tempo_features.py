"""
extract_tempo_features.py -- signer-tempo features for Idea A (see src/tempo_features.py).

For every video of every split, builds a sign segmentation, computes the 17 signer-relative
timing channels and saves them where the CUE STREAM can load them (no model / dataset change):

  <out-dir>/<video>.npy        (T, 17) float16, T = length of the v2 phrase labels
  <out-dir>/_names.json        channel names in file order (src/cue_sets.py reads this)
  <out-dir>/_norm.json         train mean / SD of the 4 video-level channels + all settings
  <out-dir>/_report.json       per-split channel means, outside-signing share, skips,
                               and (predicted signs) the train/val mismatch check

Two sources of sign segmentation:
  --source gold   gold gloss labels processed_data/BIO_tags (phrase_oracle_analysis.sign_segments).
                  An ORACLE: report it as an upper bound. v2 phrase starts are snapped to gold
                  sign onsets, so the positional channels alone give away the candidate frames.
  --source pred   the gloss model's exported logits (decoder_study/export_logits.py), decoded
                  into signs with --sign-decoder:
                    argmax_merge3     argmax -> segments -> duplicate-start merge (3 frames)
                    viterbi_collapse  constrained Viterbi (no O->I) + collapse of B runs
                    auto (default)    whichever of the two has the higher sign-onset F1@2 on VAL
                  sign_conf = mean P(B or I) over each predicted sign.
                  Train logits are IN-SAMPLE for the gloss model. The script therefore checks
                  the train/val mismatch (sign-onset F1@2 against gold, and the KS distance of
                  gap_prev_rel, with gold-vs-gold as the baseline) and says whether
                  cross-fitting is needed. With --xfit-runs A B the TRAIN features come
                  out-of-fold from two gloss models trained on the two halves of train
                  (val / test always from the full model).

Run from the repo root:
  # gold upper bound
  python data_processing/extract_tempo_features.py --source gold
  # predicted signs (export the gloss model first, incl. train)
  python decoder_study/export_logits.py --run stgcn_bilstm-44 --splits train val test
  python data_processing/extract_tempo_features.py --source pred \
         --sign-logits decoder_study/exports/stgcn_bilstm-44
  # only if the mismatch check says so: cross-fitted train features
  python data_processing/extract_tempo_features.py --make-xfit-splits
  python data_processing/extract_tempo_features.py --queue-xfit stgcn_bilstm-44 901 902
  python train.py                                     # trains stgcn_bilstm-901 / -902
  python decoder_study/export_logits.py --run stgcn_bilstm-901 --splits train
  python decoder_study/export_logits.py --run stgcn_bilstm-902 --splits train
  python data_processing/extract_tempo_features.py --source pred \
         --sign-logits decoder_study/exports/stgcn_bilstm-44 \
         --xfit-runs stgcn_bilstm-901 stgcn_bilstm-902       # -> processed_data/tempo_features_pred_xfit
"""
import argparse
import json
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(1, os.path.join(REPO_ROOT, "decoder_study"))

from src.tempo_features import (TEMPO_CHANNELS, VIDEO_CHANNELS, LOG1P_CHANNELS, clean_segments,  # noqa: E402
                                tempo_features, transition_table, video_level_raw, video_stats)

B_TAG = 2
SIGN_DECODERS = ("argmax_merge3", "viterbi_collapse")


# ============================================================================ sign sources
def sign_segments(gloss_labels):
    """phrase_oracle_analysis.sign_segments (imported lazily: that module pulls sklearn)."""
    from phrase_oracle_analysis import sign_segments as _ss
    return _ss(gloss_labels)


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def decode_signs(logits, decoder):
    """(T, 3) gloss-model logits [O, I, B] -> ((N, 2) inclusive segments, (N,) confidence)."""
    from src.metrics import extract_segments, merge_close_starts
    from study_decoders import to_logp, decode_viterbi, collapse_b_runs
    p = softmax(logits)
    if decoder == "argmax_merge3":
        segs = extract_segments(p.argmax(axis=1))
        segs = merge_close_starts(segs, p[:, B_TAG], 3)
        segs = np.array(segs, dtype=np.int64).reshape(-1, 2)
    elif decoder == "viterbi_collapse":
        logp = to_logp(logits)
        segs = sign_segments(collapse_b_runs(decode_viterbi(logp), logp))
    else:
        raise ValueError(f"unknown sign decoder '{decoder}'")
    pin = p[:, 1] + p[:, 2]
    conf = np.array([pin[s:e + 1].mean() for s, e in segs]) if len(segs) else np.zeros(0)
    return segs, conf


class ExportReader:
    """Lazy reader of decoder_study/exports/<run>/<split>.npz (keys logits__<vid>)."""

    def __init__(self, run_dir):
        self.run_dir = run_dir
        self._z = {}

    def has(self, split):
        return os.path.exists(os.path.join(self.run_dir, f"{split}.npz"))

    def get(self, split, vid):
        if split not in self._z:
            path = os.path.join(self.run_dir, f"{split}.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found -- run: python decoder_study/export_logits.py "
                    f"--run {os.path.basename(self.run_dir)} --splits {split}")
            self._z[split] = np.load(path)
        key = f"logits__{vid}"
        return self._z[split][key] if key in self._z[split].files else None


def train_ids_of_run(run):
    """Video ids the gloss run was trained on (its hyperparameters' split_file, train split)."""
    hp_path = os.path.join(REPO_ROOT, "experiments", run, "hyperparameters.json")
    with open(hp_path) as f:
        hp = json.load(f)
    split_file = hp.get("split_file", "dataset_splits.json")
    if not os.path.isabs(split_file):
        split_file = os.path.join(REPO_ROOT, split_file)
    with open(split_file) as f:
        return {strip_ext(v) for v in json.load(f)["train"]}, split_file


def strip_ext(v):
    return re.sub(r"\.(npy|pt)$", "", v)


# ============================================================================ checks
def onset_f1(pred_segs, gold_segs, tol=2):
    """Pooled sign-onset counts (tp, n_pred, n_gold) with one-to-one matching within +-tol."""
    from src.metrics import match_boundaries
    gs = np.asarray(gold_segs, dtype=np.int64).reshape(-1, 2)[:, 0]
    ps = np.asarray(pred_segs, dtype=np.int64).reshape(-1, 2)[:, 0]
    g_ok, _, _ = match_boundaries(gs, ps, tol)
    return int(g_ok.sum()), len(ps), len(gs)


def f1(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def gap_rel_values(segs, T, fps):
    """Raw gap_prev_rel of every sign except the first (the transitions the oracle uses)."""
    s = clean_segments(segs, T)
    if len(s) < 2:
        return np.zeros(0)
    table, _ = transition_table(s, fps)
    return table[1:, 2]


def mismatch_check(seg_pred, seg_gold, lengths, splits, fps):
    from scipy.stats import ks_2samp
    res = {}
    for split in ("train", "val", "test"):
        vids = [v for v in splits.get(split, []) if v in seg_pred and v in seg_gold]
        if not vids:
            continue
        c = np.array([onset_f1(seg_pred[v][0], seg_gold[v], 2) for v in vids]).sum(0)
        res[split] = {"videos": len(vids), "sign_onset_f1_at2": round(f1(*c), 4),
                      "n_pred_signs": int(c[1]), "n_gold_signs": int(c[2])}

    def gaps(src, split, is_pred):
        vs = [v for v in splits.get(split, []) if v in src]
        return np.concatenate([gap_rel_values(src[v][0] if is_pred else src[v], lengths[v], fps) for v in vs]
                              or [np.zeros(0)])
    if "train" in res and "val" in res:
        ks_p = ks_2samp(gaps(seg_pred, "train", True), gaps(seg_pred, "val", True))
        ks_g = ks_2samp(gaps(seg_gold, "train", False), gaps(seg_gold, "val", False))
        res["ks_gap_prev_rel_pred_train_vs_val"] = {"stat": round(float(ks_p.statistic), 4),
                                                    "p": float(ks_p.pvalue)}
        res["ks_gap_prev_rel_gold_train_vs_val"] = {"stat": round(float(ks_g.statistic), 4),
                                                    "p": float(ks_g.pvalue)}
        gap = res["train"]["sign_onset_f1_at2"] - res["val"]["sign_onset_f1_at2"]
        ks_flag = ks_p.pvalue < 0.01 and ks_p.statistic > 2 * ks_g.statistic + 0.02
        res["train_minus_val_f1"] = round(gap, 4)
        res["xfit_recommended"] = bool(gap > 0.05 or ks_flag)
        res["rule"] = ("cross-fit if train F1@2 - val F1@2 > 0.05, or KS(pred train vs val) is "
                       "significant (p < .01) and > 2 x KS(gold train vs val) + 0.02")
    return res


# ============================================================================ xfit helpers
def make_xfit_splits(split_file, seed=0):
    """Two halves of TRAIN, split by TRANSCRIPT (both participants of a conversation stay in
    the same half); val / test unchanged. Writes dataset_splits_xfit_a.json / _b.json."""
    with open(os.path.join(REPO_ROOT, split_file)) as f:
        sp = json.load(f)
    train = list(sp["train"])
    transcripts = sorted({strip_ext(v).rsplit("_", 1)[0] for v in train})
    rng = np.random.default_rng(seed)
    rng.shuffle(transcripts)
    half_a = set(transcripts[: len(transcripts) // 2])
    out = {}
    for name, keep_a in (("a", True), ("b", False)):
        part = [v for v in train if (strip_ext(v).rsplit("_", 1)[0] in half_a) == keep_a]
        new = dict(sp)
        new["train"] = part
        path = os.path.join(REPO_ROOT, f"dataset_splits_xfit_{name}.json")
        with open(path, "w") as f:
            json.dump(new, f, indent=2)
        out[name] = (path, len(part))
        print(f"wrote {path}: {len(part)} train videos (val {len(sp.get('val', []))}, test {len(sp.get('test', []))})")
    return out


def queue_xfit(base_run, prefix_a, prefix_b, queue_file="train_queue.json"):
    """Append two copies of a gloss run's recipe, trained on the two train halves, to train.py's queue."""
    hp_path = os.path.join(REPO_ROOT, "experiments", base_run, "hyperparameters.json")
    with open(hp_path) as f:
        base = json.load(f)
    qpath = os.path.join(REPO_ROOT, queue_file)
    queue = json.load(open(qpath)) if os.path.exists(qpath) else [{"prefixes": {}}]
    tracker = queue[0].setdefault("prefixes", {})
    name = base["basename"]
    for half, p in (("a", prefix_a), ("b", prefix_b)):
        sf = f"dataset_splits_xfit_{half}.json"
        if not os.path.exists(os.path.join(REPO_ROOT, sf)):
            raise FileNotFoundError(f"{sf} missing -- run with --make-xfit-splits first.")
        if int(p) in tracker.get(name, []):
            raise ValueError(f"prefix {p} is already tracked for {name} in {queue_file}")
        cfg = dict(base)
        cfg["prefix"] = f"{int(p):02d}"
        cfg["split_file"] = sf
        cfg["description"] = f"XFIT half {half} of {base_run} (recipe copied; train = {sf})"
        queue.append(cfg)
        tracker.setdefault(name, []).append(int(p))
        print(f"queued {name}-{cfg['prefix']}: {cfg['description']}")
    with open(qpath, "w") as f:
        json.dump(queue, f, indent=4)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Signer-tempo features for the cue stream (Idea A).")
    ap.add_argument("--source", choices=["gold", "pred"])
    ap.add_argument("--sign-logits", default=os.path.join("decoder_study", "exports", "stgcn_bilstm-44"),
                    help="export dir of the gloss model (pred): <dir>/<split>.npz")
    ap.add_argument("--sign-decoder", default="auto", choices=("auto",) + SIGN_DECODERS)
    ap.add_argument("--xfit-runs", nargs=2, default=None, metavar=("RUN_A", "RUN_B"),
                    help="gloss runs trained on the two train halves: train features out-of-fold")
    ap.add_argument("--exports-root", default=os.path.join("decoder_study", "exports"))
    ap.add_argument("--out-dir", default=None,
                    help="default processed_data/tempo_features_{gold|pred|pred_xfit}")
    ap.add_argument("--split-file", default="dataset_splits.json")
    ap.add_argument("--gloss-labels", default=os.path.join("processed_data", "BIO_tags"))
    ap.add_argument("--phrase-labels", default=os.path.join("processed_data", "BIO_tags_phrase_v2"))
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--transform", default="log1p", choices=["log1p", "none"],
                    help="log1p of the 4 unbounded ratio channels (default) or raw ratios")
    ap.add_argument("--no-fallback", action="store_true",
                    help="skip videos without a sign source (default: write an all-outside file so the "
                         "training set stays identical to the reference)")
    ap.add_argument("--max-videos", type=int, default=None, help="per split, for a quick test")
    ap.add_argument("--make-xfit-splits", action="store_true", help="write the two train halves and exit")
    ap.add_argument("--queue-xfit", nargs=3, default=None, metavar=("BASE_RUN", "PREFIX_A", "PREFIX_B"),
                    help="queue two half-train copies of a gloss run for train.py and exit")
    args = ap.parse_args()
    os.chdir(REPO_ROOT)

    if args.make_xfit_splits:
        make_xfit_splits(args.split_file)
        return
    if args.queue_xfit:
        queue_xfit(*args.queue_xfit)
        return
    if args.source is None:
        ap.error("--source is required")
    if args.xfit_runs and args.source != "pred":
        ap.error("--xfit-runs needs --source pred")
    tag = args.source + ("_xfit" if args.xfit_runs else "")
    out_dir = args.out_dir or os.path.join("processed_data", f"tempo_features_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.split_file) as f:
        sp = json.load(f)
    splits = {}
    for split in ("train", "val", "test"):
        vids = sorted(strip_ext(v) for v in sp.get(split, []))
        splits[split] = vids[: args.max_videos] if args.max_videos else vids
    split_of = {v: s for s, vs in splits.items() for v in vs}

    # ---------------------------------------------------------------- frame counts + gold signs
    lengths, seg_gold = {}, {}
    skips = {"missing_phrase_labels": [], "no_sign_source": [], "gold_length_mismatch": [], "pred_length_mismatch": []}
    for v in split_of:
        pp = os.path.join(args.phrase_labels, f"{v}.npy")
        if not os.path.exists(pp):
            skips["missing_phrase_labels"].append(v)
            continue
        lengths[v] = len(np.load(pp, mmap_mode="r"))
        gp = os.path.join(args.gloss_labels, f"{v}.npy")
        if os.path.exists(gp):
            g = np.load(gp)
            if len(g) != lengths[v]:
                skips["gold_length_mismatch"].append(v)
            seg_gold[v] = sign_segments(g)

    # ---------------------------------------------------------------- sign segmentation
    seg = {}                                         # vid -> (segs, conf or None)
    decoder_choice = None
    if args.source == "gold":
        seg = {v: (s, None) for v, s in seg_gold.items()}
    else:
        full = ExportReader(args.sign_logits)
        xfit = None
        if args.xfit_runs:
            xfit = []
            for run in args.xfit_runs:
                ids, sf = train_ids_of_run(run)
                xfit.append((run, ids, ExportReader(os.path.join(args.exports_root, run))))
                print(f"xfit model {run}: trained on {len(ids)} videos ({os.path.relpath(sf, REPO_ROOT)})")

        def logits_for(v):
            if split_of[v] == "train" and xfit:
                cands = [r for r in xfit if v not in r[1]]
                if not cands:
                    skips.setdefault("xfit_in_both_halves", []).append(v)
                    return None
                return cands[0][2].get("train", v)
            return full.get(split_of[v], v)

        decoders = SIGN_DECODERS if args.sign_decoder == "auto" else (args.sign_decoder,)
        cache = {d: {} for d in decoders}
        if len(decoders) > 1:
            scores = {}
            for d in decoders:
                c = np.zeros(3)
                for v in splits["val"]:
                    lg = logits_for(v) if v in lengths else None
                    if lg is not None and v in seg_gold:
                        cache[d][v] = decode_signs(lg, d)
                        c += onset_f1(cache[d][v][0], seg_gold[v], 2)
                scores[d] = f1(*c)
            decoder_choice = max(scores, key=scores.get)
            print("sign decoder on VAL (sign-onset F1@2): " +
                  ", ".join(f"{d} {s:.3f}" for d, s in scores.items()) + f" -> {decoder_choice}")
        else:
            decoder_choice = decoders[0]
        for split in ("val", "test", "train"):
            for v in splits[split]:
                if v not in lengths:
                    continue
                lg = logits_for(v)
                if lg is None:
                    continue
                if len(lg) != lengths[v]:
                    skips["pred_length_mismatch"].append(v)
                seg[v] = cache[decoder_choice].get(v) or decode_signs(lg, decoder_choice)

    # ---------------------------------------------------------------- video-level norm (train)
    raw_train = [video_level_raw(video_stats(clean_segments(seg[v][0], lengths[v]), args.fps))
                 for v in splits["train"] if v in seg and len(clean_segments(seg[v][0], lengths[v]))]
    if not raw_train:
        raise SystemExit("no train video with a sign segmentation -- nothing to normalize with")
    raw_train = np.array(raw_train)
    norm = {"channels": VIDEO_CHANNELS, "mean": raw_train.mean(0).tolist(), "sd": raw_train.std(0).tolist(),
            "n_train_videos": int(len(raw_train)), "transform": args.transform,
            "log1p_channels": LOG1P_CHANNELS if args.transform == "log1p" else [],
            "source": args.source, "sign_decoder": decoder_choice,
            "sign_logits": args.sign_logits if args.source == "pred" else None,
            "xfit_runs": args.xfit_runs, "fps": args.fps, "phrase_labels": args.phrase_labels,
            "gloss_labels": args.gloss_labels}

    # ---------------------------------------------------------------- features
    report = {"splits": {}, "skips": {k: len(x) for k, x in skips.items()},
              "skipped_videos": {k: x[:20] for k, x in skips.items() if x}}
    ci_out = TEMPO_CHANNELS.index("outside_signing")
    for split, vids in splits.items():
        sums, frames, n_written, n_fallback = np.zeros(len(TEMPO_CHANNELS)), 0, 0, 0
        for v in vids:
            if v not in lengths:
                continue
            T = lengths[v]
            if v in seg:
                segs, conf = seg[v]
            elif args.no_fallback:
                skips["no_sign_source"].append(v)
                continue
            else:
                segs, conf = np.zeros((0, 2), dtype=np.int64), None
                skips["no_sign_source"].append(v)
                n_fallback += 1
            if conf is not None:      # keep conf aligned with the cleaned segments
                s_arr = np.asarray(segs, dtype=np.int64).reshape(-1, 2)
                keep = (s_arr[:, 0] < T)
                segs, conf = s_arr[keep], np.asarray(conf)[keep]
            F = tempo_features(segs, T, args.fps, sign_conf=conf, video_norm=norm, transform=args.transform)
            np.save(os.path.join(out_dir, f"{v}.npy"), F.astype(np.float16))
            sums += F.sum(0)
            frames += T
            n_written += 1
        means = sums / max(frames, 1)
        report["splits"][split] = {"videos_written": n_written, "fallback_no_signs": n_fallback,
                                   "outside_signing_share": round(float(means[ci_out]), 4),
                                   "channel_means": {c: round(float(m), 4) for c, m in zip(TEMPO_CHANNELS, means)}}
        print(f"[{split}] {n_written} videos written ({n_fallback} without signs), "
              f"outside-signing share {means[ci_out]:.3f}")
    report["skips"] = {k: len(x) for k, x in skips.items()}
    report["skipped_videos"] = {k: x[:20] for k, x in skips.items() if x}

    if args.source == "pred":
        mm = mismatch_check(seg, seg_gold, lengths, splits, args.fps)
        report["mismatch"] = mm
        print("\nTRAIN / VAL MISMATCH (predicted signs)")
        for split in ("train", "val", "test"):
            if split in mm:
                print(f"  {split:5s} sign-onset F1@2 {mm[split]['sign_onset_f1_at2']:.3f}  "
                      f"({mm[split]['n_pred_signs']} predicted / {mm[split]['n_gold_signs']} gold signs)")
        if "xfit_recommended" in mm:
            kp, kg = mm["ks_gap_prev_rel_pred_train_vs_val"], mm["ks_gap_prev_rel_gold_train_vs_val"]
            print(f"  KS gap_prev_rel train vs val: predicted {kp['stat']:.3f} (p={kp['p']:.2g}), "
                  f"gold baseline {kg['stat']:.3f} (p={kg['p']:.2g})")
            verdict = "RECOMMENDED (see the module docstring)" if mm["xfit_recommended"] else "not needed"
            if args.xfit_runs:
                verdict += " -- these features already use cross-fitted train signs"
            print(f"  cross-fitting: {verdict}")

    with open(os.path.join(out_dir, "_names.json"), "w") as f:
        json.dump(TEMPO_CHANNELS, f, indent=2)
    with open(os.path.join(out_dir, "_norm.json"), "w") as f:
        json.dump(norm, f, indent=2)
    with open(os.path.join(out_dir, "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {sum(s['videos_written'] for s in report['splits'].values())} files + _names/_norm/_report.json "
          f"to {out_dir}  (skips: {report['skips']})")


if __name__ == "__main__":
    main()
