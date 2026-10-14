"""
check_keypoint_importance.py -- which keypoints (and HaMeR joints) does a trained model use?

Two complementary analyses per run:

1. OCCLUSION (causal, "what breaks if this keypoint carries no information?")
   A keypoint group is replaced by its per-video temporal mean (it stays at a plausible
   position but no longer moves), the whole split is re-predicted, and the change in
   metrics is measured:
     dFrameF1, dF1_B, dIoU   -- the usual metrics (argmax decoding)
     dBndF1@k                -- boundary F1: predicted phrase starts vs gold starts,
                                one-to-one within +-k frames
     dPeakPB                 -- mean of max P(B) within +-peak_window of each gold start
     changed                 -- share of frames whose argmax prediction changes
   Groups come at two levels: COARSE (head, shoulders, arms, each hand, face, each HaMeR
   hand) and FINE (fingers, eyes/ears/nose/mouth, eyebrows, HaMeR finger joints ...).
   Groups of a disabled stream (multistream use_stream_* = False) are skipped.

2. GRADIENT ATTRIBUTION around gold phrase starts ("when does each keypoint matter?")
   For every gold phrase start, log P(B) at that frame is differentiated w.r.t. the input
   window; attribution = (input - per-video mean) * gradient (or integrated gradients with
   --ig-steps N, same baseline). Per group the signed contribution is summed, its absolute
   value is averaged over boundaries for every time offset in [-R, +R] frames. The same is
   done at CONTROL frames (inside phrases, far from any boundary) for comparison.
   Output: a heatmap (fine groups x time offset), a line plot (coarse groups), CSVs.

Results go to evaluation_phrase/keypoint_importance/<run>/.

Usage (repo root):
  python check_keypoint_importance.py --model multistream_bilstm --prefixes 52
  python check_keypoint_importance.py --model multistream_bilstm --prefixes 52 55 --ig-steps 16
  python check_keypoint_importance.py --model multistream_bilstm --no-gradient --per-vertex

Caveats: importance is about THIS model, not about sign language; redundant keypoints
share credit (occluding one finger point understates the finger -- use groups); the HaMeR
mean of rotation matrices is not itself a rotation, it only serves as an uninformative input.
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import evaluate_phrase as ep  # noqa: E402
from src.evaluation import predict_video_probs, video_windows  # noqa: E402,F401
from src.face_subsets import (face_subset_raw_indices, LIPS_INDICES, LEFT_EYE_INDICES,  # noqa: E402
                              RIGHT_EYE_INDICES, LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES)
from src.metrics import evaluate_videos, extract_segments, match_boundaries, B_TAG, I_TAG  # noqa: E402
from src.model_factory import build_model_kwargs, call_model  # noqa: E402

OUT_ROOT = os.path.join(ep.OUT_DIR, "keypoint_importance")

# MediaPipe hand: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky
HAND_PARTS = [("wrist", [0]), ("thumb", [1, 2, 3, 4]), ("index", [5, 6, 7, 8]),
              ("middle", [9, 10, 11, 12]), ("ring", [13, 14, 15, 16]), ("pinky", [17, 18, 19, 20])]
# MANO joint order of HaMeR's 15 finger joints: index, middle, pinky, ring, thumb (3 each)
MANO_FINGERS = [("index", [0, 1, 2]), ("middle", [3, 4, 5]), ("pinky", [6, 7, 8]),
                ("ring", [9, 10, 11]), ("thumb", [12, 13, 14])]


# ==============================================================================
# Keypoint groups
# ==============================================================================
def stream_enabled(config):
    if config["basename"].startswith("multistream"):
        return {"body_hands": bool(config.get("use_stream_body_hands", True)),
                "face": bool(config.get("use_stream_face", False)),
                "hamer": bool(config.get("use_stream_hamer", False))}
    face_only = config.get("face_only", False)
    return {"body_hands": not face_only,
            "face": bool(config.get("use_face_keypoints", False)),
            "hamer": bool(config.get("use_hamer_features", False))}


def build_groups(config, hamer_dim):
    """List of {name, level, stream, vertices | hamer_dims}. Names use MediaPipe's left/right."""
    on = stream_enabled(config)
    face_only = config.get("face_only", False)
    groups = []

    def add(name, level, stream, vertices=None, dims=None):
        groups.append({"name": name, "level": level, "stream": stream,
                       "vertices": vertices or [], "hamer_dims": dims or []})

    if on["body_hands"] and not face_only:
        add("body: head (nose/eyes/ears/mouth)", "coarse", "body_hands", list(range(0, 11)))
        add("body: shoulders", "coarse", "body_hands", [11, 12])
        add("body: arms (elbows + pose wrists)", "coarse", "body_hands", list(range(13, 23)))
        add("left hand (21 pts)", "coarse", "body_hands", list(range(23, 44)))
        add("right hand (21 pts)", "coarse", "body_hands", list(range(44, 65)))
        for name, v in [("head: nose", [0]), ("head: eyes", [1, 2, 3, 4, 5, 6]), ("head: ears", [7, 8]),
                        ("head: mouth corners", [9, 10]), ("L shoulder", [11]), ("R shoulder", [12]),
                        ("L elbow", [13]), ("R elbow", [14]),
                        ("L pose wrist+hand pts", [15, 17, 19, 21]), ("R pose wrist+hand pts", [16, 18, 20, 22])]:
            add(name, "fine", "body_hands", v)
        for side, off in (("L", 23), ("R", 44)):
            for part, local in HAND_PARTS:
                add(f"{side} hand: {part}", "fine", "body_hands", [off + i for i in local])

    if on["face"]:
        off = 0 if face_only else 65
        raw = face_subset_raw_indices(config.get("face_subset", "full"))
        pos = {r: i for i, r in enumerate(raw)}
        add("face (all)", "coarse", "face", [off + i for i in range(len(raw))])
        for name, region in [("face: left eye", LEFT_EYE_INDICES), ("face: right eye", RIGHT_EYE_INDICES),
                             ("face: left eyebrow", LEFT_EYEBROW_INDICES),
                             ("face: right eyebrow", RIGHT_EYEBROW_INDICES), ("face: mouth", LIPS_INDICES)]:
            v = sorted({off + pos[r] for r in region if r in pos})
            if v:
                add(name, "fine", "face", v)

    if on["hamer"] and hamer_dim:
        if hamer_dim == 288:
            for h, side in ((0, "L"), (1, "R")):
                pose = list(range(h * 135, h * 135 + 135))
                orient = list(range(270 + h * 9, 270 + h * 9 + 9))
                add(f"HaMeR {side} hand (all)", "coarse", "hamer", dims=pose + orient)
                for finger, joints in MANO_FINGERS:
                    add(f"HaMeR {side}: {finger}", "fine", "hamer",
                        dims=[h * 135 + j * 9 + k for j in joints for k in range(9)])
                add(f"HaMeR {side}: wrist orientation", "fine", "hamer", dims=orient)
        else:
            add("HaMeR (all)", "coarse", "hamer", dims=list(range(hamer_dim)))
    return groups


# ==============================================================================
# Metrics shared by both analyses
# ==============================================================================
def gold_starts(gold):
    return np.array([s for s, _ in extract_segments(gold)], dtype=np.int64)


def summarize(probs, golds, tol, peak_window, base_pred=None):
    ev = evaluate_videos(probs, golds, decoder="argmax")
    tp = n_pred = n_gold = 0
    peaks, changed, total = [], 0, 0
    for vid, p in probs.items():
        g = np.asarray(golds[vid])[:p.shape[1]]
        gs = gold_starts(g)
        pred = p.argmax(0)
        ps = np.array([s for s, _ in extract_segments(pred)], dtype=np.int64)
        g_ok, _, _ = match_boundaries(gs, ps, tol)
        tp += int(g_ok.sum()); n_pred += len(ps); n_gold += len(gs)
        T = p.shape[1]
        for s in gs:
            peaks.append(float(p[B_TAG, max(0, s - peak_window):min(T, s + peak_window + 1)].max()))
        if base_pred is not None:
            changed += int((base_pred[vid] != pred).sum()); total += T
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return {
        "FrameF1": ev["Frame_F1"], "F1_B": ev["F1_B"], "IoU": ev["IoU"],
        "BndF1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "PeakPB": float(np.mean(peaks)) if peaks else float("nan"),
        "changed": changed / total if total else 0.0,
    }


# ==============================================================================
# 1. Occlusion
# ==============================================================================
def occlusion(model, config, dataset, device, groups, args, extra_vertex_groups):
    golds = {v: np.asarray(c["hard_labels"]).astype(np.int64) for v, c in dataset.video_cache.items()}
    bs = args.batch_size or config.get("batch_size", 16)

    def run(cache_fn):
        return {v: predict_video_probs(model, config, cache_fn(v, c), device, bs)
                for v, c in dataset.video_cache.items()}

    base_probs = run(lambda v, c: c)
    base_pred = {v: p.argmax(0) for v, p in base_probs.items()}
    base = summarize(base_probs, golds, args.tolerance, args.peak_window)

    means = {}
    for v, c in dataset.video_cache.items():
        means[v] = {"feat": c["features"].to(torch.float32).mean(0).to(c["features"].dtype)}
        if "hamer_features" in c and c["hamer_features"] is not None:
            means[v]["hamer"] = c["hamer_features"].to(torch.float32).mean(0).to(c["hamer_features"].dtype)

    def occluded(group):
        def fn(v, c):
            out = dict(c)
            if group["vertices"]:
                f = c["features"].clone()
                f[:, group["vertices"], :] = means[v]["feat"][group["vertices"], :]
                out["features"] = f
            if group["hamer_dims"]:
                h = c["hamer_features"].clone()
                h[:, group["hamer_dims"]] = means[v]["hamer"][group["hamer_dims"]]
                out["hamer_features"] = h
            return out
        return fn

    rows = []
    all_groups = groups + extra_vertex_groups
    for i, g in enumerate(all_groups):
        m = summarize(run(occluded(g)), golds, args.tolerance, args.peak_window, base_pred)
        rows.append({"level": g["level"], "group": g["name"], "stream": g["stream"],
                     "size": len(g["vertices"]) or len(g["hamer_dims"]),
                     **{f"d{k}": m[k] - base[k] for k in ("FrameF1", "F1_B", "IoU", "BndF1", "PeakPB")},
                     "changed": m["changed"]})
        print(f"\r  occlusion {i + 1}/{len(all_groups)}", end="", flush=True)
    print()
    return base, rows


def print_occlusion(base, rows, tol):
    print(f"  base (argmax): FrameF1 {base['FrameF1']:.4f} | F1_B {base['F1_B']:.4f} | IoU {base['IoU']:.4f} | "
          f"BndF1@{tol} {base['BndF1']:.4f} | PeakPB {base['PeakPB']:.4f}")
    for level in ("coarse", "fine", "vertex"):
        lr = [r for r in rows if r["level"] == level]
        if not lr:
            continue
        lr.sort(key=lambda r: (r["dBndF1"], r["dPeakPB"]))
        print(f"\n  [{level}] sorted by drop in boundary F1 (most important first)")
        hdr = (f"  {'group':<34} {'n':>4} {'dFrameF1':>9} {'dF1_B':>8} {'dIoU':>8} "
               f"{'dBndF1@' + str(tol):>10} {'dPeakPB':>8} {'changed':>8}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for r in lr:
            print(f"  {r['group']:<34} {r['size']:>4} {r['dFrameF1']:>+9.4f} {r['dF1_B']:>+8.4f} "
                  f"{r['dIoU']:>+8.4f} {r['dBndF1']:>+10.4f} {r['dPeakPB']:>+8.4f} {r['changed']:>7.2%}")


# ==============================================================================
# 2. Gradient attribution around gold phrase starts
# ==============================================================================
def target_positions(gold, rng, margin):
    """Gold phrase starts, and the same number of control frames inside phrases >= margin from any boundary."""
    segs = extract_segments(gold)
    starts = np.array([s for s, _ in segs], dtype=np.int64)
    edges = np.array(sorted({s for s, _ in segs} | {e for _, e in segs}), dtype=np.int64)
    inside = np.flatnonzero(gold == I_TAG)
    if len(edges) and len(inside):
        dist = np.abs(inside[:, None] - edges[None, :]).min(1) if len(inside) * len(edges) < 5e7 else None
        if dist is None:  # very long video: compute distance chunk-wise
            dist = np.concatenate([np.abs(ch[:, None] - edges[None, :]).min(1)
                                   for ch in np.array_split(inside, max(1, len(inside) // 20000))])
        cand = inside[dist >= margin]
    else:
        cand = inside
    controls = rng.choice(cand, size=min(len(starts), len(cand)), replace=False) if len(cand) else np.array([], int)
    return starts, np.sort(controls)


def gradient_attribution(model, config, dataset, device, groups, args):
    W = int(config["window_size"])
    if int(config.get("temporal_downsample_factor", 1)) != 1:
        print("  (gradient attribution skipped: temporal_downsample_factor > 1 is not supported)")
        return None
    use_hamer = config.get("use_hamer_features", False)
    R = args.time_radius
    G = len(groups)
    rng = np.random.default_rng(0)
    acc = {"boundary": np.zeros((G, 2 * R + 1)), "control": np.zeros((G, 2 * R + 1))}
    cnt = {"boundary": np.zeros(2 * R + 1), "control": np.zeros(2 * R + 1)}
    n_targets = {"boundary": 0, "control": 0}
    vert_idx = [torch.tensor(g["vertices"], dtype=torch.long) for g in groups]
    dim_idx = [torch.tensor(g["hamer_dims"], dtype=torch.long) for g in groups]

    for vid, c in dataset.video_cache.items():
        gold = np.asarray(c["hard_labels"]).astype(np.int64)
        feats = c["features"].to(torch.float32)                       # (T, V, C)
        T = feats.shape[0]
        f_mean = feats.mean(0)                                         # (V, C)
        ham = c["hamer_features"].to(torch.float32) if use_hamer else None
        h_mean = ham.mean(0) if use_hamer else None
        starts, controls = target_positions(gold, rng, args.control_margin)
        if args.max_boundaries:
            starts = starts[:args.max_boundaries]
            controls = controls[:args.max_boundaries]

        for kind, positions in (("boundary", starts), ("control", controls)):
            for b0 in range(0, len(positions), args.grad_batch):
                pos = positions[b0:b0 + args.grad_batch]
                xs, bs_, hs, hbs, rel = [], [], [], [], []
                for p in pos:
                    s = int(min(max(p - W // 2, 0), max(T - W, 0)))
                    e = min(s + W, T)
                    x = torch.zeros(W, *feats.shape[1:]); x[:e - s] = feats[s:e]
                    xb = torch.zeros_like(x); xb[:e - s] = f_mean
                    xs.append(x.permute(2, 0, 1)); bs_.append(xb.permute(2, 0, 1))
                    if use_hamer:
                        h = torch.zeros(W, ham.shape[1]); h[:e - s] = ham[s:e]
                        hb = torch.zeros_like(h); hb[:e - s] = h_mean
                        hs.append(h.t()); hbs.append(hb.t())
                    rel.append(int(p - s))
                X = torch.stack(xs).to(device); XB = torch.stack(bs_).to(device)
                H = torch.stack(hs).to(device) if use_hamer else None
                HB = torch.stack(hbs).to(device) if use_hamer else None
                rel_t = torch.tensor(rel, device=device)
                steps = args.ig_steps if args.ig_steps > 0 else 1
                gx = torch.zeros_like(X); gh = torch.zeros_like(H) if use_hamer else None
                for k in range(steps):
                    alpha = (k + 1) / steps if args.ig_steps > 0 else 1.0
                    xi = (XB + alpha * (X - XB)).detach().requires_grad_(True)
                    hi = (HB + alpha * (H - HB)).detach().requires_grad_(True) if use_hamer else None
                    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
                        logits = call_model(model, config, xi, hi)          # (B, 3, W)
                        logp = torch.log_softmax(logits.float(), dim=1)
                        target = logp[torch.arange(len(rel)), B_TAG, rel_t].sum()
                        grads = torch.autograd.grad(target, [xi] + ([hi] if use_hamer else []))
                    gx += grads[0]
                    if use_hamer:
                        gh += grads[1]
                attr_x = ((X - XB) * gx / steps).sum(1)                   # (B, W, V): signed per vertex
                attr_h = ((H - HB) * gh / steps).transpose(1, 2) if use_hamer else None   # (B, W, D)

                per_group = []
                for gi in range(G):
                    v = torch.zeros(attr_x.shape[:2], device=device)
                    if len(vert_idx[gi]):
                        v = v + attr_x[:, :, vert_idx[gi].to(device)].sum(-1)
                    if len(dim_idx[gi]) and use_hamer:
                        v = v + attr_h[:, :, dim_idx[gi].to(device)].sum(-1)
                    per_group.append(v.abs())
                per_group = torch.stack(per_group, 1).cpu().numpy()      # (B, G, W)

                for bi, r0 in enumerate(rel):
                    lo, hi_ = max(0, r0 - R), min(W, r0 + R + 1)
                    a, b = lo - r0 + R, hi_ - r0 + R
                    acc[kind][:, a:b] += per_group[bi][:, lo:hi_]
                    cnt[kind][a:b] += 1
                n_targets[kind] += len(pos)
        print(f"\r  gradients: {n_targets['boundary']} boundaries, {n_targets['control']} controls", end="", flush=True)
    print()
    maps = {k: acc[k] / np.maximum(cnt[k], 1)[None, :] for k in acc}
    return maps, n_targets


def save_gradient_outputs(out_dir, run_name, groups, maps, n_targets, R):
    offsets = np.arange(-R, R + 1)
    names = [g["name"] for g in groups]
    with open(os.path.join(out_dir, "gradient_time_map.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "level", "group"] + [str(o) for o in offsets])
        for kind in ("boundary", "control"):
            for g, row in zip(groups, maps[kind]):
                w.writerow([kind, g["level"], g["name"]] + [f"{x:.6g}" for x in row])

    # summary: share of attribution per group (within its level), boundary vs control
    summary = []
    for level in ("coarse", "fine"):
        idx = [i for i, g in enumerate(groups) if g["level"] == level]
        if not idx:
            continue
        tot_b = maps["boundary"][idx].sum() or 1.0
        tot_c = maps["control"][idx].sum() or 1.0
        for i in idx:
            sb = maps["boundary"][i].sum() / tot_b
            sc = maps["control"][i].sum() / tot_c
            peak = offsets[int(np.argmax(maps["boundary"][i]))]
            summary.append({"level": level, "group": names[i], "share_boundary": sb, "share_control": sc,
                            "ratio": sb / sc if sc > 0 else float("nan"), "peak_offset": int(peak)})
    with open(os.path.join(out_dir, "gradient_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    print(f"  gradient attribution ({n_targets['boundary']} gold phrase starts, {n_targets['control']} control frames)")
    for level in ("coarse", "fine"):
        lr = sorted([s for s in summary if s["level"] == level], key=lambda s: -s["share_boundary"])
        if not lr:
            continue
        print(f"\n  [{level}] share of attribution for P(B) at boundaries vs at control frames")
        print(f"  {'group':<34} {'boundary':>9} {'control':>8} {'ratio':>6} {'peak @':>7}")
        for s in lr:
            print(f"  {s['group']:<34} {s['share_boundary']:>9.1%} {s['share_control']:>8.1%} "
                  f"{s['ratio']:>6.2f} {s['peak_offset']:>+7d}")

    coarse = [i for i, g in enumerate(groups) if g["level"] == "coarse"]
    fine = [i for i, g in enumerate(groups) if g["level"] == "fine"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 4 + 0.28 * max(len(fine), 1)),
                             gridspec_kw={"height_ratios": [1, max(1.5, 0.28 * len(fine))]})
    for i in coarse:
        axes[0].plot(offsets, maps["boundary"][i], label=names[i], linewidth=2)
    axes[0].axvline(0, color="black", linestyle=":", linewidth=1)
    axes[0].set_title(f"{run_name}: |attribution| of each group to P(B) at gold phrase starts (offset 0)")
    axes[0].set_xlabel("frames relative to gold phrase start")
    axes[0].set_ylabel("mean |attribution|")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    axes[0].grid(alpha=0.3)
    if fine:
        data = maps["boundary"][fine]
        im = axes[1].imshow(data / (data.max() or 1.0), aspect="auto", cmap="magma",
                            extent=[offsets[0] - 0.5, offsets[-1] + 0.5, len(fine) - 0.5, -0.5])
        axes[1].set_yticks(range(len(fine)))
        axes[1].set_yticklabels([names[i] for i in fine], fontsize=8)
        axes[1].axvline(0, color="white", linestyle=":", linewidth=1)
        axes[1].set_xlabel("frames relative to gold phrase start")
        fig.colorbar(im, ax=axes[1], label="relative |attribution|")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gradient_time_map.png"), dpi=150)
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Keypoint-level importance (occlusion + gradient attribution).")
    parser.add_argument("--model", help="Model basename (asks if omitted).")
    parser.add_argument("--prefixes", nargs="*", help="Only these prefixes (default: all).")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--labels-dir", default=None, help="Override phrase labels for all runs.")
    parser.add_argument("--no-occlusion", action="store_true")
    parser.add_argument("--no-gradient", action="store_true")
    parser.add_argument("--per-vertex", action="store_true",
                        help="Also occlude every single vertex (body/hands/face) one by one.")
    parser.add_argument("--tolerance", type=int, default=5, help="+-frames for boundary F1 (default 5).")
    parser.add_argument("--peak-window", type=int, default=5, help="+-frames for peak P(B) (default 5).")
    parser.add_argument("--time-radius", type=int, default=50, help="+-frames shown around boundaries (default 50).")
    parser.add_argument("--ig-steps", type=int, default=0,
                        help="Integrated-gradients steps (0 = plain gradient x (input - mean)).")
    parser.add_argument("--control-margin", type=int, default=25,
                        help="Control frames are >= this many frames from any phrase boundary.")
    parser.add_argument("--max-boundaries", type=int, default=0, help="Per video cap for speed (0 = all).")
    parser.add_argument("--grad-batch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size for occlusion.")
    args = parser.parse_args()

    ep.LABELS_OVERRIDE = args.labels_dir
    runs = ep.discover_runs(ep.EXP_DIR, ep.MODEL_DIR)
    if not runs:
        print("No runs found.")
        return
    model_name = args.model or ep.choose_model_interactively(runs)
    if model_name not in runs:
        print(f"No saved runs for '{model_name}'. Available: {sorted(runs)}")
        return
    selected = runs[model_name]
    if args.prefixes:
        wanted = {int(p) for p in args.prefixes}
        selected = [(p, r) for p, r in selected if int(p) in wanted]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _, run_name in selected:
        print(f"\n--- {run_name} ---")
        with open(os.path.join(ep.EXP_DIR, run_name, "hyperparameters.json")) as f:
            config = json.load(f)
        print(f"  {config.get('description', '')}")
        if "pure_hamer" in config.get("base_features", []):
            print("  skipped: pure_hamer runs are not supported")
            continue
        dataset = ep.get_dataset(config, args.split)
        model_class, kwargs = build_model_kwargs(config, detected_hamer_dim=dataset.detected_hamer_dim,
                                                 detected_dinov2_dim=dataset.detected_dinov2_dim)
        model = model_class(**kwargs).to(device)
        model.load_state_dict(torch.load(os.path.join(ep.MODEL_DIR, f"{run_name}.pth"), map_location=device),
                              strict=True)
        model.eval()

        hamer_dim = dataset.detected_hamer_dim if config.get("use_hamer_features", False) else None
        groups = build_groups(config, hamer_dim)
        if not groups:
            print("  no keypoint groups for this configuration")
            continue
        out_dir = os.path.join(OUT_ROOT, run_name)
        os.makedirs(out_dir, exist_ok=True)

        if not args.no_occlusion:
            vertex_groups = []
            if args.per_vertex:
                seen = sorted({v for g in groups for v in g["vertices"]})
                vertex_groups = [{"name": f"vertex {v}", "level": "vertex", "stream": "-",
                                  "vertices": [v], "hamer_dims": []} for v in seen]
            base, rows = occlusion(model, config, dataset, device, groups, args, vertex_groups)
            print_occlusion(base, rows, args.tolerance)
            with open(os.path.join(out_dir, "occlusion.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            with open(os.path.join(out_dir, "occlusion_base.json"), "w") as f:
                json.dump(base, f, indent=2)

        if not args.no_gradient:
            res = gradient_attribution(model, config, dataset, device, groups, args)
            if res is not None:
                maps, n_targets = res
                save_gradient_outputs(out_dir, run_name, groups, maps, n_targets, args.time_radius)
        print(f"  saved to {out_dir}/")


if __name__ == "__main__":
    main()