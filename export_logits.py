"""
Freeze a trained encoder and export its per-frame B/I/O logits for the decoder
study. Run from anywhere:

    python export_logits.py --run stgcn_bilstm-08 --splits val test

--run is the exact run name you trained under (experiments/<run>/hyperparameters.json
and saved_models/<run>.pth must exist). Output goes to
decoder_study/exports/<run>/<split>.npz (+ <split>_meta.json).

Design notes
  * Inference is CHUNKED at the encoder's TRAINED window size (the regime this
    project's inference-scaling experiment found safe for every architecture),
    on whole videos, with overlapping windows averaged -- not concatenated.
  * Gold labels are read from processed_data/BIO_tags/*.npy DIRECTLY. The
    dataset's smoothed labels are never used: argmax of them dilates Begin.
  * Default splits are val+test. Exporting train is possible (--splits train)
    but train logits are IN-SAMPLE for the encoder (overconfident), so do not
    fit thresholds/calibration on them; the duration prior only needs gold
    labels and is fit without any encoder outputs.
  * Hidden features are not exported yet (decoder 2, boundary regression, is
    the first one that needs them; train-split hidden features for ~600 long
    videos are tens of GB, so that is a deliberate later step).
"""
import os
import sys
import json
import argparse
import inspect
import numpy as np

from study_common import (PROJECT_ROOT, LABELS_DIR, SPLIT_FILE, plan_windows,
                          stitch_logits, save_export)

BASENAME_TO_CLASS = {
    "pure_mamba": "PureMambaBaseline", "bi_mamba": "BiMambaBaseline",
    "stgcn_mamba": "STGCN_Mamba", "stgcn_mlp_mamba": "STGCN_MLP_Mamba",
    "stgcn_bimamba": "STGCN_BiMamba", "decoupled_stgcn_mamba": "Decoupled_STGCN_Mamba",
    "bilstm_baseline": "BiLSTM_Baseline", "stgcn_bilstm": "STGCN_BiLSTM",
    "transformer_baseline": "TransformerBaseline", "stgcn_transformer": "STGCN_Transformer",
    "latent_stgcn_mamba": "Latent_STGCN_Mamba", "ctrgcn_mamba": "CTRGCN_Mamba",
    "infogcn_mamba": "InfoGCN_Mamba", "shiftgcn_mamba": "ShiftGCN_Mamba",
    "spatial_transformer_mamba": "SpatialTransformer_Mamba", "hdgcn_mamba": "HDGCN_Mamba",
    "hypersign_mamba": "HyperSign_Mamba", "stgcn_hybrid_seq": "STGCN_HybridSequential",
    "stgcn_hybrid_parallel": "STGCN_HybridParallel", "mlpaux_mamba": "MLPAux_Mamba",
    "mlpaux_bimamba": "MLPAux_BiMamba", "mlpaux_bilstm": "MLPAux_BiLSTM",
    "mlpaux_transformer": "MLPAux_Transformer", "handson_2025": "HandsOn2025",
}


# --------------------------------------------------- pure numpy (unit-tested) --
def export_video_logits(features, extras, window, stride, batch_windows, run_batch):
    """features: (C, T, V) numpy. extras: dict name -> (D, T) numpy (hamer, dinov2).
    run_batch(feat_batch (n,C,W,V), extra_batches {name: (n,D,W)}) -> (n, 3, W).
    Short windows are zero-padded to `window` (as src/dataset.py does) and the
    padded outputs are cut before stitching. Returns (T, 3) float32."""
    C, T, V = features.shape
    wins = plan_windows(T, window, stride)
    outs = []
    for k in range(0, len(wins), batch_windows):
        chunk = wins[k:k + batch_windows]
        fb = np.zeros((len(chunk), C, window, V), dtype=np.float32)
        eb = {n: np.zeros((len(chunk), a.shape[0], window), dtype=np.float32) for n, a in extras.items()}
        for i, (s, e) in enumerate(chunk):
            fb[i, :, :e - s] = features[:, s:e]
            for n, a in extras.items():
                eb[n][i, :, :e - s] = a[:, s:e]
        lg = run_batch(fb, eb)
        for i, (s, e) in enumerate(chunk):
            outs.append(lg[i][:, :e - s])
    return stitch_logits(T, wins, outs)


# ------------------------------------------------------------- torch-side code --
def infer_in_dim(state_dict, tag):
    for k, v in state_dict.items():
        if tag in k and k.endswith("0.weight") and v.ndim == 2:
            return int(v.shape[1]), int(v.shape[0])
    return None, None


def load_custom_model_class(hp, project_root=PROJECT_ROOT):
    """Models that live outside src/models.py (e.g. novelty_crf/) record where
    they come from in hyperparameters.json: model_module_dir (relative to the
    project root), model_module, model_class and model_kwargs. Importing them by
    that recipe means the exporter never needs a hard-coded name registry."""
    import importlib
    mod_dir = os.path.join(project_root, hp["model_module_dir"])
    if mod_dir not in sys.path:
        sys.path.insert(0, mod_dir)
    return getattr(importlib.import_module(hp["model_module"]), hp["model_class"])


def build_model(hp, state_dict, device):
    if hp.get("model_module"):
        cls = load_custom_model_class(hp)
        kw = dict(hp["model_kwargs"])
        model = cls(**kw).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, kw
    import src.models as models
    name = hp["basename"]
    if name not in BASENAME_TO_CLASS:
        raise ValueError(f"basename '{name}' not in BASENAME_TO_CLASS -- add it (mirror train.py's MODEL_REGISTRY).")
    cls = getattr(models, BASENAME_TO_CLASS[name])
    d_model = hp["d_model"]
    kw = {"in_channels": hp["in_channels"], "num_vertices": hp["num_vertices"], "num_classes": 3,
          "d_model": d_model, "n_layers": hp["n_layers"], "nhead": hp.get("nhead", 8),
          "dim_feedforward": hp.get("dim_feedforward", d_model * 4),
          "mlp_expansion_factor": hp.get("mlp_expansion_factor", 4), "latent_dim": hp.get("latent_dim", 128),
          "mamba_d_state": hp.get("mamba_d_state", 16), "mamba_d_conv": hp.get("mamba_d_conv", 4),
          "mamba_expand": hp.get("mamba_expand", 2),
          "adapter_dim": hp.get("adapter_dim", 512), "adapter_hidden": hp.get("adapter_hidden"),
          "mixer_hidden": hp.get("mixer_hidden", 512), "downsample": hp.get("downsample", 2),
          "pose_stream": hp.get("pose_stream", "angles"), "angle_y_scale": hp.get("angle_y_scale", 1.0),
          "ctc_num_tokens": hp.get("ctc_num_tokens", 1), "norm_first": hp.get("norm_first", True)}
    hd, hp_proj = infer_in_dim(state_dict, "hamer")
    if hd is not None:
        kw["hamer_dim"], kw["hamer_proj_dim"] = hd, hp_proj
    dd, dp_proj = infer_in_dim(state_dict, "dinov2")
    if dd is not None:
        kw["dinov2_dim"], kw["dinov2_proj_dim"] = dd, dp_proj
    accepted = inspect.signature(cls.__init__).parameters
    kw = {k: v for k, v in kw.items() if k in accepted}
    model = cls(**kw).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, kw


def get_fps(vid):
    """Best-effort frame rate from the raw video (answers the brief's open
    question about our feature frame rate). None if the video/cv2 is missing."""
    try:
        import cv2
    except Exception:
        return None
    base = vid.rsplit("_", 1)[0]
    for suffix in ("_1a1", "_1b1"):
        for ext in (".mp4", ".avi", ".mov"):
            p = os.path.join(PROJECT_ROOT, "raw_data", "videos", f"{base}{suffix}{ext}")
            if os.path.exists(p):
                cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
                if fps and fps > 0:
                    return float(fps)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. stgcn_bilstm-08")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--stride", type=int, default=None, help="window stride (default: window size, no overlap)")
    ap.add_argument("--batch-windows", type=int, default=64)
    ap.add_argument("--root", default=None, help="directory holding experiments/<run> and saved_models/<run>.pth, "
                    "relative to the project root or absolute (default: the project root itself; use "
                    "--root novelty_crf for runs trained by novelty_crf/train_novelty.py)")
    args = ap.parse_args()

    # src/dataset.py resolves its default feature directories relative to the
    # CURRENT directory, so run from the project root regardless of where this
    # script was launched.
    os.chdir(PROJECT_ROOT)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    import torch
    from src.dataset import SignSegmentationDataset

    root = PROJECT_ROOT if args.root is None else (args.root if os.path.isabs(args.root) else os.path.join(PROJECT_ROOT, args.root))
    hp_path = os.path.join(root, "experiments", args.run, "hyperparameters.json")
    ckpt = os.path.join(root, "saved_models", f"{args.run}.pth")
    hp = json.load(open(hp_path))
    if hp.get("temporal_downsample_factor", 1) != 1:
        raise NotImplementedError("temporal_downsample_factor != 1 is not supported by the exporter yet.")
    if hp.get("use_face_keypoints", False):
        raise NotImplementedError("use_face_keypoints=True is not supported by the exporter yet.")
    window = hp["window_size"]
    stride = args.stride or window
    tol = hp.get("tolerance_window", 5)
    use_h, use_d = hp.get("use_hamer_features", False), hp.get("use_dinov2_features", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device)
    model, model_kw = build_model(hp, state, device)
    print(f"Loaded {args.run}: {hp['basename']}  window={window} stride={stride}  model kwargs={model_kw}")
    extra_meta = model.export_extra() if hasattr(model, "export_extra") else {}   # e.g. learned CRF transitions
    if extra_meta:
        print(f"  exporting extra metadata from the model: {sorted(extra_meta)}")

    def run_batch(fb, eb):
        f = torch.nan_to_num(torch.from_numpy(fb).to(device), nan=0.0, posinf=0.0, neginf=0.0)
        kwargs = {}
        if use_h: kwargs["hamer"] = torch.nan_to_num(torch.from_numpy(eb["hamer"]).to(device), nan=0.0, posinf=0.0, neginf=0.0)
        if use_d: kwargs["dinov2"] = torch.nan_to_num(torch.from_numpy(eb["dinov2"]).to(device), nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            out = model(f, **kwargs)
        logits = out[0] if isinstance(out, tuple) else out
        return logits.float().cpu().numpy()

    for split in args.splits:
        ds = SignSegmentationDataset(
            keypoints_dir="processed_data/keypoints", labels_dir=LABELS_DIR, split_file=SPLIT_FILE,
            split=split, window_size=window, overlap=0, tolerance_window=tol, use_full_length=True,
            base_features=hp["base_features"], kinematic_features=hp["kinematic_features"],
            temporal_downsample_factor=1, use_hamer_features=use_h,
            hamer_dir=hp.get("hamer_dir", "processed_data/hamer_features"),
            use_dinov2_features=use_d, dinov2_dir=hp.get("dinov2_dir", "processed_data/dinov2_features"))
        records, fps_by_vid = {}, {}
        for idx in range(len(ds)):
            item = ds[idx]
            vid = ds.samples[idx]["video_id"]
            feats = item[0].numpy()
            extras, pos = {}, 2
            if use_h: extras["hamer"] = item[pos].numpy(); pos += 1
            if use_d: extras["dinov2"] = item[pos].numpy(); pos += 1
            raw = np.load(os.path.join(LABELS_DIR, f"{vid}.npy")).astype(np.int8)
            if len(raw) != feats.shape[1]:
                print(f"  skip {vid}: {len(raw)} labels vs {feats.shape[1]} frames"); continue
            logits = export_video_logits(feats, extras, window, stride, args.batch_windows, run_batch)
            records[vid] = {"logits": logits, "labels": raw}
            fps_by_vid[vid] = get_fps(vid)
            print(f"  [{split}] {vid}: T={len(raw)}")
        fps_vals = sorted({round(v, 2) for v in fps_by_vid.values() if v})
        meta = {"run": args.run, "basename": hp["basename"], "window": window, "stride": stride,
                "tolerance_window": tol, "class_weights": hp.get("class_weights"), "fps_by_vid": fps_by_vid,
                "n_videos": len(records), "hyperparameters": hp, **extra_meta}
        save_export(args.run, split, records, meta)
        print(f"Saved {len(records)} videos for split '{split}'. Distinct fps seen: {fps_vals or 'unknown'}")


if __name__ == "__main__":
    main()