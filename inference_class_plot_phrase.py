"""
inference_class_plot_phrase.py -- interactive BIO + confidence viewer for PHRASE models.

Same idea as inference_class_plot.py (gloss), adapted to the phrase pipeline:
  - Reads runs from experiments_phrase/ and saved_models_phrase/, labels from
    processed_data/BIO_tags_phrase/.
  - Builds the model with the SAME factory as train_phrase.py / evaluate_phrase.py
    (all hyperparameters come from the run's hyperparameters.json; strict loading).
  - Predicts each FULL video exactly like evaluate_phrase.py (training windows,
    overlapping windows averaged), so what you see is what the metrics measure.
  - Ground truth = raw HARD phrase labels (tolerance_window never changes it).
  - Bottom plot shows the decoded prediction: argmax, or the 2023-style threshold
    decoder (press 't' to switch).

Controls:
  Right / Left : next / previous display window (moves on to the next video at the end)
  Up / Down    : next / previous video
  t            : toggle decoder (argmax <-> threshold)
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import SignSegmentationDataset
from src.model_factory import build_model_kwargs
from src.evaluation import predict_video_probs
from src.metrics import (extract_segments, decode_threshold_2023, segment_iou,
                         segment_percentage)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
CHOSEN_MODEL = "stgcn_bilstm"
PREFIX = "10"
TARGET_SPLIT = "val"            # "train", "val", "test" or "all"

DISPLAY_WINDOW = 512            # frames shown per page (display only; inference always
                                # uses the run's trained window_size on the full video)
DECODER = "argmax"              # "argmax" or "threshold" (toggle with 't')
B_THRESHOLD = 0.5               # 2023-style decoder thresholds
O_THRESHOLD = 0.5

EXP_DIR = "experiments_phrase"
MODEL_DIR = "saved_models_phrase"
KEYPOINTS_DIR = "processed_data/keypoints"
LABELS_DIR = "processed_data/BIO_tags_phrase"
SPLIT_FILE = "dataset_splits.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_NAME = f"{CHOSEN_MODEL}-{int(PREFIX):02d}" if PREFIX.isdigit() else f"{CHOSEN_MODEL}-{PREFIX}"
WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{RUN_NAME}.pth")
HYPERPARAMETER_PATH = os.path.join(EXP_DIR, RUN_NAME, "hyperparameters.json")


# ==============================================================================
# Helpers
# ==============================================================================
def segments_to_bio(segments, length):
    """Turns decoded segments back into a per-frame BIO sequence for plotting."""
    bio = np.zeros(length, dtype=np.int64)
    for s, e in segments:
        s, e = max(0, s), min(length - 1, e)
        if s > e:
            continue
        bio[s:e + 1] = 1
        bio[s] = 2
    return bio


def build_dataset(hp, split):
    return SignSegmentationDataset(
        keypoints_dir=KEYPOINTS_DIR,
        labels_dir=LABELS_DIR,
        split_file=SPLIT_FILE,
        split=split,
        window_size=hp["window_size"],
        overlap=hp.get("overlap", 0),
        tolerance_window=1,  # ground truth is always the hard labels
        use_full_length=False,
        base_features=hp["base_features"],
        kinematic_features=hp.get("kinematic_features", []),
        temporal_downsample_factor=hp.get("temporal_downsample_factor", 1),
        use_face_keypoints=hp.get("use_face_keypoints", False),
        face_dir=hp.get("face_dir", "processed_data/face_keypoints_normalized"),
        face_subset=hp.get("face_subset", "full"),
        use_hamer_features=hp.get("use_hamer_features", False),
        hamer_dir=hp.get("hamer_dir", "processed_data/hamer_features"),
        use_dinov2_features=hp.get("use_dinov2_features", False),
        dinov2_dir=hp.get("dinov2_dir", "processed_data/dinov2_features"),
    )


def print_skipped(dataset, split):
    skipped = getattr(dataset, "skipped_videos", {}) or {}
    total = sum(len(v) for v in skipped.values())
    if total == 0:
        print(f"[{split}] all {len(dataset.video_cache)} videos loaded.")
        return
    print(f"[{split}] ⚠️ {total} video(s) not loaded (not shown):")
    for reason, items in skipped.items():
        for vid, detail in items:
            print(f"    {reason}: {vid}" + (f" ({detail})" if detail else ""))


# ==============================================================================
# 🖼️ VIEWER
# ==============================================================================
class PhraseViewer:
    def __init__(self, model, hp, videos):
        """videos: list of (split, vid, cached_entry)."""
        self.model = model
        self.hp = hp
        self.videos = videos
        self.decoder = DECODER
        self.video_idx = 0
        self.page = 0
        self._cache = {}  # vid -> (probs (3,T), gold (T,))

        self.fig, (self.ax1, self.ax2) = plt.subplots(
            2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
        # Matplotlib only keeps a WEAK reference to bound-method callbacks, so the
        # viewer must stay referenced or its key handler silently disappears.
        # Attaching it to the figure keeps it alive as long as the window is open.
        self.fig._phrase_viewer = self
        self._cid = self.fig.canvas.mpl_connect("key_press_event", self.on_press)
        self.fig.canvas.manager.set_window_title(
            f"Phrase Confidence Viewer - {RUN_NAME} [{TARGET_SPLIT.upper()}]")
        self.draw()

    # --- data -----------------------------------------------------------------
    def video_data(self, idx):
        split, vid, cached = self.videos[idx]
        if vid not in self._cache:
            probs = predict_video_probs(self.model, self.hp, cached, DEVICE,
                                        batch_size=self.hp.get("batch_size", 16))
            gold = np.asarray(cached["hard_labels"]).astype(np.int64)
            T = min(len(gold), probs.shape[1])
            self._cache[vid] = (probs[:, :T], gold[:T])
        return split, vid, self._cache[vid]

    def num_pages(self, idx):
        _, _, (probs, _) = self.video_data(idx)
        return max(1, int(np.ceil(probs.shape[1] / DISPLAY_WINDOW)))

    def decode(self, probs):
        T = probs.shape[1]
        if self.decoder == "argmax":
            argmax = probs.argmax(axis=0)
            return argmax, extract_segments(argmax)
        segments = decode_threshold_2023(probs, B_THRESHOLD, O_THRESHOLD)
        return segments_to_bio(segments, T), segments

    # --- drawing --------------------------------------------------------------
    def draw(self):
        split, vid, (probs, gold) = self.video_data(self.video_idx)
        T = probs.shape[1]
        pred_bio, pred_segments = self.decode(probs)
        gold_segments = extract_segments(gold)

        # Whole-video numbers (same definitions as evaluate_phrase.py)
        iou = segment_iou(pred_segments, gold_segments, T)
        pct = segment_percentage(pred_segments, gold_segments)

        n_pages = self.num_pages(self.video_idx)
        self.page = min(self.page, n_pages - 1)
        s = self.page * DISPLAY_WINDOW
        e = min(T, s + DISPLAY_WINDOW)
        x = np.arange(s, e)
        g = gold[s:e]
        p = probs[:, s:e]
        d = pred_bio[s:e]

        self.ax1.clear()
        self.ax2.clear()

        # TOP: probabilities over shaded ground truth
        self.ax1.fill_between(x, 0, 1.05, where=(g == 0), color="gray", alpha=0.15, label="GT: Outside", step="post")
        self.ax1.fill_between(x, 0, 1.05, where=(g == 1), color="blue", alpha=0.10, label="GT: Inside", step="post")
        self.ax1.fill_between(x, 0, 1.05, where=(g == 2), color="red", alpha=0.35, label="GT: Begin", step="post")
        self.ax1.plot(x, p[0], color="black", linewidth=2, label="P(Outside)")
        self.ax1.plot(x, p[1], color="dodgerblue", linewidth=2, label="P(Inside)")
        self.ax1.plot(x, p[2], color="red", linewidth=2.5, label="P(Begin)")
        if self.decoder == "threshold":
            self.ax1.axhline(B_THRESHOLD, color="red", linestyle=":", linewidth=1, label=f"B thr {B_THRESHOLD}")
            self.ax1.axhline(O_THRESHOLD, color="black", linestyle=":", linewidth=1, label=f"O thr {O_THRESHOLD}")
        self.ax1.set_ylim(-0.05, 1.05)
        self.ax1.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        self.ax1.set_ylabel("Confidence", fontsize=12, fontweight="bold")
        self.ax1.grid(True, linestyle="--", alpha=0.5)
        self.ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        dec_str = "argmax" if self.decoder == "argmax" else f"threshold b={B_THRESHOLD} o={O_THRESHOLD}"
        pct_str = f"{pct:.2f}" if not np.isnan(pct) else "n/a"
        title = (f"{RUN_NAME} | File: {vid} ({split}) | Frames {s}-{e - 1} of {T} | "
                 f"trained window {self.hp['window_size']}\n"
                 f"Whole video [{dec_str}]: gold segs {len(gold_segments)}, pred segs {len(pred_segments)}, "
                 f"% (ratio) {pct_str}, IoU {iou:.3f}\n"
                 f"Video {self.video_idx + 1}/{len(self.videos)}, page {self.page + 1}/{n_pages} "
                 f"(Left/Right: page, Up/Down: video, t: toggle decoder)")
        self.ax1.set_title(title, fontsize=11, fontweight="bold")

        # BOTTOM: hard prediction vs ground truth
        self.ax2.step(x, g, label="Ground Truth", color="gold", linestyle="--", alpha=0.8, linewidth=4, where="post")
        self.ax2.step(x, d + 0.05, label=f"Prediction ({self.decoder})", color="dodgerblue",
                      linestyle="-", linewidth=2.5, where="post")
        self.ax2.set_yticks([0, 1, 2])
        self.ax2.set_yticklabels(["Outside (0)", "Inside (1)", "Begin (2)"])
        self.ax2.set_ylim(-0.2, 2.2)
        self.ax2.set_ylabel("Class", fontsize=12, fontweight="bold")
        self.ax2.set_xlabel("Frame (raw frame rate)", fontsize=12, fontweight="bold")
        self.ax2.grid(True, linestyle="--", alpha=0.5)
        self.ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.)

        plt.tight_layout()
        self.fig.canvas.draw_idle()

    # --- navigation -----------------------------------------------------------
    def on_press(self, event):
        if event.key == "right":
            if self.page < self.num_pages(self.video_idx) - 1:
                self.page += 1
            elif self.video_idx < len(self.videos) - 1:
                self.video_idx += 1
                self.page = 0
            else:
                print("Already at the end.")
                return
        elif event.key == "left":
            if self.page > 0:
                self.page -= 1
            elif self.video_idx > 0:
                self.video_idx -= 1
                self.page = self.num_pages(self.video_idx) - 1
            else:
                print("Already at the beginning.")
                return
        elif event.key == "up":
            if self.video_idx >= len(self.videos) - 1:
                print("Already at the last video.")
                return
            self.video_idx += 1
            self.page = 0
        elif event.key == "down":
            if self.video_idx == 0:
                print("Already at the first video.")
                return
            self.video_idx -= 1
            self.page = 0
        elif event.key == "t":
            self.decoder = "threshold" if self.decoder == "argmax" else "argmax"
        else:
            return
        self.draw()


# ==============================================================================
# MAIN
# ==============================================================================
def free_default_keys():
    """
    Matplotlib's toolbar binds Left/Right to view back/forward by default. Remove
    those bindings so the arrow keys only do this viewer's navigation.
    """
    for name in ("keymap.back", "keymap.forward"):
        plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in ("left", "right")]


def main():
    free_default_keys()
    if not os.path.exists(HYPERPARAMETER_PATH):
        raise FileNotFoundError(f"Could not find {HYPERPARAMETER_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Could not find {WEIGHTS_PATH}")

    with open(HYPERPARAMETER_PATH) as f:
        hp = json.load(f)
    print(f"Loaded hyperparameters for {RUN_NAME}: window={hp['window_size']} "
          f"overlap={hp.get('overlap', 0)} d_model={hp['d_model']} n_layers={hp['n_layers']} "
          f"(trained tolerance_window={hp.get('tolerance_window')}, ignored for the ground truth)")

    splits = ["train", "val", "test"] if TARGET_SPLIT == "all" else [TARGET_SPLIT]
    videos = []
    detected_hamer = detected_dinov2 = None
    for split in splits:
        ds = build_dataset(hp, split)
        print_skipped(ds, split)
        detected_hamer = detected_hamer or ds.detected_hamer_dim
        detected_dinov2 = detected_dinov2 or ds.detected_dinov2_dim
        videos += [(split, vid, cached) for vid, cached in ds.video_cache.items()]
    if not videos:
        raise ValueError(f"No videos loaded for split '{TARGET_SPLIT}'.")

    model_class, model_kwargs = build_model_kwargs(
        hp, detected_hamer_dim=detected_hamer, detected_dinov2_dim=detected_dinov2)
    model = model_class(**model_kwargs).to(DEVICE)
    print(f"Loading weights from {WEIGHTS_PATH}...")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE), strict=True)
    model.eval()

    viewer = PhraseViewer(model, hp, videos)  # keep a reference (see PhraseViewer.__init__)
    plt.show()


if __name__ == "__main__":
    main()