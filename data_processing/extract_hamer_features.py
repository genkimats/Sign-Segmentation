"""
Stage 2 of 2 for HaMeR feature extraction.

Reads the per-frame hand boxes cached by extract_hand_boxes.py (stage 1) and
runs ONLY HaMeR here -- no `mediapipe` import in this process, to avoid a
known class of native-library (ABI) segfault from having MediaPipe and
PyTorch/HaMeR loaded in the same interpreter.

Videos are distributed across NUM_WORKERS worker processes. Each worker loads
its OWN copy of the HaMeR model once (via a Pool initializer), then reuses it
for every video that worker is assigned -- the model load itself is NOT
repeated per video.

IMPORTANT: this means peak GPU memory usage roughly doubles at NUM_WORKERS=2
(two independent model copies + two independent batches in flight). If you
hit an OOM, lower BATCH_SIZE first; drop NUM_WORKERS back to 1 if that's not
enough headroom, or if you're CPU-only and don't have spare cores to give it.

Run extract_hand_boxes.py FIRST, then this script as a separate invocation.
"""
import os
import pickle
import multiprocessing as mproc
import cv2
import torch
import numpy as np
from tqdm import tqdm

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to

INPUT_VIDEO_DIR = os.path.expanduser("./raw_data/videos")
INPUT_BOX_DIR = os.path.expanduser("./processed_data/hand_boxes")
OUTPUT_FEATURE_DIR = os.path.expanduser("./processed_data/hamer_features")
os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# HaMeR resolves its checkpoint/config via a path relative to ITS OWN repo root
# (e.g. "_DATA/hamer_ckpts/model_config.yaml"), regardless of where you invoke
# this script from. Point this at wherever you cloned/installed geopavlakos/hamer.
HAMER_REPO_DIR = os.path.expanduser("~/Genki_GR/hamer")

BATCH_SIZE = 32
RESCALE_FACTOR = 2.0  # matches HaMeR's own ViTDetDataset default context padding

# Number of videos processed in parallel, each with its OWN loaded HaMeR model
# (see the module docstring for the GPU-memory tradeoff this implies).
NUM_WORKERS = 4

# HaMeR always regresses in a canonical RIGHT-hand frame; left-hand crops are
# mirrored on the way in (handled by ViTDetDataset via the `right` flag passed
# through the cached boxes), so predicted rotation matrices must be mirrored
# back via reflection conjugation: R_true = M @ R_predicted @ M.
MIRROR_AXIS = 0

# Per-process globals -- set exactly once per worker by _init_worker(), then
# reused by process_one_video() for every video that worker handles.
_model = None
_model_cfg = None


def unmirror_rotation_batch(R: np.ndarray, mirror_axis: int = MIRROR_AXIS) -> np.ndarray:
    """R: array of shape (..., 3, 3)."""
    M = np.eye(3, dtype=R.dtype)
    M[mirror_axis, mirror_axis] = -1.0
    return np.einsum('ij,...jk,kl->...il', M, R, M)


def _init_worker():
    """
    Runs ONCE per worker process, right after it's spawned (see main()'s
    Pool(..., initializer=_init_worker)). Loading the model here rather than
    inside process_one_video() means each worker pays this cost exactly once,
    not once per video.
    """
    global _model, _model_cfg
    print(f"[worker pid={os.getpid()}] Loading HaMeR model on {DEVICE}...")
    _prev_cwd = os.getcwd()
    os.chdir(HAMER_REPO_DIR)
    try:
        model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    finally:
        os.chdir(_prev_cwd)
    model = model.to(DEVICE)
    model.eval()
    _model = model
    _model_cfg = model_cfg


def process_one_video(video_name):
    """
    Runs in a worker process, using that worker's already-loaded _model/_model_cfg
    (set once by _init_worker()). Returns (video_name, message_or_None) so the
    parent can report skips/warnings without multiple processes' prints
    colliding in the terminal.
    """
    model, model_cfg = _model, _model_cfg

    video_path = os.path.join(INPUT_VIDEO_DIR, video_name)
    stem = video_name.rsplit('.', 1)[0]

    box_path = os.path.join(INPUT_BOX_DIR, stem + "_boxes.pkl")
    save_path = os.path.join(OUTPUT_FEATURE_DIR, stem + "_hamer.pt")
    if os.path.exists(save_path):
        return video_name, None
    if not os.path.exists(box_path):
        return video_name, (f"Skipping {video_name}: no cached boxes at {box_path} "
                             f"(run extract_hand_boxes.py first)")

    with open(box_path, "rb") as f:
        cached = pickle.load(f)
    downsample_factor = cached["temporal_downsample_factor"]
    frame_records = cached["frames"]

    cap = cv2.VideoCapture(video_path)

    # Index 0 = left hand, index 1 = right hand.
    video_hand_pose = []
    video_global_orient = []
    kept_source_frame_idx = []

    hand_queue = []
    meta_queue = []

    def flush_queue():
        if not hand_queue:
            return
        batch = torch.utils.data.dataloader.default_collate(hand_queue)
        batch = recursive_to(batch, DEVICE)
        with torch.no_grad():
            out = model(batch)

        # First run: sanity-check these before trusting the full extraction:
        #   print(out['pred_mano_params'].keys())
        #   print(out['pred_mano_params']['hand_pose'].shape)      # expect (B, 15, 3, 3)
        #   print(out['pred_mano_params']['global_orient'].shape)  # expect (B, 1, 3, 3) or (B, 3, 3)
        mano = out['pred_mano_params']
        hand_pose = mano['hand_pose'].detach().cpu().numpy()
        global_orient = mano['global_orient'].detach().cpu().numpy()
        if global_orient.ndim == 4:
            global_orient = global_orient[:, 0]

        for i, (slot_idx, is_right) in enumerate(meta_queue):
            hp = hand_pose[i].copy()
            go = global_orient[i].copy()
            if not is_right:
                hp = unmirror_rotation_batch(hp)
                go = unmirror_rotation_batch(go)
            slot = 1 if is_right else 0
            video_hand_pose[slot_idx][slot] = hp
            video_global_orient[slot_idx][slot] = go

        hand_queue.clear()
        meta_queue.clear()

    raw_frame_idx = 0
    record_ptr = 0
    while cap.isOpened() and record_ptr < len(frame_records):
        ret, frame = cap.read()
        if not ret:
            break

        record = frame_records[record_ptr]
        if raw_frame_idx != record["source_frame_idx"]:
            # This frame was dropped by the downsample stride in stage 1; skip it.
            raw_frame_idx += 1
            continue

        slot_idx = len(video_hand_pose)
        video_hand_pose.append(np.zeros((2, 15, 3, 3), dtype=np.float32))
        video_global_orient.append(np.zeros((2, 3, 3), dtype=np.float32))
        kept_source_frame_idx.append(raw_frame_idx)

        boxes, rights = record["boxes"], record["rights"]
        if boxes:
            boxes_arr = np.array(boxes, dtype=np.float32)
            rights_arr = np.array(rights, dtype=np.float32)

            frame_ds = ViTDetDataset(model_cfg, frame, boxes_arr, rights_arr,
                                      rescale_factor=RESCALE_FACTOR)
            for i in range(len(frame_ds)):
                hand_queue.append(frame_ds[i])
                meta_queue.append((slot_idx, bool(rights_arr[i])))

            if len(hand_queue) >= BATCH_SIZE:
                flush_queue()

        record_ptr += 1
        raw_frame_idx += 1

    cap.release()
    flush_queue()

    torch.save({
        "hand_pose": np.array(video_hand_pose, dtype=np.float32),          # (T', 2, 15, 3, 3)
        "global_orient": np.array(video_global_orient, dtype=np.float32),  # (T', 2, 3, 3)
        "hand_order": "index 0 = left, index 1 = right",
        "temporal_downsample_factor": downsample_factor,
        "source_frame_indices": np.array(kept_source_frame_idx, dtype=np.int64),
    }, save_path)

    return video_name, None


def main():
    all_video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Only extract videos whose output doesn't already exist. Scanning the output
    # directory ONCE up front (by id, i.e. video stem) rather than relying solely
    # on the per-video check inside process_one_video means an already-done video
    # is never dispatched to a worker -- important here specifically, since a
    # trivially-skipped video would still have paid for a loaded HaMeR model
    # sitting idle in that worker if we didn't filter first.
    existing_ids = {
        f[:-len("_hamer.pt")] for f in os.listdir(OUTPUT_FEATURE_DIR) if f.endswith("_hamer.pt")
    }
    video_files = [f for f in all_video_files if f.rsplit('.', 1)[0] not in existing_ids]

    skipped = len(all_video_files) - len(video_files)
    print(f"Found {len(all_video_files)} videos, {skipped} already extracted -- "
          f"processing the remaining {len(video_files)}.")

    if not video_files:
        print("Nothing to do.")
        return

    # 'spawn', not the Linux default 'fork': PyTorch/CUDA explicitly does not
    # support being forked once a CUDA context may have been touched -- 'spawn'
    # gives each worker a genuinely fresh interpreter/CUDA context instead of
    # inheriting the parent's, which is the documented-safe way to combine
    # multiprocessing with CUDA.
    ctx = mproc.get_context('spawn')
    with ctx.Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
        for video_name, message in tqdm(
            pool.imap_unordered(process_one_video, video_files),
            total=len(video_files),
            desc=f"Extracting HaMeR features ({NUM_WORKERS} workers)",
        ):
            if message:
                tqdm.write(message)


if __name__ == "__main__":
    main()