"""
visualize_face_subsets_mediapipe.py -- the 2023 paper's face subset (MediaPipe
FACEMESH_CONTOURS) next to our face subsets, all on the same MediaPipe face.

Why a separate script: FACEMESH_CONTOURS has 128 points -- lips (incl. the outer UPPER
lip), both eyes, both brows and the face OVAL -- but our saved face files only hold 83
points (no face oval, no outer upper lip), so it cannot be drawn from
processed_data/face_keypoints*. This script runs MediaPipe Holistic itself on frames of
a raw video, so all 468 face-mesh points are available.

The face is normalized like face v2 (origin between the inner eye corners 133 / 362,
eye line levelled, scaled by the outer-eye-corner distance) and is the median over the
sampled frames.

Outputs (evaluation_phrase/face_subsets/):
  face_subset_2023_contours.png        the 2023 subset alone (coloured by region)
  face_subsets_vs_2023.png             2023 contours + full / compact / eyes_brows / minimal
                                       (ring = point also in our 83 saved points)

Usage (repo root, needs mediapipe + opencv; with the Tasks API also the
face_landmarker.task model file, searched for in the repo or passed with --model):
  python visualize_face_subsets_mediapipe.py                       # first video in raw_data/videos
  python visualize_face_subsets_mediapipe.py --video raw_data/videos/1234_1a1.mp4 --frames 60
  python visualize_face_subsets_mediapipe.py --labels
"""
import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402

from src.face_subsets import (FACE_SUBSETS, SAVED_FACE_INDICES, LEFT_EYE_INDICES, RIGHT_EYE_INDICES,  # noqa: E402
                              LEFT_EYEBROW_INDICES, RIGHT_EYEBROW_INDICES, LIPS_INDICES,
                              ANCHOR_LEFT_EYE_INNER, ANCHOR_RIGHT_EYE_INNER, face_subset_raw_indices)
from src.graph import SkeletonGraph  # noqa: E402

# ---- MediaPipe connections: new Tasks API (mediapipe >= 0.10.x without `solutions`) or the
# legacy Solutions API. FACEMESH_CONTOURS (2023 paper) == FACE_LANDMARKS_CONTOURS (Tasks).
def _load_connections():
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C
        def e(conns):
            return [(c.start, c.end) for c in conns]
        return {"contours": e(C.FACE_LANDMARKS_CONTOURS), "lips": e(C.FACE_LANDMARKS_LIPS),
                "left_eye": e(C.FACE_LANDMARKS_LEFT_EYE), "right_eye": e(C.FACE_LANDMARKS_RIGHT_EYE),
                "left_brow": e(C.FACE_LANDMARKS_LEFT_EYEBROW), "right_brow": e(C.FACE_LANDMARKS_RIGHT_EYEBROW),
                "oval": e(C.FACE_LANDMARKS_FACE_OVAL)}, "tasks"
    except (ImportError, AttributeError):
        pass
    if hasattr(mp, "solutions"):
        F = mp.solutions.face_mesh_connections
        return {"contours": list(mp.solutions.holistic.FACEMESH_CONTOURS), "lips": list(F.FACEMESH_LIPS),
                "left_eye": list(F.FACEMESH_LEFT_EYE), "right_eye": list(F.FACEMESH_RIGHT_EYE),
                "left_brow": list(F.FACEMESH_LEFT_EYEBROW), "right_brow": list(F.FACEMESH_RIGHT_EYEBROW),
                "oval": list(F.FACEMESH_FACE_OVAL)}, "solutions"
    raise SystemExit("This mediapipe version has neither the Tasks face-landmark connections nor "
                     "mp.solutions -- please report `pip show mediapipe`.")


CONN, MP_API = _load_connections()
CONTOUR_EDGES = sorted({tuple(sorted(e)) for e in CONN["contours"]})
CONTOUR_POINTS = sorted({i for e in CONTOUR_EDGES for i in e})


def _pts(edges):
    return {i for e in edges for i in e}


# region colours shared by both kinds of subset
REGIONS = [
    ("eyes", _pts(CONN["left_eye"]) | _pts(CONN["right_eye"])
     | set(LEFT_EYE_INDICES) | set(RIGHT_EYE_INDICES), "#1f77b4"),
    ("brows", _pts(CONN["left_brow"]) | _pts(CONN["right_brow"])
     | set(LEFT_EYEBROW_INDICES) | set(RIGHT_EYEBROW_INDICES), "#17a589"),
    ("mouth", _pts(CONN["lips"]) | set(LIPS_INDICES), "#d6336c"),
    ("face oval", _pts(CONN["oval"]), "#e67e22"),
]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
OUR_ORDER = ["full", "compact", "eyes_brows", "minimal"]


def _find_model(path):
    cands = [path] if path else []
    cands += glob.glob("**/face_landmarker*.task", recursive=True)[:5]
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise SystemExit(f"Face Landmarker model not found. Download it (e.g. into models/):\n  {MODEL_URL}\n"
                     f"then pass --model models/face_landmarker.task")


def _make_detector(model_path):
    """Returns detect(rgb_uint8) -> (468+, 2) normalized x, y or None."""
    if MP_API == "tasks":
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.FaceLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=_find_model(model_path)),
                                            running_mode=vision.RunningMode.IMAGE, num_faces=1)
        lm = vision.FaceLandmarker.create_from_options(opts)

        def detect(rgb):
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks:
                return None
            return np.array([[p.x, p.y] for p in res.face_landmarks[0]])
        return detect, lm.close
    hol = mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1)

    def detect(rgb):
        res = hol.process(rgb)
        if res.face_landmarks is None:
            return None
        return np.array([[p.x, p.y] for p in res.face_landmarks.landmark])
    return detect, hol.close


def mean_mediapipe_face(video, n_frames, model_path=None):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    targets = set(np.linspace(0, max(total - 1, 0), n_frames).astype(int).tolist())
    detect, close = _make_detector(model_path)
    faces = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i not in targets:
            continue
        lm = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if lm is None:
            continue
        xy = lm[:468] * np.array([W, H])                                     # pixels, mesh points only
        origin = (xy[133] + xy[362]) / 2
        eye = xy[263] - xy[33]
        a = -np.arctan2(eye[1], eye[0])
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        faces.append(((xy - origin) @ R.T) / np.linalg.norm(eye))
    cap.release()
    close()
    if not faces:
        raise SystemExit(f"MediaPipe found no face in the sampled frames of {video}.")
    return np.median(np.stack(faces), axis=0), len(faces)


def draw(ax, face, title, points, edges, labels, mark_saved):
    ax.scatter(face[:, 0], face[:, 1], s=2, c="#e3e3e3", zorder=0)          # whole mesh
    for a, b in edges:
        ax.plot(*zip(face[a], face[b]), c="#888888", lw=0.7, zorder=1)
    pts = set(points)
    for name, members, color in REGIONS:
        sel = sorted(pts & members)
        if sel:
            ax.scatter(face[sel, 0], face[sel, 1], s=26, c=color, edgecolors="white",
                       linewidths=0.5, zorder=3, label=f"{name} ({len(sel)})")
    if mark_saved:
        sel = sorted(pts & set(SAVED_FACE_INDICES))
        ax.scatter(face[sel, 0], face[sel, 1], s=70, facecolors="none", edgecolors="black",
                   linewidths=0.6, zorder=4, label=f"also in our 83 ({len(sel)})")
    for r in (ANCHOR_LEFT_EYE_INNER, ANCHOR_RIGHT_EYE_INNER):
        if r in pts:
            ax.scatter(*face[r], s=150, facecolors="none", edgecolors="black", linewidths=1.3, zorder=5)
    if labels:
        for r in sorted(pts):
            ax.annotate(str(r), face[r], fontsize=5, xytext=(2, 2), textcoords="offset points", zorder=6)
    ax.set_title(f"{title}  ({len(pts)} points)", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=7.5, frameon=False)


def our_edges(subset):
    raw = face_subset_raw_indices(subset)
    g = SkeletonGraph(num_vertices=len(raw), face_only=True)
    return [(raw[a], raw[b]) for a, b in g._get_face_edges(offset=0)]


def main():
    p = argparse.ArgumentParser(description="2023 FACEMESH_CONTOURS subset vs our face subsets.")
    p.add_argument("--video", default=None, help="video file (default: first in raw_data/videos)")
    p.add_argument("--frames", type=int, default=40, help="frames to sample and take the median of")
    p.add_argument("--labels", action="store_true", help="annotate MediaPipe indices")
    p.add_argument("--model", default=None, help="face_landmarker.task (Tasks API; searched for if omitted)")
    p.add_argument("--out-dir", default=os.path.join("evaluation_phrase", "face_subsets"))
    args = p.parse_args()

    video = args.video or next(iter(sorted(glob.glob(os.path.join("raw_data", "videos", "*.mp4")))), None)
    if video is None:
        raise SystemExit("No video found -- pass --video.")
    face, n = mean_mediapipe_face(video, args.frames, args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    overlap = len(set(CONTOUR_POINTS) & set(SAVED_FACE_INDICES))
    print(f"[mediapipe {MP_API} API] FACEMESH_CONTOURS: {len(CONTOUR_POINTS)} points, {len(CONTOUR_EDGES)} edges; "
          f"{overlap} of them are among our 83 saved points")

    fig, ax = plt.subplots(figsize=(5.5, 6))
    draw(ax, face, "2023: FACEMESH_CONTOURS", CONTOUR_POINTS, CONTOUR_EDGES, args.labels, mark_saved=True)
    fig.savefig(os.path.join(args.out_dir, "face_subset_2023_contours.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    ours = [s for s in OUR_ORDER if s in FACE_SUBSETS]
    fig, axes = plt.subplots(1, 1 + len(ours), figsize=(4.6 * (1 + len(ours)), 6))
    draw(axes[0], face, "2023: FACEMESH_CONTOURS", CONTOUR_POINTS, CONTOUR_EDGES, args.labels, mark_saved=True)
    for ax, s in zip(axes[1:], ours):
        draw(ax, face, f"ours: {s}", face_subset_raw_indices(s), our_edges(s), args.labels, mark_saved=False)
    fig.suptitle(f"Face subsets on one MediaPipe face (median of {n} frames, face-local coordinates)",
                 fontsize=12)
    fig.savefig(os.path.join(args.out_dir, "face_subsets_vs_2023.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {args.out_dir}/face_subset_2023_contours.png, face_subsets_vs_2023.png")


if __name__ == "__main__":
    main()