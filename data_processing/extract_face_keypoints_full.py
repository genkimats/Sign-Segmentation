"""
extract_face_keypoints_full.py -- save EVERY face landmark MediaPipe gives (not just the 83).

Why: the original extract_face_keypoints.py kept 83 points (lips without the outer upper
lip, eyes, brows). The 2023 paper's face subset (MediaPipe FACEMESH_CONTOURS, 128 points)
also needs the face OVAL and the outer upper lip, so it cannot be built from those files.
This script keeps all landmarks, so any subset (2023 contours, ours, others) can be cut
out later without touching the videos again.

Same detector setup as extract_face_keypoints.py (Face Landmarker Tasks API, VIDEO
running mode, one face, confidences 0.5, one row per video frame so T matches the labels),
with three differences:
  - all 478 landmarks (468 face mesh + 10 iris) with x, y (image-normalized) and z
  - missing frames are NOT forward-filled: they stay 0 and a detection mask is saved
  - optionally the 52 face blendshape scores (eyeBlinkLeft, browInnerUp, jawOpen, ...)

Outputs
  processed_data/face_keypoints_full/<video>.npy                 (T, 478, 3)  float32 (or --dtype float16)
  processed_data/face_keypoints_full/_detection/<video>.npy      (T,) bool
  processed_data/face_keypoints_full/_blendshapes/<video>.npy    (T, 52) float16   (--blendshapes)
  processed_data/face_keypoints_full/_blendshape_names.json
  processed_data/face_keypoints_full/_contours_2023.json         FACEMESH_CONTOURS points + edges
                                                                  (taken from mediapipe itself, so
                                                                  training code needs no mediapipe)
  processed_data/face_keypoints_full/_report.json

Disk: about 5.7 KB per frame in float32 (2.9 KB in float16); the script prints an estimate.
Next step: data_processing/normalize_face_keypoints_v2.py --layout full (face-local 2D).

Run from the repo root:
  python data_processing/extract_face_keypoints_full.py --workers 4
  python data_processing/extract_face_keypoints_full.py --dtype float16 --blendshapes --workers 6
"""
import argparse
import glob
import json
import multiprocessing as mproc
import os
import urllib.request

import cv2
import numpy as np
from tqdm import tqdm

RAW_VIDEO_DIR = "raw_data/videos"
OUTPUT_DIR = "processed_data/face_keypoints_full"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
NUM_LANDMARKS = 478
_ARGS = None


def contour_spec():
    """FACEMESH_CONTOURS (2023 paper) from whichever mediapipe API is installed."""
    import mediapipe as mp
    try:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C
        groups = {"contours": C.FACE_LANDMARKS_CONTOURS, "lips": C.FACE_LANDMARKS_LIPS,
                  "left_eye": C.FACE_LANDMARKS_LEFT_EYE, "right_eye": C.FACE_LANDMARKS_RIGHT_EYE,
                  "left_eyebrow": C.FACE_LANDMARKS_LEFT_EYEBROW, "right_eyebrow": C.FACE_LANDMARKS_RIGHT_EYEBROW,
                  "face_oval": C.FACE_LANDMARKS_FACE_OVAL}
        edges = {k: sorted({tuple(sorted((c.start, c.end))) for c in v}) for k, v in groups.items()}
        api = "tasks"
    except (ImportError, AttributeError):
        F = mp.solutions.face_mesh_connections
        groups = {"contours": mp.solutions.holistic.FACEMESH_CONTOURS, "lips": F.FACEMESH_LIPS,
                  "left_eye": F.FACEMESH_LEFT_EYE, "right_eye": F.FACEMESH_RIGHT_EYE,
                  "left_eyebrow": F.FACEMESH_LEFT_EYEBROW, "right_eyebrow": F.FACEMESH_RIGHT_EYEBROW,
                  "face_oval": F.FACEMESH_FACE_OVAL}
        edges = {k: sorted({tuple(sorted(e)) for e in v}) for k, v in groups.items()}
        api = "solutions"
    points = sorted({i for e in edges["contours"] for i in e})
    return {"source": f"mediapipe {api} API", "num_points": len(points), "points": points,
            "edges": [list(e) for e in edges["contours"]],
            "regions": {k: sorted({i for e in v for i in e}) for k, v in edges.items() if k != "contours"}}


def process(video_path):
    a = _ARGS
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    vid = os.path.basename(video_path).split(".")[0]          # same naming as extract_face_keypoints.py
    out_path = os.path.join(a.out_dir, f"{vid}.npy")
    if os.path.exists(out_path) and not a.overwrite:
        return vid, "exists", None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return vid, "cannot_open", None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps):
        fps = 50.0
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5,
        output_face_blendshapes=a.blendshapes)
    pts, det, blend, names = [], [], [], None
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                      int(i / fps * 1000))
            kp = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
            found = bool(res.face_landmarks)
            if found:
                f = res.face_landmarks[0]
                n = min(len(f), NUM_LANDMARKS)
                kp[:n] = [[p.x, p.y, p.z] for p in f[:n]]
            pts.append(kp)
            det.append(found)
            if a.blendshapes:
                b = np.zeros(52, dtype=np.float32)
                if found and res.face_blendshapes:
                    s = [c.score for c in res.face_blendshapes[0]][:52]
                    b[:len(s)] = s
                    if names is None:
                        names = [c.category_name for c in res.face_blendshapes[0]][:52]
                blend.append(b)
            i += 1
    cap.release()
    if not pts:
        return vid, "no_frames", None
    np.save(out_path, np.stack(pts).astype(a.dtype))
    np.save(os.path.join(a.out_dir, "_detection", f"{vid}.npy"), np.array(det, dtype=bool))
    if a.blendshapes:
        np.save(os.path.join(a.out_dir, "_blendshapes", f"{vid}.npy"), np.stack(blend).astype(np.float16))
    return vid, None, {"frames": len(pts), "detected": int(np.sum(det)), "blend_names": names}


def _init(args):
    global _ARGS
    _ARGS = args


def main():
    p = argparse.ArgumentParser(description="Extract ALL MediaPipe face landmarks (478) for every video.")
    p.add_argument("--video-dir", default=RAW_VIDEO_DIR)
    p.add_argument("--out-dir", default=OUTPUT_DIR)
    p.add_argument("--model", default="face_landmarker.task", help="Face Landmarker .task file (downloaded if missing)")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                   help="float16 halves the disk use (~0.5 px precision on 1280 px video)")
    p.add_argument("--blendshapes", action="store_true", help="also save the 52 blendshape scores")
    p.add_argument("--workers", type=int, default=4, help="parallel videos (each loads its own model)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"Downloading the Face Landmarker model -> {args.model}")
        urllib.request.urlretrieve(MODEL_URL, args.model)
    os.makedirs(os.path.join(args.out_dir, "_detection"), exist_ok=True)
    if args.blendshapes:
        os.makedirs(os.path.join(args.out_dir, "_blendshapes"), exist_ok=True)

    spec = contour_spec()
    with open(os.path.join(args.out_dir, "_contours_2023.json"), "w") as f:
        json.dump(spec, f, indent=1)
    print(f"FACEMESH_CONTOURS: {spec['num_points']} points, {len(spec['edges'])} edges ({spec['source']})")

    videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))
    total_frames = sum(int(cv2.VideoCapture(v).get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos[:20])
    per_frame = NUM_LANDMARKS * 3 * (4 if args.dtype == "float32" else 2)
    if videos[:20]:
        est = total_frames / len(videos[:20]) * len(videos) * per_frame / 1e9
        print(f"{len(videos)} videos -> {args.out_dir}  (estimated disk use ~{est:.1f} GB)")

    skips, infos = {}, {}
    if args.workers <= 1:
        _init(args)
        it = map(process, videos)
    else:
        pool = mproc.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args,))
        it = pool.imap_unordered(process, videos)
    for vid, reason, info in tqdm(it, total=len(videos)):  # noqa: B007
        if reason == "exists":
            continue
        if reason:
            skips.setdefault(reason, []).append(vid)
        else:
            infos[vid] = info
    if args.workers > 1:
        pool.close()
        pool.join()

    if args.blendshapes:
        names = next((i["blend_names"] for i in infos.values() if i.get("blend_names")), None)
        with open(os.path.join(args.out_dir, "_blendshape_names.json"), "w") as f:
            json.dump(names, f, indent=1)
    frames = sum(i["frames"] for i in infos.values())
    detected = sum(i["detected"] for i in infos.values())
    report = {"written": len(infos), "skipped": skips, "frames": frames, "detected_frames": detected,
              "detected_share": detected / frames if frames else None, "dtype": args.dtype,
              "landmarks": NUM_LANDMARKS, "blendshapes": args.blendshapes}
    for i in infos.values():
        i.pop("blend_names", None)
    with open(os.path.join(args.out_dir, "_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Written {len(infos)} videos | skipped {({k: len(v) for k, v in skips.items()})} | "
          f"face detected in {report['detected_share']:.1%} of frames" if frames else "Nothing new written.")


if __name__ == "__main__":
    main()