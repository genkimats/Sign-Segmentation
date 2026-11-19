"""
draw_keypoints.py -- slide 5 figure: the model's inputs drawn on one video frame (body + hand keypoints, and the
compact face points), side by side with the same keypoints on a white background.

Runs the MediaPipe Tasks landmarkers (pose, hands, face) on a single image. The .task model files are downloaded
next to the script on first use if they are not there yet (the same files the extraction scripts use).

  python draw_keypoints.py --image frame.png
  python draw_keypoints.py --image frame.png --face-indices 70 105 107 336 334 300 33 133 362 263 61 291 13 14
Output: evaluation_phrase/figures/keypoints.png

FACE_INDICES below is a placeholder close to the compact 31-point subset (eyebrows, eyes, mouth). Replace it with
the exact 31 MediaPipe Face Mesh indices your compact face files use, so the slide shows the real input.
"""
import argparse
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELS = {
    "pose": ("pose_landmarker_full.task", "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task"),
    "hand": ("hand_landmarker.task", "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task"),
    "face": ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task"),
}
FACE_INDICES = [  # placeholder: eyebrows (5 + 5), eyes (6 + 6), mouth (9) = 31
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    33, 159, 145, 133, 160, 144, 263, 386, 374, 362, 387, 373,
    61, 291, 0, 17, 13, 14, 78, 308, 81,
]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22), (0, 1), (1, 2), (2, 3), (3, 7),
              (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
BODY_C, HAND_C, FACE_C = (181, 109, 47), (87, 139, 46), (57, 123, 224)   # BGR: blue, green, orange


def model_path(key):
    fname, url = MODELS[key]
    if not os.path.exists(fname):
        print(f"downloading {fname} ...")
        urllib.request.urlretrieve(url, fname)
    return fname


def detect(rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    opt = lambda key: mp_python.BaseOptions(model_asset_path=model_path(key))
    with vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=opt("pose"))) as m:
        pose = m.detect(img).pose_landmarks
    with vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=opt("hand"), num_hands=2)) as m:
        hands = m.detect(img).hand_landmarks
    with vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=opt("face"), num_faces=1)) as m:
        face = m.detect(img).face_landmarks
    return (pose[0] if pose else None), hands, (face[0] if face else None)


def draw(canvas, pose, hands, face, face_idx, w, h, scale):
    px = lambda lm: (int(lm.x * w), int(lm.y * h))
    t = max(2, int(3 * scale))
    if pose is not None:
        for a, b in POSE_EDGES:
            if a <= 22 and b <= 22:                      # upper body only (the model keeps landmarks 0-22)
                cv2.line(canvas, px(pose[a]), px(pose[b]), BODY_C, t, cv2.LINE_AA)
        for i in range(23):
            cv2.circle(canvas, px(pose[i]), t + 2, BODY_C, -1, cv2.LINE_AA)
    for hand in hands:
        for a, b in HAND_EDGES:
            cv2.line(canvas, px(hand[a]), px(hand[b]), HAND_C, max(1, t - 1), cv2.LINE_AA)
        for lm in hand:
            cv2.circle(canvas, px(lm), t, HAND_C, -1, cv2.LINE_AA)
    if face is not None:
        for i in face_idx:
            cv2.circle(canvas, px(face[i]), max(2, t - 1), FACE_C, -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--face-indices", type=int, nargs="+", default=FACE_INDICES)
    ap.add_argument("--out", default=os.path.join("evaluation_phrase", "figures", "keypoints.png"))
    a = ap.parse_args()
    bgr = cv2.imread(a.image)
    if bgr is None:
        raise SystemExit(f"cannot read {a.image}")
    h, w = bgr.shape[:2]
    pose, hands, face = detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"pose: {'yes' if pose else 'no'}, hands: {len(hands)}, face: {'yes' if face else 'no'}")
    over = bgr.copy()
    draw(over, pose, hands, face, a.face_indices, w, h, w / 1000)
    blank = np.full_like(bgr, 255)
    draw(blank, pose, hands, face, a.face_indices, w, h, w / 1000)
    gap = np.full((h, 30, 3), 255, np.uint8)
    out = np.hstack([over, gap, blank])
    legend_y = 40
    for txt, c in (("body keypoints", BODY_C), ("hand keypoints", HAND_C), (f"face ({len(a.face_indices)} points)", FACE_C)):
        cv2.putText(out, txt, (w + 60, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2, cv2.LINE_AA)
        legend_y += 42
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(a.out, out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
