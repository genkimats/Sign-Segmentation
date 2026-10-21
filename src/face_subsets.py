"""
Face-keypoint subsets, selected at LOAD time (the saved face files are unchanged).

The saved face files (extract_face_keypoints.py -> normalize_face_keypoints.py) hold
83 MediaPipe Face Mesh points per frame: lips, both eyes, both eyebrows, stored in
ascending raw-index order (SAVED_FACE_INDICES). A subset keeps only some of them,
so the face no longer outweighs body + hands (65 vertices) in the model.

Subsets (point counts):
  "full"       83  -- everything that was saved
  "compact"    31  -- eyes (6 per eye: the standard eye-aspect-ratio points -> blinks),
                      eyebrows (5 upper points each -> raises / furrows),
                      mouth (9: corners + inner-lip points -> opening / mouthing)
  "eyes_brows" 22  -- compact without the mouth (blinks + brows only)
  "minimal"    18  -- eyes (corners + upper/lower lid centre), eyebrows (3 each),
                      mouth (corners + inner-lip centres)

Every subset keeps the inner eye corners (raw 133 / 362), which anchor the face to
the nose in the skeleton graph. Each subset has a DIFFERENT size: graph.py infers
the subset from num_vertices - 65, so models.py needs no extra argument.

NOTE: the saved lips points cover the outer LOWER lip and the inner lips only (the
outer upper lip was never extracted), so mouth points are chosen from those.
"""
import json
import os


# ---- Must match extract_face_keypoints.py's SELECTED_INDICES construction exactly ----
LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LEFT_EYEBROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

SAVED_FACE_INDICES = sorted(set(
    LIPS_INDICES + LEFT_EYE_INDICES + RIGHT_EYE_INDICES + LEFT_EYEBROW_INDICES + RIGHT_EYEBROW_INDICES
))  # 83 -- the vertex order inside the saved face .npy files

# Regions in CONTOUR order (used for graph edges); closed loops for lips and eyes.
FACE_REGIONS = [
    (LIPS_INDICES, True),
    (LEFT_EYE_INDICES, True),
    (RIGHT_EYE_INDICES, True),
    (LEFT_EYEBROW_INDICES, False),
    (RIGHT_EYEBROW_INDICES, False),
]

ANCHOR_LEFT_EYE_INNER = 133
ANCHOR_RIGHT_EYE_INNER = 362

_COMPACT_EYES = [33, 160, 158, 133, 153, 144,      # left eye: EAR points
                 263, 387, 385, 362, 380, 373]     # right eye: EAR points
_COMPACT_BROWS = [70, 63, 105, 66, 107,            # left eyebrow, upper edge
                  300, 293, 334, 296, 336]         # right eyebrow, upper edge
_COMPACT_MOUTH = [61, 291,                         # mouth corners
                  13, 81, 311,                     # inner upper lip
                  14, 87, 317,                     # inner lower lip
                  17]                              # outer lower lip centre

FACE_SUBSETS = {
    "full": list(SAVED_FACE_INDICES),
    "compact": _COMPACT_EYES + _COMPACT_BROWS + _COMPACT_MOUTH,
    "eyes_brows": _COMPACT_EYES + _COMPACT_BROWS,
    "minimal": [33, 159, 133, 145,                 # left eye: corners, upper/lower lid centre
                263, 386, 362, 374,                # right eye
                70, 105, 107,                      # left eyebrow
                300, 334, 336,                     # right eyebrow
                61, 291, 13, 14],                  # mouth corners, inner lip centres
}


# ==============================================================================
# FULL-MESH layout (data_processing/extract_face_keypoints_full.py): all 478 MediaPipe
# landmarks are saved, so the vertex index in the file IS the MediaPipe index.
# Subsets that need points outside the 83 only exist in this layout.
# ==============================================================================
FACE_LAYOUTS = {"saved83": len(SAVED_FACE_INDICES), "full478": 478}

# Iris (Face Landmarker refinement): 468 = centre of the iris next to 33/133 (the eye
# called LEFT_EYE_INDICES here), 469-472 around it; 473 = centre next to 263/362, 474-477.
IRIS_A = [468, 469, 470, 471, 472]
IRIS_B = [473, 474, 475, 476, 477]

# The 2023 paper's subset = MediaPipe FACEMESH_CONTOURS (lips incl. outer upper lip, eyes,
# brows, face oval). Read from the file extract_face_keypoints_full.py writes (taken from
# mediapipe itself), so training needs no mediapipe and nothing is typed from memory.
_CONTOUR_JSON_PATHS = [os.environ.get("FACE_CONTOURS_JSON", ""),
                       "src/facemesh_contours_2023.json",
                       "processed_data/face_keypoints_full/_contours_2023.json"]
CONTOURS_2023 = None   # {"points", "edges", "regions"} once found
for _p in _CONTOUR_JSON_PATHS:
    if _p and os.path.exists(_p):
        with open(_p) as _f:
            CONTOURS_2023 = json.load(_f)
        break

FULL_MESH_SUBSETS = {
    # ours (compact) + both irises -> gaze direction relative to the eye corners
    "compact_iris": FACE_SUBSETS["compact"] + IRIS_A + IRIS_B,
}
if CONTOURS_2023 is not None:
    _oval = set(CONTOURS_2023["regions"].get("face_oval", []))
    FULL_MESH_SUBSETS["contours2023"] = list(CONTOURS_2023["points"])
    # the 2023 subset WITHOUT the face oval: is the oval (head shape / pose) what matters?
    FULL_MESH_SUBSETS["contours2023_no_oval"] = [i for i in CONTOURS_2023["points"] if i not in _oval]
    # the face oval alone (+ the two anchors, which every subset keeps)
    FULL_MESH_SUBSETS["face_oval"] = sorted(_oval | {ANCHOR_LEFT_EYE_INNER, ANCHOR_RIGHT_EYE_INNER})
# subsets whose graph uses the MediaPipe contour connections instead of our region chains
CONTOUR_GRAPH_SUBSETS = {"contours2023", "contours2023_no_oval", "face_oval"}

FACE_SUBSETS.update(FULL_MESH_SUBSETS)


def _validate():
    sizes = {}
    for name, raw in FACE_SUBSETS.items():
        if name not in FULL_MESH_SUBSETS:
            missing = sorted(set(raw) - set(SAVED_FACE_INDICES))
            if missing:
                raise ValueError(f"Face subset '{name}' uses points that were never saved: {missing}")
        if len(set(raw)) != len(raw):
            raise ValueError(f"Face subset '{name}' contains duplicate points.")
        for anchor in (ANCHOR_LEFT_EYE_INNER, ANCHOR_RIGHT_EYE_INNER):
            if anchor not in raw:
                raise ValueError(f"Face subset '{name}' must keep anchor point {anchor}.")
        n = len(raw)
        if n in sizes:
            raise ValueError(f"Face subsets '{sizes[n]}' and '{name}' have the same size ({n}); "
                             f"sizes must be unique so graph.py can infer the subset from num_vertices.")
        sizes[n] = name


_validate()


def face_subset_raw_indices(name):
    """Raw MediaPipe indices of a subset, in vertex order (= ascending, like the saved files)."""
    if name not in FACE_SUBSETS:
        hint = ("" if CONTOURS_2023 is not None or not name.startswith(("contours", "face_oval")) else
                " (the 2023 contour subsets need processed_data/face_keypoints_full/_contours_2023.json "
                "from data_processing/extract_face_keypoints_full.py)")
        raise ValueError(f"Unknown face_subset '{name}'. Options: {sorted(FACE_SUBSETS)}{hint}")
    return sorted(FACE_SUBSETS[name])


def face_subset_positions(name, layout="saved83"):
    """Positions of the subset's points inside the saved face array (for slicing at load time).
    layout "saved83": the 83-point files; "full478": the 478-point files (position = index)."""
    raw = face_subset_raw_indices(name)
    if layout == "full478":
        return list(raw)
    if layout != "saved83":
        raise ValueError(f"Unknown face_layout '{layout}'. Options: {sorted(FACE_LAYOUTS)}")
    if name in FULL_MESH_SUBSETS:
        raise ValueError(f"Face subset '{name}' needs points that are not in the 83-point files -- use "
                         f"face_layout='full478' with the face_keypoints_full files.")
    pos = {r: i for i, r in enumerate(SAVED_FACE_INDICES)}
    return [pos[r] for r in raw]


def face_subset_size(name):
    return len(face_subset_raw_indices(name))


def face_subset_for_vertex_count(num_face_vertices):
    """Infers the subset name from the number of face vertices (sizes are unique)."""
    for name in FACE_SUBSETS:
        if face_subset_size(name) == num_face_vertices:
            return name
    options = {name: face_subset_size(name) for name in FACE_SUBSETS}
    raise ValueError(f"No face subset has {num_face_vertices} points. Known subsets: {options}")