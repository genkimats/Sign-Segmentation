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


def _validate():
    sizes = {}
    for name, raw in FACE_SUBSETS.items():
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
        raise ValueError(f"Unknown face_subset '{name}'. Options: {sorted(FACE_SUBSETS)}")
    return sorted(FACE_SUBSETS[name])


def face_subset_positions(name):
    """Positions of the subset's points inside the SAVED face array (for slicing at load time)."""
    pos = {raw: i for i, raw in enumerate(SAVED_FACE_INDICES)}
    return [pos[raw] for raw in face_subset_raw_indices(name)]


def face_subset_size(name):
    return len(face_subset_raw_indices(name))


def face_subset_for_vertex_count(num_face_vertices):
    """Infers the subset name from the number of face vertices (sizes are unique)."""
    for name in FACE_SUBSETS:
        if face_subset_size(name) == num_face_vertices:
            return name
    options = {name: face_subset_size(name) for name in FACE_SUBSETS}
    raise ValueError(f"No face subset has {num_face_vertices} points. Known subsets: {options}")