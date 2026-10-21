"""
Interpretable boundary cues: names, families and the cue SETS used for training.

Single source of truth for
  boundary_cue_analysis.py                    (model-free analysis)
  data_processing/extract_cue_features.py     (writes processed_data/cue_features/)
  src/dataset.py                              (loads a cue set as the "cues" stream)

The saved files hold EVERY cue (+ a "face detected" channel) in CUE_FILE_ORDER; a cue set
is selected at load time by name, so changing the set never needs a re-extraction.
"""

FAMILIES = ["eyes", "brows", "mouth", "face", "head", "torso", "hands", "handshape", "hamer", "body"]
FAMILY_COLORS = {"eyes": "#1f77b4", "brows": "#17becf", "mouth": "#e377c2", "face": "#9467bd",
                 "head": "#8c564b", "torso": "#7f7f7f", "hands": "#d62728", "handshape": "#ff7f0e",
                 "hamer": "#2ca02c", "body": "#bcbd22"}

# (name, family, kind, description). kind: "z" = continuous, robust-z per video;
# "rate" = 0/1 indicator, kept as a rate.
CUES = [
    ("EAR (eye openness)", "eyes", "z", "eye aspect ratio, mean of both eyes"),
    ("blink", "eyes", "rate", "short closure: EAR < blink_ratio x median for 2 frames .. 0.4 s"),
    ("eyes narrowed (long)", "eyes", "rate", "EAR < blink_ratio x median for longer (looking down / squint)"),
    ("eye asymmetry", "eyes", "z", "|EAR left - EAR right|"),
    ("brow height", "brows", "z", "upper brow edge above the eye centre (inter-ocular units)"),
    ("inner brow height", "brows", "z", "inner brow ends above the inner eye corners"),
    ("brow furrow (inner distance)", "brows", "z", "distance between the inner brow ends (small = furrowed)"),
    ("brow vertical velocity", "brows", "z", "d(brow height)/dt, signed (+ = raising)"),
    ("MAR (mouth aspect ratio)", "mouth", "z", "inner-lip opening / mouth width"),
    ("mouth opening", "mouth", "z", "inner-lip distance 13-14"),
    ("mouth width", "mouth", "z", "mouth corner distance 61-291"),
    ("lower lip / jaw", "mouth", "z", "outer lower lip centre (17) below the eyes"),
    ("mouth motion energy", "mouth", "z", "mean speed of all lip points (mouthing activity)"),
    ("mouth opening velocity", "mouth", "z", "d(MAR)/dt, signed (+ = opening)"),
    ("face deformation energy", "face", "z", "mean speed of all face points (face-local)"),
    ("head height", "head", "z", "nose above the shoulder midpoint"),
    ("head pitch (nod)", "head", "z", "nose above the ear midpoint / ear distance (- = chin down)"),
    ("head yaw deviation", "head", "z", "|yaw - video median| (turning away from the usual direction)"),
    ("head roll deviation", "head", "z", "|ear-line angle - video median|"),
    ("head speed", "head", "z", "nose speed"),
    ("head vertical velocity", "head", "z", "d(head height)/dt, signed (- = moving down)"),
    ("head rotation speed", "head", "z", "speed of (yaw, pitch)"),
    ("shoulder tilt", "torso", "z", "|shoulder-line angle - video median|"),
    ("shoulder width change", "torso", "z", "shoulder width / video median (lean / turn)"),
    ("dom wrist height", "hands", "z", "dominant pose wrist above the shoulder midpoint"),
    ("nondom wrist height", "hands", "z", "non-dominant pose wrist height"),
    ("dom wrist speed", "hands", "z", "dominant pose wrist speed"),
    ("nondom wrist speed", "hands", "z", "non-dominant pose wrist speed"),
    ("dom wrist acceleration", "hands", "z", "dominant wrist acceleration magnitude"),
    ("dom hold", "hands", "rate", "dominant wrist speed in the video's lowest 20 %"),
    ("hand distance", "hands", "z", "distance between the two pose wrists"),
    ("dom hand-to-face", "hands", "z", "dominant wrist to nose distance"),
    ("dom reach (signing space)", "hands", "z", "dominant wrist to shoulder midpoint distance"),
    ("two-handedness", "hands", "z", "non-dominant share of the summed wrist speed"),
    ("velocity symmetry (mirrored)", "hands", "z", "cosine of the wrist velocities, x mirrored"),
    ("MP dom hand detected", "handshape", "rate", "MediaPipe found the dominant hand"),
    ("MP nondom hand detected", "handshape", "rate", "MediaPipe found the non-dominant hand"),
    ("dom hand openness", "handshape", "z", "fingertips to wrist / palm size"),
    ("dom finger spread", "handshape", "z", "index tip to pinky tip / palm size"),
    ("dom handshape change", "handshape", "z", "speed of the wrist-relative, palm-scaled hand shape"),
    ("nondom handshape change", "handshape", "z", "same for the non-dominant hand"),
    ("HaMeR dom detected", "hamer", "rate", "HaMeR found the dominant hand"),
    ("HaMeR dom finger flexion", "hamer", "z", "mean rotation angle of the 15 finger joints"),
    ("HaMeR dom articulation speed", "hamer", "z", "mean geodesic change of the 15 joint rotations / s"),
    ("HaMeR dom wrist rotation speed", "hamer", "z", "geodesic change of global_orient / s"),
    ("HaMeR nondom articulation speed", "hamer", "z", "same for the non-dominant hand"),
    ("whole-body motion energy", "body", "z", "mean speed of all body + hand points"),
]
CUE_NAMES = [c[0] for c in CUES]
CUE_INFO = {c[0]: {"family": c[1], "kind": c[2], "desc": c[3]} for c in CUES}

# Extra channel written by the extractor: 1 where the face was really detected (face cues
# are 0 = "video median" on the other frames, so the model can tell missing from neutral).
FACE_DETECTED = "face detected"
CUE_FILE_ORDER = CUE_NAMES + [FACE_DETECTED]

# ---- Cue sets for training --------------------------------------------------------
# "selected": the cues that showed boundary information in boundary_cue_analysis.py
# (delta AUC CI clear of 0.5 in the pause-matched / phrase-end / continuous comparisons).
# Dropped: torso (no signal), MediaPipe detection rates (~constant 0.999), eye asymmetry,
# finger spread, two-handedness, velocity symmetry, mouth opening velocity (noise-level).
SELECTED = [
    "EAR (eye openness)", "blink", "eyes narrowed (long)",
    "brow height", "inner brow height", "brow vertical velocity",
    "MAR (mouth aspect ratio)", "mouth opening", "lower lip / jaw", "mouth motion energy",
    "face deformation energy",
    "head height", "head pitch (nod)", "head yaw deviation", "head roll deviation",
    "head speed", "head vertical velocity", "head rotation speed",
    "dom wrist height", "nondom wrist height", "dom wrist speed", "nondom wrist speed",
    "dom wrist acceleration", "dom hold", "hand distance", "dom hand-to-face",
    "dom reach (signing space)",
    "dom hand openness", "dom handshape change", "nondom handshape change",
    "HaMeR dom finger flexion", "HaMeR dom articulation speed", "HaMeR dom wrist rotation speed",
    "HaMeR nondom articulation speed",
    "whole-body motion energy",
    FACE_DETECTED,
]

_FACE_FAMS = ("eyes", "brows", "mouth", "face")
_HEAD_FAMS = ("head",)
_MANUAL_FAMS = ("hands", "handshape", "hamer", "body")


def _fam(names, fams):
    return [n for n in names if n in CUE_INFO and CUE_INFO[n]["family"] in fams]


CUE_SETS = {
    "all": list(CUE_FILE_ORDER),
    "selected": list(SELECTED),
    # facial expression cues only (what the raw face stream sees) -- the direct replacement
    # for the face ST-GCN stream
    "face": _fam(SELECTED, _FACE_FAMS) + [FACE_DETECTED],
    # all non-manual cues: face + head
    "nonmanual": _fam(SELECTED, _FACE_FAMS + _HEAD_FAMS) + [FACE_DETECTED],
    # manual cues only: hand kinematics, handshape, HaMeR, whole-body energy
    "manual": _fam(SELECTED, _MANUAL_FAMS),
}


def cue_names_path(cue_dir):
    """Channel-name file of a cue directory: <dir>/_names.json (extract_cue_features.py), or
    for the blendshape scores of extract_face_keypoints_full.py (<face dir>/_blendshapes/)
    the parent's _blendshape_names.json."""
    import os
    own = os.path.join(cue_dir, "_names.json")
    if os.path.exists(own):
        return own
    parent = os.path.join(os.path.dirname(os.path.normpath(cue_dir)), "_blendshape_names.json")
    return parent if os.path.exists(parent) else own


def resolve_cue_set(cue_set, file_names=None):
    """
    cue_set: a name from CUE_SETS, or an explicit list of cue names.
    file_names: the channel names stored in the cue files (_names.json); used to check
    availability and to return channel indices. Returns (names, indices or None).
    """
    if cue_set == "file":            # every channel the files hold (e.g. the 52 blendshapes)
        if file_names is None:
            raise ValueError("cue_set='file' needs the cue directory's channel-name file.")
        return list(file_names), list(range(len(file_names)))
    if isinstance(cue_set, str):
        if cue_set not in CUE_SETS:
            raise ValueError(f"Unknown cue_set '{cue_set}'. Options: {sorted(CUE_SETS)} or a list of names.")
        names = list(CUE_SETS[cue_set])
    else:
        names = list(cue_set)
    if file_names is None:
        return names, None
    pos = {n: i for i, n in enumerate(file_names)}
    missing = [n for n in names if n not in pos]
    if missing:
        raise ValueError(f"Cue(s) not in the cue files: {missing}. Re-run "
                         f"data_processing/extract_cue_features.py (the files have {len(file_names)} channels).")
    return names, [pos[n] for n in names]