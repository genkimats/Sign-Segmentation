"""
Auxiliary, stream-specific supervision targets for Idea B (no torch, no sklearn).

Two frame-level tasks, built by data_processing/make_aux_labels.py from the ELAN files with the
SAME parsing as phrase_oracle_analysis.py:

  mouth    (face stream)   0 = no mouth annotation, 1 = mouthing (spoken-word mouth picture),
                           2 = mouth gesture ("[...]" values). -100 for the WHOLE video when the
                           participant has no Mundbild tier at all (missing tier != no mouthing).
  signcat  (HaMeR stream)  one class per sign (all frames of the sign), -100 outside signs and for
                           signs without a gloss annotation:
                             0 lexical (other)        4 fingerspelling / number ($ALPHA, $NUM)
                             1 palm-up ($GEST-OFF)    5 other gesture / special ($GEST, $ORAL,
                             2 pointing ($INDEX)        $LIST, $PMS, $)
                             3 depicting ($PROD)      6 connective / discourse marker

IGNORE = -100 everywhere (the ignore_index of the auxiliary cross-entropy).
"""

IGNORE = -100
AUX_TASKS = ("mouth", "signcat")

MOUTH_CLASSES = ["none", "mouthing", "mouth gesture"]
SIGNCAT_CLASSES = ["lexical", "palm-up", "pointing", "depicting", "fingerspelling/number",
                   "other gesture/special", "connective"]
CLASS_NAMES = {"mouth": MOUTH_CLASSES, "signcat": SIGNCAT_CLASSES}
NUM_CLASSES = {t: len(c) for t, c in CLASS_NAMES.items()}

# Connectives / discourse markers: the positive "gloss_cur=" weights >= 0.9 in
# evaluation_phrase/oracle/oracle_lexical_weights.csv (phrase_oracle_analysis.py, lexical oracle
# FITTED ON TRAIN ONLY -> no val/test leakage). Fixed here on purpose: never rebuild this list from
# val/test. make_aux_labels.py drops items with fewer than --min-connective-count (20) train
# occurrences and records the kept list in processed_data/aux_labels/_meta.json.
CONNECTIVE_CANDIDATES = ["ABER", "WENN", "DANN", "UND", "DARUM", "UND-DANN", "SPÄTER", "GRUND", "PLÖTZLICH",
                         "DESHALB", "JETZT", "BEISPIEL", "JA", "FRÜHER", "HEUTE", "MEISTENS", "ACHTUNG"]

_CATEGORY_TO_CLASS = {"$GEST-OFF": 1, "$INDEX": 2, "$PROD": 3, "$ALPHA": 4, "$NUM": 4,
                      "$GEST": 5, "$ORAL": 5, "$LIST": 5, "$PMS": 5, "$": 5}


def signcat_id(category, base, connectives):
    """category = phrase_oracle_analysis.gloss_category(g), base = gloss_base(g)."""
    if base == "?":                                   # no gloss annotation overlaps the sign
        return IGNORE
    if category in _CATEGORY_TO_CLASS:
        return _CATEGORY_TO_CLASS[category]
    return 6 if base in connectives else 0


def mouth_id(kind):
    """kind = phrase_oracle_analysis.mouth_kind(value): 'word' | 'gesture' | 'none'."""
    return {"word": 1, "gesture": 2}.get(kind, 0)


def apply_label_map(arr, mapping):
    """Remap class ids of one task, e.g. {6: 0} merges connectives into lexical (arm B6).
    mapping keys may be strings (JSON). IGNORE stays IGNORE."""
    if not mapping:
        return arr
    out = arr.copy()
    for src, dst in mapping.items():
        out[arr == int(src)] = int(dst)
    return out


def mapped_class_counts(counts, mapping, n_classes):
    """Train frame counts per class after apply_label_map, truncated/padded to n_classes."""
    c = list(counts) + [0] * max(0, n_classes - len(counts))
    if mapping:
        c = list(c)
        for src, dst in mapping.items():
            src, dst = int(src), int(dst)
            if src != dst and src < len(c):
                c[dst] += c[src]
                c[src] = 0
    return c[:n_classes]


def inv_sqrt_weights(counts):
    """w_c proportional to 1 / sqrt(freq_c), normalized to mean 1 over the classes that occur;
    classes that never occur get weight 0."""
    import math
    raw = [1.0 / math.sqrt(n) if n > 0 else 0.0 for n in counts]
    present = [w for w in raw if w > 0]
    if not present:
        raise ValueError("no class of this auxiliary task occurs in the training labels")
    mean = sum(present) / len(present)
    return [w / mean for w in raw]
