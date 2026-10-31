"""
Sanity tests for the C / A / B implementation (phrase_plan_A_B_C.md). Synthetic data only, no
real files touched. Run from the repo root:

    python test_phrase_ideas.py            # everything; torch tests are skipped without torch
    python -m pytest test_phrase_ideas.py  # same, with pytest

  C  count_oracle_decoding.selftest()  (oracle-K on one-hot = perfect, peak picker, segments)
  A  tempo features == phrase_oracle_analysis timing features on the same transitions; the
     extractor end to end (gold + predicted signs, mismatch check, files the cue stream loads)
  B  aux labels from a synthetic ELAN file (mouth tier / missing tier, palm-up, pointing,
     connective), class-map / weights, the --by-context table, and (torch) the model heads,
     FiLM identity at initialization, strict loading of a config without the new options.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
FPS = 50.0


# ----------------------------------------------------------------------------- helpers
def _synthetic_signs(rng, T, start=37):
    lab = np.zeros(T, dtype=np.int8)
    t, signs = start, []
    while t < T - 60:
        d = int(rng.integers(5, 40))
        lab[t] = 2
        lab[t + 1:t + d] = 1
        signs.append((t, t + d - 1))
        t += d + int(rng.choice([0, 0, 2, 5, 10, 30, 80]))
    return lab, np.array(signs, dtype=np.int64)


def _phrase_labels(signs, T, every=4):
    ph = np.zeros(T, dtype=np.int8)
    for i, (s, e) in enumerate(signs):
        ph[s:e + 1] = 1
        if i % every == 0:
            ph[s] = 2
    for i in range(len(signs) - 1):
        if (i + 1) % every != 0:
            ph[signs[i][1] + 1:signs[i + 1][0]] = 1
    return ph


def _eaf(signs, glosses, participant="A", mouth=None):
    """Minimal ELAN file: right-hand gloss tier (+ optional Mundbild tier) for one participant."""
    slots, tiers, k = [], [], 0

    def ann(s, e, v):
        nonlocal k
        slots.append(f'<TIME_SLOT TIME_SLOT_ID="t{k}" TIME_VALUE="{int(round(s / FPS * 1000))}"/>'
                     f'<TIME_SLOT TIME_SLOT_ID="t{k + 1}" TIME_VALUE="{int(round((e + 1) / FPS * 1000))}"/>')
        a = (f'<ALIGNABLE_ANNOTATION TIME_SLOT_REF1="t{k}" TIME_SLOT_REF2="t{k + 1}">'
             f'<ANNOTATION_VALUE>{v}</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION>')
        k += 2
        return a
    g = "".join(ann(s, e, v) for (s, e), v in zip(signs, glosses))
    tiers.append(f'<TIER TIER_ID="Lexem_Gebärde_r_{participant}">{g}</TIER>')
    if mouth:
        m = "".join(ann(s, e, v) for s, e, v in mouth)
        tiers.append(f'<TIER TIER_ID="Mundbild_Mundgestik_{participant}">{m}</TIER>')
    return f'<ANNOTATION_DOCUMENT><TIME_ORDER>{"".join(slots)}</TIME_ORDER>{"".join(tiers)}</ANNOTATION_DOCUMENT>'


# ----------------------------------------------------------------------------- C
def test_count_oracle_selftest():
    import count_oracle_decoding
    count_oracle_decoding.selftest()


# ----------------------------------------------------------------------------- A
def test_tempo_matches_oracle():
    import phrase_oracle_analysis as poa
    from src.tempo_features import TEMPO_CHANNELS, tempo_features, transition_table
    rng = np.random.default_rng(1)
    T = 4000
    lab, signs = _synthetic_signs(rng, T)
    assert (poa.sign_segments(lab) == signs).all()
    ph = _phrase_labels(signs, T)
    d = tempfile.mkdtemp()
    try:
        for sub in ("g", "p", "a"):
            os.makedirs(os.path.join(d, sub))
        np.save(os.path.join(d, "g", "X_A.npy"), lab)
        np.save(os.path.join(d, "p", "X_A.npy"), ph)
        with open(os.path.join(d, "a", "X.eaf"), "w") as f:
            f.write(_eaf(signs, [f"G{i}" for i in range(len(signs))]))
        args = types.SimpleNamespace(gloss_labels=os.path.join(d, "g"), phrase_labels=os.path.join(d, "p"),
                                     annotations=os.path.join(d, "a"), fps=FPS, short_gap=12)
        rows, why = poa.video_transitions("X_A", args, {})
        assert why is None, why
    finally:
        shutil.rmtree(d)
    tab, st = transition_table(signs, FPS)
    names = ["log_gap_prev", "log_gap_next", "gap_prev_rel", "gap_prev_rank", "gap_prev_local_max",
             "dur_prev_rel", "dur_cur_rel", "signs_since_long_gap", "local_rate_rel"]
    for r in rows:
        m = dict(zip(names, tab[r["sign_index"]]))
        for k in ("log_gap_prev", "log_gap_next", "gap_prev_rel", "gap_prev_local_max", "dur_prev_rel", "dur_cur_rel"):
            assert abs(m[k] - r[k]) < 1e-9, (k, m[k], r[k])
        assert abs(m["signs_since_long_gap"] * 30 - r["signs_since_long_gap"]) < 1e-9
        assert abs(m["local_rate_rel"] * st["rate"] - r["local_rate"]) < 1e-9
    F = tempo_features(signs, T, FPS, video_norm={"mean": [0] * 4, "sd": [1] * 4})
    ci = {c: k for k, c in enumerate(TEMPO_CHANNELS)}
    assert F.shape == (T, 17) and np.isfinite(F).all()
    for i, (s, e) in enumerate(signs):            # each sign onset carries its own transition
        assert abs(F[s, ci["log_gap_prev"]] - tab[i, 0]) < 1e-5
        assert abs(F[s, ci["onset_bump"]] - 1.0) < 1e-6
    assert F[:signs[0, 0], ci["outside_signing"]].all() and F[signs[-1, 1] + 1:, ci["outside_signing"]].all()
    assert F[signs[0, 0]:signs[-1, 1] + 1, ci["outside_signing"]].sum() == 0
    gaps = sum(max(0, signs[i + 1, 0] - signs[i, 1] - 1) for i in range(len(signs) - 1))
    assert F[:, ci["in_gap"]].sum() == gaps
    F0 = tempo_features(np.zeros((0, 2)), 100, FPS)
    assert F0[:, ci["outside_signing"]].all() and F0.sum() == 100


def test_tempo_extractor_end_to_end():
    rng = np.random.default_rng(0)
    root = tempfile.mkdtemp()
    try:
        for sub in ("BIO_tags", "BIO_tags_phrase_v2", "exp"):
            os.makedirs(os.path.join(root, sub))
        splits = {"train": [f"T{i}_A.npy" for i in range(8)], "val": [f"V{i}_A.npy" for i in range(3)],
                  "test": [f"S{i}_B.npy" for i in range(3)]}
        with open(os.path.join(root, "splits.json"), "w") as f:
            json.dump(splits, f)
        recs = {s: {} for s in splits}
        for s, vs in splits.items():
            for v in vs:
                v = v[:-4]
                T = int(rng.integers(1500, 2500))
                lab, signs = _synthetic_signs(rng, T, start=int(rng.integers(0, 80)))
                np.save(os.path.join(root, "BIO_tags", v + ".npy"), lab)
                np.save(os.path.join(root, "BIO_tags_phrase_v2", v + ".npy"), _phrase_labels(signs, T))
                noise = 0.5 if s == "train" else 1.5                     # in-sample train logits are cleaner
                recs[s][v] = (np.eye(3)[lab] * 4 + rng.normal(0, noise, (T, 3))).astype(np.float32)
        for s in splits:
            np.savez_compressed(os.path.join(root, "exp", f"{s}.npz"), **{f"logits__{v}": a for v, a in recs[s].items()})
        for source in ("gold", "pred"):
            out = os.path.join(root, f"out_{source}")
            r = subprocess.run([sys.executable, os.path.join(REPO, "data_processing", "extract_tempo_features.py"),
                                "--source", source, "--split-file", os.path.join(root, "splits.json"),
                                "--gloss-labels", os.path.join(root, "BIO_tags"),
                                "--phrase-labels", os.path.join(root, "BIO_tags_phrase_v2"),
                                "--sign-logits", os.path.join(root, "exp"), "--out-dir", out],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr[-2000:]
            names = json.load(open(os.path.join(out, "_names.json")))
            assert len(names) == 17
            from src.cue_sets import resolve_cue_set
            assert resolve_cue_set("file", names)[1] == list(range(17))
            for s, vs in splits.items():
                for v in vs:
                    a = np.load(os.path.join(out, v))
                    T = len(np.load(os.path.join(root, "BIO_tags_phrase_v2", v)))
                    assert a.shape == (T, 17) and a.dtype == np.float16 and np.isfinite(a.astype(float)).all()
            rep = json.load(open(os.path.join(out, "_report.json")))
            if source == "pred":
                mm = rep["mismatch"]
                assert mm["train"]["sign_onset_f1_at2"] > mm["val"]["sign_onset_f1_at2"]
                assert "xfit_recommended" in mm
    finally:
        shutil.rmtree(root)


# ----------------------------------------------------------------------------- B
def test_aux_labels():
    from src.aux_labels import IGNORE
    rng = np.random.default_rng(2)
    root = tempfile.mkdtemp()
    try:
        for sub in ("BIO_tags", "BIO_tags_phrase_v2", "ann"):
            os.makedirs(os.path.join(root, sub))
        T = 3000
        lab, signs = _synthetic_signs(rng, T)
        cycle = ["HAUS1", "$GEST-OFF^", "$INDEX1", "ABER1", "$PROD", "$ALPHA1:A", "$GEST^"]
        glosses = [cycle[i % len(cycle)] for i in range(len(signs))]
        mouth = [(int(signs[0, 0]), int(signs[0, 1]), "haus"), (int(signs[2, 0]), int(signs[2, 1]), "[MG]")]
        # participant A has a Mundbild tier, participant B has none
        with open(os.path.join(root, "ann", "X.eaf"), "w") as f:
            doc_a = _eaf(signs, glosses, "A", mouth)
            doc_b = _eaf(signs, glosses, "B", None)
            tiers_b = doc_b.split("</TIME_ORDER>")[1].replace("</ANNOTATION_DOCUMENT>", "")
            slots_b = doc_b.split("<TIME_ORDER>")[1].split("</TIME_ORDER>")[0].replace('"t', '"u')
            tiers_b = tiers_b.replace('"t', '"u')
            doc = doc_a.replace("</TIME_ORDER>", slots_b + "</TIME_ORDER>").replace("</ANNOTATION_DOCUMENT>",
                                                                                 tiers_b + "</ANNOTATION_DOCUMENT>")
            f.write(doc)
        for p in ("A", "B"):
            np.save(os.path.join(root, "BIO_tags", f"X_{p}.npy"), lab)
            np.save(os.path.join(root, "BIO_tags_phrase_v2", f"X_{p}.npy"), _phrase_labels(signs, T))
        with open(os.path.join(root, "splits.json"), "w") as f:
            json.dump({"train": ["X_A.npy", "X_B.npy"], "val": [], "test": []}, f)
        out = os.path.join(root, "aux")
        r = subprocess.run([sys.executable, os.path.join(REPO, "data_processing", "make_aux_labels.py"),
                            "--annotations", os.path.join(root, "ann"), "--gloss-labels", os.path.join(root, "BIO_tags"),
                            "--phrase-labels", os.path.join(root, "BIO_tags_phrase_v2"),
                            "--split-file", os.path.join(root, "splits.json"), "--out-dir", out,
                            "--min-connective-count", "1"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        za, zb = np.load(os.path.join(out, "X_A.npz")), np.load(os.path.join(out, "X_B.npz"))
        assert (zb["mouth"] == IGNORE).all(), "missing Mundbild tier must be ignored, not 'no mouthing'"
        assert set(np.unique(za["mouth"])) <= {0, 1, 2} and (za["mouth"][signs[0, 0]] == 1) and (za["mouth"][signs[2, 0]] == 2)
        expect = {"HAUS1": 0, "$GEST-OFF^": 1, "$INDEX1": 2, "ABER1": 6, "$PROD": 3, "$ALPHA1:A": 4, "$GEST^": 5}
        for (s, e), g in zip(signs, glosses):
            assert (za["signcat"][s:e + 1] == expect[g]).all(), (g, za["signcat"][s])
        assert (za["signcat"][:signs[0, 0]] == IGNORE).all()
        meta = json.load(open(os.path.join(out, "_meta.json")))
        assert "ABER" in meta["connectives"] and len(meta["train_class_frames"]["signcat"]) == 7
    finally:
        shutil.rmtree(root)


def test_label_map_and_weights():
    from src.aux_labels import apply_label_map, inv_sqrt_weights, mapped_class_counts
    a = np.array([0, 6, 6, -100, 3], dtype=np.int8)
    assert apply_label_map(a, {"6": 0}).tolist() == [0, 0, 0, -100, 3]
    c = mapped_class_counts([100, 10, 10, 10, 10, 10, 40], {"6": 0}, 6)
    assert c == [140, 10, 10, 10, 10, 10]
    w = inv_sqrt_weights([400, 100, 0])
    assert w[2] == 0 and abs(np.mean(w[:2]) - 1) < 1e-9 and w[1] == 2 * w[0]


def test_by_context_table():
    import boundary_error_analysis as bea
    keys = [("v", 10), ("v", 50), ("v", 90), ("v", 130)]
    trans = {("v", 10): {"gloss_prev": "$GEST-OFF^", "gloss_cur": "HAUS1", "boundary_gap_type": "short_pause"},
             ("v", 50): {"gloss_prev": "HAUS1", "gloss_cur": "$INDEX1", "boundary_gap_type": "continuous"},
             ("v", 90): {"gloss_prev": "HAUS1", "gloss_cur": "ABER1", "boundary_gap_type": "long_pause"},
             ("v", 130): {"gloss_prev": "HAUS1", "gloss_cur": "GEHEN1", "boundary_gap_type": "long_pause"}}
    det = np.array([[1, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int8)
    runs = [{"config": "cfg"}, {"config": "cfg"}]
    d = tempfile.mkdtemp()
    try:
        args = types.SimpleNamespace(aux_meta=None, tolerance=5, out_dir=d, split="test")
        bea.by_context_table(args, keys, det, runs, trans, lambda s="": None)
        import csv
        row = next(csv.DictReader(open(os.path.join(d, "by_context_test.csv"))))
        assert float(row["prev = palm-up"]) == 0.5 and float(row["cur = pointing"]) == 0.0
        assert float(row["continuous (no pause)"]) == 0.0 and float(row["cur = connective"]) == 1.0
        assert float(row["other"]) == 1.0 and abs(float(row["all"]) - 5 / 8) < 1e-9
    finally:
        shutil.rmtree(d)


# ----------------------------------------------------------------------------- torch (model)
def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _tiny_model(**kw):
    from src.models import MultiStream_BiLSTM
    base = dict(in_channels=2, num_vertices=65 + 31, num_classes=3, d_model=32, n_layers=1, stgcn_channels=8,
                use_stream_body_hands=True, use_stream_face=True, use_stream_hamer=True, use_stream_cues=True,
                body_hands_proj_dim=16, face_proj_dim=16, hamer_proj_dim=16, hamer_dim=288,
                cue_dim=17, cue_proj_dim=8, cue_temporal_kernel=9)
    base.update(kw)
    return MultiStream_BiLSTM(**base)


def test_model_aux_heads_and_film():
    if not _have_torch():
        print("  (torch not installed: model test skipped)")
        return
    import torch
    torch.manual_seed(0)
    B, T = 2, 40
    x, hamer, cues = torch.randn(B, 2, T, 96), torch.randn(B, 288, T), torch.randn(B, 17, T)
    ref = _tiny_model()
    ref_keys = set(ref.state_dict())
    heads = {"mouth": {"stream": "face", "classes": 3}, "signcat": {"stream": "hamer", "classes": 7},
             "ctl": {"stream": "fused", "classes": 3}}
    m = _tiny_model(aux_heads=heads, film_cue_idx=[13, 14, 15, 16])
    out = m(x, hamer=hamer, cues=cues)
    assert isinstance(out, tuple) and out[0].shape == (B, 3, T)
    assert {k: tuple(v.shape) for k, v in m._aux_logits.items()} == {"mouth": (B, 3, T), "signcat": (B, 7, T),
                                                                     "ctl": (B, 3, T)}
    # old configs: identical parameter names -> strict loading of old checkpoints keeps working
    assert set(_tiny_model().state_dict()) == ref_keys
    # FiLM starts as identity: same weights with FiLM switched off give the same logits
    m.eval()
    with torch.no_grad():
        a = m(x, hamer=hamer, cues=cues)[0]
        idx = m.film_cue_idx
        m.film_cue_idx = None
        b = m(x, hamer=hamer, cues=cues)[0]
        m.film_cue_idx = idx
    assert torch.allclose(a, b, atol=1e-6)
    # auxiliary loss: finite with ignored frames, NaN-free gradient into the face branch only for 'mouth'
    m.train()
    m.zero_grad()
    m(x, hamer=hamer, cues=cues)
    tgt = torch.randint(0, 3, (B, T))
    tgt[:, :10] = -100
    lt = torch.nn.functional.cross_entropy(m._aux_logits["mouth"], tgt, ignore_index=-100)
    lt.backward()
    assert torch.isfinite(lt)
    assert m.branches["face"][0].weight.grad.abs().sum() > 0
    assert m.branches["hamer"][0].weight.grad is None or m.branches["hamer"][0].weight.grad.abs().sum() == 0


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
