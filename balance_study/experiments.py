"""
Arms of the modality-imbalance study (imported by queue_train_phrase.py). See balance_study/README.md.

Every arm is the B+H+Fc CE recipe of the reference runs 373-375 (_MS_BEST: multistream_bilstm, window 512,
v2 labels, weights [1.0, 1.1, 8.5], tolerance 1) with exactly one change, and selects its best epoch by
validation Frame F1 like 373-375 did. First pass: 1 seed each (press Enter at the seed prompt); reference =
multistream_bilstm-373 (seed 42), seed spread from 374/375.
"""
import os

ALL_STREAMS = ["body_hands", "face", "hamer"]

# Single-stream teachers for UMT: the existing single-stream runs with the same stream encoders as _MS_BEST
# (branch width 256, compact face, plain HaMeR). After the single-stream reruns below have trained, the
# teachers can be switched to them (UMT_TEACHERS_RERUN).
UMT_TEACHERS = {"body_hands": "multistream_bilstm-82", "hamer": "multistream_bilstm-133",
                "face": "multistream_bilstm-136"}


def _heads(detach):
    return {"balance_heads": list(ALL_STREAMS), "balance_head_detach": bool(detach)}


def imbalance_arms(base):
    b = {**base, "force_selection_metric": "frame_f1"}
    return [
        {**b, **_heads(True), "balance": {"method": "monitor"},
         "description": "IMB0 monitor: B+H+Fc + detached per-stream BIO heads (logging only, no balancing)"},
        {**b, "balance": {"method": "opm", "q_base": 0.3, "lam": 0.5},
         "description": "IMB-OPM: adaptive stream dropout (OPM; q_base 0.3, lambda 0.5; mono-modal scores by masking)"},
        {**b, "balance": {"method": "umt", "teachers": dict(UMT_TEACHERS), "lam": 1.0},
         "description": "IMB-UMT-all: feature distillation from single-stream teachers (body 82, HaMeR 133, face 136), lambda 1"},
        {**b, "balance": {"method": "umt", "teachers": {"face": UMT_TEACHERS["face"]}, "lam": 1.0},
         "description": "IMB-UMT-face: feature distillation from the face-only teacher (136) only, lambda 1"},
        {**b, **_heads(True), "balance": {"method": "cggm", "window": 100, "lam_gm": 0.0},
         "description": "IMB-CGGM-mag: per-stream LR scaled by relative head improvement (CGGM magnitude term)"},
        {**b, **_heads(True), "balance": {"method": "cggm", "window": 100, "lam_gm": 0.1},
        #  "description": "IMB-CGGM-full: CGGM magnitude + direction term (lambda_gm 0.1, first-order approximation)"},
        # {**b, "balance": {"method": "agm", "alpha": 1.0},
         "description": "IMB-AGM: Shapley mono-modal responses (7 coalitions, mean replacement), alpha 1"},
        {**b, "balance": {"method": "wu", "Q": 5, "alpha": "auto", "alpha_margin": 0.1, "warm_epochs": 1},
         "description": "IMB-Wu: conditional learning speed + guided re-balancing (Q 5, tolerance from warm-up + 0.1)"},
        {**b, "balance": {"method": "inforeg", "beta": 1.0, "K": 0.0, "c": 1.0},
         "description": "IMB-InfoReg: proximal slowing of information-sufficient streams in the prime window (beta 1, K 0)"},
        # {**b, **_heads(False), "balance": {"method": "uni", "lam": 1.0},
        #  "description": "IMB-UNI: + unimodal BIO loss on every stream, plain sum (MMPareto's uniform baseline)"},
        # {**b, **_heads(False), "balance": {"method": "mmpareto", "gamma": 1.5},
        #  "description": "IMB-MMPareto: Pareto integration of multimodal + unimodal gradients per encoder (gamma 1.5)"},
        {**b, **_heads(False), "balance": {"method": "dgl"},
         "description": "IMB-DGL (diagnostic): fusion sees detached streams; encoders trained by unimodal losses only"},
    ]


def single_stream_arms(cue_study_base):
    """Step 0 of the plan: the three single-stream models on THIS machine with the current code (learning
    speed curves; later also UMT teachers)."""
    b = {**cue_study_base, "force_selection_metric": "frame_f1"}
    return [
        {**b, "use_stream_hamer": False,
         "description": "SS-body: Body (+MediaPipe hands) only, same recipe as 373-375"},
        {**b, "use_stream_body_hands": False,
         "description": "SS-hamer: HaMeR only, same recipe as 373-375"},
        {**b, "use_stream_body_hands": False, "use_stream_hamer": False, "use_stream_face": True,
         "face_subset": "compact",
         "description": "SS-face: Face (compact) only, same recipe as 373-375"},
    ]


def missing_teachers(exp, exp_dir="experiments_phrase", model_dir="saved_models_phrase"):
    """Teacher runs of a UMT arm whose config or checkpoint is not on this machine."""
    teachers = ((exp.get("balance") or {}).get("teachers") or {}).values()
    return [t for t in teachers if not (os.path.exists(os.path.join(exp_dir, t, "hyperparameters.json"))
                                        and os.path.exists(os.path.join(model_dir, f"{t}.pth")))]
