"""
fusion_experiments.py -- the fusion-study arms for queue_train_phrase.py.

    from fusion_study.fusion_experiments import fusion_study_arms
    FUSION_STUDY = fusion_study_arms(R0)

Every arm is R0 (B+H+Fc, count loss w=2 r=5, best epoch by boundary F1@5, window 512) with ONE
change in how the streams are fused. Reference = the existing R0 runs (multistream_bilstm 310-312).

  F1a  global scalar gates (existing fusion "gated_sum")    control for F1: is a frame-wise gate
                                                           better than one fixed weight per stream?
  F1   frame-wise gate                                     idea 1
  F2   time-lagged face, content attention, +-25 frames    idea 2 (+-0.5 s at 50 fps)
  F2b  time-lagged face, content-free learned lag filter   ablation of F2: is a fixed learned lag
                                                           enough, or does the alignment depend on content?
  F3   late fusion, one BiLSTM + head per stream,           idea 3; per-stream heads also trained
       learned stream weights, deep supervision 0.5        alone -> per-stream predictions for the
                                                           complementarity analysis
  F5   face FiLM-modulates body+hands, face not            idea 5
       concatenated

Optional arms (not in the default list): F3g late fusion with frame-wise weights,
F5k FiLM + face also concatenated, F5f FiLM on the fused representation.
"""


def fusion_study_arms(r0):
    fusion = {**r0, "basename": "multistream_fusion"}
    return [
        # {**r0, "fusion": "gated_sum",
        #  "description": "F1a: R0 + global scalar stream gates (gated_sum) -- control for F1"},
        # {**fusion, "fusion_mode": "frame_gate",
        #  "description": "F1: R0 + frame-wise stream gate"},
        # {**fusion, "fusion_mode": "lag_face", "lag_attention": "content", "lag_max_frames": 25,
        #  "description": "F2: R0 + time-lagged face (content attention, +-25 frames)"},
        {**fusion, "fusion_mode": "lag_face", "lag_attention": "fixed", "lag_max_frames": 25,
         "description": "F2b: R0 + time-lagged face (fixed learned lag filter, +-25 frames)"},
        {**fusion, "fusion_mode": "late", "late_combine": "weighted", "late_aux_weight": 0.5,
         "description": "F3: late fusion, per-stream BiLSTM heads, learned stream weights, deep supervision 0.5"},
        {**fusion, "fusion_mode": "film_face", "film_target": "body_hands", "film_keep_face": False,
         "description": "F5: face FiLM-modulates body+hands (face not concatenated)"},
    ]


def fusion_optional_arms(r0):
    fusion = {**r0, "basename": "multistream_fusion"}
    return [
        {**fusion, "fusion_mode": "late", "late_combine": "gated", "late_aux_weight": 0.5,
         "description": "F3g: late fusion with frame-wise stream weights, deep supervision 0.5"},
        {**fusion, "fusion_mode": "film_face", "film_target": "body_hands", "film_keep_face": True,
         "description": "F5k: face FiLM-modulates body+hands AND is concatenated"},
        {**fusion, "fusion_mode": "film_face", "film_target": "fused", "film_keep_face": False,
         "description": "F5f: face FiLM-modulates the fused representation (face not concatenated)"},
    ]
