"""
Single source of truth for building a model from a run's config.

Used by BOTH train_phrase.py and evaluate_phrase.py, so an evaluated model is
always constructed with exactly the same hyperparameters (d_model, n_layers,
nhead, dim_feedforward, mamba settings, latent_dim, hamer_dim, dinov2_dim, ...)
as the one that was trained. The logic is moved here unchanged from
train_phrase.py.
"""
import inspect
import json
import os

from src.models import (PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_MLP_Mamba,
                        STGCN_BiMamba, Decoupled_STGCN_Mamba, BiLSTM_Baseline, STGCN_BiLSTM,
                        TransformerBaseline, STGCN_Transformer, Latent_STGCN_Mamba,
                        CTRGCN_Mamba, InfoGCN_Mamba, ShiftGCN_Mamba, SpatialTransformer_Mamba,
                        HDGCN_Mamba, HyperSign_Mamba, STGCN_HybridSequential, STGCN_HybridParallel,
                        MultiStreamSegmenter, MultiStream_BiLSTM, MultiStream_Transformer,
                        MultiStream_BiLSTMAttn, Moryossef2023_BiLSTM)

MODEL_REGISTRY = {
    "pure_mamba": PureMambaBaseline,
    "bi_mamba": BiMambaBaseline,
    "stgcn_mamba": STGCN_Mamba,
    "stgcn_mlp_mamba": STGCN_MLP_Mamba,
    "stgcn_bimamba": STGCN_BiMamba,
    "decoupled_stgcn_mamba": Decoupled_STGCN_Mamba,
    "bilstm_baseline": BiLSTM_Baseline,
    "stgcn_bilstm": STGCN_BiLSTM,
    "transformer_baseline": TransformerBaseline,
    "stgcn_transformer": STGCN_Transformer,
    "latent_stgcn_mamba": Latent_STGCN_Mamba,
    "ctrgcn_mamba": CTRGCN_Mamba,
    "infogcn_mamba": InfoGCN_Mamba,
    "shiftgcn_mamba": ShiftGCN_Mamba,
    "spatial_transformer_mamba": SpatialTransformer_Mamba,
    "hdgcn_mamba": HDGCN_Mamba,
    "hypersign_mamba": HyperSign_Mamba,
    "stgcn_hybrid_seq": STGCN_HybridSequential,
    "stgcn_hybrid_parallel": STGCN_HybridParallel,
    "multistream_bilstm": MultiStream_BiLSTM,
    "multistream_transformer": MultiStream_Transformer,
    "multistream_bilstm_attn": MultiStream_BiLSTMAttn,
    # Moryossef et al. 2023 E1s / E2s (phrase head), see src/models.py
    "bilstm_2023": Moryossef2023_BiLSTM,
}

MULTISTREAM_MODELS = ["multistream_bilstm", "multistream_transformer", "multistream_bilstm_attn"]

MAMBA_BASED_MODELS = ["pure_mamba", "bi_mamba", "stgcn_mamba", "stgcn_mlp_mamba", "stgcn_bimamba",
                      "decoupled_stgcn_mamba", "latent_stgcn_mamba", "ctrgcn_mamba", "infogcn_mamba",
                      "shiftgcn_mamba", "spatial_transformer_mamba", "hdgcn_mamba", "hypersign_mamba",
                      "stgcn_hybrid_seq", "stgcn_hybrid_parallel"]

HAMER_SUPPORTED_MODELS = ["stgcn_mamba", "latent_stgcn_mamba", "ctrgcn_mamba", "infogcn_mamba",
                          "shiftgcn_mamba", "spatial_transformer_mamba", "hdgcn_mamba", "hypersign_mamba",
                          "stgcn_bilstm", "stgcn_transformer", "bilstm_2023",
                          "stgcn_mlp_mamba", "stgcn_bimamba", "decoupled_stgcn_mamba",
                          "stgcn_hybrid_seq", "stgcn_hybrid_parallel",
                          "multistream_bilstm", "multistream_transformer", "multistream_bilstm_attn"]

# Own list (not an alias of HAMER_SUPPORTED_MODELS, so appending can't mutate it):
# the graph-free baselines also accept DINOv2, e.g. for pure_hamer + DINOv2 runs.
DINOV2_SUPPORTED_MODELS = list(HAMER_SUPPORTED_MODELS) + ["bilstm_baseline", "transformer_baseline"]


def resolve_cue_dim(config, detected_cue_dim=None):
    """Number of cue channels of the run's cue_set: config "cue_dim" > the dataset's count >
    the cue files' _names.json (so evaluation scripts need no extra argument)."""
    if config.get("cue_dim"):
        return int(config["cue_dim"])
    if detected_cue_dim:
        return int(detected_cue_dim)
    from src.cue_sets import resolve_cue_set
    from src.cue_sets import cue_names_path
    names_path = cue_names_path(config.get("cue_dir", "processed_data/cue_features"))
    file_names = None
    if os.path.exists(names_path):
        with open(names_path) as f:
            file_names = json.load(f)
    names, _ = resolve_cue_set(config.get("cue_set", "selected"), file_names)
    return len(names)


def resolve_film_cue_idx(config):
    """Idea A3: positions of config["film_cue_names"] within the run's cue_set channels."""
    from src.cue_sets import resolve_cue_set, cue_names_path
    names_path = cue_names_path(config.get("cue_dir", "processed_data/cue_features"))
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"film_cue_names needs {names_path} (run the cue / tempo extractor first).")
    with open(names_path) as f:
        file_names = json.load(f)
    names, _ = resolve_cue_set(config.get("cue_set", "selected"), file_names)
    missing = [n for n in config["film_cue_names"] if n not in names]
    if missing:
        raise ValueError(f"film_cue_names {missing} are not among the run's cue channels {names}.")
    return [names.index(n) for n in config["film_cue_names"]]


def build_model_kwargs(config, detected_hamer_dim=None, detected_dinov2_dim=None, detected_cue_dim=None):
    """
    Returns (model_class, model_kwargs) for a run config.

    detected_hamer_dim / detected_dinov2_dim: the dimensions observed in the data
    (SignSegmentationDataset.detected_hamer_dim / detected_dinov2_dim). An explicit
    "hamer_dim" / "dinov2_dim" in the config overrides them, exactly as before.
    """
    model_name = config["basename"]
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in MODEL_REGISTRY.")

    d_model = config["d_model"]
    model_kwargs = {
        "in_channels": config["in_channels"],
        "num_vertices": config["num_vertices"],
        "num_classes": 3,
        "d_model": d_model,
        "n_layers": config["n_layers"],
    }

    if model_name in ["transformer_baseline", "stgcn_transformer", "multistream_transformer",
                      "multistream_bilstm_attn"]:
        model_kwargs["nhead"] = config.get("nhead", 8)
        model_kwargs["dim_feedforward"] = config.get("dim_feedforward", d_model * 4)

    if model_name == "stgcn_mlp_mamba":
        model_kwargs["mlp_expansion_factor"] = config.get("mlp_expansion_factor", 4)

    if model_name in ["latent_stgcn_mamba", "ctrgcn_mamba", "infogcn_mamba", "shiftgcn_mamba",
                      "spatial_transformer_mamba", "hdgcn_mamba", "hypersign_mamba"]:
        model_kwargs["latent_dim"] = config.get("latent_dim", 128)

    if model_name in MAMBA_BASED_MODELS:
        model_kwargs["mamba_d_state"] = config.get("mamba_d_state", 16)
        model_kwargs["mamba_d_conv"] = config.get("mamba_d_conv", 4)
        model_kwargs["mamba_expand"] = config.get("mamba_expand", 2)

    if model_name == "bilstm_2023":
        # HaMeR only: the keypoint tensor is still loaded but not fed to the model
        model_kwargs["use_pose_input"] = bool(config.get("use_pose_input", True))
        # optional temporal conv on the fused vector (1 = off, the faithful 2023 model)
        model_kwargs["temporal_kernel"] = int(config.get("temporal_kernel", 1))
        model_kwargs["temporal_layers"] = int(config.get("temporal_layers", 2))
        model_kwargs["temporal_dropout"] = float(config.get("temporal_dropout", 0.0))

    if config.get("use_hamer_features", False):
        if model_name not in HAMER_SUPPORTED_MODELS:
            raise ValueError(
                f"use_hamer_features=True but model '{model_name}' doesn't have a hamer_dim "
                f"argument implemented yet. Supported models: {HAMER_SUPPORTED_MODELS}."
            )
        hamer_dim = config.get("hamer_dim", detected_hamer_dim)
        if hamer_dim is None:
            raise RuntimeError("use_hamer_features=True but no hamer_dim was detected in the data "
                               "and none is set in the config.")
        model_kwargs["hamer_dim"] = hamer_dim

    if config.get("use_dinov2_features", False):
        if model_name not in DINOV2_SUPPORTED_MODELS:
            raise ValueError(
                f"use_dinov2_features=True but model '{model_name}' doesn't have a dinov2_dim "
                f"argument implemented yet. Supported models: {DINOV2_SUPPORTED_MODELS}."
            )
        dinov2_dim = config.get("dinov2_dim", detected_dinov2_dim)
        if dinov2_dim is None:
            raise RuntimeError("use_dinov2_features=True but no dinov2_dim was detected in the data "
                               "and none is set in the config.")
        model_kwargs["dinov2_dim"] = dinov2_dim

    # Optional per-stream branch sizes (stream balancing). Only passed when set in the
    # config, so old configs/checkpoints build exactly as before. A model that doesn't
    # support a requested size raises instead of silently ignoring it.
    sig_class = MultiStreamSegmenter if issubclass(model_class, MultiStreamSegmenter) else model_class
    accepted = inspect.signature(sig_class.__init__).parameters
    for key in ("stgcn_proj_dim", "hamer_proj_dim", "dinov2_proj_dim"):
        value = config.get(key)
        if value is None:
            continue
        if key not in accepted:
            raise ValueError(f"Config sets {key}={value}, but model '{model_name}' has no "
                             f"'{key}' argument. Remove it from the config or use a model that supports it.")
        model_kwargs[key] = int(value)

    # Multi-stream models: which streams are on, how they are fused, branch sizes.
    if model_name in MULTISTREAM_MODELS:
        if config.get("face_only", False):
            raise ValueError("face_only=True is not used with multistream models -- turn the other "
                             "streams off instead (use_stream_body_hands / use_stream_hamer = False).")
        streams = {
            "use_stream_body_hands": bool(config.get("use_stream_body_hands", True)),
            "use_stream_face": bool(config.get("use_stream_face", False)),
            "use_stream_hamer": bool(config.get("use_stream_hamer", False)),
            "use_stream_cues": bool(config.get("use_stream_cues", False)),
        }
        if not any(streams.values()):
            raise ValueError("All streams are off -- turn on at least one of use_stream_body_hands / "
                             "use_stream_face / use_stream_hamer / use_stream_cues.")
        if streams["use_stream_cues"]:
            if not config.get("use_cue_features", False):
                raise ValueError("use_stream_cues=True needs use_cue_features=True (the queue sets it automatically).")
            model_kwargs["cue_dim"] = resolve_cue_dim(config, detected_cue_dim)
            model_kwargs["cue_proj_dim"] = int(config.get("cue_proj_dim", 128))
            model_kwargs["cue_temporal_kernel"] = int(config.get("cue_temporal_kernel", 1))
        if streams["use_stream_face"] and not config.get("use_face_keypoints", False):
            raise ValueError("use_stream_face=True needs use_face_keypoints=True (the queue sets it automatically).")
        if streams["use_stream_hamer"] and not config.get("use_hamer_features", False):
            raise ValueError("use_stream_hamer=True needs use_hamer_features=True (the queue sets it automatically).")
        model_kwargs.update(streams)
        model_kwargs["fusion"] = config.get("fusion", "concat")
        if streams["use_stream_face"] and int(config.get("face_dims", 3)) == 2:
            # 2D face: the face stream reads only the x / y channels (see dataset.py).
            coord_order = [f for f in config["base_features"] if f in ("x-cord", "y-cord", "z-cord")]
            model_kwargs["face_channels"] = [coord_order.index(f) for f in ("x-cord", "y-cord")
                                             if f in coord_order]
        for key in ("body_hands_proj_dim", "face_proj_dim"):  # hamer_proj_dim is handled above
            if config.get(key) is not None:
                model_kwargs[key] = int(config[key])
        # Front-end options (src/frontend.py). Defaults = the original model, so older
        # configs / checkpoints rebuild unchanged.
        model_kwargs["vertex_pooling"] = config.get("vertex_pooling", "flatten")
        model_kwargs["stream_encoder"] = config.get("stream_encoder", "stgcn")
        model_kwargs["stream_encoder_layers"] = int(config.get("stream_encoder_layers", 2))
        model_kwargs["blockgcn_groups"] = int(config.get("blockgcn_groups", 8))
        model_kwargs["blockgcn_ph"] = bool(config.get("blockgcn_ph", True))
        model_kwargs["skate_local_frames"] = int(config.get("skate_local_frames", 32))
        model_kwargs["stream_dropout"] = float(config.get("stream_dropout", 0.0))
        if config.get("change_point_scales"):
            model_kwargs["change_point_scales"] = [int(w) for w in config["change_point_scales"]]
        # Idea B: stream-specific auxiliary heads; Idea A3: FiLM on video-level cues.
        # Both are absent from older configs, so their checkpoints rebuild unchanged.
        if config.get("aux_heads"):
            model_kwargs["aux_heads"] = {t: {"stream": spec["stream"], "classes": int(spec["classes"])}
                                         for t, spec in config["aux_heads"].items()}
            model_kwargs["aux_temporal_kernel"] = int(config.get("aux_temporal_kernel", 9))
        if config.get("film_cue_names"):
            if not streams["use_stream_cues"]:
                raise ValueError("film_cue_names needs use_stream_cues=True.")
            model_kwargs["film_cue_idx"] = resolve_film_cue_idx(config)
        if streams["use_stream_hamer"]:
            model_kwargs["hamer_rep"] = config.get("hamer_rep", "matrix")
            model_kwargs["hamer_mask"] = bool(config.get("hamer_mask", False))
            model_kwargs["hamer_motion"] = bool(config.get("hamer_motion", False))
            model_kwargs["hamer_motion_lag"] = int(config.get("hamer_motion_lag", 1))
            model_kwargs["hamer_temporal_kernel"] = int(config.get("hamer_temporal_kernel", 1))
        if not streams["use_stream_hamer"]:
            model_kwargs.pop("hamer_dim", None)
        if model_name == "multistream_bilstm_attn":
            # Attention head on top of the BiLSTM: pre-norm by default, 1 layer by default.
            model_kwargs["pos_encoding"] = config.get("pos_encoding", "rope")
            model_kwargs["transformer_norm"] = config.get("transformer_norm", "pre")
            model_kwargs["rope_base"] = float(config.get("rope_base", 10000.0))
            model_kwargs["attn_layers"] = int(config.get("attn_layers", 1))
        if model_name == "multistream_transformer":
            # Position encoding / norm placement of the Transformer encoder. Defaults
            # (sinusoidal + post) = the original encoder, so older runs rebuild unchanged.
            model_kwargs["pos_encoding"] = config.get("pos_encoding", "sinusoidal")
            model_kwargs["transformer_norm"] = config.get("transformer_norm", "post")
            model_kwargs["rope_base"] = float(config.get("rope_base", 10000.0))

    if model_name not in MULTISTREAM_MODELS and (config.get("aux_heads") or config.get("film_cue_names")):
        raise ValueError(f"aux_heads / film_cue_names are implemented for multistream models only, not '{model_name}'.")

    # Face-only input (dataset face_only mode): graph models must build a face-only graph.
    if config.get("face_only", False):
        if "face_only" in accepted:
            model_kwargs["face_only"] = True
        elif model_name not in ("bilstm_baseline", "transformer_baseline", "bilstm_2023"):  # graph-free: any vertex set works
            raise ValueError(f"face_only=True but model '{model_name}' has no face-only graph. "
                             f"Use stgcn_bilstm, stgcn_transformer, bilstm_baseline or transformer_baseline.")

    return model_class, model_kwargs


def call_model(model, config, features, hamer=None, dinov2=None, cues=None):
    """Passes hamer/dinov2/cues only when the run uses them; returns the LOGITS (B, 3, T)."""
    kwargs = {}
    if config.get("use_hamer_features", False):
        kwargs["hamer"] = hamer
    if config.get("use_dinov2_features", False):
        kwargs["dinov2"] = dinov2
    if config.get("use_cue_features", False):
        kwargs["cues"] = cues
    output = model(features, **kwargs)
    return output[0] if isinstance(output, tuple) else output