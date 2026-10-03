"""
Single source of truth for building a model from a run's config.

Used by BOTH train_phrase.py and evaluate_phrase.py, so an evaluated model is
always constructed with exactly the same hyperparameters (d_model, n_layers,
nhead, dim_feedforward, mamba settings, latent_dim, hamer_dim, dinov2_dim, ...)
as the one that was trained. The logic is moved here unchanged from
train_phrase.py.
"""
from src.models import (PureMambaBaseline, BiMambaBaseline, STGCN_Mamba, STGCN_MLP_Mamba,
                        STGCN_BiMamba, Decoupled_STGCN_Mamba, BiLSTM_Baseline, STGCN_BiLSTM,
                        TransformerBaseline, STGCN_Transformer, Latent_STGCN_Mamba,
                        CTRGCN_Mamba, InfoGCN_Mamba, ShiftGCN_Mamba, SpatialTransformer_Mamba,
                        HDGCN_Mamba, HyperSign_Mamba, STGCN_HybridSequential, STGCN_HybridParallel)

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
}

MAMBA_BASED_MODELS = ["pure_mamba", "bi_mamba", "stgcn_mamba", "stgcn_mlp_mamba", "stgcn_bimamba",
                      "decoupled_stgcn_mamba", "latent_stgcn_mamba", "ctrgcn_mamba", "infogcn_mamba",
                      "shiftgcn_mamba", "spatial_transformer_mamba", "hdgcn_mamba", "hypersign_mamba",
                      "stgcn_hybrid_seq", "stgcn_hybrid_parallel"]

HAMER_SUPPORTED_MODELS = ["stgcn_mamba", "latent_stgcn_mamba", "ctrgcn_mamba", "infogcn_mamba",
                          "shiftgcn_mamba", "spatial_transformer_mamba", "hdgcn_mamba", "hypersign_mamba",
                          "stgcn_bilstm", "stgcn_transformer",
                          "stgcn_mlp_mamba", "stgcn_bimamba", "decoupled_stgcn_mamba",
                          "stgcn_hybrid_seq", "stgcn_hybrid_parallel"]

# Own list (not an alias of HAMER_SUPPORTED_MODELS, so appending can't mutate it):
# the graph-free baselines also accept DINOv2, e.g. for pure_hamer + DINOv2 runs.
DINOV2_SUPPORTED_MODELS = list(HAMER_SUPPORTED_MODELS) + ["bilstm_baseline", "transformer_baseline"]


def build_model_kwargs(config, detected_hamer_dim=None, detected_dinov2_dim=None):
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

    if model_name in ["transformer_baseline", "stgcn_transformer"]:
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

    return model_class, model_kwargs


def call_model(model, config, features, hamer=None, dinov2=None):
    """Passes hamer/dinov2 only when the run uses them; returns the LOGITS (B, 3, T)."""
    kwargs = {}
    if config.get("use_hamer_features", False):
        kwargs["hamer"] = hamer
    if config.get("use_dinov2_features", False):
        kwargs["dinov2"] = dinov2
    output = model(features, **kwargs)
    return output[0] if isinstance(output, tuple) else output