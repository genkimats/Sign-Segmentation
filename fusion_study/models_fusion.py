"""
models_fusion.py -- MultiStream_Fusion: the multi-stream BiLSTM with alternative ways of fusing
the streams. Registered as basename "multistream_fusion" (src/model_factory.py), so the normal
queue -> train_phrase.py -> evaluate_phrase.py path trains and evaluates it like any other model.

The streams are computed EXACTLY as in src.models.MultiStreamSegmenter (same ST-GCN branches,
HaMeR front end, cue stream, stream dropout, ablation hook); only the fusion step differs:

  fusion_mode = "frame_gate"  (idea 1)  per-frame softmax weights over the streams (FrameGate);
                                        the global scalar-gate version is the existing
                                        fusion "gated_sum" of multistream_bilstm (control arm)
  fusion_mode = "lag_face"    (idea 2)  the face stream is re-aligned to the body stream by local
                                        attention over +-lag_max_frames (LocalLagAttention), then
                                        concatenated as usual
  fusion_mode = "late"        (idea 3)  every stream has its own BiLSTM + B/I/O head; the heads'
                                        log-probabilities are combined with learned weights
                                        (late_combine "weighted": one weight per stream; "gated":
                                        per-frame weights); every head is also trained on its own
                                        (deep supervision, extra_loss)
  fusion_mode = "film_face"   (idea 5)  the face stream modulates the body+hands stream (or the
                                        fused representation) by FiLM; by default it is NOT
                                        concatenated, so the face can act only as a modulator

forward() returns (logits (B, 3, T), embedding (B, D, T)) like every model in the project, and
leaves the analysis signals in model._trace (see fusion_analysis.py).

Options (config keys, all optional; defaults in FUSION_DEFAULTS):
  frame_gate : gate_hidden, gate_temporal_kernel, gate_temperature
  lag_face   : lag_max_frames, lag_heads, lag_head_dim, lag_attention ("content" | "fixed"),
               lag_bias_slope, lag_query ("body_hands" | "hamer")
  late       : late_d_model, late_n_layers, late_combine ("weighted" | "gated"), late_aux_weight
  film_face  : film_target ("body_hands" | "fused"), film_keep_face, film_hidden, film_temporal_kernel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_study.fusion_layers import FaceFiLM, FrameGate, LocalLagAttention, StreamHead
from src.models import MultiStreamSegmenter

FUSION_MODES = ("frame_gate", "lag_face", "late", "film_face")

FUSION_DEFAULTS = {
    "gate_hidden": 64, "gate_temporal_kernel": 9, "gate_temperature": 1.0,
    "lag_max_frames": 25, "lag_heads": 4, "lag_head_dim": 32, "lag_attention": "content",
    "lag_bias_slope": 0.1, "lag_query": "body_hands",
    "late_d_model": 128, "late_n_layers": 4, "late_combine": "weighted", "late_aux_weight": 0.5,
    "film_target": "body_hands", "film_keep_face": False, "film_hidden": 128, "film_temporal_kernel": 9,
}


def fusion_model_kwargs(config):
    """Model kwargs for a multistream_fusion run config (used by src/model_factory.py)."""
    mode = config.get("fusion_mode")
    if mode not in FUSION_MODES:
        raise ValueError(f"multistream_fusion needs fusion_mode in {FUSION_MODES}, got {mode!r}")
    opts = {k: config.get(k, v) for k, v in FUSION_DEFAULTS.items()}
    return {"fusion_mode": mode, "fusion_opts": opts}


class MultiStream_Fusion(MultiStreamSegmenter):
    def __init__(self, fusion_mode="frame_gate", fusion_opts=None, **kwargs):
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode!r}")
        o = dict(FUSION_DEFAULTS, **(fusion_opts or {}))
        unknown = set(o) - set(FUSION_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown fusion options {sorted(unknown)}")
        if kwargs.get("aux_heads") or kwargs.get("film_cue_idx"):
            raise ValueError("aux_heads / film_cue_idx are not combined with multistream_fusion -- use multistream_bilstm.")
        kwargs.pop("encoder", None)
        kwargs.pop("fusion", None)
        # frame_gate projects every branch to d_model (like gated_sum); the others keep concat widths
        super().__init__(encoder="bilstm", fusion="gated_sum" if fusion_mode == "frame_gate" else "concat", **kwargs)
        self.fusion_mode, self.fo = fusion_mode, o
        d_model, dropout = self.d_model, kwargs.get("dropout", 0.2)
        if fusion_mode == "late" and getattr(self, "change_point", None) is not None:
            raise ValueError("change_point_scales is not supported with late fusion.")

        if fusion_mode == "frame_gate":
            del self.gates                                          # replaced by the frame-wise gate
            self.frame_gate = FrameGate(len(self.streams), d_model, o["gate_hidden"], o["gate_temporal_kernel"],
                                        o["gate_temperature"])
            self.fusion = "frame_gate"

        elif fusion_mode == "lag_face":
            self._require("face", o["lag_query"])
            if o["lag_query"] == "face":
                raise ValueError("lag_query must be another stream than the face")
            self.lag = LocalLagAttention(self.stream_dims[o["lag_query"]], self.stream_dims["face"],
                                         self.stream_dims["face"], max_lag=o["lag_max_frames"], heads=o["lag_heads"],
                                         head_dim=o["lag_head_dim"], attention=o["lag_attention"],
                                         bias_slope=o["lag_bias_slope"], dropout=dropout)
            # self.fusion stays "concat": the aligned face replaces the face in the usual concatenation

        elif fusion_mode == "film_face":
            target = o["film_target"]
            if target not in ("body_hands", "fused"):
                raise ValueError("film_target must be 'body_hands' or 'fused'")
            self._require("face", "body_hands" if target == "body_hands" else None)
            self.concat_streams = [s for s in self.streams if s != "face" or o["film_keep_face"]]
            tdim = self.stream_dims["body_hands"] if target == "body_hands" else d_model
            self.film_face = FaceFiLM(self.stream_dims["face"], tdim, o["film_hidden"], o["film_temporal_kernel"])
            cat_dim = sum(self.stream_dims[s] for s in self.concat_streams)
            self.fusion_proj = nn.Sequential(nn.Linear(cat_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
                                             nn.Dropout(dropout))
            self.fusion = "film"

        elif fusion_mode == "late":
            del self.fusion_proj, self.lstm, self.classifier        # one encoder + head per stream instead
            self.stream_heads = nn.ModuleDict({s: StreamHead(self.stream_dims[s], o["late_d_model"],
                                                             o["late_n_layers"], dropout) for s in self.streams})
            if o["late_combine"] == "weighted":
                self.late_w = nn.Parameter(torch.zeros(len(self.streams)))
            elif o["late_combine"] == "gated":
                h = 2 * o["late_d_model"]
                self.late_gate = nn.Sequential(nn.Linear(len(self.streams) * h, 64), nn.GELU(),
                                               nn.Linear(64, len(self.streams)))
                nn.init.zeros_(self.late_gate[-1].weight)
                nn.init.zeros_(self.late_gate[-1].bias)
            else:
                raise ValueError("late_combine must be 'weighted' or 'gated'")
            self.fusion = "late"
        self._trace = {}
        self._stream_logits = {}

    def _require(self, *streams):
        for s in streams:
            if s is not None and s not in self.streams:
                raise ValueError(f"fusion_mode {self.fusion_mode!r} needs the {s!r} stream (use_stream_{s}=True)")

    # ---- the streams: identical to MultiStreamSegmenter.forward up to the fusion step ----------
    def _stream_features(self, x, hamer, cues):
        B = x.shape[0]
        z = {}
        if "body_hands" in self.streams:
            z["body_hands"] = self.branches["body_hands"](self._graph_stream(
                self.body_hands_stgcn, x[..., :getattr(self, "n_body_vertices", 65)],
                getattr(self, "body_hands_pool", None)))
        if "face" in self.streams:
            xf = x[..., 65:]
            if self.face_channels is not None:
                xf = xf[:, self.face_channels]
            z["face"] = self.branches["face"](self._graph_stream(self.face_stgcn, xf, getattr(self, "face_pool", None)))
        if "hamer" in self.streams:
            if hamer is None:
                raise ValueError("hamer stream is on but forward() got no `hamer` tensor.")
            if hasattr(self, "hamer_frontend"):
                z["hamer"] = self.branches["hamer"](self.hamer_frontend(hamer))
            else:
                z["hamer"] = self.branches["hamer"](hamer.permute(0, 2, 1))
            if hasattr(self, "hamer_temporal"):
                z["hamer"] = self.hamer_temporal(z["hamer"])
        if "cues" in self.streams:
            if cues is None:
                raise ValueError("cues stream is on but forward() got no `cues` tensor.")
            if self.cue_temporal_kernel > 1:
                cues = cues + self.cue_temporal(cues)
            z["cues"] = self.branches["cues"](cues.permute(0, 2, 1))

        if self.training and self.stream_dropout > 0 and len(self.streams) > 1:
            keep = (torch.rand(B, len(self.streams), device=x.device) >= self.stream_dropout)
            none_kept = ~keep.any(dim=1)
            if none_kept.any():
                pick = torch.randint(len(self.streams), (int(none_kept.sum()),), device=x.device)
                keep[none_kept.nonzero(as_tuple=True)[0], pick] = True
            for j, name in enumerate(self.streams):
                z[name] = z[name] * keep[:, j].to(z[name].dtype)[:, None, None]

        if self._ablation is not None:
            name, value = self._ablation
            z[name] = value.to(z[name].dtype).expand_as(z[name])
        return z

    def _concat_project(self, z, names):
        """Concatenation + shared projection over `names`; reports contributions to the capture hook."""
        fused = self.fusion_proj(torch.cat([z[n] for n in names], dim=-1))
        contrib = {}
        if self._capture is not None:
            W, start = self.fusion_proj[0].weight, 0
            for n in names:
                d = self.stream_dims[n]
                contrib[n] = z[n] @ W[:, start:start + d].T
                start += d
        return fused, contrib

    def _encode(self, fused):
        if hasattr(self, "change_point"):
            fused = self.change_point(fused)
        emb, _ = self.lstm(fused)
        return self.classifier(emb).permute(0, 2, 1), emb.permute(0, 2, 1)

    def forward(self, x, hamer=None, dinov2=None, cues=None):
        z = self._stream_features(x, hamer, cues)
        trace = {}
        self._aux_logits = {}

        if self.fusion_mode == "frame_gate":
            fused, w = self.frame_gate([z[s] for s in self.streams])
            fused = self.fusion_norm(fused)
            trace.update(self.frame_gate.trace)
            if self._capture is not None:
                S = len(self.streams)
                self._capture(z, {s: S * w[..., i:i + 1] * z[s] for i, s in enumerate(self.streams)})
            logits, emb = self._encode(fused)

        elif self.fusion_mode == "lag_face":
            z = dict(z)
            z["face"] = self.lag(z[self.fo["lag_query"]], z["face"])
            trace.update(self.lag.trace)
            fused, contrib = self._concat_project(z, self.streams)
            if self._capture is not None:
                self._capture(z, contrib)
            logits, emb = self._encode(fused)

        elif self.fusion_mode == "film_face":
            if self.fo["film_target"] == "body_hands":
                zb = z["body_hands"]
                zm = dict(z)
                zm["body_hands"] = self.film_face(z["face"], zb)
                fused, contrib = self._concat_project(zm, self.concat_streams)
                if self._capture is not None and "face" not in contrib:
                    # the face's contribution = how much it moved the body's part of the fused input
                    start = sum(self.stream_dims[n] for n in self.concat_streams[:self.concat_streams.index("body_hands")])
                    W = self.fusion_proj[0].weight[:, start:start + self.stream_dims["body_hands"]]
                    contrib["face"] = (zm["body_hands"] - zb) @ W.T
                    contrib["body_hands"] = zb @ W.T
            else:
                fused0, contrib = self._concat_project(z, self.concat_streams)
                fused = self.film_face(z["face"], fused0)
                if self._capture is not None and "face" not in contrib:
                    contrib["face"] = fused - fused0
            trace.update(self.film_face.trace)
            if self._capture is not None:
                self._capture(z, contrib)
            logits, emb = self._encode(fused)

        else:  # late
            outs = {s: self.stream_heads[s](z[s]) for s in self.streams}
            self._stream_logits = {s: o[0] for s, o in outs.items()}
            logp = torch.stack([F.log_softmax(o[0].float(), dim=1) for o in outs.values()], dim=1)  # (B, S, 3, T)
            if self.fo["late_combine"] == "weighted":
                w = torch.softmax(self.late_w, dim=0).view(1, -1, 1, 1).expand(logp.shape[0], -1, 1, logp.shape[-1])
            else:
                hcat = torch.cat([o[1] for o in outs.values()], dim=-1)                          # (B, T, S*2d)
                w = torch.softmax(self.late_gate(hcat), dim=-1).permute(0, 2, 1).unsqueeze(2)    # (B, S, 1, T)
            logits = (w * logp).sum(1)                                                           # (B, 3, T)
            emb = torch.cat([o[1] for o in outs.values()], dim=-1).permute(0, 2, 1)
            trace["stream_probs"] = logp.detach().exp()                                          # (B, S, 3, T)
            trace["late_weight"] = w.detach()[:, :, 0, :].permute(0, 2, 1)                       # (B, T, S)
            if self._capture is not None:
                self._capture(z, {s: (w[:, i] * logp[:, i]).permute(0, 2, 1) for i, s in enumerate(self.streams)})

        self._trace = trace
        return logits, emb

    # ---- hooks used by train_phrase.py / check_stream_balance.py ------------------------------
    def extra_loss(self, loss_fn):
        """Late fusion: every stream head is also trained on its own with the run's loss (deep
        supervision), weighted late_aux_weight / n_streams. Zero for the other modes."""
        if self.fusion_mode != "late" or not self._stream_logits:
            return torch.zeros((), device=next(self.parameters()).device)
        lam = float(self.fo["late_aux_weight"])
        if lam <= 0:
            return torch.zeros((), device=next(self.parameters()).device)
        return lam * sum(loss_fn(lg) for lg in self._stream_logits.values()) / len(self._stream_logits)

    def stream_gates(self):
        """check_stream_balance.py reads this for non-concat fusions ('W col norm' column)."""
        if self.fusion_mode == "late" and hasattr(self, "late_w"):
            return {s: float(w) for s, w in zip(self.streams, torch.softmax(self.late_w.detach(), 0).cpu())}
        return {s: 1.0 for s in self.streams}
