"""
balance_study/balancer.py -- modality-imbalance ("greedy learner") methods for multistream_bilstm.

train_phrase.py hands every training batch to Balancer.step() when a run's config has
    "balance": {"method": <name>, ...options}
Everything else (data, validation, checkpoint selection, saving) is the normal train_phrase.py path.

Methods (balance_study/README.md has the details and the deviations from each paper):
  monitor   per-stream BIO heads that only watch (detached); no balancing -- the control, and the
            per-stream learning curves inside the joint model
  opm       OPM (Wei et al., TPAMI 2024): drop the dominant streams with a probability set by their
            mono-modal scores (non-late fusion: keep one stream, zero the others)
  umt       Uni-Modal Teacher (Du et al., ICML 2023): MSE between each stream's branch output and a frozen
            single-stream model's branch output
  cggm      CGGM (Guo et al., NeurIPS 2024): per-stream learning rate scaled by the relative improvement of
            the stream heads (magnitude term); lam_gm > 0 adds the direction term (approximation, README)
  agm       AGM (Li et al., ICCV 2023): Shapley mono-modal responses from all stream coalitions (others
            replaced by their running mean); each response's gradient scaled by kappa_m
  wu        Wu et al. (ICML 2022): conditional learning speed + guided re-balancing steps (the other streams
            replaced by their running mean, so the model must learn from the lagging stream)
  inforeg   InfoReg (Huang et al., CVPR 2025): in the prime learning window, a proximal term slows the
            information-sufficient streams
  uni       unimodal BIO losses on every stream, plainly added (the "uniform baseline" of MMPareto)
  mmpareto  MMPareto (Wei & Hu, ICML 2024): per encoder, Pareto integration of the multimodal and unimodal
            gradients, rescaled to gamma * |g_m + g_u|
  dgl       DGL (Wei et al., ICCV 2025), diagnostic: the fusion sees detached stream features; the encoders
            learn only from their own unimodal losses

Gradient-MAGNITUDE methods (CGGM, and InfoReg's slowing) act through per-stream LEARNING RATES, not by
scaling gradients: with AdamW a constant gradient scale is divided out by the second-moment estimate within
~1/(1 - beta2) = 1000 steps, so scaling gradients would do almost nothing. Scaling the step size is the
optimizer-independent equivalent of "scale this stream's update".

Every method also logs, per epoch, Wu's effective updates mu = |grad|^2 / |theta|^2 for each stream's encoder
and its fusion columns, the conditional learning speeds and the gradient energy (InfoReg's Fisher-trace
proxy) to experiments_phrase/<run>/balance_log.csv.
"""
import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from balance_study import core

METHODS = ("monitor", "opm", "umt", "cggm", "agm", "wu", "inforeg", "uni", "mmpareto", "dgl")
# methods with per-stream heads, and whether the heads must be detached (monitor-only)
HEAD_METHODS = {"monitor": True, "cggm": True, "uni": False, "mmpareto": False, "dgl": False}

# parameter-name prefixes of each stream's encoder (everything before the fusion projection)
STREAM_PREFIXES = {
    "body_hands": ("body_hands_stgcn.", "body_hands_pool.", "branches.body_hands.", "lhand_stgcn.",
                   "rhand_stgcn.", "hand_branches.", "hand_stgcn.", "hand_branch."),
    "face": ("face_stgcn.", "face_pool.", "branches.face."),
    "hamer": ("hamer_shape.", "hamer_tree.", "hamer_frontend.", "branches.hamer.", "hamer_temporal."),
    "cues": ("cue_temporal.", "branches.cues."),
}


def validate_balance_config(config):
    """Raises if the run combines "balance" with something it does not support."""
    b = config.get("balance") or {}
    method = b.get("method")
    if method not in METHODS:
        raise ValueError(f"balance.method must be one of {METHODS}, got '{method}'.")
    if config.get("basename") != "multistream_bilstm":
        raise ValueError("balance is implemented for basename 'multistream_bilstm' only.")
    if config.get("fusion", "concat") != "concat":
        raise ValueError("balance needs fusion 'concat' (the fusion-column statistics).")
    if config.get("loss_function") != "weighted_ce":
        raise ValueError("balance needs loss_function 'weighted_ce'.")
    if config.get("use_full_length"):
        raise ValueError("balance is implemented for windowed training only.")
    if config.get("soft_label_dir") or float(config.get("count_loss_weight", 0.0) or 0.0) > 0:
        raise ValueError("balance is not combined with soft labels / the count loss (use the CE recipe).")
    if config.get("aux_heads") or config.get("sign_head") or config.get("use_dinov2_features"):
        raise ValueError("balance is not combined with aux_heads / sign_head / DINOv2.")
    if int(config.get("temporal_downsample_factor", 1) or 1) != 1:
        raise ValueError("balance assumes temporal_downsample_factor 1.")
    if method in HEAD_METHODS:
        if not config.get("balance_heads"):
            raise ValueError(f"balance method '{method}' needs balance_heads (one head per stream).")
        if bool(config.get("balance_head_detach", False)) != HEAD_METHODS[method]:
            raise ValueError(f"balance method '{method}' needs balance_head_detach={HEAD_METHODS[method]}.")
    if method == "umt" and not b.get("teachers"):
        raise ValueError("balance method 'umt' needs balance.teachers = {stream: run_name}.")
    if method == "cggm" and float(b.get("lam_gm", 0.0)) > 0:
        if config.get("balance_head_hidden") not in (None, config["d_model"]):
            raise ValueError("CGGM's direction term needs balance_head_hidden == d_model (same classifier shape).")


def load_run_model(run_name, device, detected=None, exp_dir="experiments_phrase", model_dir="saved_models_phrase"):
    """Rebuilds a trained run exactly as evaluate_phrase.py does (strict load), frozen, in eval mode."""
    from src.model_factory import build_model_kwargs
    with open(os.path.join(exp_dir, run_name, "hyperparameters.json")) as f:
        cfg = json.load(f)
    detected = detected or {}
    cls, kwargs = build_model_kwargs(cfg, detected_hamer_dim=detected.get("hamer"),
                                     detected_dinov2_dim=detected.get("dinov2"),
                                     detected_cue_dim=detected.get("cue"))
    model = cls(**kwargs).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, f"{run_name}.pth"), map_location=device), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


class Balancer:
    def __init__(self, config, model, device, class_weights, exp_dir=None, detected=None, teachers=None):
        validate_balance_config(config)
        self.cfg = dict(config["balance"])
        self.method = self.cfg["method"]
        self.model = model
        self.device = device
        self.w = torch.tensor(class_weights, dtype=torch.float, device=device)
        self.streams = list(model.streams)
        self.M = len(self.streams)
        self.use_hamer = bool(config.get("use_hamer_features", False))
        self.use_cues = bool(config.get("use_cue_features", False))
        self.log_path = os.path.join(exp_dir, "balance_log.csv") if exp_dir else None
        self._header = None
        self.global_step = 0
        self.epoch = 0

        if self.method in HEAD_METHODS:
            if sorted(model.bal_head_streams) != sorted(self.streams):
                raise ValueError(f"balance method '{self.method}' needs a head on every stream {self.streams}, "
                                 f"got {model.bal_head_streams}.")
            if model.bal_head_detach != HEAD_METHODS[self.method]:
                raise ValueError(f"balance method '{self.method}': heads must have detach={HEAD_METHODS[self.method]}.")

        # parameter groups: one per stream encoder, the heads, the rest (fusion, BiLSTM, classifier)
        self.groups = {k: [] for k in self.streams + ["heads", "rest"]}
        for name, p in model.named_parameters():
            if not p.requires_grad:      # frozen (e.g. the ST-GCN adjacency matrices): never updated, no gradient
                continue
            g = "rest"
            if name.startswith("bal_heads."):
                g = "heads"
            else:
                for s in self.streams:
                    if name.startswith(STREAM_PREFIXES.get(s, ())):
                        g = s
                        break
            self.groups[g].append(p)
        for s in self.streams:
            if not self.groups[s]:
                raise RuntimeError(f"no encoder parameters found for stream '{s}' (STREAM_PREFIXES out of date?).")
        start = 0
        self.fusion_slices = {}
        for s in self.streams:
            d = model.stream_dims[s]
            self.fusion_slices[s] = (start, start + d)
            start += d
        n_par = {k: sum(p.numel() for p in v) for k, v in self.groups.items()}
        print(f"⚖️  balance '{self.method}': parameters per group {n_par}")

        # running mean of every stream's features (mean replacement for AGM / Wu)
        self.fill = {s: None for s in self.streams}
        self.fill_momentum = float(self.cfg.get("fill_momentum", 0.01))
        self.cum_enc = np.zeros(self.M)       # cumulative mu of the encoders (Wu: regular steps only)
        self.cum_fus = np.zeros(self.M)

        m = self.method
        if m == "opm":
            self.q_base = float(self.cfg.get("q_base", 0.3))
            self.lam = float(self.cfg.get("lam", 0.5))
            self.score_every = int(self.cfg.get("score_every", 1))
            self.opm_q = None
            self.opm_fallback = self.M - 1
        elif m == "umt":
            self.teacher_lam = {s: float(self.cfg.get("lam", 1.0)) for s in self.cfg["teachers"]}
            for s in self.cfg["teachers"]:
                if s not in self.streams:
                    raise ValueError(f"UMT teacher for stream '{s}', which is off.")
            self.teachers = teachers if teachers is not None else {
                s: load_run_model(run, device, detected)[0] for s, run in self.cfg["teachers"].items()}
            self._cap_s, self._cap_t, self._hooks = {}, {}, []
            for s, tm in self.teachers.items():
                if s not in tm.streams or tm.stream_dims[s] != model.stream_dims[s]:
                    raise ValueError(f"UMT teacher for '{s}' has no such stream or a different width.")
                self._hooks.append(self._pre_dropout_hook(model, s, self._cap_s))
                self._hooks.append(self._pre_dropout_hook(tm, s, self._cap_t))
        elif m == "cggm":
            self.rho = self.cfg.get("rho")
            self.lam_gm = float(self.cfg.get("lam_gm", 0.0))
            self.tracker = core.ImprovementTracker(self.M, beta=float(self.cfg.get("ema", 0.98)),
                                                   window=int(self.cfg.get("window", 100)))
            self.B = np.ones(self.M)
            self.b_lo, self.b_hi = float(self.cfg.get("b_min", 0.1)), float(self.cfg.get("b_max", 3.0))
        elif m == "agm":
            self.alpha = float(self.cfg.get("alpha", 1.0))
            self.coals, Wm = core.shapley_matrix(self.M)
            if self.coals[-1] != tuple(range(self.M)):
                raise RuntimeError("the full coalition must come last")
            self.shapley_W = torch.tensor(Wm, dtype=torch.float, device=device)
            self.s_hat = None
            self.agm_t = 0
        elif m == "wu":
            self.sched = core.WuSchedule(int(self.cfg.get("Q", 5)))
            self.warm_epochs = int(self.cfg.get("warm_epochs", 1))
            a = self.cfg.get("alpha", "auto")
            self.alpha_auto = (a == "auto")
            self.alpha = None if self.alpha_auto else float(a)
            self.alpha_margin = float(self.cfg.get("alpha_margin", 0.1))
            self.wu_d = 0.0
        elif m == "inforeg":
            self.beta = float(self.cfg.get("beta", 1.0))
            self.K = float(self.cfg.get("K", 0.0))
            self.c = float(self.cfg.get("c", 1.0))
            self.score_every = int(self.cfg.get("score_every", 1))
            self.tr_hist = {}                 # epoch -> per-stream mean |grad|^2 of the joint loss
            self.ir_active = np.zeros(self.M, dtype=bool)
            self.anchors = {}
            self.ir_delta = np.zeros(self.M)
        elif m == "uni":
            self.lam = float(self.cfg.get("lam", 1.0))
        elif m == "mmpareto":
            self.gamma = float(self.cfg.get("gamma", 1.5))
        self._reset_epoch()

    # ------------------------------------------------------------------ setup helpers
    def optimizer_param_groups(self):
        """For the optimizer: one group per stream encoder + heads + rest (same lr / weight decay for all;
        the per-stream learning-rate scaling happens only inside step())."""
        return [{"params": ps, "bal_name": k} for k, ps in self.groups.items() if ps]

    @staticmethod
    def _pre_dropout_hook(model, stream, store):
        """Captures the INPUT of the branch's final Dropout = the stream feature before dropout."""
        drop = model.branches[stream][-1]
        if not isinstance(drop, torch.nn.Dropout):
            raise RuntimeError(f"branch '{stream}' does not end in Dropout")

        def hook(module, inputs):
            store[stream] = inputs[0]
        return drop.register_forward_pre_hook(hook)

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _hard(labels):
        hard = torch.argmax(labels, dim=1)
        return hard.masked_fill(labels.sum(dim=1) == 0, -100)

    def _wce(self, logits, hard):
        return F.cross_entropy(logits.float(), hard, weight=self.w, ignore_index=-100)

    def _encode(self, features, hamer, cues):
        return self.model._encode_streams(features, hamer=hamer if self.use_hamer else None,
                                          cues=cues if self.use_cues else None)

    def _decode(self, z, cues_in, aux, override=None):
        self.model._stream_override = override
        try:
            return self.model._decode_streams(z, cues_in, aux)
        finally:
            self.model._stream_override = None

    @torch.no_grad()
    def _update_fill(self, z, hard):
        valid = (hard != -100)
        for s in self.streams:
            zs = z[s].detach().float()
            if valid.shape[:2] == zs.shape[:2]:
                m = valid.unsqueeze(-1).float()
                mean = (zs * m).sum((0, 1)) / m.sum().clamp_min(1.0)
            else:
                mean = zs.mean((0, 1))
            self.fill[s] = mean if self.fill[s] is None else (1 - self.fill_momentum) * self.fill[s] + self.fill_momentum * mean

    def _ov_coalition(self, members):
        """Streams not in `members` (indices) replaced by their running mean (zeros before the first update)."""
        drop = [s for i, s in enumerate(self.streams) if i not in members]

        def fn(z):
            z = dict(z)
            for s in drop:
                f = self.fill[s]
                z[s] = torch.zeros_like(z[s]) if f is None else f.to(z[s].dtype).expand_as(z[s])
            return z
        return fn

    def _ov_keep_only_zero(self, i_keep):
        """OPM / InfoReg mono-modal prediction: keep one stream, zero the others (OPM's non-late-fusion rule)."""
        keep = self.streams[i_keep]

        def fn(z):
            return {s: (v if s == keep else torch.zeros_like(v)) for s, v in z.items()}
        return fn

    def _score(self, logits, hard, kind):
        """Class-weighted mono-modal score: 'prob' = mean p(gold) (OPM), 'ce' = weighted CE (InfoReg)."""
        if kind == "ce":
            return float(self._wce(logits, hard))
        p = F.softmax(logits.float(), dim=1)
        valid = hard != -100
        g = hard.clamp_min(0)
        pg = p.gather(1, g.unsqueeze(1)).squeeze(1)
        w = self.w[g] * valid
        return float((w * pg).sum() / w.sum().clamp_min(1e-8))

    @torch.no_grad()
    def _mono_scores(self, z, cues_in, aux, hard, kind):
        was = self.model.training
        self.model.eval()                    # no dropout in the scoring passes (encoder is not re-run)
        try:
            out = []
            for i in range(self.M):
                logits, _ = self._decode(z, cues_in, aux, self._ov_keep_only_zero(i))
                out.append(self._score(logits, hard, kind))
            return np.array(out)
        finally:
            self.model.train(was)

    def _heads_loss(self, hard):
        return {s: self._wce(self.model._bal_logits[s], hard) for s in self.model.bal_head_streams}

    # ------------------------------------------------------------------ statistics
    def _reset_epoch(self):
        z = lambda: np.zeros(self.M)            # noqa: E731
        self.ep = {"n": 0, "mu_enc": z(), "mu_fus": z(), "gsq": z(), "head": z(), "head_n": 0,
                   "mono": z(), "mono_n": 0, "x1": z(), "x2": z(), "x_n": 0, "rebal": z(), "extra": {}}

    @torch.no_grad()
    def _grad_stats(self):
        """Per stream: (|g_enc|^2, |theta_enc|^2, |g_fus|^2, |W_fus|^2) as numpy arrays (one sync)."""
        vals = []
        for s in self.streams:
            ps = self.groups[s]
            gsq = sum((p.grad.float().pow(2).sum() for p in ps if p.grad is not None),
                      torch.zeros((), device=self.device))
            psq = sum((p.detach().float().pow(2).sum() for p in ps), torch.zeros((), device=self.device))
            vals += [gsq, psq]
        W = self.model.fusion_proj[0].weight
        for s in self.streams:
            a, b = self.fusion_slices[s]
            gsq = W.grad[:, a:b].float().pow(2).sum() if W.grad is not None else torch.zeros((), device=self.device)
            vals += [gsq, W.detach()[:, a:b].float().pow(2).sum()]
        v = torch.stack(vals).cpu().numpy()
        enc = v[:2 * self.M].reshape(self.M, 2)
        fus = v[2 * self.M:].reshape(self.M, 2)
        return enc[:, 0], enc[:, 1], fus[:, 0], fus[:, 1]

    def _record(self, cumulative=True):
        g_enc, p_enc, g_fus, p_fus = self._grad_stats()
        mu_e = np.where(p_enc > 0, g_enc / np.maximum(p_enc, 1e-30), 0.0)
        mu_f = np.where(p_fus > 0, g_fus / np.maximum(p_fus, 1e-30), 0.0)
        self.ep["n"] += 1
        self.ep["mu_enc"] += mu_e
        self.ep["mu_fus"] += mu_f
        self.ep["gsq"] += g_enc
        if cumulative:
            self.cum_enc += mu_e
            self.cum_fus += mu_f
        return g_enc

    def _finish(self, optimizer, grad_clip, lr_scales=None, record=True, cumulative=True):
        if record:
            self._record(cumulative)
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip))
        saved = []
        for g in optimizer.param_groups:
            saved.append(g["lr"])
            g["lr"] = g["lr"] * float((lr_scales or {}).get(g.get("bal_name"), 1.0))
        optimizer.step()
        for g, lr in zip(optimizer.param_groups, saved):
            g["lr"] = lr
        self.global_step += 1

    def _bad(self, *losses):
        return any(not torch.isfinite(l) for l in losses)

    # ------------------------------------------------------------------ public API
    def start_epoch(self, epoch):
        self.epoch = epoch
        self._reset_epoch()
        if self.method == "inforeg":
            # Algorithm 1: no regulation in the first two epochs; afterwards the window opens per stream when
            # its Fisher-trace proxy rose by more than K between the two previous epochs
            prev, prev2 = self.tr_hist.get(epoch - 1), self.tr_hist.get(epoch - 2)
            for i, s in enumerate(self.streams):
                self.ir_active[i] = bool(epoch >= 3 and prev is not None and prev2 is not None
                                         and core.inforeg_window(prev[i], prev2[i], self.K))
            self.anchors = {s: [p.detach().clone() for p in self.groups[s]]
                            for i, s in enumerate(self.streams) if self.ir_active[i]}

    def step(self, features, labels, hamer, cues, optimizer, grad_clip):
        """One training step. Returns the multimodal weighted-CE loss (float, comparable with the baseline's
        train_loss) or None when the loss is not finite (the step is skipped)."""
        optimizer.zero_grad(set_to_none=True)
        hard = self._hard(labels)
        try:
            out = getattr(self, f"_step_{self.method}")(features, hard, hamer, cues, optimizer, grad_clip)
        finally:
            self.model._stream_override = None
        if out is None:
            optimizer.zero_grad(set_to_none=True)
        return out

    def end_epoch(self, epoch, val_metrics=None):
        n = max(self.ep["n"], 1)
        row = {"epoch": epoch, "method": self.method, "steps": self.ep["n"]}
        if val_metrics is not None:
            row["val_frame_f1"] = round(float(val_metrics["Frame_F1"]), 5)
            row["val_bnd_f1_5"] = round(float(val_metrics["BndF1_5"]), 5)
        speed_ep = core.wu_speeds(self.ep["mu_fus"], self.ep["mu_enc"])
        speed_cum = core.wu_speeds(self.cum_fus, self.cum_enc)
        for i, s in enumerate(self.streams):
            row[f"mu_enc_{s}"] = float(self.ep["mu_enc"][i] / n)
            row[f"mu_fus_{s}"] = float(self.ep["mu_fus"][i] / n)
            row[f"gradsq_{s}"] = float(self.ep["gsq"][i] / n)
            row[f"speed_{s}"] = round(float(speed_ep[i]), 5)
            row[f"cumspeed_{s}"] = round(float(speed_cum[i]), 5)
        if self.ep["head_n"]:
            for i, s in enumerate(self.streams):
                row[f"head_loss_{s}"] = round(float(self.ep["head"][i] / self.ep["head_n"]), 5)
        if self.ep["mono_n"]:
            for i, s in enumerate(self.streams):
                row[f"mono_{s}"] = round(float(self.ep["mono"][i] / self.ep["mono_n"]), 5)
        xn = max(self.ep["x_n"], 1)
        names = {"opm": ("q", "rho"), "cggm": ("B", "dimpr"), "agm": ("kappa", "logp"), "umt": ("distill", None),
                 "wu": ("rebal_frac", None), "inforeg": ("reg_frac", "delta"), "mmpareto": ("conflict_frac", "k_u")}
        a, b = names.get(self.method, (None, None))
        for i, s in enumerate(self.streams):
            if a == "rebal_frac":
                row[f"rebal_frac_{s}"] = round(float(self.ep["rebal"][i] / n), 4)
            elif a:
                row[f"{a}_{s}"] = round(float(self.ep["x1"][i] / xn), 5)
            if b:
                row[f"{b}_{s}"] = round(float(self.ep["x2"][i] / xn), 5)
        if self.method == "wu":
            row["wu_d"] = round(float(self.wu_d), 5)
            if self.alpha_auto and epoch == self.warm_epochs:
                self.alpha = core.wu_decide(speed_cum, 0.0)[2] + self.alpha_margin
                print(f"   wu: imbalance tolerance set from the warm-up: alpha = {self.alpha:.4f}")
            row["wu_alpha"] = round(float(self.alpha), 5) if self.alpha is not None else ""
        if self.method == "inforeg":
            self.tr_hist[epoch] = list(self.ep["gsq"] / n)
            for i, s in enumerate(self.streams):
                row[f"window_{s}"] = int(self.ir_active[i])
        row.update(self.ep["extra"])
        msg = " | ".join(f"{s}: mu {row[f'mu_enc_{s}']:.2e}, speed {row[f'speed_{s}']:+.2f}"
                         + (f", head {row[f'head_loss_{s}']:.3f}" if f"head_loss_{s}" in row else "")
                         + (f", {a} {row[f'{a}_{s}']:.3f}" if a and f"{a}_{s}" in row else "")
                         for s in self.streams)
        print(f"   balance [{self.method}] {msg}")
        if self.log_path:
            if self._header is None:
                self._header = list(row.keys())
                with open(self.log_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self._header).writeheader()
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self._header, extrasaction="ignore").writerow(row)
        return row

    # ------------------------------------------------------------------ methods
    def _log_heads(self, hl):
        self.ep["head"] += np.array([float(hl[s]) for s in self.streams])
        self.ep["head_n"] += 1

    def _step_monitor(self, features, hard, hamer, cues, optimizer, grad_clip, override=None, lam=1.0):
        z, ci, aux = self._encode(features, hamer, cues)
        logits, _ = self._decode(z, ci, aux, override)
        main = self._wce(logits, hard)
        hl = self._heads_loss(hard)
        total = main + lam * sum(hl.values())
        if self._bad(total):
            return None
        total.backward()
        self._log_heads(hl)
        self._finish(optimizer, grad_clip)
        return float(main)

    def _step_uni(self, features, hard, hamer, cues, optimizer, grad_clip):
        return self._step_monitor(features, hard, hamer, cues, optimizer, grad_clip, lam=self.lam)

    def _step_dgl(self, features, hard, hamer, cues, optimizer, grad_clip):
        detach_all = lambda z: {s: v.detach() for s, v in z.items()}     # noqa: E731
        return self._step_monitor(features, hard, hamer, cues, optimizer, grad_clip, override=detach_all)

    def _step_opm(self, features, hard, hamer, cues, optimizer, grad_clip):
        z, ci, aux = self._encode(features, hamer, cues)
        if self.opm_q is None or self.global_step % self.score_every == 0:
            scores = self._mono_scores(z, ci, aux, hard, "prob")
            rho = core.opm_ratios(scores)
            self.opm_q = core.opm_drop_prob(rho, self.q_base, self.lam)
            self.opm_fallback = int(np.argmin(scores))           # never drop the weakest stream of a sample
            self.ep["mono"] += scores
            self.ep["mono_n"] += 1
            self.ep["x2"] += rho
        B = next(iter(z.values())).shape[0]
        keep = core.opm_keep_mask(self.opm_q, torch.rand(B, self.M).numpy(), self.opm_fallback)
        keep_t = torch.tensor(keep, dtype=torch.float, device=self.device)

        col = {s: i for i, s in enumerate(self.streams)}

        def mask(zz):
            return {s: v * keep_t[:, col[s]].to(v.dtype)[:, None, None] for s, v in zz.items()}
        logits, _ = self._decode(z, ci, aux, mask)
        main = self._wce(logits, hard)
        if self._bad(main):
            return None
        main.backward()
        self.ep["x1"] += self.opm_q
        self.ep["x_n"] += 1
        self._finish(optimizer, grad_clip)
        return float(main)

    def _step_umt(self, features, hard, hamer, cues, optimizer, grad_clip):
        self._cap_s.clear()
        self._cap_t.clear()
        z, ci, aux = self._encode(features, hamer, cues)
        with torch.no_grad():
            for s, tm in self.teachers.items():
                tm._encode_streams(features, hamer=hamer if "hamer" in tm.streams else None,
                                   cues=cues if "cues" in tm.streams else None)
        logits, _ = self._decode(z, ci, aux)
        main = self._wce(logits, hard)
        valid = (hard != -100).unsqueeze(-1).float()
        dl = {}
        for s in self.teachers:
            zs, zt = self._cap_s[s].float(), self._cap_t[s].float()
            v = valid if valid.shape[:2] == zs.shape[:2] else torch.ones_like(zs[..., :1])
            dl[s] = ((zs - zt) ** 2 * v).sum() / (v.sum() * zs.shape[-1]).clamp_min(1.0)
        total = main + sum(self.teacher_lam[s] * dl[s] for s in dl)
        if self._bad(total):
            return None
        total.backward()
        self.ep["x1"] += np.array([float(dl[s]) if s in dl else 0.0 for s in self.streams])
        self.ep["x_n"] += 1
        self._finish(optimizer, grad_clip)
        return float(main)

    def _step_cggm(self, features, hard, hamer, cues, optimizer, grad_clip):
        z, ci, aux = self._encode(features, hamer, cues)
        logits, emb = self._decode(z, ci, aux)
        main = self._wce(logits, hard)
        hl = self._heads_loss(hard)
        if self._bad(main, *hl.values()):
            return None
        delta = self.tracker.update([float(hl[s]) for s in self.streams])
        if delta is not None:
            self.B = core.cggm_balance(delta, self.rho, lo=self.b_lo, hi=self.b_hi)
            self.ep["x2"] += delta
        total = main + sum(hl.values())
        if self.lam_gm > 0:
            # Direction term (CGGM Eq. 11-12) on the classifier weights. The fusion-head gradient is
            # dL/dW = sum_t delta_t emb_t^T; emb is detached so the term is first-order (a double backward
            # through the cuDNN BiLSTM is not available). The stream heads' gradients are constants.
            d_logits = torch.autograd.grad(main, logits, retain_graph=True, create_graph=True)[0]
            gF = torch.einsum("bct,bdt->cd", d_logits.float(), emb.detach().float()).flatten()
            lgm = 0.0
            for i, s in enumerate(self.streams):
                gi = torch.autograd.grad(hl[s], self.model.bal_heads[s].classifier.weight,
                                         retain_graph=True)[0].detach().flatten()
                cos = F.cosine_similarity(gF, gi.float(), dim=0, eps=1e-12)
                lgm = lgm + abs(float(self.B[i])) - float(self.B[i]) * cos
            lgm = lgm / self.M
            total = total + self.lam_gm * lgm
            self.ep["extra"]["L_gm"] = round(float(lgm), 5)
        total.backward()
        self._log_heads(hl)
        self.ep["x1"] += self.B
        self.ep["x_n"] += 1
        self._finish(optimizer, grad_clip, lr_scales={s: float(self.B[i]) for i, s in enumerate(self.streams)})
        return float(main)

    def shapley_phi(self, z, ci, aux):
        """AGM mono-modal responses: (phi (M, B, 3, T), full logits (B, 3, T)); phi sums to the full logits."""
        outs = []
        for c in self.coals:
            ov = None if len(c) == self.M else self._ov_coalition(set(c))
            lg, _ = self._decode(z, ci, aux, ov)
            outs.append(lg.float())
        V = torch.stack(outs, 0)                                           # (C, B, 3, T)
        phi = torch.einsum("mc,cbkt->mbkt", self.shapley_W, V)
        return phi, V[-1]

    def _step_agm(self, features, hard, hamer, cues, optimizer, grad_clip):
        z, ci, aux = self._encode(features, hamer, cues)
        self._update_fill(z, hard)
        phi, full = self.shapley_phi(z, ci, aux)
        with torch.no_grad():
            s = np.array([-float(self._wce(phi[m], hard)) for m in range(self.M)])   # weighted mean log p(gold)
        if not np.all(np.isfinite(s)):
            return None
        self.agm_t += 1
        self.s_hat = core.running_mean(self.s_hat, s, self.agm_t)
        kappa, r, tau = core.agm_kappa(s, self.s_hat, self.alpha)
        k = torch.tensor(kappa, dtype=phi.dtype, device=self.device).view(-1, 1, 1, 1)
        out = full.detach() + (k * (phi - phi.detach())).sum(0)          # value = full; gradient: sum kappa_m dphi_m
        loss = self._wce(out, hard)
        if self._bad(loss):
            return None
        loss.backward()
        self.ep["x1"] += kappa
        self.ep["x2"] += s
        self.ep["x_n"] += 1
        self._finish(optimizer, grad_clip)
        return float(loss)

    def _step_wu(self, features, hard, hamer, cues, optimizer, grad_clip):
        kind, target = self.sched.next_step(warm=self.epoch <= self.warm_epochs or self.alpha is None)
        z, ci, aux = self._encode(features, hamer, cues)
        self._update_fill(z, hard)
        ov = self._ov_coalition({target}) if kind == "rebalance" else None
        logits, _ = self._decode(z, ci, aux, ov)
        main = self._wce(logits, hard)
        if self._bad(main):
            return None
        main.backward()
        regular = kind != "rebalance"
        self._record(cumulative=regular)              # Wu accumulate the speeds on regular steps only
        if kind == "rebalance":
            self.ep["rebal"][target] += 1
        if kind == "check":
            trigger, tgt, self.wu_d = core.wu_decide(core.wu_speeds(self.cum_fus, self.cum_enc), self.alpha)
            self.sched.after_check(trigger, tgt)
        self._finish(optimizer, grad_clip, record=False)
        return float(main)

    def _step_inforeg(self, features, hard, hamer, cues, optimizer, grad_clip):
        z, ci, aux = self._encode(features, hamer, cues)
        if self.global_step % self.score_every == 0 or not self.ep["mono_n"]:
            scores = self._mono_scores(z, ci, aux, hard, "ce")
            self.ir_delta = core.inforeg_delta(scores)
            self.ep["mono"] += scores
            self.ep["mono_n"] += 1
        logits, _ = self._decode(z, ci, aux)
        main = self._wce(logits, hard)
        if self._bad(main):
            return None
        main.backward()
        self._record()                                 # Fisher-trace proxy from the joint loss only
        alpha = core.inforeg_alpha(self.ir_delta, self.beta)
        regulated = np.zeros(self.M)
        with torch.no_grad():
            for i, s in enumerate(self.streams):
                if self.ir_active[i] and self.ir_delta[i] > 0:
                    coef = self.c * float(alpha[i])    # d/dw (c * alpha / 2 * |w - w_prev_epoch|^2)
                    for p, a in zip(self.groups[s], self.anchors[s]):
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad.add_(coef * (p.detach() - a))
                    regulated[i] = 1.0
        self.ep["x1"] += regulated
        self.ep["x2"] += self.ir_delta
        self.ep["x_n"] += 1
        self._finish(optimizer, grad_clip, record=False)
        return float(main)

    def _step_mmpareto(self, features, hard, hamer, cues, optimizer, grad_clip):
        z, ci, aux = self._encode(features, hamer, cues)
        logits, _ = self._decode(z, ci, aux)
        main = self._wce(logits, hard)
        hl = self._heads_loss(hard)
        if self._bad(main, *hl.values()):
            return None
        params = [p for p in self.model.parameters() if p.requires_grad]
        g_main = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
        gmap = {id(p): g for p, g in zip(params, g_main)}
        done = set()
        conflict, k_u_log = np.zeros(self.M), np.zeros(self.M)
        for i, s in enumerate(self.streams):
            enc = self.groups[s]
            head = list(self.model.bal_heads[s].parameters())
            gu = torch.autograd.grad(hl[s], enc + head, retain_graph=(i < self.M - 1), allow_unused=True)
            gu_enc, gu_head = gu[:len(enc)], gu[len(enc):]
            gm_enc = [gmap[id(p)] for p in enc]
            zero = torch.zeros((), device=self.device)
            mm = sum((g.float().pow(2).sum() for g in gm_enc if g is not None), zero)
            uu = sum((g.float().pow(2).sum() for g in gu_enc if g is not None), zero)
            mu = sum(((a.float() * b.float()).sum() for a, b in zip(gm_enc, gu_enc)
                      if a is not None and b is not None), zero)
            mm, uu, mu = torch.stack([mm, uu, mu]).tolist()
            k_m, k_u, conf = core.mmpareto_coeffs(mm, uu, mu, self.gamma)
            conflict[i], k_u_log[i] = float(conf), k_u
            for p, a, b in zip(enc, gm_enc, gu_enc):
                g = torch.zeros_like(p)
                if a is not None:
                    g = g + k_m * a
                if b is not None:
                    g = g + k_u * b
                p.grad = g
                done.add(id(p))
            for p, g in zip(head, gu_head):
                p.grad = torch.zeros_like(p) if g is None else g
                done.add(id(p))
        for p in params:
            if id(p) not in done:
                p.grad = gmap[id(p)]
        self._log_heads(hl)
        self.ep["x1"] += conflict
        self.ep["x2"] += k_u_log
        self.ep["x_n"] += 1
        self._finish(optimizer, grad_clip)
        return float(main)
