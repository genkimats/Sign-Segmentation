# balance_study: modality-imbalance ("greedy learner") methods

Claim under test: *the multi-stream model exploits whichever stream reduces the training loss fastest.*

Two explanations compete:
- **H-opt (greedy / optimization):** the face is under-learned because body and HaMeR reduce the loss first.
- **H-info (information):** the face carries little phrase-boundary information beyond the hands.

This study does two things:
- **intervenes**, with published imbalance methods, each as one change to the B+H+Fc CE recipe of 373-375;
- **measures**, with per-stream learning speed and utilization inside the joint model, plus frozen-branch probes.

Background, and the fit of each method to this setting: `claude/modality_imbalance_review.md` in the GR project.

## Files

| File | What it does |
|---|---|
| `core.py` | Torch-free math of every method (Shapley matrix, OPM probabilities, CGGM balance terms, AGM kappa, MMPareto coefficients, InfoReg Delta / alpha / window, Wu speeds and schedule) |
| `test_balance_core.py` | Numpy tests of `core.py` against brute-force references |
| `heads.py` | `StreamBIOHead`: a per-stream BiLSTM(1 layer, d_model) → Linear(3) head, training-only |
| `balancer.py` | `Balancer`: runs each training step for the chosen method and writes `experiments_phrase/<run>/balance_log.csv` |
| `experiments.py` | The arms (`imbalance_arms`, `single_stream_arms`), used by `queue_train_phrase.py` (`IMBALANCE_STUDY`) |
| `smoke_test.py` | Every arm, end to end, on random data (CPU), plus checks that the model refactor changed nothing |
| `stream_shapley.py` | Diagnostic: coalition table (all 8 stream subsets), Wu's conditional utilization, per-frame Shapley attribution of the B log-odds, event-locked to phrase starts |
| `laziness_probe.py` | Diagnostic: the same probe trained on a frozen stream branch of the joint model vs the single-stream model |
| `summarize_balance.py` | Arm table vs 373 (seed spread of 373-375, paired video bootstrap), per-stream learning curves, single-stream loss curves |

### Edits to existing files

All are inactive unless a config sets them, so older configs and checkpoints rebuild unchanged.

**`src/models.py` (`MultiStreamSegmenter`)**
- `forward` is split into `_encode_streams` (encoders, branches, stream heads) and `_decode_streams` (dropout, ablation, override, fusion, BiLSTM, heads). Together they are the old `forward`, line for line; `smoke_test.py` checks equality.
- New `_stream_override` hook, applied right before the fusion.
- `set_ablations([...])`: replace several streams at once.
- Optional `balance_heads`, created last in `__init__`, so every other module keeps exactly its usual initialisation.

**Other files**
- `src/model_factory.py` passes `balance_heads` / `balance_head_hidden` / `balance_head_layers` / `balance_head_detach`.
- `train_phrase.py`: when a config has `"balance"`, the `Balancer` builds the optimizer's parameter groups and runs each training step. Validation, selection and saving are unchanged.
- `queue_train_phrase.py`: `IMBALANCE_STUDY` (now `EXPERIMENTS_TO_RUN`). It leaves out UMT arms whose teacher runs are missing.

## Run order (repo root)

```bash
python balance_study/test_balance_core.py      # numpy, seconds
python balance_study/smoke_test.py             # torch, CPU ok, 1-3 min -- run this before queueing

python queue_train_phrase.py                   # EXPERIMENTS_TO_RUN = IMBALANCE_STUDY; press Enter at the seed prompt (1 seed)
python train_phrase.py                         # every arm forces Frame F1 selection, like 373-375

# evaluate (prefixes = the ones the queue printed)
python evaluate_phrase.py --model multistream_bilstm --prefixes <new runs> 373 --split val  --save-predictions
python evaluate_phrase.py --model multistream_bilstm --prefixes <new runs> 373 --split test --save-predictions
python balance_study/summarize_balance.py --split test --plot

# diagnostics on existing models (no training)
python balance_study/stream_shapley.py --runs multistream_bilstm-373 multistream_bilstm-428 --split test --plot
python balance_study/laziness_probe.py --probe conv9 --pairs \
    multistream_bilstm-373:face multistream_bilstm-136:face \
    multistream_bilstm-373:hamer multistream_bilstm-133:hamer \
    multistream_bilstm-373:body_hands multistream_bilstm-82:body_hands
```

After the SS-* reruns finish, repeat the probe with them as the single-stream side (same machine), and run `stream_shapley.py` on the arms that moved.

## Arms (1 seed, seed 42; reference multistream_bilstm-373)

| Arm | Method (paper) | Changes | Cost vs a normal run |
|---|---|---|---|
| IMB0 monitor | -- | detached per-stream heads, logging only | ~1.3× |
| IMB-OPM | OPM (TPAMI 2024) | stream m dropped with q = q_base(1 + λ tanh(ρ_m − 1)) when ρ_m > 1; ρ from mono-modal scores (keep m, zero the others) | ~1.5-2× |
| IMB-UMT-all | UMT (ICML 2023) | + Σ λ·MSE(branch_s, teacher branch_s), teachers 82 / 133 / 136 | ~1.3× |
| IMB-UMT-face | UMT | face teacher only | ~1.1× |
| IMB-CGGM-mag | CGGM (NeurIPS 2024) | per-stream LR × B_m, B from the heads' smoothed loss decrease | ~1.3× |
| IMB-CGGM-full | CGGM | + direction term (λ_gm 0.1) | ~1.4× |
| IMB-AGM | AGM (ICCV 2023) | Shapley mono-modal responses (7 coalitions); gradient of each × κ_m | ~5-7× |
| IMB-Wu | Wu et al. (ICML 2022) | guided re-balancing steps (other streams → running mean) when the conditional-speed gap exceeds α | ~1× |
| IMB-InfoReg | InfoReg (CVPR 2025) | proximal pull to last epoch's weights for information-sufficient streams in the prime window | ~1.5-2× |
| IMB-UNI | (MMPareto's uniform baseline) | + unimodal BIO loss on every stream, plain sum | ~1.3× |
| IMB-MMPareto | MMPareto (ICML 2024) | per encoder: Pareto integration of g_multi and g_uni, rescaled to γ‖g_m + g_u‖ | ~2× |
| IMB-DGL | DGL (ICCV 2025), diagnostic | fusion sees detached streams; encoders learn from unimodal losses only | ~1.3× |
| SS-body / SS-hamer / SS-face | -- | single-stream reruns on this machine (learning-speed curves, later teachers) | ~0.5-1× |

## Design decisions and deviations from the papers

**Per-stream learning rates instead of gradient scaling.** This applies to the gradient-magnitude methods.
- With AdamW, a gradient multiplied by a constant is divided out by the second-moment estimate within about 1/(1 − β2) = 1000 steps (less than one epoch here). OGM-GE's analysis assumes SGD for this reason.
- So CGGM's B_m scales the **learning rate** of stream m's parameter group for that step. Under SGD the two are identical; under Adam only this version has a lasting effect.
- The groups are only temporarily rescaled around `optimizer.step()`, so the cosine schedule is untouched.

**Unimodal predictions in a model without per-stream outputs.**
- OPM and InfoReg use OPM's own non-late-fusion rule: keep one stream, zero the others, decode (no gradient, eval mode, encoder not re-run).
- AGM, Wu and the diagnostics replace streams by their **running mean**. That is the AGM / Wu / `check_stream_balance.py` ablation, a realistic input that carries no information.
- CGGM, UNI, MMPareto and DGL use `StreamBIOHead`s. A 1-layer BiLSTM gives them context comparable to the real model; the aux heads' k=9 conv sees only 0.3 s and would make every stream look weak.

**Class weighting.** BIO is O-dominated. Every mono-modal score and unimodal loss uses the training class weights [1.0, 1.1, 8.5]. Otherwise the scores mostly measure "is the person signing", which the face can partly answer.

**OPM.**
- q_base 0.3, λ 0.5. The paper sweeps 0.2-0.6 and shows q_base 0.5 / λ 0.5 in one figure; 0.3 is gentler, because both manual streams will look "dominant" relative to the face.
- A sample never loses every stream: the weakest stream is kept.

**CGGM.**
- Δε_m = decrease of the EMA (0.98) of head m's loss over the last 100 steps. Per-iteration differences on random 512-frame windows are noise.
- ρ = M/(M−1), so equal improvement gives B = 1. B is clipped to [0.1, 3].
- The heads are detached monitors, so the arm isolates the magnitude modulation.
- Direction term:
  - The fusion-classifier gradient is written as δᵀ·emb with emb detached. A true second-order term would need a double backward through the cuDNN BiLSTM, which PyTorch does not provide.
  - The stream heads' classifier gradients are constants.
  - It is an approximation; CGGM's own ablation finds the magnitude term gives most of the gain.

**AGM.**
- Implemented as the standard Shapley value with v(∅) = 0, so the responses add up to the logits. The paper's 2-modality example does the same.
- s_m is the weighted mean **log**-probability of the gold class, the sign of the paper's Algorithm 1. Its Eq. 4 has the opposite sign, which would invert "too strong → κ < 1".
- The forward value is exactly the full model's: out = full.detach() + Σ κ_m (φ_m − φ_m.detach()).

**InfoReg.**
- Fisher trace ≈ epoch mean of ‖∇_θm L_joint‖² (batch gradients).
- The window opens from epoch 3 when that trace rose by more than K = 0 between the two previous epochs. If the trace only falls, the arm never regulates; the log's `window_*` columns say so.
- Δ_m follows the paper's text: a stream that is *ahead* of another is information-sufficient, and the weakest has Δ = 0. The paper's Eq. 10 is written the other way round.
- The proximal term is c·(α/2)‖θ − θ_prev_epoch‖² with c = 1. Under AdamW it slows the regulated encoders strongly within a few hundred steps; c is the knob.

**Wu et al.**
- Their speed s(m1|m0) compares the fusion parameters carrying m1 to m0's branch. With concat fusion: s_m = log(Σ μ(fusion columns of the other streams) / Σ μ(encoder m)). For two streams this is exactly their definition.
- Re-balancing step for m: the other streams are replaced by their running mean, so the BiLSTM must use m. This is stronger than their MMTM version, which only removes the cross-talk.
- Heterogeneous encoders give a non-zero gap even without imbalance, so α = the gap after the warm-up epoch + 0.1.

**UMT.**
- MSE on the branch output **before** its dropout, captured with a pre-hook, so dropout noise does not bias the target.
- Teachers are the existing single-stream runs: 82 body, 133 HaMeR, 136 face. Same branch width, compact face, plain HaMeR.

**MMPareto.**
- Exact per-encoder integration: `autograd.grad` of the multimodal loss once, then of each unimodal loss w.r.t. its encoder and head.
- γ = 1.5. Under AdamW the γ rescale is mostly normalized away; the direction rule is what remains.

**DGL.** A diagnostic arm. It tests whether the joint gradient helps or suppresses the encoders, at the cost of features that are only useful in combination (expected to hurt the face).

## Reading the outcome

Compare with 373 using `summarize_balance.py`. A single-seed difference should at least exceed the 373-375 spread and show in the paired video bootstrap; re-queue the promising arms with 3 seeds before claiming anything.

| Face use (Shapley / probe / balance_log) | Boundary F1@5 (esp. non-manually marked starts) | Reading |
|---|---|---|
| up | up | H-opt: greedy learning limited the face |
| up | flat / down | H-info: the face can be forced into use, but it adds little; reliance on the manual streams is near-optimal (ARL's view) |
| unchanged | unchanged | the method does not act in this architecture |

The `balance_log.csv` of the monitor arm (IMB0) is the direct evidence for the claim's wording. It shows each stream's head loss per epoch (how fast each stream alone is learnt *inside* the joint model) and Wu's conditional speeds.
