# Fusion study: other ways of fusing the streams

R0 fuses body+hands / HaMeR / face by **concatenation + one shared projection**. This study replaces
only that step, four ways. Each change tests a question about the streams, so the analysis yields
findings even if the numbers barely move.

| Arm | Fusion | Question |
|---|---|---|
| F1a | global scalar gate per stream (existing `gated_sum`) | control for F1 |
| **F1** | **frame-wise gate**: per-frame softmax weights over the streams, from the streams themselves | Does the model rely on the face *at* phrase boundaries and on the hands inside phrases? |
| **F2** | **time-lagged face**: each body frame attends to face frames within ±25 frames (±0.5 s), with a learned per-offset bias | Do non-manual boundary cues lead or lag the manual signal, and is the face more useful once realigned? |
| F2b | same, content-free (one learned temporal filter = a fixed learned lag) | Is a fixed lag enough, or does the alignment depend on content? |
| **F3** | **late fusion**: one BiLSTM + B/I/O head per stream, log-probabilities combined with learned weights; every head also trained alone (deep supervision 0.5) | Does early fusion let the body dominate? Which boundaries does only the face (or only HaMeR) find? |
| **F5** | **face FiLM**: the face produces a per-frame scale and shift for the body+hands features and is *not* concatenated | Are non-manuals better used as modulators of the manual signal than as extra input? |

Optional arms: `FUSION_OPTIONAL` in the queue.
- F3g: late fusion with frame-wise weights.
- F5k: FiLM, and the face is also concatenated.
- F5f: FiLM on the fused representation.

Everything else is R0: B+H+Fc, count loss w=2 r=5, best epoch by boundary F1@5, window 512, weights
[1.0, 1.1, 8.5]. The reference is the existing R0 runs, `multistream_bilstm-310..312`.

## Files

| File | What it does |
|---|---|
| `models_fusion.py` | `MultiStream_Fusion`, a subclass of `MultiStreamSegmenter`. Streams are computed exactly as in R0; only the fusion differs. Registered as basename `multistream_fusion`. |
| `fusion_layers.py` | `FrameGate`, `LocalLagAttention`, `FaceFiLM`, `StreamHead`. |
| `fusion_experiments.py` | The arms, derived from R0 (imported by `queue_train_phrase.py`). |
| `fusion_analysis.py` | Event-locked analysis of what each model learned (gates, lags, FiLM strength, per-stream predictions + complementarity). |
| `trace_core.py`, `test_trace_core.py` | Numpy core of the analysis + tests. |
| `smoke_test.py` | Builds every arm through the factory on random inputs. Checks shapes, gradients, traces, hooks and initialisation, and compares the lag attention against a brute-force version. |

## Small edits to existing files (all marked "fusion_study")

- `src/model_factory.py`
  - registers `multistream_fusion` (in `MODEL_REGISTRY`, `MULTISTREAM_MODELS` and `HAMER_SUPPORTED_MODELS`);
  - passes the fusion options (`fusion_model_kwargs`).
- `queue_train_phrase.py`
  - `multistream_fusion` is added to the multi-stream list that sets the data flags;
  - `FUSION_STUDY` / `FUSION_OPTIONAL` are defined;
  - `EXPERIMENTS_TO_RUN = FUSION_STUDY` (the A2 line is kept as a comment).
- `train_phrase.py`
  - if the model has `extra_loss`, it is added to the loss, with the run's own loss function passed
    in (late fusion's per-stream deep supervision);
  - all other models are unaffected.

Training, evaluation and analysis need no other changes:
- `evaluate_phrase.py --model multistream_fusion` works;
- `--save-predictions`, `phrase_stats.py` and `count_oracle_decoding.py` work;
- `check_stream_balance.py` works (each mode reports per-stream contributions to its capture hook).

## How to run (from the repo root)

```bash
python fusion_study/test_trace_core.py        # numpy tests
python fusion_study/smoke_test.py             # every arm, random inputs, ~1 min (needs the training env)

python queue_train_phrase.py                  # EXPERIMENTS_TO_RUN = FUSION_STUDY
#   seed prompt: press Enter for 1 seed per arm first (6 runs, ~3 h each), then re-queue the
#   promising arms with "all" (3 seeds)
python train_phrase.py

python evaluate_phrase.py --model multistream_fusion --split val  --save-predictions
python evaluate_phrase.py --model multistream_fusion --split test --save-predictions
python evaluate_phrase.py --model multistream_bilstm --prefixes <F1a prefixes> --split test --save-predictions

python phrase_stats.py --a multistream_fusion-01 ... --b multistream_bilstm-310 multistream_bilstm-311 multistream_bilstm-312
python fusion_study/fusion_analysis.py --runs multistream_fusion-01 multistream_fusion-02 ... --split test
python check_stream_balance.py ...            # as before, also for multistream_fusion runs
```

## Reading the analysis (`evaluation_phrase/fusion_analysis/<run>/`)

All profiles are locked to **gold** events. Every phrase-start profile has a **control**: sign
starts inside phrases, at least 10 frames from any phrase boundary. That separates "something new
starts" from "a new phrase starts". `summary_<split>.txt` prints, per signal, the value at the event
(offsets −2..+2) for every group, and *phrase start − sign start*.

**F1 `gate_<stream>`.** If the face weight rises at phrase starts but not at sign starts, the model
uses the face specifically for phrase boundaries. Compare the continuous / short / long profiles:
continuous boundaries are where the oracle found pauses useless.

**F2 `lag_<split>.csv`.** This is the attention distribution over offsets −25..+25:
- at all frames;
- at phrase starts;
- at sign starts.

A peak away from 0 at phrase starts is a learned non-manual lead or lag, which you can compare with
the pre/post deltas in `boundary_cue_analysis.py`. `lag_mean` gives the per-frame expected offset;
"+" means the face information used comes after the frame. If F2b (fixed lag) matches F2, a constant
offset is enough.

**F3 `complementarity_<split>.csv`.** Per-stream recall@5 of gold phrase starts, union recall,
boundaries only one stream finds, and pairwise Jaccard. These are given overall and per boundary
type. The per-stream predictions are saved as `<run>_<stream>`, so `phrase_stats.py` can compare,
e.g., the face head with the body head. A high "only face" share at continuous boundaries would be a
direct argument for keeping the face.

**F5 `film_gamma`, `film_beta`.** How strongly the face modulates the body over time. If the
modulation peaks at phrase starts more than at sign starts, the face acts as a prosodic marker.

## Notes

- **Size.** F1, F2 and F5 add well under 0.5 M parameters to R0. F3 replaces the shared 4 × 256
  BiLSTM with three 4 × 128 BiLSTMs (similar size). `smoke_test.py` prints every arm's size relative
  to R0.
- **Initialisation.**
  - The frame gate starts uniform, which equals the plain sum.
  - FiLM starts as the identity.
  - The lag attention starts with a mild preference for offset 0. Offsets outside a training window
    are masked, so the first and last 25 frames of each 512-frame window see a one-sided range.
- **Not combinable.** `multistream_fusion` cannot be combined with `aux_heads` or `film_cue_names`
  (ideas A3 and B); those stay on `multistream_bilstm`.
- **Not executed where it was written.** The numpy analysis core is tested, but the torch code could
  not be run in that environment. Run `smoke_test.py` first.
