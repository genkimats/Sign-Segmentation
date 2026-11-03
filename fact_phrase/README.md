# FACT pilot for phrase segmentation (frozen front end)

Does a modern segment-level encoder beat the BiLSTM? FACT (Lu & Elhamifar, CVPR 2024) runs a
frame branch (MS-TCN++ convolutions) and a set of action tokens (transformer) in parallel and lets
them exchange information through cross-attention. Each token is trained to represent exactly one
segment, so the model reasons explicitly about how many phrases there are and where. That ties
directly to the count-vs-placement analysis (idea C).

This is a **two-stage pilot**: R0's trained front end (ST-GCN streams + fusion) is frozen. Its fused
features are exported, and FACT replaces exactly the BiLSTM. A BiLSTM retrained on the same frozen
features is the fair reference.

## Files

| File | What it does |
|---|---|
| `export_features.py` | Exports a run's encoder-input features (256-d, 50 fps) for train/val/test. Checks itself against `src.evaluation`. |
| `fact_core.py` | Torch-free core: BIO → segments, soft IoU, token decoding, window tiling/stitching, frame-rate conversion, token statistics. |
| `test_fact_core.py` | Numpy tests for the core (brute-force references, plus the project's own `src.metrics`). |
| `fact_model.py` | FACT adapted to BIO (vendored from the official MIT code) + the frozen-BiLSTM reference. |
| `data.py` | Memory-mapped feature store, random training windows, full-video inference and stitching. |
| `train_fact.py` | Queue-driven training. Keeps the best checkpoint per decoder by val boundary F1@5, then evaluates. |
| `evaluate_fact.py` | Metrics CSV (same columns as `evaluate_phrase.py`) + predictions in `evaluation_phrase/predictions/`. |
| `queue_fact.py` | The experiment design (arms P0–P4, 3 seeds). |
| `summarize_fact.py` | Go/no-go table: mean ± SD, paired bootstraps (FACT vs frozen BiLSTM, sanity vs R0). |
| `smoke_test.py` | End-to-end check on synthetic data (no exported features needed). |

## How to run (from the repo root)

```bash
# 0. sanity: numpy core tests + end-to-end smoke test on synthetic data (CPU ok, a few minutes)
python fact_phrase/test_fact_core.py
python fact_phrase/smoke_test.py

# 1. export frozen features (R0 seed 42; add the CE reference only if stage 2 is needed)
python fact_phrase/export_features.py --run multistream_bilstm-310
#    the last line per split prints R0's own boundary F1@5 / IoU / % from the exported
#    probabilities -- it must match evaluate_phrase.py's argmax row for run 310

# 2. stage 1 (go/no-go): P0 frozen BiLSTM @512, P1 FACT @1500, P2 frozen BiLSTM @1500, 3 seeds each
python fact_phrase/queue_fact.py --stage 1
python fact_phrase/train_fact.py                     # trains, then evaluates val + test automatically

# 3. results
python fact_phrase/summarize_fact.py                 # test split
python fact_phrase/summarize_fact.py --split val

# 4. only if FACT wins stage 1: robustness on the CE front end
python fact_phrase/export_features.py --run multistream_bilstm-208
python fact_phrase/queue_fact.py --stage 2 --overwrite
python fact_phrase/train_fact.py
```

The existing analysis tools work on the saved predictions, e.g.
`python count_oracle_decoding.py --runs fact_head-01 fact_head-02 fact_head-03` or
`python phrase_stats.py --a fact_token-01 fact_token-02 fact_token-03 --b frozen_bilstm-01 frozen_bilstm-02 frozen_bilstm-03`.
Prediction names: `fact_token-NN` (FACT's token decoding), `fact_head-NN` (argmax of FACT's B/I/O
frame branch), `frozen_bilstm-NN`.

## Decision rule

- **Sanity first.** P0 (BiLSTM retrained on R0's frozen features, R0 recipe) should land close to R0
  (runs 310–312). If it is clearly worse, something in the export or data path is off. Check that
  before reading anything else.
- **Go.** FACT-token or FACT-head beats P0 on test boundary F1@5 with a paired-bootstrap 95% CI above 0,
  and is not worse on val. Then run stage 2. If FACT also wins on the CE front end, the gain is not
  tied to one set of features.
- **P2 separates "longer context" from "FACT".** If P2 matches FACT, the gain comes from the 30 s
  window, not the architecture.
- **No-go is a result too.** "A segment-token encoder does not move the ceiling either" belongs in the
  performance-limits chapter, together with the earlier encoder studies.

## Design decisions (and why)

**Exported features.** The input of R0's BiLSTM, captured with a forward pre-hook: the fused 256-d
streams after the fusion projection (and change-point / FiLM when a run uses them). Inference uses
`src.evaluation`'s windows and stitching, and the first video of each split is checked against
`predict_video_probs`. If they differ, the export stops.
- **Source run:** 310 (R0 seed 42) for stage 1. 208 (CE reference, best epoch by boundary F1@5) for
  stage 2, because a front end trained with the count loss might pre-shape the features toward sharp
  B peaks.
- **Seeds:** the seeds vary only the new encoder. The front end is one fixed run.

**Adjacent phrases.** FACT's original builds target segments from runs of the same class, so two
adjacent phrases (no O frames between them) would merge into one segment. Those continuous
boundaries are exactly the hard ones. Here:
- segments come from the BIO labels: every B starts a phrase, and every O run is its own segment;
- tokens predict O-segment / phrase / null;
- the frame branch predicts B/I/O;
- token decoding starts a new phrase where the owning token changes.

**Losses.** FACT's matching, token and cross-attention losses are unchanged in spirit. The frame
losses are R0's: class-weighted CE with [1.0, 1.1, 8.5], plus the count loss at weight 2,
radius 5. FACT's temporal smoothing applies to the phrase/non-phrase marginal, because smoothing the
B/I/O probabilities would suppress the one-frame B spike.

**Two decoders, two checkpoints.**
- `token`: FACT's own decoding.
- `head`: argmax of the last frame branch's B/I/O output, directly comparable to R0's argmax rows.

Each keeps its best epoch by val boundary F1@5 (R0's selection rule), and early stopping waits until
neither improves. FACT-token wins only if the tokens themselves add something beyond a stronger
frame branch.

**Windows.**
- FACT trains on random 30 s windows (1500 frames). It infers on windows with 50% overlap: head
  probabilities are averaged; token labels are taken from the window that owns each frame (overlap
  midpoints).
- The BiLSTM reference uses R0's 512-frame windows and R0's tiling (no overlap), plus P2 at 1500 frames.

**Number of tokens.** 1.25 × the most segments (O runs + phrases) in any training window, rounded up
to a multiple of 8. The null-class weight follows FACT's formula. Both are printed and stored in
`hyperparameters.json`.

**Model size.** hid 256, frame maps 128, token dim 128, blocks `iuUU` (input, update, two
temporal-pooling updates). Kept near the BiLSTM's size rather than FACT's 512-d default, so that
"bigger" is not the explanation. `smoke_test.py` prints both parameter counts, and every run stores
its own.

**Recipes.**
- FACT keeps FACT's: Adam 1e-4, constant LR, gradient clipping 10, batch of 4 windows, half the
  training frames per epoch, up to 60 epochs, patience 10.
- The frozen BiLSTM keeps R0's: AdamW 1e-4, cosine schedule, batch 16, patience 10. A dropout of 0.2
  on the input stands in for the dropout R0's fusion layer applies before its BiLSTM.

**Fixes to the original code.**
- The segment weights in the cross-attention loss are indexed by the matched segments. The original
  only works when every segment is matched and all weights are equal.
- Channel masking is an explicit per-channel mask (newer PyTorch handles `Dropout2d` on 3-D input
  differently).
- The soft IoU uses an exact closed form instead of a frames × tokens × segments tensor.

## Caveats

- **In-sample train features.** R0 was trained on the train videos, so the train-split features are
  cleaner than val/test. FACT and the frozen BiLSTM see the same features, so the comparison is fair,
  but a gain here is evidence for FACT on frozen features, not a full end-to-end result.
- **Token predictions are one-hot.** For `fact_token-*` the saved "probabilities" are the decoded
  labels, because FACT's decoding has no per-frame B score. Use `fact_head-*` for analyses that need
  scores (`count_oracle_decoding.py`'s peak picking, AP).
- **Disk.** Features at 50 fps cost 512 bytes per frame (256 × float16), i.e. roughly 15 MB per
  10 minutes of video. The export prints the total per split. `feature_stride: 2` in a job halves
  FACT's sequence length if training is too slow; predictions are upsampled back to 50 fps for
  evaluation.
- **Not executed where it was written.** The numpy core is tested, but the torch code could not be
  run in that environment (no PyTorch). Run `smoke_test.py` before the real runs.

## Reference

Z. Lu, E. Elhamifar. FACT: Frame-Action Cross-Attention Temporal Modeling for Efficient Action
Segmentation. CVPR 2024. Code: https://github.com/ZijiaLewisLu/CVPR2024-FACT (MIT License).
