"""
smoke_test.py -- end-to-end check of the TORCH code on a tiny synthetic corpus (needs torch; ~1-2 minutes on CPU).

    python smoke_test.py

Nothing is read from or written to your real data: split lists, gold labels and keypoints are faked in memory, and every
output goes to a temp directory. It exercises: Stage A (padded/packed BiLSTM, training, validation, early-stopping
bookkeeping, fold models, export), Stage C (all training sources incl. out-of-fold, Transformer and BiLSTM, checkpoints),
and evaluate.py (threshold tuning on val, flat / hier / oracle rows, JSON output). Exit code 0 = everything ran and the
invariants held. Run it before queueing a real run.
"""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import segments as S
import stage_a as A
import stage_c as SC
import evaluate as EV

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        fails.append(name)


tmp = tempfile.mkdtemp()
for m in (C, A, SC, EV):
    m.RUNS_DIR = os.path.join(tmp, "runs")
EV.HERE = tmp                                                       # results/ goes to the temp dir
rng = np.random.default_rng(0)
CACHE = {}


def fake_gold(vid, stride=2):
    if vid not in CACHE:
        r = np.random.default_rng(abs(hash(vid)) % 2 ** 32)
        T = 200 if vid.endswith("0_A") else int(r.integers(450, 800))      # one video shorter than the crop -> exercises padding
        t, sg = 5, []
        while t + 30 < T:
            L = int(r.integers(5, 25)); sg.append((t, t + L)); t += L + (int(r.integers(1, 5)) if r.random() < .5 else 0)
        sg = [(s, e) for s, e in sg if e <= T]
        cuts = [0] + sorted(r.choice(np.arange(1, len(sg)), size=max(1, len(sg) // 5), replace=False).tolist()) + [len(sg)]
        ph = [(sg[a][0], sg[b - 1][1]) for a, b in zip(cuts[:-1], cuts[1:])]
        CACHE[vid] = {"sign": sg, "phrase": ph, "T": T, "T_native": 2 * T}
    return CACHE[vid]


def fake_kp(vid, stride=2):
    g = fake_gold(vid)
    r = np.random.default_rng(abs(hash(vid + "kp")) % 2 ** 32)
    x = r.normal(size=(g["T"], 65, 3)).astype(np.float32)
    for s, e in g["sign"]:                                          # make signs detectable: hands move inside signs
        x[s:e, 23:44] += 1.0
    return x


A.split_ids = lambda split, *a: [f"{split}{i}_{p}" for i in range(4) for p in "AB"]
A.have_all_files = lambda v: True
A.load_gold_segments = fake_gold
A.load_keypoints = fake_kp

common_a = dict(seed=0, stride=2, epochs=2, patience=5, lr=1e-3, hidden=32, layers=2, dropout=0.1, crop=256, batch=2,
                crops_per_video=1, weight_power=0.5, no_phrase_head=False, n_folds=2)

# ---------------------------------------------------------------- Stage A
torch.manual_seed(0)
m = A.StageA(in_dim=195, hidden=32, layers=2, dropout=0.0).eval()
x_long, x_short = torch.randn(1, 120, 195), torch.randn(1, 50, 195)
with torch.no_grad():
    alone = m(x_short)[0]
    batch = torch.zeros(2, 120, 195); batch[0] = x_long[0]; batch[1, :50] = x_short[0]
    packed = m(batch, torch.tensor([120, 50]))[0]
check("packed/padded BiLSTM equals the unpadded forward on the short video", torch.allclose(alone[0], packed[1, :50], atol=1e-4),
      f"max diff {float((alone[0] - packed[1, :50]).abs().max()):.2e}")
check("sign and phrase heads output (B, T, 3)", m(x_long)[0].shape == (1, 120, 3) and m(x_long)[1].shape == (1, 120, 3))

A.train(SimpleNamespace(name="sa_t", fold=None, **common_a))
check("Stage A wrote a checkpoint and a log", os.path.exists(os.path.join(tmp, "runs", "sa_t", "stage_a.pt"))
      and os.path.exists(os.path.join(tmp, "runs", "sa_t", "stage_a_log.json")))
A.export(SimpleNamespace(name="sa_t", stride=2, splits=["train", "val", "test"]))
z = C.load_cache_video(os.path.join(tmp, "runs", "sa_t", "cache", "val", "val1_B.npz"))
T = int(z["T"])
check("cache has the expected arrays and shapes", z["h"].shape == (T, 64) and z["sign_logits"].shape == (T, 3)
      and z["xyz"].shape == (T, 65, 3) and z["gold_sign"].shape[1] == 2)
for f in range(2):
    A.train(SimpleNamespace(name=f"sa_f{f}", fold=f, **common_a))
    A.export(SimpleNamespace(name=f"sa_f{f}", stride=2, splits=["train", "val", "test"]))
oof_n = sum(len(os.listdir(os.path.join(tmp, "runs", f"sa_f{f}", "cache", "oof"))) for f in range(2))
check("fold models export exactly the held-out train videos as OOF (and no train cache)", oof_n == 8
      and not os.path.exists(os.path.join(tmp, "runs", "sa_f0", "cache", "train")), f"{oof_n} OOF videos")

# ---- chunked inference (window + overlap) and cache under a new run name
sg_full, ph_full, h_full = A.infer_video(m, x_long[0].numpy(), torch.device("cpu"))
sg_ch, ph_ch, h_ch = A.infer_video(m, x_long[0].numpy(), torch.device("cpu"), window=48, keep=0.75)
check("chunked inference (window 48, keep 0.75) returns full-length outputs of the same shapes",
      sg_ch.shape == sg_full.shape == (120, 3) and ph_ch.shape == (120, 3) and h_ch.shape == h_full.shape == (120, 64) and np.isfinite(sg_ch).all())
sg_big, _, _ = A.infer_video(m, x_long[0].numpy(), torch.device("cpu"), window=500, keep=0.75)
check("window longer than the video == whole-video inference", np.allclose(sg_big, sg_full, atol=1e-5))
A.export(SimpleNamespace(name="sa_t", stride=2, splits=["val"], infer_window=64, infer_keep=0.75, as_name="sa_t_w64k75"))
zz = C.load_cache_video(os.path.join(tmp, "runs", "sa_t_w64k75", "cache", "val", "val1_B.npz"))
check("export --as writes a separate cache that records the inference mode", int(zz["infer_window"]) == 64 and abs(float(zz["infer_keep"]) - 0.75) < 1e-6
      and os.path.exists(os.path.join(tmp, "runs", "sa_t_w64k75", "stage_a.pt")) and zz["h"].shape[0] == int(zz["T"]))
for arch in ("stgcn_bilstm", "stgcn_bimamba"):
    try:
        ms = A.build_stage_a(dict(arch=arch, in_dim=195, hidden=32, layers=2, dropout=0.0, no_phrase_head=False)).eval()
    except Exception as e:                                           # e.g. mamba_ssm missing
        print(f"[SKIP] {arch}: cannot build here ({type(e).__name__}: {str(e)[:80]})")
        continue
    dev = torch.device("cuda" if (arch == "stgcn_bimamba" and torch.cuda.is_available()) else "cpu")
    if arch == "stgcn_bimamba" and dev.type == "cpu":
        print("[SKIP] stgcn_bimamba forward: Mamba kernels need a GPU (it built fine)")
        continue
    ms = ms.to(dev)
    with torch.no_grad():
        o = ms(torch.randn(2, 80, 195, device=dev))
    check(f"{arch}: outputs (B,T,3) x2 and features (B,T,64)", o[0].shape == (2, 80, 3) and o[1].shape == (2, 80, 3) and o[2].shape == (2, 80, 64))
A.train(SimpleNamespace(name="sa_g", fold=None, **{**common_a, "no_phrase_head": False}, arch="stgcn_bilstm", stgcn_channels=8, equal_frames=True,
                        infer_window=64, infer_keep=0.75))
check("Stage A trains with the ST-GCN encoder, equal-frames and chunked validation", os.path.exists(os.path.join(tmp, "runs", "sa_g", "stage_a.pt")))

import window_study as WS
WS.HERE = tmp
sys.argv = ["window_study.py", "--run", "sa_t", "--windows", "32", "--keeps", "1.0", "0.75"]
WS.main()
ws = json.load(open(os.path.join(tmp, "results", "window_study_sa_t.json")))
check("window_study ran whole-video and chunked modes", {"0x1.0", "32x1.0", "32x0.75"} <= set(ws["val"]) and "phrase" in ws["val"]["0x1.0"])

# ---------------------------------------------------------------- Stage C
base = dict(stage_a="sa_t", tag="t", seed=0, groups=list(SC.ALL_GROUPS), oof_runs=[], arch="transformer", d_model=32, layers=1,
            heads=2, lstm_layers=1, lstm_hidden=16, dropout=0.1, feat_dropout=0.1, window=16, batch=4, epochs=2, patience=5,
            lr=1e-3, wd=0.01, weight_power=1.0, focal_gamma=0.0, ramp=2, jitter=1.0, b_thr=0.5, o_thr=0.5, limit=None, source="mix",
            tag_rule="contain_or_next", end_rule="last_sign_end")
for source in ("gold", "jitter", "schedule", "mix", "pred"):
    SC.train(SimpleNamespace(**{**base, "source": source, "tag": source}))
    check(f"Stage C trains with source '{source}'", os.path.exists(os.path.join(tmp, "runs", "sa_t", "stage_c", f"{source}_s0", "stage_c.pt")))
SC.train(SimpleNamespace(**{**base, "arch": "bilstm", "tag": "bilstm"}))
SC.train(SimpleNamespace(**{**base, "source": "pred", "tag": "oof", "groups": ["sign_probs", "prosody"], "oof_runs": ["sa_f0", "sa_f1"]}))
SC.train(SimpleNamespace(**{**base, "tag": "focal", "focal_gamma": 2.0}))
check("BiLSTM control, out-of-fold variant and focal loss all train", all(
    os.path.exists(os.path.join(tmp, "runs", "sa_t", "stage_c", f"{t}_s0", "stage_c.pt")) for t in ("bilstm", "oof", "focal")))
try:
    SC.train(SimpleNamespace(**{**base, "tag": "bad", "oof_runs": ["sa_f0"]}))
    check("--oof-runs with h_pool is refused", False)
except SystemExit:
    check("--oof-runs with h_pool is refused", True)

# ---------------------------------------------------------------- evaluate
for tag in ("mix", "oof"):
    sys.argv = ["evaluate.py", "--stage-a", "sa_t", "--tag", tag, "--seed", "0"]
    EV.main()
res = json.load(open(os.path.join(tmp, "results", "sa_t__mix_s0.json")))
rows = set(res["test"])
check("evaluate.py wrote every row for val and test", {"flat_default", "flat_tuned", "hier_default", "hier_tuned",
      "hier_oracle_default", "hier_oracle_tuned", "sign_stage_tuned"} <= rows and set(res["val"]) == rows)
check("metrics are finite and in range", all(0 <= res["test"][r]["frame_f1"] <= 1 for r in rows) and
      all(np.isfinite(res["test"][r]["ratio"]) for r in rows))

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)