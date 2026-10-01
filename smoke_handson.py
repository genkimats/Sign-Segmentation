"""
smoke_handson.py -- ~15-second check of HandsOn2025, the angle features and the CTC loss BEFORE queueing a long run.
Run from the project root (the folder containing src/):

    python smoke_handson.py

Checks (torch needed):
  1. parameter count for all three pose streams vs an independent analytic formula
  2. output shapes for even AND odd window lengths; model.ctc_logits has shape (B, ceil(T/2), 2)
  3. torch angle features == the numpy reference implementation (what test_handson.py verifies exhaustively)
  4. the torch HandsOnCTCLoss equals ce + w * mean(per-token CTC NLL) computed with the independent numpy CTC
     forward algorithm, including a window with NO signs and a window that is INFEASIBLE (must be excluded)
  5. forward/backward through that loss with gradient clipping 0.1; the CTC head receives gradient
  6. ReduceLROnPlateau(mode="max", patience=5) lowers the LR when the score stalls
  7. the model refuses to run without HaMeR
Exit code 0 = all passed.
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.optim as optim

from src.models import HandsOn2025
from src.handson_loss import HandsOnCTCLoss
from src.skeleton_angles import skeleton_angle_features, ANGLE_FEATURE_DIM
from src.handson_ctc_core import count_signs, ctc_feasible, ctc_nll_numpy

fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        fails.append(name)


def lin(i, o): return i * o + o
def ln(d): return 2 * d
def mlp3(i, h, o): return lin(i, h) + ln(h) + lin(h, h) + ln(h) + lin(h, o) + ln(o)
def encoder_layer(d, ff): return (3 * d * d + 3 * d) + (d * d + d) + lin(d, ff) + lin(ff, d) + 2 * ln(d)


def expected(pose_in, hamer_dim=288, d=256, L=4, ff=1024, ad=512, mh=512, classes=3, ctc_k=2):
    return (mlp3(hamer_dim, ad, ad) + mlp3(pose_in, ad, ad) + mlp3(2 * ad, mh, d)
            + L * encoder_layer(d, ff) + lin(d, classes) + lin(d, ctc_k))


torch.manual_seed(0)
V = 65
pose_dim = {"angles": ANGLE_FEATURE_DIM, "xyz": 3 * V, "xyz+angles": 3 * V + ANGLE_FEATURE_DIM}
print(f"angle feature dim = {ANGLE_FEATURE_DIM}\n")

for stream, pin in pose_dim.items():
    net = HandsOn2025(in_channels=3, num_vertices=V, d_model=256, n_layers=4, nhead=8, dim_feedforward=1024,
                      pose_stream=stream, hamer_dim=288, ctc_num_tokens=1)
    n = sum(p.numel() for p in net.parameters())
    check(f"param count pose_stream={stream}", n == expected(pin), f"real={n:,} analytic={expected(pin):,}")
    for T in (64, 63):
        x, ham = torch.randn(2, 3, T, V), torch.randn(2, 288, T)
        net.eval()
        with torch.no_grad():
            logits, emb = net(x, hamer=ham)
        Tp = (T + 1) // 2
        check(f"shapes stream={stream} T={T}",
              tuple(logits.shape) == (2, 3, T) and tuple(emb.shape) == (2, 256, T) and tuple(net.ctc_logits.shape) == (2, Tp, 2),
              f"logits {tuple(logits.shape)} ctc {tuple(net.ctc_logits.shape)}")

# ---------------------------------------------------------------- angles: torch == numpy
xyz = torch.randn(2, 5, V, 3)
a_t = skeleton_angle_features(xyz).numpy()
a_n = skeleton_angle_features(xyz.double().numpy())
check("torch angle features == numpy reference", np.allclose(a_t, a_n, atol=5e-4), f"max diff {np.abs(a_t - a_n).max():.2e}")
a_ys = skeleton_angle_features(xyz, y_scale=0.5625).numpy()
check("torch y_scale path works", a_ys.shape == a_t.shape and np.isfinite(a_ys).all() and not np.allclose(a_ys, a_t))
check("angle features finite for all-zero input", bool(torch.isfinite(skeleton_angle_features(torch.zeros(1, 2, V, 3))).all()))

# ---------------------------------------------------------------- CTC loss: torch vs independent numpy reference
B, Tp, w = 4, 32, 0.5
ctc_logits = torch.randn(B, Tp, 2) * 2
bio_logits = torch.randn(B, 3, 64)
labels = torch.zeros(B, 64, dtype=torch.long)
labels[0, 5] = 2; labels[0, 6:20] = 1; labels[0, 30] = 2; labels[0, 31:45] = 1        # 2 signs
# labels[1]: no sign at all
labels[2, 0:60:3] = 2                                                                  # 20 signs -> needs 39 steps > 32: infeasible
labels[3, 10] = 2; labels[3, 11:30] = 1                                                # 1 sign
crit = HandsOnCTCLoss(ctc_weight=w)
total, ce, ctc = crit(bio_logits, ctc_logits, labels)
lab = labels.numpy()
lp = torch.log_softmax(ctc_logits, -1).numpy()
per, kept = [], []
for b in range(B):
    L = count_signs(lab[b])
    if ctc_feasible(L, Tp):
        per.append(ctc_nll_numpy(lp[b].astype(np.float64), L) / max(L, 1)); kept.append(b)
ce_ref = torch.nn.functional.cross_entropy(bio_logits, labels).item()
check("infeasible window excluded, others kept", kept == [0, 1, 3] and abs(crit.excluded_fraction() - 0.25) < 1e-9,
      f"kept={kept} excluded_fraction={crit.excluded_fraction():.2f}")
check("CTC term equals the independent numpy CTC (incl. a no-sign window)", abs(ctc.item() - float(np.mean(per))) < 1e-4,
      f"torch={ctc.item():.6f} numpy={float(np.mean(per)):.6f}")
check("total = CE + w * CTC", abs(total.item() - (ce_ref + w * float(np.mean(per)))) < 1e-4)

# ---------------------------------------------------------------- backward / clip / plateau
net = HandsOn2025(in_channels=3, num_vertices=V, d_model=256, n_layers=4, nhead=8, dim_feedforward=1024,
                  pose_stream="angles", hamer_dim=288)
net.train()
opt = optim.Adam(net.parameters(), lr=3e-4)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5)
x, ham = torch.randn(4, 3, 64, V), torch.randn(4, 288, 64)
labels = torch.zeros(4, 64, dtype=torch.long)
labels[:, 5] = 2; labels[:, 6:20] = 1; labels[:, 30] = 2; labels[:, 31:45] = 1
opt.zero_grad()
logits, _ = net(x, hamer=ham)
loss, ce, ctc = HandsOnCTCLoss(ctc_weight=0.5)(logits, net.ctc_logits, labels)
loss.backward()
gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1)
grads_ok = all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
check("loss finite, grads finite", bool(torch.isfinite(loss)) and grads_ok,
      f"loss={loss.item():.4f} ce={float(ce):.4f} ctc={float(ctc):.4f} grad_norm={float(gn):.3f}")
check("CTC head receives gradient", net.ctc_head.weight.grad is not None and float(net.ctc_head.weight.grad.abs().sum()) > 0)
check("HaMeR adapter receives gradient", float(net.hamer_adapter.net[0].weight.grad.abs().sum()) > 0)
opt.step()
lrs = []
for _ in range(8):
    sched.step(0.5)
    lrs.append(opt.param_groups[0]["lr"])
check("ReduceLROnPlateau lowers LR after patience", lrs[-1] < 3e-4, f"lr trace {['%.0e' % v for v in lrs]}")

# ---------------------------------------------------------------- guard rails
try:
    HandsOn2025(in_channels=3, num_vertices=V, hamer_dim=None)
    check("refuses to build without HaMeR", False)
except ValueError:
    check("refuses to build without HaMeR", True)
try:
    net(x, hamer=None)
    check("refuses forward without hamer tensor", False)
except ValueError:
    check("refuses forward without hamer tensor", True)

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)