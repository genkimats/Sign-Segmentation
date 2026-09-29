"""
Self-tests for the decoder study. Run:  python test_decoder_study.py
Needs only numpy/scipy (plus the project's src/dataset.py for the dilation
tests; if that can't be imported here, the smoothing function is read straight
out of the project file instead, and a note is printed).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_common import PROJECT_ROOT, plan_windows, stitch_logits, O, I, B
import study_metrics as M
import study_decoders as D
import study_calibration as C

_results = []
def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

def load_smoothing():
    try:
        from study_common import training_targets
        training_targets(np.array([0, 2, 1, 0]))
        return training_targets
    except Exception as e:
        print(f"  note: src.dataset not importable here ({type(e).__name__}); reading apply_label_smoothing from file text")
    for p in [os.path.join(PROJECT_ROOT, "src", "dataset.py"), "/mnt/user-data/outputs/dataset.py"]:
        if os.path.exists(p):
            src = open(p).read().replace("\r", "")
            ns = {"np": np}
            exec(src[src.index("def apply_label_smoothing"):src.index("class SignSegmentationDataset")], ns)
            f = ns["apply_label_smoothing"]
            return lambda raw, tol=5: (lambda s: (s, s.argmax(1).astype(np.int8)))(f(np.asarray(raw).astype(np.int64), tol))
    raise RuntimeError("cannot locate apply_label_smoothing")

def synth_gold(rng, T, med=20, gap_p=0.4):
    lab = np.zeros(T, np.int8); t = int(rng.integers(0, 10))
    while t < T - 2:
        d = min(int(np.clip(rng.lognormal(np.log(med), 0.5), 3, 120)), T - t)
        lab[t] = B; lab[t + 1:t + d] = I; t += d
        if rng.random() < gap_p: t += int(rng.integers(1, 15))
    return lab

def legal(x):
    x = np.asarray(x)
    return x[0] != I and not np.any((x[:-1] == O) & (x[1:] == I))

rng = np.random.default_rng(0)

# ---------------------------------------------------------------- windows ----
print("\n== windows / stitching ==")
w = plan_windows(150, 64, 64)
check("plan_windows tiles with flush tail", w == [(0, 64), (64, 128), (86, 150)], str(w))
check("plan_windows short video -> one window", plan_windows(40, 64, 64) == [(0, 40)])
truth = rng.normal(size=(150, 3)).astype(np.float32)
st = stitch_logits(150, w, [truth[s:e].T for s, e in w])
check("stitching recovers signal exactly (overlaps averaged, not duplicated)", np.allclose(st, truth, atol=1e-6))
check("stitched length == T (old concat gave 192)", len(st) == 150)

# ---------------------------------------------- metrics vs the project's code ---
print("\n== metrics ==")
mp = next((p for p in [os.path.join(PROJECT_ROOT, "src", "metrics.py"), "/mnt/user-data/uploads/metrics.py"] if os.path.exists(p)), None)
gold = np.array([0]*5 + [2] + [1]*29 + [0]*5)
pred2 = np.array([0]*5 + [2] + [1]*14 + [2] + [1]*14 + [0]*5)
s2 = M.summarize(M.aggregate([M.video_stats(pred2, gold)]))
if mp:
    src = open(mp).read().replace("\r", ""); ns = {"np": np}
    exec(src[src.index("def extract_segments"):src.index("def evaluate_batch")], ns)
    bad = 0
    for _ in range(1000):
        a = rng.integers(0, 3, rng.integers(1, 70))
        bad += M.bio_to_segments(a) != [(s, e + 1) for s, e in ns["extract_segments"](a.tolist())]
    check("bio_to_segments == project extract_segments (1000 random)", bad == 0)
    proj = ns["calculate_segment_metrics"](pred2.tolist(), gold.tolist(), 0.5)
    check("legacy_metrics reproduces project Segment F1", abs(proj - s2["legacy_segF1"]) < 1e-9, f"{proj:.4f}")
else:
    print("  note: project metrics.py not found, skipping cross-check")
check("LEGACY Segment F1 exceeds 1.0 on a sign split in two (project metric flaw)", s2["legacy_segF1"] > 1.0, f"{s2['legacy_segF1']:.3f}")
check("one-to-one segF1@0.5 is honest (0.667) and ratio shows 2.0", abs(s2["segF1@0.5"] - 2/3) < 1e-9 and s2["segment_ratio"] == 2.0)
perf = M.summarize(M.aggregate([M.video_stats(gold, gold)]))
check("perfect prediction -> all 1.0", perf["frame_macro_f1"] == 1.0 and perf["segF1@0.5"] == 1.0 and perf["startF1@2"] == 1.0 and perf["segment_ratio"] == 1.0)
shift = np.array([0]*7 + [2] + [1]*29 + [0]*3)          # start moved +2 frames
sh = M.summarize(M.aggregate([M.video_stats(shift, gold, tols=(1, 2))]), tols=(1, 2))
check("boundary tolerance: start off by 2 frames fails @1, passes @2", sh["startF1@1"] == 0.0 and sh["startF1@2"] == 1.0)
sts = [M.video_stats(synth_gold(rng, 800), synth_gold(rng, 800)) for _ in range(12)]
pt = M.summarize(M.aggregate(sts)); ci = M.bootstrap_ci(sts, n_boot=200)
check("bootstrap CI brackets the point estimate", all(ci[k][0] - 1e-9 <= pt[k] <= ci[k][1] + 1e-9 for k in ("frame_macro_f1", "segF1@0.5", "segment_ratio")))

# --------------------------------------------------------------- decoders ----
print("\n== decoders ==")
ok_eq, ok_legal = True, True
uniform = np.concatenate([[-np.inf], np.zeros(60)])
for _ in range(40):
    T = int(rng.integers(5, 45))
    lp = D.to_logp(rng.normal(size=(T, 3)) * 1.5)
    v = D.decode_viterbi(lp)
    sm = D.decode_semi_markov(lp, np.concatenate([[-np.inf], np.zeros(T)]))
    ok_eq &= np.array_equal(v, sm); ok_legal &= legal(v) and legal(sm)
check("semi-Markov (uniform prior, dmax>=T) == constrained Viterbi on 40 random inputs", ok_eq)
check("Viterbi and semi-Markov outputs are always legal BIO (no O->I, no leading I)", ok_legal)

goldv = {i: synth_gold(rng, 600) for i in range(6)}
prior = D.fit_duration_prior(list(goldv.values()))
allok = {}
for name, fn in {"argmax": D.decode_argmax, "threshold": lambda l: D.decode_threshold(l, .5, .5),
                 "hysteresis": D.decode_hysteresis, "viterbi": D.decode_viterbi,
                 "semi_markov": lambda l: D.decode_semi_markov(l, prior["dur_logp"])}.items():
    allok[name] = all(np.array_equal(fn(D.oracle_logp(g)), g) for g in goldv.values())
check("ORACLE (gold logits): every decoder reproduces gold exactly", all(allok.values()), str(allok))

# dilation oracle: what a PERFECT model of the training objective would emit
tt = load_smoothing()
keys = {"argmax": lambda lp: D.decode_argmax(lp),
        "argmax+collapse": lambda lp: D.collapse_b_runs(D.decode_argmax(lp), lp),
        "viterbi": lambda lp: D.decode_viterbi(lp),
        "semi_markov": lambda lp: D.decode_semi_markov(lp, prior["dur_logp"])}
res = {k: [] for k in keys}; golds = {}
for i, g in goldv.items():
    soft, _ = tt(g, 5); lp = D.oracle_soft_logp(soft); golds[i] = g
    for k, fn in keys.items(): res[k].append(fn(lp))
print("  perfect-model-of-dilated-targets, scored vs RAW gold:")
sm = {}
for k in keys:
    s, _, _ = M.evaluate({i: res[k][n] for n, i in enumerate(goldv)}, golds)
    sm[k] = s; print(f"    {k:16s} segF1@0.5={s['segF1@0.5']:.3f}  ratio={s['segment_ratio']:.2f}  startF1@2={s['startF1@2']:.3f}  frameF1={s['frame_macro_f1']:.3f}")
check("greedy argmax over-segments badly on dilated targets (ratio > 1.5)", sm["argmax"]["segment_ratio"] > 1.5, f"{sm['argmax']['segment_ratio']:.2f}")
check("collapse_b_runs repairs it (ratio < 1.15)", sm["argmax+collapse"]["segment_ratio"] < 1.15, f"{sm['argmax+collapse']['segment_ratio']:.2f}")
check("=> a fair greedy baseline MUST include collapse (else semi-Markov gets unearned credit)", True)

# noisy-emission benefit under a well-specified generative model
print("  noisy emissions (gold one-hot*2 + N(0,1.5)), prior fit on independent gold:")
prior2 = D.fit_duration_prior([synth_gold(rng, 800) for _ in range(20)])
tg = {i: synth_gold(rng, 1500) for i in range(8)}
lgs = {}
for i, g in tg.items():
    z = rng.normal(0, 1.5, (len(g), 3)); z[np.arange(len(g)), g] += 2.0; lgs[i] = z
out = {"argmax": {}, "argmax+collapse": {}, "viterbi": {}, "semi_markov": {}}
for i, z in lgs.items():
    lp = D.to_logp(z)
    out["argmax"][i] = D.decode_argmax(lp)
    out["argmax+collapse"][i] = D.collapse_b_runs(D.decode_argmax(lp), lp)
    out["viterbi"][i] = D.decode_viterbi(lp)
    out["semi_markov"][i] = D.decode_semi_markov(lp, prior2["dur_logp"])
nz = {}
for k, pr in out.items():
    s, _, _ = M.evaluate(pr, tg); nz[k] = s
    print(f"    {k:16s} segF1@0.5={s['segF1@0.5']:.3f}  ratio={s['segment_ratio']:.2f}  frameF1={s['frame_macro_f1']:.3f}")
check("semi-Markov beats argmax on segF1 under noise, and has ratio nearer 1",
      nz["semi_markov"]["segF1@0.5"] > nz["argmax"]["segF1@0.5"] and abs(nz["semi_markov"]["segment_ratio"] - 1) < abs(nz["argmax"]["segment_ratio"] - 1))

# hysteresis port behaviour
lp = np.log(np.array([[.9,.05,.05]]*3 + [[.2,.7,.1]] + [[.9,.05,.05]]*3))       # O O O (I-like blip) O O O
h = D.decode_hysteresis(lp, 0.6)
check("hysteresis: O->I blip with weak B blocked (stays O)", (h == 0).all(), str(h.tolist()))

# speed at realistic scale
T = 20000; lp = D.to_logp(rng.normal(size=(T, 3)))
t0 = time.time(); D.decode_semi_markov(lp, prior["dur_logp"]); dt = time.time() - t0
check(f"semi-Markov speed: T={T}, dmax={prior['dmax']} in {dt:.1f}s", dt < 60, f"~{dt/T*1e5:.1f}s per 100k frames")

# ------------------------------------------------------------ calibration ----
print("\n== calibration ==")
T0 = 2.0; n = 60000
true_logits = rng.normal(size=(n, 3)) * 2
p = np.exp(D.to_logp(true_logits)); cdf = p.cumsum(1)
y = (rng.random((n, 1)) > cdf).sum(1).clip(0, 2)
fit = C.fit_temperature([true_logits * T0], [y])       # model logits are T0x too sharp
check("fit_temperature recovers the injected over-confidence factor", abs(fit - T0) < 0.1, f"fit={fit:.2f} vs {T0}")
check("ECE drops after temperature scaling", C.calibration_report([true_logits * T0], [y], fit)["ece_top"] < C.calibration_report([true_logits * T0], [y], 1.0)["ece_top"])


# ------------------------------------------------------ exporter windowing ----
print("\n== exporter windowing (fake model, no torch) ==")
from export_logits import export_video_logits
def fake_run_batch(fb, eb):
    # logits channel 0 <- frame index encoded in feature[0,:,0]; channel 1 <- hamer[0]; channel 2 <- zeros
    n, C, W, V = fb.shape
    out = np.zeros((n, 3, W), np.float32)
    out[:, 0] = fb[:, 0, :, 0]
    if "hamer" in eb: out[:, 1] = eb["hamer"][:, 0]
    return out
for T in (40, 64, 150, 1000):
    feats = np.zeros((2, T, 3), np.float32); feats[0, :, 0] = np.arange(T)
    ham = np.zeros((5, T), np.float32); ham[0] = np.arange(T) * 2
    lg = export_video_logits(feats, {"hamer": ham}, 64, 64, 4, fake_run_batch)
    check(f"exporter T={T}: length, frame alignment and extras alignment exact",
          lg.shape == (T, 3) and np.array_equal(lg[:, 0], np.arange(T)) and np.array_equal(lg[:, 1], np.arange(T) * 2))
lg2 = export_video_logits(np.zeros((2, 150, 3), np.float32), {}, 64, 32, 3, fake_run_batch)
check("exporter with overlapping stride 32 still covers every frame", lg2.shape == (150, 3))

# ------------------------------------------------ full pipeline, synthetic ----
print("\n== run_evaluation end-to-end on synthetic exports ==")
import run_decoder_eval as R
def enc_like(g, rng):
    """Encoder-like logits: log of the DILATED soft targets (what the net is trained on) plus
    temporally CORRELATED noise (AR(1)), not iid -- closer to real encoder errors."""
    soft, _ = tt(g, 5)
    base = np.log(np.clip(soft, 1e-4, 1)) * 0.8
    noise = np.zeros_like(base); eps = rng.normal(0, 1.0, base.shape)
    for t in range(1, len(g)): noise[t] = 0.85 * noise[t - 1] + 0.53 * eps[t]
    return (base + noise).astype(np.float32)
def make_records(n, T):
    recs = {}
    for i in range(n):
        g = synth_gold(rng, T); recs[f"v{rng.integers(1e9)}"] = {"logits": enc_like(g, rng), "labels": g}
    return recs
sel_rec, ev_rec = make_records(4, 2500), make_records(4, 2500)
train_gold = [synth_gold(rng, 2500) for _ in range(15)]
small = {"t_b": (0.3, 0.5, 0.7), "t_o": (0.4, 0.6), "hyst": (0.6,), "sm_w": (0.5, 1.0), "sm_pen": (0.0, 2.0)}
lines = []
res = R.run_evaluation(sel_rec, ev_rec, train_gold, 5, "auto", 40, targets_fn=tt, grids=small, log=lambda m: lines.append(m))
check("pipeline runs and returns all 8 decoder families", len(res["decoders"]) == 8, ", ".join(res["decoders"])[:90])
check("oracle-1 (gold logits) passes for every decoder in the runner", all(res["oracle_gold_ok"].values()))
od = res["oracle_dilated"]
check("oracle-2 shows the dilation problem and that collapse/semi-Markov remove it",
      od["argmax"]["segment_ratio"] > 1.5 and od["argmax+collapse"]["segment_ratio"] < 1.15 and od["semi-Markov"]["segment_ratio"] < 1.15)
check("viterbi alone does NOT fix dilation (ratio stays > 1.5)", od["viterbi"]["segment_ratio"] > 1.5)
check("every decoder result carries bootstrap CIs and duration-bucket recalls",
      all(d["ci"] is not None and "recall_bucket0" in d["eval"] for d in res["decoders"].values()))
check("temperature was fit and is positive", res["temperature"] > 0, f"T={res['temperature']:.2f}")
print("  ---- excerpt of the runner's report ----")
for l in lines:
    if l.startswith(("decoder ", "argmax", "threshold", "hysteresis", "viterbi", "semi-Markov")) and "segF1" not in l[:6] and len(l) > 60:
        print("  " + l)


# ------------------------------------- save path / grouping / grid edges / cap ----
print("\n== saving, document-level CIs, grid-edge warnings, dmax-aware oracle ==")
import json as _json, tempfile, csv as _csv
# (a) the exact failure seen on real data: numpy bools + NaN must survive a STRICT json dump
res["oracle_gold_ok"]["_np_bool_probe"] = np.bool_(True)
res["decoders"]["argmax"]["eval"]["_nan_probe"] = float("nan")
with tempfile.TemporaryDirectory() as td:
    R.save_results(res, td)
    back = _json.load(open(os.path.join(td, "results.json")), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    rows = list(_csv.reader(open(os.path.join(td, "results.csv"))))
check("save_results writes STRICT JSON (numpy bool_ and NaN handled) and a CSV", back["oracle_gold_ok"]["_np_bool_probe"] is True
      and back["decoders"]["argmax"]["eval"]["_nan_probe"] is None and len(rows) == 1 + len(res["decoders"]))

# (b) document-level bootstrap must be WIDER than per-video when A and B of a recording are duplicates
docs = {}
for d in range(8):
    g = synth_gold(rng, 1500); z = rng.normal(0, 1.5, (1500, 3)); z[np.arange(1500), g] += 2.0
    p_ = D.decode_argmax(D.to_logp(z)); docs[f"doc{d}_A"] = (p_, g); docs[f"doc{d}_B"] = (p_.copy(), g.copy())
pr_ = {k: v[0] for k, v in docs.items()}; go_ = {k: v[1] for k, v in docs.items()}
_, _, ci_v = M.evaluate(pr_, go_, n_boot=400)
_, _, ci_d = M.evaluate(pr_, go_, n_boot=400, group_fn=R.doc_id)
w_v = ci_v["segF1@0.5"][1] - ci_v["segF1@0.5"][0]; w_d = ci_d["segF1@0.5"][1] - ci_d["segF1@0.5"][0]
check("document-level CI is wider than naive per-video CI when A/B are dependent", w_d > w_v, f"{w_d:.4f} vs {w_v:.4f}")
check("doc_id groups A and B of one recording", R.doc_id("1429910-16075041-16115817_A") == R.doc_id("1429910-16075041-16115817_B") == "1429910-16075041-16115817")

# (c) grid-edge warnings
g_ = R.DEFAULT_GRIDS
check("edge_warnings flags a value on the grid boundary", len(R.edge_warnings({"dur_w": g_["sm_w"][0], "seg_pen": 0.0}, g_)) == 1)
check("edge_warnings is silent for interior values", R.edge_warnings({"dur_w": 0.25, "seg_pen": -2.0, "t_b": 0.5, "t_o": 0.5}, g_) == [])
check("the old optimum (dur_w=0.25, seg_pen=-2.0) is now interior to the extended grid", R.edge_warnings({"dur_w": 0.25, "seg_pen": -2.0}, g_) == [])

# (d) oracle with signs longer than dmax: semi-Markov cannot represent them; the runner must EXPLAIN, not cry wolf
def gold_with(durs, T=4000):
    lab = np.zeros(T, np.int8); t = 5
    for d in durs:
        lab[t] = B; lab[t + 1:t + d] = I; t += d + 6
    return lab
tr = [gold_with(rng.integers(6, 50, 60).tolist()) for _ in range(6)]           # dmax will be ~60
ev_g = {"e1": gold_with([20] * 30 + [130] + [15] * 30, 6000), "e2": gold_with([25] * 40 + [400], 6000)}   # 130 (<=2*dmax), 400 (>2*dmax)
ev_r = {k: {"logits": D.oracle_logp(g).astype(np.float32), "labels": g} for k, g in ev_g.items()}
sel_r = {k: {"logits": D.oracle_logp(gold_with([20] * 40, 3000)).astype(np.float32), "labels": gold_with([20] * 40, 3000)} for k in ("s1",)}
lines2 = []
r2 = R.run_evaluation(sel_r, ev_r, tr, 5, 1.0, 0, targets_fn=tt, grids={"t_b": (0.5,), "t_o": (0.5,), "hyst": (0.6,), "sm_w": (1.0,), "sm_pen": (0.0,)}, log=lambda m: lines2.append(m))
notes = r2["oracle_notes"]
check("runner counts gold signs beyond dmax and the forced splits", notes["n_gold_longer_than_dmax"] == 2 and notes["forced_extra_splits"] >= 2, str(notes))
sm_line = next(l for l in lines2 if l.strip().startswith("semi-Markov") and "segF1@0.5=" in l and "ORACLE" not in l and "ok" in l)
check("semi-Markov's oracle shortfall is reported as explained by the dmax cap (not 'LOSES INFORMATION')",
      r2["oracle_gold_ok"]["semi-Markov"] and "explained" in sm_line and all(r2["oracle_gold_ok"][k] for k in ("argmax", "threshold", "hysteresis", "viterbi")), sm_line.strip()[:110])


# (e) grids bracket both real optima; hysteresis-off is not a warning; P/R consistent with F1
check("both real semi-Markov optima (0.25,-2.0) and (1.0,+2.0) are interior to the grids", R.edge_warnings({"dur_w": 0.25, "seg_pen": -2.0}, g_) == [] and R.edge_warnings({"dur_w": 1.0, "seg_pen": 2.0}, g_) == [])
check("threshold optimum t_b=0.2 is now interior", R.edge_warnings({"t_b": 0.2, "t_o": 0.5}, g_) == [])
check("hysteresis thr<=1/3 (rule disabled) raises no warning", R.edge_warnings({"thr": 0.3}, g_) == [])
_s = M.summarize(M.aggregate([M.video_stats(pred2, gold)]))
check("segP/segR are consistent with segF1", abs(2 * _s["segP@0.5"] * _s["segR@0.5"] / (_s["segP@0.5"] + _s["segR@0.5"]) - _s["segF1@0.5"]) < 1e-9)


# ------------------------------------------------ paired bootstrap ----
print("\n== paired bootstrap: resolves small differences marginal CIs cannot ==")
def synth_gold2(rng, T, med=15):
    lab = np.zeros(T, np.int8); t = int(rng.integers(0, 10))
    while t < T - 2:
        d = min(int(np.clip(rng.lognormal(np.log(med), 0.4), 3, 60)), T - t)
        lab[t] = 2; lab[t+1:t+d] = 1; t += d + int(rng.integers(1, 10))
    return lab

A_, B_, gold_ = {}, {}, {}
for d in range(9):
    g = synth_gold2(rng, 2000); noise = rng.normal(0, 1.3, (2000, 3))
    la = noise.copy(); la[np.arange(2000), g] += 1.85       # A: small true edge
    lb = noise.copy(); lb[np.arange(2000), g] += 1.80       # B: same shared noise draw
    A_[f"d{d}"] = D.decode_argmax(D.to_logp(la))
    B_[f"d{d}"] = D.decode_argmax(D.to_logp(lb))
    gold_[f"d{d}"] = g

_, stats_a, ci_a = M.evaluate(A_, gold_, n_boot=600, group_fn=lambda v: v)
_, stats_b, ci_b = M.evaluate(B_, gold_, n_boot=600, group_fn=lambda v: v)
marg_overlap = not (ci_a["segF1@0.5"][1] < ci_b["segF1@0.5"][0] or ci_b["segF1@0.5"][1] < ci_a["segF1@0.5"][0])
pd_ab = M.paired_bootstrap_diff(stats_a, stats_b, groups=list(gold_), n_boot=1500)
lo, hi, p_gt = pd_ab["segF1@0.5"]
check("paired test on correlated decoders: marginal CIs overlap but paired CI still excludes 0",
      marg_overlap and (lo > 0 or hi < 0) and p_gt > 0.9, f"marginal_overlap={marg_overlap}, paired=[{lo:.4f},{hi:.4f}], P={p_gt:.3f}")
pd_aa = M.paired_bootstrap_diff(stats_a, stats_a, groups=list(gold_), n_boot=200)
check("paired diff of a decoder against ITSELF is exactly zero", pd_aa["segF1@0.5"] == (0.0, 0.0, 0.0), str(pd_aa["segF1@0.5"]))

# ------------------------------------------- runner surfaces paired comparisons ----
print("\n== runner: paired_vs_best section ==")
sel3, ev3 = make_records(4, 2500), make_records(4, 2500)   # reuse helpers from the earlier full-pipeline test
res3 = R.run_evaluation(sel3, ev3, [synth_gold(rng, 2500) for _ in range(15)], 5, "auto", 300,
                        targets_fn=tt, grids=small, log=lambda m: None)
check("runner records a paired_vs_best block naming the top decoder", "paired_vs_best" in res3 and "reference" in res3["paired_vs_best"])
best_ref = res3["paired_vs_best"]["reference"]
others = [k for k in res3["paired_vs_best"] if k != "reference"]
check("every non-best decoder gets a verdict against the reference", len(others) == len(res3["decoders"]) - 1
      and all("verdict" in res3["paired_vs_best"][k] for k in others))
check(f"reference '{best_ref}' is indeed the top decoder by point estimate",
      best_ref == max(res3["decoders"], key=lambda k: res3["decoders"][k]["eval"][R.SELECT_METRIC]))

print(f"\n{sum(_results)}/{len(_results)} checks passed")
sys.exit(0 if all(_results) else 1)