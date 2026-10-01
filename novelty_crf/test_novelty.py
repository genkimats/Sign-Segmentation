"""
Self-tests for the novelty_crf package. Runs with numpy only (no torch needed):
the algorithmic cores execute the REAL code via the seg_ops.NP_OPS adapter and are
checked against brute-force / reference implementations. Run:

    python test_novelty.py
"""
import os
import sys
import itertools
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from seg_ops import NP_OPS as xp
import seg_cores as C

_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""))


rng = np.random.default_rng(0)
K = 3

# ============================================================== CRF ==========
print("\n== CRF: forward algorithm / gold score / Viterbi vs brute force ==")


def path_score(E, y, trans, start, end):
    s = start[y[0]] + E[0, y[0]]
    for t in range(1, len(y)):
        s += trans[y[t - 1], y[t]] + E[t, y[t]]
    return s + end[y[-1]]


def brute(E, trans, start, end):
    T = E.shape[0]
    paths = list(itertools.product(range(K), repeat=T))
    scores = np.array([path_score(E, p, trans, start, end) for p in paths])
    return paths, scores


Bsz, T = 4, 5
E = rng.normal(0, 1.5, (Bsz, T, K))
trans = rng.normal(0, 1.0, (K, K)); start = rng.normal(0, 1, K); end = rng.normal(0, 1, K)
logZ = C.crf_log_partition(xp, E, trans, start, end)
ok_Z, ok_gold, ok_vit, ok_prob = True, True, True, True
y = rng.integers(0, K, (Bsz, T))
gold = C.crf_gold_score(xp, E, y, trans, start, end)
vit = C.crf_viterbi(xp, E, trans, start, end)
for b in range(Bsz):
    paths, scores = brute(E[b], trans, start, end)
    m = scores.max(); ref_logZ = m + np.log(np.exp(scores - m).sum())
    ok_Z &= abs(logZ[b] - ref_logZ) < 1e-9
    ok_gold &= abs(gold[b] - path_score(E[b], y[b], trans, start, end)) < 1e-9
    ok_vit &= tuple(vit[b]) == paths[int(scores.argmax())]
    ok_prob &= abs(np.exp(scores - ref_logZ).sum() - 1.0) < 1e-9
check("log-partition == logsumexp over all 3^5 paths (batch of 4)", ok_Z)
check("gold-path score == explicit sum", ok_gold)
check("Viterbi path == brute-force argmax path", ok_vit)
check("path probabilities exp(score - logZ) sum to exactly 1", ok_prob)

nll = C.crf_nll_per_frame(xp, E, y, trans, start, end)
check("per-frame NLL is non-negative and equals mean of (logZ - gold)/T",
      nll >= 0 and abs(nll - ((logZ - gold).sum() / (Bsz * T))) < 1e-12, f"{float(nll):.4f}")

# forbidden O->I transition
fm = C.forbid_mask(K, ((C.O, C.I),))
trans_f = trans + fm * (-20.0)
E2 = rng.normal(0, 1.0, (3, 6, K))
vit_f = C.crf_viterbi(xp, E2, trans_f, start, end)
no_oi = all(not any(v[t] == C.O and v[t + 1] == C.I for t in range(len(v) - 1)) for v in vit_f)
paths, scores = brute(E2[0], trans_f, start, end)
mass_bad = sum(np.exp(s - (scores.max() + np.log(np.exp(scores - scores.max()).sum())))
               for p, s in zip(paths, scores) if any(p[t] == 0 and p[t + 1] == 1 for t in range(len(p) - 1)))
check("with the O->I penalty, Viterbi never emits an illegal O->I", no_oi)
check("with the O->I penalty, total probability of illegal paths is negligible", mass_bad < 1e-6, f"{mass_bad:.2e}")

# masking == running on the truncated sequence
E3 = rng.normal(0, 1.2, (2, 6, K)); y3 = rng.integers(0, K, (2, 6)); lens = [6, 3]
mask = np.array([[t < L for t in range(6)] for L in lens])
lz = C.crf_log_partition(xp, E3, trans, start, end, mask)
gs = C.crf_gold_score(xp, E3, y3, trans, start, end, mask)
ok_mask = True
for b, L in enumerate(lens):
    lz_ref = C.crf_log_partition(xp, E3[b:b + 1, :L], trans, start, end)[0]
    gs_ref = C.crf_gold_score(xp, E3[b:b + 1, :L], y3[b:b + 1, :L], trans, start, end)[0]
    ok_mask &= abs(lz[b] - lz_ref) < 1e-9 and abs(gs[b] - gs_ref) < 1e-9
check("padding mask: results identical to running each sequence at its true length", ok_mask)
nll_m = C.crf_nll_per_frame(xp, E3, y3, trans, start, end, mask)
ref = sum(C.crf_log_partition(xp, E3[b:b+1, :L], trans, start, end)[0] - C.crf_gold_score(xp, E3[b:b+1, :L], y3[b:b+1, :L], trans, start, end)[0]
          for b, L in enumerate(lens)) / sum(lens)
check("masked per-frame NLL normalises by REAL frame count", abs(nll_m - ref) < 1e-9)

# a CRF with zero transitions and a peaked emission reduces to per-frame softmax CE
E4 = rng.normal(0, 2, (2, 7, K)); y4 = rng.integers(0, K, (2, 7)); z = np.zeros((K, K)); z1 = np.zeros(K)
crf_loss = C.crf_nll_per_frame(xp, E4, y4, z, z1, z1)
logp = E4 - np.log(np.exp(E4).sum(-1, keepdims=True))
ce = -np.take_along_axis(logp, y4[..., None], -1).mean()
check("with all-zero transitions the CRF NLL equals per-frame softmax cross-entropy", abs(crf_loss - ce) < 1e-9)

# ============================================================= mLSTM =========
print("\n== mLSTM parallel form vs the defining recurrence ==")


def mlstm_recurrent(q, k, v, i_pre, f_pre):
    """float64 reference straight from the definition (no stabiliser)."""
    Bq, H, T, dh = q.shape
    out = np.zeros_like(q)
    for b in range(Bq):
        for h in range(H):
            Cm = np.zeros((dh, dh)); n = np.zeros(dh)
            for t in range(T):
                f = 1.0 / (1.0 + np.exp(-f_pre[b, h, t])); i = np.exp(i_pre[b, h, t])
                kt = k[b, h, t] / np.sqrt(dh)
                Cm = f * Cm + i * np.outer(v[b, h, t], kt)
                n = f * n + i * kt
                out[b, h, t] = Cm @ q[b, h, t] / max(abs(n @ q[b, h, t]), 1.0)
    return out


Bq, H, T, dh = 2, 3, 9, 8
q, k, v = (rng.normal(0, 1, (Bq, H, T, dh)) for _ in range(3))
ip, fp = rng.normal(0, 1.5, (Bq, H, T)), rng.normal(1.0, 1.5, (Bq, H, T))
par = C.mlstm_parallel(xp, q, k, v, ip, fp)
check("parallel mLSTM == recurrent definition (float64)", np.allclose(par, mlstm_recurrent(q, k, v, ip, fp), atol=1e-9),
      f"max err {np.abs(par - mlstm_recurrent(q, k, v, ip, fp)).max():.2e}")

ip_big = rng.normal(0, 1.0, (Bq, H, T)) + 6.0
par_big = C.mlstm_parallel(xp, q, k, v, ip_big, fp)
check("agrees with the recurrence with strongly positive exponential input gates",
      np.allclose(par_big, mlstm_recurrent(q, k, v, ip_big, fp), rtol=1e-7, atol=1e-9))

q32, k32, v32 = (a.astype(np.float32) for a in (q, k, v))
ip_ext = np.full((Bq, H, T), 100.0, dtype=np.float32) + rng.normal(0, 1, (Bq, H, T)).astype(np.float32)
with np.errstate(over="raise", invalid="raise", divide="raise"):
    try:
        out32 = C.mlstm_parallel(xp, q32, k32, v32, ip_ext, fp.astype(np.float32))
        finite = bool(np.isfinite(out32).all())
    except FloatingPointError:
        finite = False
ref_ext = mlstm_recurrent(q, k, v, ip_ext.astype(np.float64), fp)
check("float32 with input-gate pre-activation ~100 (exp overflows at 88): stays finite", finite)
check("... and still matches the float64 reference", finite and np.allclose(out32, ref_ext, rtol=2e-3, atol=2e-3),
      f"max err {np.abs(out32 - ref_ext).max():.2e}" if finite else "")

v_pert = v.copy(); v_pert[:, :, 5:] += 7.0
par_p = C.mlstm_parallel(xp, q, k, v_pert, ip, fp)
check("mLSTM is causal: perturbing the future leaves earlier outputs unchanged", np.allclose(par[:, :, :5], par_p[:, :, :5]))

# ============================================================= sLSTM =========
print("\n== sLSTM stabilised scan vs unstabilised reference ==")


def slstm_naive(pre, R):
    Bq, T, _, H, dh = pre.shape
    h = np.zeros((Bq, H, dh)); c = np.zeros_like(h); n = np.zeros_like(h); out = []
    for t in range(T):
        a = pre[:, t] + np.einsum('bhd,ghde->bghe', h, R)
        z = np.tanh(a[:, 0]); i = np.exp(a[:, 1]); f = 1 / (1 + np.exp(-a[:, 2])); o = 1 / (1 + np.exp(-a[:, 3]))
        c = f * c + i * z; n = f * n + i; h = o * c / n; out.append(h)
    return np.stack(out, 1)


Bq, T, H, dh = 2, 11, 2, 6
pre = rng.normal(0, 1.2, (Bq, T, 4, H, dh)); R = rng.normal(0, 0.4, (4, H, dh, dh))
check("stabilised sLSTM == naive recurrence (float64)", np.allclose(C.slstm_scan(xp, pre, R), slstm_naive(pre, R), atol=1e-10))
pre_x = pre.copy(); pre_x[:, :, 1] += 60.0
check("... with very large input-gate pre-activations (exp(60))", np.allclose(C.slstm_scan(xp, pre_x, R), slstm_naive(pre_x, R), rtol=1e-8, atol=1e-10))
pre32 = (pre_x + 40.0 * (np.arange(4) == 1)[None, None, :, None, None]).astype(np.float32)
with np.errstate(over="raise", invalid="raise", divide="raise"):
    try:
        o32 = C.slstm_scan(xp, pre32, R.astype(np.float32)); fin = bool(np.isfinite(o32).all())
    except FloatingPointError:
        fin = False
check("float32 sLSTM with input-gate pre-activation ~100 stays finite", fin)
pre_p = pre.copy(); pre_p[:, 7:] += 5.0
check("sLSTM is causal", np.allclose(C.slstm_scan(xp, pre, R)[:, :7], C.slstm_scan(xp, pre_p, R)[:, :7]))

# ============================================================ similarity =====
print("\n== similarity rows and checkerboard novelty ==")
Bq, T, D, Kr = 2, 14, 5, 4
e = rng.normal(0, 1, (Bq, T, D))
S = C.cosine_ssm(xp, e)
en = e / np.linalg.norm(e, axis=-1, keepdims=True)
check("cosine SSM matches explicit normalised dot products", np.allclose(S, np.einsum('btd,bsd->bts', en, en), atol=1e-7))
check("cosine SSM has unit diagonal and is symmetric", np.allclose(np.diagonal(S, axis1=1, axis2=2), 1, atol=1e-6) and np.allclose(S, S.transpose(0, 2, 1)))
rows = C.row_similarity(xp, S, Kr)
ok_rows = True
for b in range(Bq):
    for t in range(T):
        for j, o in enumerate(range(-Kr, Kr + 1)):
            want = S[b, t, t + o] if 0 <= t + o < T else 0.0
            ok_rows &= abs(rows[b, t, j] - want) < 1e-12
check("row_similarity[b,t,o] == S[b,t,t+o], zero outside the sequence", ok_rows and rows.shape == (Bq, T, 2 * Kr + 1))
check("centre offset of every row is the self-similarity (=1)", np.allclose(rows[:, :, Kr], 1, atol=1e-6))

T2 = 40; bnd = 20
blk = np.zeros((1, T2, 4)); blk[0, :bnd, 0] = 1; blk[0, bnd:, 1] = 1                      # two perfectly homogeneous blocks
S2 = C.cosine_ssm(xp, blk)
bands = [C.make_novelty_bands(T2, L) for L in (2, 4, 8)]
nov = C.checkerboard_novelty(xp, S2, bands)
check("band matrices are normalised: sum |a| = 1 per row (interior)", np.allclose(np.abs(bands[1][20]).sum(), 1.0))
peaks = [int(nov[0, :, i].argmax()) for i in range(3)]
check("novelty on two homogeneous blocks peaks at the block boundary (+-1 frame) for every scale",
      all(p in (bnd - 1, bnd) for p in peaks), f"peaks at {peaks}, boundary at {bnd}")
quiet = [np.abs(nov[0, :bnd - 9, i]).max() for i in range(3)]
check("... and is exactly 0 everywhere inside a homogeneous block, INCLUDING at the sequence edges", max(quiet) < 1e-6, f"{max(quiet):.1e}")
edge_ok = all(np.all(nov[0, :L_, i] == 0) and np.all(nov[0, T2 - L_:, i] == 0) for i, L_ in enumerate((2, 4, 8)))
check("novelty is forced to 0 within L frames of either edge (a truncated kernel would give a spurious 0.25)", edge_ok)
tiny = C.checkerboard_novelty(xp, C.cosine_ssm(xp, rng.normal(0, 1, (1, 5, 3))), [C.make_novelty_bands(5, 8)])
check("a sequence shorter than the kernel gives all-zero novelty without error", np.all(tiny == 0) and tiny.shape == (1, 5, 1))
check("peak value is 0.5 (two perfectly self-similar halves, zero cross-similarity)", np.allclose([nov[0, p, i] for i, p in enumerate(peaks)], 0.5, atol=1e-6))

noisy = blk + rng.normal(0, 0.05, blk.shape)
nov_n = C.checkerboard_novelty(xp, C.cosine_ssm(xp, noisy), bands)
check("boundary is still the strongest novelty response under noise", all(int(nov_n[0, :, i].argmax()) in (bnd - 1, bnd) for i in range(3)))
check("similarity features are invariant to a global rescaling of the embeddings",
      np.allclose(C.cosine_ssm(xp, 7.3 * e), S, atol=1e-6))

# smooth transition (handshape drifting inside ONE sign) should score far lower than an abrupt change
ramp = np.zeros((1, T2, 2)); a_ = np.clip((np.arange(T2) - 10) / 20.0, 0, 1); ramp[0, :, 0] = 1 - a_; ramp[0, :, 1] = a_
nov_ramp = C.checkerboard_novelty(xp, C.cosine_ssm(xp, ramp), [bands[1]])
nov_step = C.checkerboard_novelty(xp, S2, [bands[1]])
check("a gradual feature drift gives much weaker novelty than an abrupt change (sign-internal change vs boundary)",
      nov_ramp[0, :, 0].max() < 0.5 * nov_step[0, :, 0].max(), f"{nov_ramp[0,:,0].max():.3f} vs {nov_step[0,:,0].max():.3f}")


# ======================================================= static audit ========
print("\n== static audit of the torch modules (cannot be executed without torch) ==")
import ast
import inspect

TORCH_FILES = ["models_novelty.py", "train_novelty.py"]


def audit_class_attrs(path):
    tree = ast.parse(open(os.path.join(_HERE, path)).read())
    problems = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is None:
            continue
        defined = {n.attr for n in ast.walk(init) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                   and isinstance(n.value, ast.Name) and n.value.id == "self"}
        defined |= {n.args[0].value for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr in ("register_buffer", "register_parameter") and n.args
                    and isinstance(n.args[0], ast.Constant)}
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        for m in cls.body:
            if isinstance(m, ast.FunctionDef) and m.name != "__init__":
                for n in ast.walk(m):
                    if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self" \
                            and isinstance(n.ctx, ast.Load) and n.attr not in defined | methods | {"training"}:
                        problems.append(f"{path}:{cls.name}.{m.name} uses self.{n.attr} never set in __init__")
    return problems


def audit_core_calls(path):
    import seg_cores
    tree = ast.parse(open(os.path.join(_HERE, path)).read())
    problems, n_calls = [], 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) \
                and n.func.value.id == "C":
            fn = getattr(seg_cores, n.func.attr, None)
            n_calls += 1
            if fn is None:
                problems.append(f"{path}: C.{n.func.attr} does not exist in seg_cores"); continue
            sig = inspect.signature(fn)
            req = sum(1 for p in sig.parameters.values() if p.default is p.empty)
            tot = len(sig.parameters)
            npos = len(n.args) + len(n.keywords)
            if not (req <= npos <= tot):
                problems.append(f"{path}: C.{n.func.attr} called with {npos} args, expects {req}..{tot}")
    return problems, n_calls


attr_problems = [p for f in TORCH_FILES if os.path.exists(os.path.join(_HERE, f)) for p in audit_class_attrs(f)]
check("every self.<attr> used outside __init__ is defined in __init__ (models + training)", not attr_problems, "; ".join(attr_problems))
call_problems, n_core_calls = [], 0
for f in TORCH_FILES:
    if os.path.exists(os.path.join(_HERE, f)):
        pr, n = audit_core_calls(f); call_problems += pr; n_core_calls += n
check(f"all {n_core_calls} calls into seg_cores match the real function signatures", not call_problems and n_core_calls >= 6, "; ".join(call_problems))


import symtable
import builtins


def undefined_names(path):
    """Poor man's pyflakes: names that are referenced somewhere but never bound
    (module level, enclosing function, or builtin). Catches typo'd variables in
    code paths the numpy tests cannot execute."""
    src = open(os.path.join(_HERE, path)).read()
    top = symtable.symtable(src, path, "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    bad = set()

    def walk(tab):
        for sym in tab.get_symbols():
            nm = sym.get_name()
            if sym.is_referenced() and sym.is_global() and nm not in defined and not hasattr(builtins, nm) and nm not in ("__file__", "__name__"):
                bad.add(f"{path}:{tab.get_name()} -> {nm}")
        for ch in tab.get_children():
            walk(ch)
    walk(top)
    return sorted(bad)


undef = [u for f in ("seg_ops.py", "seg_cores.py", "models_novelty.py", "train_novelty.py", "queue_train_novelty.py")
         if os.path.exists(os.path.join(_HERE, f)) for u in undefined_names(f)]
check("no undefined names in any module (symtable scan: catches typo'd variables in un-runnable code paths)", not undef, "; ".join(undef))

# ============================== numpy re-enactment of the wrapper logic =======
print("\n== numpy re-enactment of the nn.Module wrappers (shape flow, head splitting, direction handling) ==")


def lin(x, W, b=None):
    y = x @ W.T
    return y if b is None else y + b


def layernorm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)


class SimMLSTM:
    """mirrors MLSTMCell.forward line by line, with numpy weights"""
    def __init__(self, D, H, r):
        self.D, self.H, self.dh = D, H, D // H
        mk = lambda o, i: r.normal(0, 0.3, (o, i))
        self.q, self.k, self.v, self.o, self.out = mk(D, D), mk(D, D), mk(D, D), mk(D, D), mk(D, D)
        self.gi, self.gf = mk(H, D), mk(H, D)
        self.bf = np.linspace(3.0, 6.0, H)

    def __call__(self, x):
        B, T, D = x.shape
        split = lambda t: t.reshape(B, T, self.H, self.dh).transpose(0, 2, 1, 3)
        q, k, v = split(lin(x, self.q)), split(lin(x, self.k)), split(lin(x, self.v))
        i_pre = lin(x, self.gi).transpose(0, 2, 1)
        f_pre = (lin(x, self.gf) + self.bf).transpose(0, 2, 1)
        h = C.mlstm_parallel(xp, q, k, v, i_pre, f_pre)
        h = layernorm(h.transpose(0, 2, 1, 3)).reshape(B, T, D)
        return lin((1 / (1 + np.exp(-lin(x, self.o)))) * h, self.out)


class SimSLSTM:
    def __init__(self, D, H, r):
        self.D, self.H, self.dh = D, H, D // H
        self.inp = r.normal(0, 0.3, (4 * D, D)); self.b = np.zeros(4 * D); self.b[2 * D:3 * D] = 3.0
        self.R = r.normal(0, 0.1 / np.sqrt(self.dh), (4, H, self.dh, self.dh)); self.out = r.normal(0, 0.3, (D, D))

    def __call__(self, x):
        B, T, D = x.shape
        pre = lin(x, self.inp, self.b).reshape(B, T, 4, self.H, self.dh)
        h = C.slstm_scan(xp, pre, self.R)
        return lin(layernorm(h).reshape(B, T, D), self.out)


r2 = np.random.default_rng(5)
Bq, T, D, H = 2, 12, 16, 4
x = r2.normal(0, 1, (Bq, T, D))
for name, cell in (("mLSTM", SimMLSTM(D, H, r2)), ("sLSTM", SimSLSTM(D, H, r2))):
    y = cell(x)
    check(f"{name} wrapper: (B,T,D) in -> (B,T,D) out, finite", y.shape == x.shape and np.isfinite(y).all())
    xf = x.copy(); xf[:, 8:] += 3.0
    check(f"{name} wrapper is causal end-to-end (head split / norm / gating introduce no future leakage)",
          np.allclose(cell(xf)[:, :8], y[:, :8]))
    h_fwd = cell(x); h_bwd = np.flip(cell(np.flip(x, 1)), 1)
    xp_past = x.copy(); xp_past[:, :4] += 3.0
    bwd_past = np.flip(cell(np.flip(xp_past, 1)), 1)
    check(f"{name}: the reversed-sequence branch depends only on frames >= t (flip/unflip is correct)",
          np.allclose(bwd_past[:, 5:], h_bwd[:, 5:]) and not np.allclose(bwd_past[:, :4], h_bwd[:, :4]))

for streams_cfg in ((1, False), (2, False), (3, True), (2, True)):
    n_streams, sim = streams_cfg
    Kc, scales, d_sim = 16, (2, 4, 8, 16), 64
    feat = n_streams * ((2 * Kc + 1) + len(scales))
    Tw = 64
    embs = [r2.normal(0, 1, (2, Tw, 8)) for _ in range(n_streams)]
    parts = []
    for e in embs:
        S_ = C.cosine_ssm(xp, e)
        parts += [C.row_similarity(xp, S_, Kc), C.checkerboard_novelty(xp, S_, [C.make_novelty_bands(Tw, L) for L in scales])]
    cat = np.concatenate(parts, -1)
    check(f"SimilarityNovelty input width == n_streams*((2K+1)+n_scales) for {n_streams} stream(s)", cat.shape[-1] == feat and np.isfinite(cat).all(), f"{cat.shape[-1]}")

# model width bookkeeping for every config the queue uses
def fused_width(d_model, hamer, dino, sim, hp=128, dp=128, ds=64):
    return d_model + (hp if hamer else 0) + (dp if dino else 0) + (ds if sim else 0)
check("mixer width bookkeeping: p + hamer + dino + sim", fused_width(256, True, False, True) == 256 + 128 + 64
      and fused_width(256, False, False, False) == 256)



# ================================ config <-> trainer <-> model consistency =====
print("\n== queue defaults vs. keys the trainer reads vs. model constructor ==")
import re
import importlib.util
spec = importlib.util.spec_from_file_location("queue_train_novelty", os.path.join(_HERE, "queue_train_novelty.py"))
Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)
train_src = open(os.path.join(_HERE, "train_novelty.py")).read()
read_keys = set(re.findall(r'cfg\.get\("([A-Za-z_0-9]+)"', train_src)) | set(re.findall(r'cfg\["([A-Za-z_0-9]+)"\]', train_src))
model_cfg_keys = set(re.findall(r'"([A-Za-z_0-9]+)"', re.search(r"MODEL_CONFIG_KEYS = \[(.*?)\]", train_src, re.S).group(1)))
allowed = set(Q.NOVELTY_DEFAULTS) | {"seed", "prefix", "description"}
check("every config key the trainer reads exists in the queue defaults (no key the queue can't set)", read_keys <= allowed, str(sorted(read_keys - allowed)))
dead = set(Q.NOVELTY_DEFAULTS) - read_keys - model_cfg_keys
check("every queue default is actually consumed by the trainer or passed to the model (no dead settings)", not dead, str(sorted(dead)))
tree = ast.parse(open(os.path.join(_HERE, "models_novelty.py")).read())
ctor = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "STGCN_Novelty")
init = next(n for n in ctor.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
ctor_args = {a.arg for a in init.args.args} - {"self"}
check("every MODEL_CONFIG_KEY is a real STGCN_Novelty constructor argument", model_cfg_keys <= ctor_args, str(sorted(model_cfg_keys - ctor_args)))
for arm in Q.EXPERIMENTS_TO_RUN:
    unknown = set(arm) - allowed
    check(f"ablation arm has only known keys: {arm['description'][:42]}", not unknown, str(unknown))


if __name__ == "__main__":
    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    sys.exit(0 if all(_results) else 1)