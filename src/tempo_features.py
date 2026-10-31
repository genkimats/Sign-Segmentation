"""
Signer-tempo features (Idea A): per-frame channels that describe each sign-to-sign transition
RELATIVE TO THE SIGNER'S OWN TIMING, computed over the whole video from a sign segmentation.

Pure numpy, no torch. Used by data_processing/extract_tempo_features.py (which writes the
files the cue stream loads) and by its tests.

Why: the timing-only oracle of phrase_oracle_analysis.py (AP .671 with gold signs) beats the
trained model scored on the same transitions (.594). Its features are normalized per video
or relative to the neighbourhood ("is this pause long FOR THIS SIGNER / the longest of the
last +-5 signs"); a BiLSTM on a 512-frame window does not learn such rank statistics.

Input: a sign segmentation S = [(s_i, e_i)], i = 0..N-1, inclusive ends, sorted, disjoint
(frames at 50 fps), plus the frame count T of the phrase labels.

Channels (17, in this file order -- TEMPO_CHANNELS):

  transition (9), written on frames [e_{i-1}+1, e_i] = the gap before sign i plus sign i, so
  every phrase B frame (a sign onset in the v2 labels) carries the features of its own
  transition; names as in phrase_oracle_analysis.TIMING where they exist:
    log_gap_prev          log1p(g_i)
    log_gap_next          log1p(g_{i+1}); 0 for the last sign
    gap_prev_rel          g_i / max(med_gap, 1)                                   [log1p, *]
    gap_prev_rank         mid-rank empirical CDF of g_i among this video's gaps (0..1)
    gap_prev_local_max    g_i / max(max g_j for j in [i-5, i+5], j != i; 1)       [log1p, *]
    dur_prev_rel          dur_{i-1} / med_dur; 0 for the first sign                [log1p, *]
    dur_cur_rel           dur_i / med_dur                                          [log1p, *]
    signs_since_long_gap  min(signs since the last gap >= p90_gap, 30) / 30
    local_rate_rel        (onsets within +-2 s of s_i / 4 s) / video rate
  positional (4):
    onset_bump            exp(-(t - nearest onset)^2 / (2 * 5^2))
    in_gap                1 between two signs (e_{i-1} < t < s_i), else 0
    sign_conf             gold: 1; predicted: mean P_sign(B or I) over sign i (on the
                          transition span, like the transition channels); 0 after the last sign
    outside_signing       1 before the first and after the last sign
  video-level (4), constant over the video, z-scored with the TRAIN-video mean / SD (SD
  floored at 0.01, z clipped to +-8):
    log_med_gap, log_p90_gap, log_med_dur (log1p of frames), log_rate (log signs / s)

Gaps: g_i = s_i - e_{i-1} - 1 (>= 0 for a BIO segmentation). The per-video statistics
(med_gap, p90_gap) use the inter-sign gaps i >= 1 only, exactly like the oracle. The lead-in
before the first sign is not a pause between signs: g_0 = min(s_0, p90_gap) ("clipped"), so a
long silence at the start of the recording looks like an ordinary long pause, and the
neighbourhood maximum of gap_prev_local_max skips j = 0 (as in the oracle).

[*] Deliberate, documented deviation from the plan's raw formulas: the four unbounded ratio
channels are stored as log1p(ratio) (transform="log1p", recorded in _norm.json). The cue
stream applies no normalization, and a single 60x pause would otherwise dominate the first
Linear layer of the branch. log1p is monotone, so rank information (what the oracle uses) is
unchanged. transform="none" stores the raw ratios.

Frames after the last sign: transition channels 0, outside_signing 1, video-level unchanged.
A video without any sign: transition and positional channels 0, outside_signing 1,
video-level 0 (= the train mean).
"""
import numpy as np

TRANSITION_CHANNELS = ["log_gap_prev", "log_gap_next", "gap_prev_rel", "gap_prev_rank", "gap_prev_local_max",
                       "dur_prev_rel", "dur_cur_rel", "signs_since_long_gap", "local_rate_rel"]
POSITIONAL_CHANNELS = ["onset_bump", "in_gap", "sign_conf", "outside_signing"]
VIDEO_CHANNELS = ["log_med_gap", "log_p90_gap", "log_med_dur", "log_rate"]
TEMPO_CHANNELS = TRANSITION_CHANNELS + POSITIONAL_CHANNELS + VIDEO_CHANNELS          # 17, file order
LOG1P_CHANNELS = ["gap_prev_rel", "gap_prev_local_max", "dur_prev_rel", "dur_cur_rel"]

# channel subsets for the TEMPO_STUDY arms (queue_train_phrase.py)
TEMPO_SUBSETS = {
    "full": list(TEMPO_CHANNELS),
    "pos": list(POSITIONAL_CHANNELS),
    "video": list(VIDEO_CHANNELS),
    "trans": TRANSITION_CHANNELS + POSITIONAL_CHANNELS,       # transition + positional, no video-level
}

LOCAL_MAX_RADIUS = 5        # +-signs for gap_prev_local_max (oracle: 5)
SINCE_LONG_CAP = 30         # signs_since_long_gap cap (oracle: 30)
RATE_WINDOW_S = 2.0         # +-seconds for local_rate_rel (oracle: 2.0)
BUMP_SIGMA = 5.0            # frames
DEFAULT_P90_GAP = 25.0      # oracle fallback when a video has no positive gap
DEFAULT_MED_GAP = 1.0
VIDEO_SD_FLOOR = 1e-2       # a near-constant video-level statistic must not blow up the z-score
VIDEO_Z_CLIP = 8.0          # same clip as extract_cue_features.py


def clean_segments(signs, T):
    """(N, 2) inclusive segments -> sorted, cropped to [0, T), empty ones dropped."""
    s = np.asarray(signs, dtype=np.int64).reshape(-1, 2)
    if len(s) == 0:
        return s
    s = s[np.argsort(s[:, 0], kind="stable")]
    s = s[s[:, 0] < T]
    s[:, 1] = np.minimum(s[:, 1], T - 1)
    s = s[s[:, 1] >= s[:, 0]]
    return s


def video_stats(signs, fps=50.0):
    """Per-video timing statistics of a (non-empty, clean) segmentation (no phrase labels)."""
    signs = np.asarray(signs, dtype=np.int64).reshape(-1, 2)
    n = len(signs)
    dur = (signs[:, 1] - signs[:, 0] + 1).astype(np.float64)
    inter = (signs[1:, 0] - signs[:-1, 1] - 1).astype(np.float64)           # gaps before signs 1..N-1
    inter = np.maximum(inter, 0.0)
    pos = inter[inter > 0]
    med_gap = float(np.median(pos)) if len(pos) else DEFAULT_MED_GAP
    p90_gap = float(np.percentile(pos, 90)) if len(pos) else DEFAULT_P90_GAP
    med_dur = float(np.median(dur))
    span_s = (signs[-1, 1] - signs[0, 0] + 1) / float(fps)
    rate = n / max(span_s, 1.0 / fps)
    gaps = np.r_[min(float(signs[0, 0]), p90_gap), inter]                     # g_0 clipped lead-in
    return {"n_signs": n, "gaps": gaps, "dur": dur, "med_gap": med_gap, "p90_gap": p90_gap,
            "med_dur": med_dur, "rate": float(rate)}


def video_level_raw(stats):
    """The 4 video-level values before z-scoring (VIDEO_CHANNELS order)."""
    return np.array([np.log1p(stats["med_gap"]), np.log1p(stats["p90_gap"]), np.log1p(stats["med_dur"]),
                     np.log(max(stats["rate"], 1e-6))], dtype=np.float64)


def midrank_ecdf(values):
    """Mid-rank empirical CDF of every value among `values`: (#< + 0.5 #=) / n, in (0, 1)."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    if n == 0:
        return v
    srt = np.sort(v)
    lo = np.searchsorted(srt, v, side="left")
    hi = np.searchsorted(srt, v, side="right")
    return (lo + 0.5 * (hi - lo)) / n


def transition_table(signs, fps=50.0):
    """
    Per-sign transition features, (N, 9) in TRANSITION_CHANNELS order, RAW ratios (before the
    log1p transform), plus the stats dict. Same definitions as phrase_oracle_analysis.py for the
    shared names (checked in the tests).
    """
    st = video_stats(signs, fps)
    n, g, dur = st["n_signs"], st["gaps"], st["dur"]
    starts = signs[:, 0].astype(np.float64)
    rank = midrank_ecdf(g)
    out = np.zeros((n, len(TRANSITION_CHANNELS)), dtype=np.float64)
    since = 0
    win = RATE_WINDOW_S * fps
    for i in range(n):
        gi = max(g[i], 0.0)
        g_next = max(g[i + 1], 0.0) if i + 1 < n else None
        lo, hi = max(1, i - LOCAL_MAX_RADIUS), min(n - 1, i + LOCAL_MAX_RADIUS)
        nb = [g[j] for j in range(lo, hi + 1) if j != i]
        nb_max = max(nb) if nb else 0.0
        # the video start counts as a long pause (the oracle restarts its counter there too)
        since = 0 if (i == 0 or g[i] >= st["p90_gap"]) else since + 1
        local = np.sum(np.abs(starts - starts[i]) <= win) / (2 * RATE_WINDOW_S)
        out[i] = [np.log1p(gi),
                  np.log1p(g_next) if g_next is not None else 0.0,
                  gi / max(st["med_gap"], 1.0),
                  rank[i],
                  gi / max(nb_max, 1.0),
                  dur[i - 1] / st["med_dur"] if i > 0 else 0.0,
                  dur[i] / st["med_dur"],
                  min(since, SINCE_LONG_CAP) / SINCE_LONG_CAP,
                  local / st["rate"]]
    return out, st


def tempo_features(signs, T, fps=50.0, sign_conf=None, video_norm=None, transform="log1p"):
    """
    signs: (N, 2) inclusive segments; T: number of frames (= phrase-label length).
    sign_conf: (N,) per-sign confidence or None (gold: all 1).
    video_norm: {"mean": [4], "sd": [4]} train statistics of video_level_raw, or None (-> 0).
    Returns (T, 17) float32 in TEMPO_CHANNELS order.
    """
    if transform not in ("log1p", "none"):
        raise ValueError(f"transform must be 'log1p' or 'none', got '{transform}'")
    F = np.zeros((T, len(TEMPO_CHANNELS)), dtype=np.float32)
    ci = {c: k for k, c in enumerate(TEMPO_CHANNELS)}
    signs = clean_segments(signs, T)
    n = len(signs)
    if n == 0:
        F[:, ci["outside_signing"]] = 1.0
        return F

    table, st = transition_table(signs, fps)
    if transform == "log1p":
        for c in LOG1P_CHANNELS:
            j = TRANSITION_CHANNELS.index(c)
            table[:, j] = np.log1p(table[:, j])
    conf = np.ones(n) if sign_conf is None else np.asarray(sign_conf, dtype=np.float64).reshape(-1)
    if len(conf) != n:
        raise ValueError(f"sign_conf has {len(conf)} entries for {n} signs")

    # transition channels + sign_conf on [e_{i-1}+1, e_i]
    span_start = np.r_[0, signs[:-1, 1] + 1]
    span_end = signs[:, 1]
    idx = np.full(T, -1, dtype=np.int64)
    for i in range(n):
        idx[span_start[i]:span_end[i] + 1] = i
    on = idx >= 0
    tcols = [ci[c] for c in TRANSITION_CHANNELS]
    F[np.ix_(on, tcols)] = table[idx[on]].astype(np.float32)
    F[on, ci["sign_conf"]] = conf[idx[on]].astype(np.float32)

    # positional
    t = np.arange(T)
    onsets = signs[:, 0]
    k = np.clip(np.searchsorted(onsets, t), 0, n - 1)
    d = np.minimum(np.abs(t - onsets[k]), np.abs(t - onsets[np.maximum(k - 1, 0)]))
    F[:, ci["onset_bump"]] = np.exp(-(d.astype(np.float64) ** 2) / (2 * BUMP_SIGMA ** 2)).astype(np.float32)
    for i in range(1, n):
        a, b = signs[i - 1, 1] + 1, signs[i, 0] - 1
        if b >= a:
            F[a:b + 1, ci["in_gap"]] = 1.0
    F[:signs[0, 0], ci["outside_signing"]] = 1.0
    F[signs[-1, 1] + 1:, ci["outside_signing"]] = 1.0

    # video-level, constant
    if video_norm is not None:
        raw = video_level_raw(st)
        mean = np.asarray(video_norm["mean"], dtype=np.float64)
        sd = np.maximum(np.asarray(video_norm["sd"], dtype=np.float64), VIDEO_SD_FLOOR)
        z = np.clip((raw - mean) / sd, -VIDEO_Z_CLIP, VIDEO_Z_CLIP)
        for j, c in enumerate(VIDEO_CHANNELS):
            F[:, ci[c]] = np.float32(z[j])
    return F
