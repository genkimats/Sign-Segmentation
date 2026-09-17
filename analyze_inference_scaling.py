"""
Analyzes results from evaluate_inference_scaling.py -- both numerically
(fitted scaling exponents, crossover points, OOM/error ceilings, F1 deltas)
and visually (time/memory vs length plots, success/failure maps, F1 bars).

The core analytical move: fitting time ≈ a × length^b via log-log linear
regression turns "eyeballing whether a curve looks linear or quadratic" into
a specific number you can state directly (e.g. "Transformer's fitted
exponent was 1.8, Mamba's was 1.05") -- the difference between an impression
and a claim you can defend.

Reads:
    inference_scaling/real_video_results.csv
    inference_scaling/synthetic_sweep_results.csv
    (either can be missing -- whatever's available gets analyzed)

Writes:
    inference_scaling/plots/*.png
    (numerical summaries print to console)

Requires pandas (pip install pandas if not already present -- matplotlib is
already a dependency elsewhere in this codebase).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "inference_scaling"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_COLORS = {
    "stgcn_mamba": "#1f77b4",
    "stgcn_bimamba": "#ff7f0e",
    "stgcn_bilstm": "#2ca02c",
    "stgcn_transformer": "#d62728",
}
MODEL_ORDER = list(MODEL_COLORS.keys())


def load_results():
    real_path = os.path.join(RESULTS_DIR, "real_video_results.csv")
    synth_path = os.path.join(RESULTS_DIR, "synthetic_sweep_results.csv")
    real_df = pd.read_csv(real_path) if os.path.exists(real_path) else None
    synth_df = pd.read_csv(synth_path) if os.path.exists(synth_path) else None
    return real_df, synth_df


# ==============================================================================
# NUMERICAL ANALYSIS
# ==============================================================================

def fit_scaling_exponent(lengths, values):
    """
    Fits value ≈ a × length^b via log-log linear regression (polyfit on
    log-log data). Returns (a, b, r_squared). b~1 indicates linear (O(n))
    scaling -- expected for Mamba/BiLSTM; b~2 indicates quadratic (O(n^2)) --
    expected for attention. r_squared close to 1 means the power-law fit
    actually describes the data well (worth checking before trusting b).
    """
    lengths = np.asarray(lengths, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = (lengths > 0) & (values > 0) & np.isfinite(values)
    if mask.sum() < 2:
        return None, None, None

    log_l, log_v = np.log(lengths[mask]), np.log(values[mask])
    b, log_a = np.polyfit(log_l, log_v, 1)
    a = np.exp(log_a)

    pred = log_a + b * log_l
    ss_res = np.sum((log_v - pred) ** 2)
    ss_tot = np.sum((log_v - log_v.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r_squared


def print_scaling_exponents(df, mode, value_col, label):
    print(f"\n--- Fitted {label} scaling exponent ({mode} mode): value ~ a * length^b ---")
    print(f"{'Model':<20} {'exponent b':>12} {'R^2':>8}   interpretation")
    for model in MODEL_ORDER:
        sub = df[(df["model"] == model) & (df["mode"] == mode) & (df["status"] == "ok")]
        if sub.empty:
            print(f"{model:<20} {'--':>12} {'--':>8}   no successful runs in this mode")
            continue
        a, b, r2 = fit_scaling_exponent(sub["length_frames"], sub[value_col])
        if b is None:
            print(f"{model:<20} {'--':>12} {'--':>8}   not enough distinct data points to fit")
            continue
        if b < 1.3:
            interp = "~linear (O(n)) -- matches Mamba/RNN theory"
        elif b > 1.7:
            interp = "~quadratic (O(n^2)) -- matches attention theory"
        else:
            interp = "ambiguous -- between linear and quadratic"
        print(f"{model:<20} {b:>12.3f} {r2:>8.3f}   {interp}")


def find_crossover(df, mode, model_a, model_b, value_col):
    """
    Given two fitted power-law curves a1*L^b1 and a2*L^b2, solves for the
    length L where they're equal: L = (a1/a2)^(1/(b2-b1)). Returns None if
    either fit fails, the exponents are too close to distinguish (near-
    parallel curves produce numerically unstable, meaningless extrapolations
    -- e.g. two models that are BOTH genuinely linear will have b1~b2, and
    dividing by that near-zero difference blows up into billions of frames,
    which isn't a real crossover, just noise in the fit), or the solved
    length falls far outside the range of lengths actually tested (a
    power-law fit isn't trustworthy extrapolated orders of magnitude beyond
    its data).
    """
    sub_a = df[(df["model"] == model_a) & (df["mode"] == mode) & (df["status"] == "ok")]
    sub_b = df[(df["model"] == model_b) & (df["mode"] == mode) & (df["status"] == "ok")]
    a1, b1, _ = fit_scaling_exponent(sub_a["length_frames"], sub_a[value_col])
    a2, b2, _ = fit_scaling_exponent(sub_b["length_frames"], sub_b[value_col])
    if a1 is None or a2 is None:
        return None

    # Exponents too close to reliably distinguish -- these curves are both
    # effectively the same complexity class in this data; a "crossover" would
    # be fit noise, not signal.
    if abs(b1 - b2) < 0.15:
        return None

    try:
        crossover_length = (a1 / a2) ** (1.0 / (b2 - b1))
    except (ZeroDivisionError, ValueError):
        return None
    if crossover_length <= 0 or not np.isfinite(crossover_length):
        return None

    # Only trust the crossover if it falls within a reasonable extrapolation
    # of the ACTUAL tested range -- a couple orders of magnitude beyond the
    # data is already a stretch for a power-law fit; billions of frames
    # beyond it is not a finding, it's extrapolation noise.
    all_lengths = pd.concat([sub_a["length_frames"], sub_b["length_frames"]])
    min_len, max_len = all_lengths.min(), all_lengths.max()
    if not (min_len / 100 <= crossover_length <= max_len * 100):
        return None

    return crossover_length


def print_crossover_analysis(df, mode, value_col, label):
    print(f"\n--- Estimated crossover points ({mode} mode, {label}) ---")
    print("(length at which the two fitted curves cross -- i.e. which model is faster flips;")
    print(" omitted when both models fit the same complexity class, or no reliable crossover")
    print(" exists within a reasonable extrapolation of the tested length range)")
    for i, model_a in enumerate(MODEL_ORDER):
        for model_b in MODEL_ORDER[i + 1:]:
            crossover = find_crossover(df, mode, model_a, model_b, value_col)
            if crossover is None:
                print(f"  {model_a} vs {model_b}: no reliable crossover (same complexity class, "
                      f"or one model wins throughout the tested range)")
            else:
                print(f"  {model_a} vs {model_b}: crossover at ~{crossover:.0f} frames")


def print_ceiling_analysis(df):
    print("\n--- Maximum length successfully processed before failure (OOM or other error) ---")
    print(f"{'Model':<20} {'Mode':<12} {'Max OK length':>15} {'First failure len':>20} {'Failure type'}")
    for model in MODEL_ORDER:
        for mode in ["chunked", "streaming"]:
            sub = df[(df["model"] == model) & (df["mode"] == mode)]
            if sub.empty:
                continue
            ok_sub = sub[sub["status"] == "ok"]
            fail_sub = sub[sub["status"] != "ok"]
            max_ok = ok_sub["length_frames"].max() if not ok_sub.empty else "--"
            first_fail_len = fail_sub["length_frames"].min() if not fail_sub.empty else "--"
            fail_type = (fail_sub.sort_values("length_frames")["status"].iloc[0][:40]
                         if not fail_sub.empty else "none")
            print(f"{model:<20} {mode:<12} {str(max_ok):>15} {str(first_fail_len):>20} {fail_type}")


def print_f1_delta_analysis(real_df):
    print("\n--- F1: streaming vs chunked (per model, averaged across real videos) ---")
    print(f"{'Model':<20} {'F1 chunked':>12} {'F1 streaming':>14} {'delta (stream-chunk)':>22}")
    for model in MODEL_ORDER:
        chunked = real_df[(real_df["model"] == model) & (real_df["mode"] == "chunked") & (real_df["status"] == "ok")]
        streaming = real_df[(real_df["model"] == model) & (real_df["mode"] == "streaming") & (real_df["status"] == "ok")]
        if chunked.empty or streaming.empty:
            print(f"{model:<20} {'--':>12} {'--':>14} {'--':>22}")
            continue
        f1_c, f1_s = chunked["f1"].mean(), streaming["f1"].mean()
        delta = f1_s - f1_c
        flag = "  notable drop" if delta < -0.02 else ("  improved" if delta > 0.02 else "")
        print(f"{model:<20} {f1_c:>12.4f} {f1_s:>14.4f} {delta:>+22.4f}{flag}")


# ==============================================================================
# VISUALIZATIONS
# ==============================================================================

def plot_scaling_curves(real_df, synth_df, value_col, ylabel, filename, log_y=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, mode in zip(axes, ["chunked", "streaming"]):
        for model in MODEL_ORDER:
            color = MODEL_COLORS[model]

            if real_df is not None:
                sub = real_df[(real_df["model"] == model) & (real_df["mode"] == mode) & (real_df["status"] == "ok")]
                if not sub.empty:
                    ax.scatter(sub["length_frames"], sub[value_col], color=color, alpha=0.5, s=25,
                               label=f"{model} (real)", marker="o")

            if synth_df is not None:
                sub = synth_df[(synth_df["model"] == model) & (synth_df["mode"] == mode) & (synth_df["status"] == "ok")]
                if not sub.empty:
                    sub = sub.sort_values("length_frames")
                    ax.plot(sub["length_frames"], sub[value_col], color=color, linewidth=2,
                           linestyle="--", marker="s", label=f"{model} (synthetic)")

        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Sequence length (frames)")
        ax.set_title(f"{mode.capitalize()} inference")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=8)
    fig.suptitle(f"{ylabel} vs sequence length", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_f1_comparison(real_df, filename="f1_chunked_vs_streaming.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    chunked_means, streaming_means = [], []
    for model in MODEL_ORDER:
        c = real_df[(real_df["model"] == model) & (real_df["mode"] == "chunked") & (real_df["status"] == "ok")]["f1"]
        s = real_df[(real_df["model"] == model) & (real_df["mode"] == "streaming") & (real_df["status"] == "ok")]["f1"]
        chunked_means.append(c.mean() if not c.empty else 0)
        streaming_means.append(s.mean() if not s.empty else 0)

    ax.bar(x - width / 2, chunked_means, width, label="Chunked", color="#888888")
    ax.bar(x + width / 2, streaming_means, width, label="Streaming", color="#e07b39")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Mean F1 (real videos)")
    ax.set_title("F1: chunked vs streaming inference")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_failure_map(real_df, synth_df, filename="failure_map.png"):
    """Visualizes WHERE (model, length, mode) each run succeeded vs failed --
    directly shows OOM/error ceilings per architecture at a glance."""
    combined = []
    if real_df is not None:
        combined.append(real_df[["model", "mode", "length_frames", "status"]])
    if synth_df is not None:
        combined.append(synth_df[["model", "mode", "length_frames", "status"]])
    if not combined:
        return
    df = pd.concat(combined, ignore_index=True)
    df["ok"] = df["status"] == "ok"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mode in zip(axes, ["chunked", "streaming"]):
        sub = df[df["mode"] == mode]
        for i, model in enumerate(MODEL_ORDER):
            model_sub = sub[sub["model"] == model]
            ok_pts = model_sub[model_sub["ok"]]
            fail_pts = model_sub[~model_sub["ok"]]
            ax.scatter(ok_pts["length_frames"], [i] * len(ok_pts), color="green", marker="o", s=40)
            ax.scatter(fail_pts["length_frames"], [i] * len(fail_pts), color="red", marker="x", s=60)
        ax.set_xscale("log")
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(MODEL_ORDER)
        ax.set_xlabel("Sequence length (frames)")
        ax.set_title(f"{mode.capitalize()}: green=ok, red=failed")
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Success/failure map across lengths", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    real_df, synth_df = load_results()
    if real_df is None and synth_df is None:
        print(f"No result CSVs found in {RESULTS_DIR} -- run evaluate_inference_scaling.py first.")
        return

    print("=" * 70)
    print("NUMERICAL ANALYSIS")
    print("=" * 70)

    if synth_df is not None:
        for mode in ["chunked", "streaming"]:
            print_scaling_exponents(synth_df, mode, "time_seconds", "TIME")
        for mode in ["chunked", "streaming"]:
            print_scaling_exponents(synth_df, mode, "peak_mem_gb", "MEMORY")
        for mode in ["chunked", "streaming"]:
            print_crossover_analysis(synth_df, mode, "time_seconds", "time")
        print_ceiling_analysis(synth_df)
    else:
        print("\n(no synthetic_sweep_results.csv found -- skipping scaling-exponent and crossover analysis)")

    if real_df is not None:
        print_f1_delta_analysis(real_df)
    else:
        print("\n(no real_video_results.csv found -- skipping F1 delta analysis)")

    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    if synth_df is not None or real_df is not None:
        plot_scaling_curves(real_df, synth_df, "time_seconds", "Time (seconds)", "time_vs_length.png")
        plot_scaling_curves(real_df, synth_df, "peak_mem_gb", "Peak GPU memory (GB)", "memory_vs_length.png", log_y=False)
        plot_failure_map(real_df, synth_df)

    if real_df is not None:
        plot_f1_comparison(real_df)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()