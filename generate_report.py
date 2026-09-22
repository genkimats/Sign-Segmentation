"""
generate_report.py

Scans experiments/<basename>-<prefix>/ dirs, groups runs that share identical
hyperparameters (ignoring the "prefix" field -- i.e. multi-seed reruns of the
same experiment), and prints a Markdown-style report to the terminal:

    # <basename>
    ## <description>
    F1 Score: ...
    Best epoch number: ...
    Total train time: ...
    Time per epoch: ...
    GPU Memory: ...
    GPU Utilization percentage: ...

Usage:
    python generate_report.py                     # interactive: pick model, then prefixes
    python generate_report.py stgcn_bimamba-37 stgcn_bimamba-38 stgcn_bimamba-39
    python generate_report.py --exp-dir experiments --basename stgcn_bimamba --prefixes "3-6, 8, 10, 14-19, 21"
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict


def parse_prefix_ranges(spec):
    """
    Parses a prefix spec like "3-6, 8, 10, 14-19, 21" into a sorted set of ints.
    """
    prefixes = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", 1)
            start, end = int(start_str.strip()), int(end_str.strip())
            if start > end:
                start, end = end, start
            prefixes.update(range(start, end + 1))
        else:
            prefixes.add(int(chunk))
    return prefixes


def discover_basenames(exp_dir):
    """Returns a sorted list of unique basenames found in experiments/<basename>-<prefix>/ dirs."""
    basenames = set()
    for entry in os.listdir(exp_dir):
        if "-" not in entry:
            continue
        basename, _, prefix_part = entry.rpartition("-")
        if basename and prefix_part.isdigit():
            basenames.add(basename)
    return sorted(basenames)


def discover_prefixes_for_basename(exp_dir, basename):
    """Returns a sorted list of int prefixes that exist on disk for the given basename."""
    prefixes = []
    for entry in os.listdir(exp_dir):
        prefix = entry.rpartition("-")[0], entry.rpartition("-")[2]
        entry_basename, entry_prefix = prefix
        if entry_basename == basename and entry_prefix.isdigit():
            prefixes.append(int(entry_prefix))
    return sorted(prefixes)


def resolve_run_names(exp_dir, basename, chosen_prefixes):
    """
    Matches chosen integer prefixes against actual directory names for this
    basename, regardless of zero-padding width (e.g. chosen prefix 8 matches
    directory 'stgcn_bimamba-08').
    """
    run_names = []
    for entry in sorted(os.listdir(exp_dir)):
        entry_basename, sep, entry_prefix = entry.rpartition("-")
        if not sep or entry_basename != basename or not entry_prefix.isdigit():
            continue
        if int(entry_prefix) in chosen_prefixes:
            run_names.append(entry)
    return run_names


def interactive_select(exp_dir):
    basenames = discover_basenames(exp_dir)
    if not basenames:
        print(f"❌ No runs found under '{exp_dir}'.")
        return []

    print("Available models:")
    for i, name in enumerate(basenames, 1):
        print(f"  [{i}] {name}")

    while True:
        choice = input("\nChoose a model (number or name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(basenames):
            basename = basenames[int(choice) - 1]
            break
        elif choice in basenames:
            basename = choice
            break
        else:
            print("⚠️ Invalid choice, try again.")

    available_prefixes = discover_prefixes_for_basename(exp_dir, basename)
    print(f"\nAvailable prefixes for '{basename}': {', '.join(str(p) for p in available_prefixes)}")

    while True:
        spec = input("Enter prefixes (e.g. '3-6, 8, 10, 14-19, 21'): ").strip()
        try:
            chosen = parse_prefix_ranges(spec)
            break
        except ValueError:
            print("⚠️ Could not parse that. Use ranges like '3-6' and/or individual numbers separated by commas.")

    run_names = resolve_run_names(exp_dir, basename, chosen)

    missing = chosen - {int(r.rpartition('-')[2]) for r in run_names}
    if missing:
        print(f"⚠️ No experiment folder found for prefixes: {sorted(missing)}")

    return run_names


def parse_time_to_seconds(time_str):
    """Parses strings like '3m 27s' or '1h 12m 3s' into total seconds."""
    if not time_str:
        return 0
    pattern = r"(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?"
    match = re.match(pattern, time_str.strip())
    h, m, s = (int(g) if g else 0 for g in match.groups())
    return h * 3600 + m * 60 + s


def seconds_to_time_str(total_seconds):
    total_seconds = int(round(total_seconds))
    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def load_run(exp_dir, run_name):
    run_path = os.path.join(exp_dir, run_name)

    hp_path = os.path.join(run_path, "hyperparameters.json")
    hw_path = os.path.join(run_path, "hardware_summary.json")
    metrics_path = os.path.join(run_path, "training_metrics.csv")

    if not (os.path.exists(hp_path) and os.path.exists(hw_path) and os.path.exists(metrics_path)):
        return None

    with open(hp_path) as f:
        hyperparams = json.load(f)
    with open(hw_path) as f:
        hardware = json.load(f)

    best_row = None
    best_combined = -1.0
    with open(metrics_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combined = float(row["frame_f1"]) + float(row["mean_iou"]) + float(row["segment_f1"])
            if combined > best_combined:
                best_combined = combined
                best_row = row

    if best_row is None:
        return None

    return {
        "run_name": run_name,
        "basename": hyperparams.get("basename", "unknown"),
        "description": hyperparams.get("description", "(no description)"),
        "hyperparams_key": {k: v for k, v in hyperparams.items() if k not in ("prefix", "description")},
        "f1_score": float(best_row["frame_f1"]),
        "best_epoch": int(best_row["epoch"]),
        "total_train_seconds": hardware.get("total_training_seconds", 0),
        "avg_epoch_time_str": hardware.get("average_time_per_epoch", "0m 0s"),
        "gpu_memory_gb": hardware.get("max_gpu_memory_used_gb", 0.0),
        "gpu_util_pct": hardware.get("average_gpu_utilization_percent", 0.0),
    }


def group_runs(runs):
    """Groups runs sharing identical hyperparameters (minus prefix/description)."""
    groups = defaultdict(list)
    for run in runs:
        key = (run["basename"], json.dumps(run["hyperparams_key"], sort_keys=True))
        groups[key].append(run)
    return groups


def format_group(basename, description, runs):
    f1_scores = ", ".join(f"{r['f1_score']:.3f}" for r in runs)
    best_epochs = ", ".join(str(r["best_epoch"]) for r in runs)

    train_times_h = [r["total_train_seconds"] / 3600 for r in runs]
    train_times_str = ", ".join(f"{t:.2f}h" for t in train_times_h)

    avg_epoch_seconds = [parse_time_to_seconds(r["avg_epoch_time_str"]) for r in runs]
    mean_epoch_time = seconds_to_time_str(sum(avg_epoch_seconds) / len(avg_epoch_seconds))

    mean_gpu_mem = sum(r["gpu_memory_gb"] for r in runs) / len(runs)
    mean_gpu_util = sum(r["gpu_util_pct"] for r in runs) / len(runs)

    lines = [
        f"## {description}",
        f"F1 Score: {f1_scores}",
        f"Best epoch number: {best_epochs}",
        f"Total train time: {train_times_str}",
        f"Time per epoch: {mean_epoch_time}",
        f"GPU Memory: {mean_gpu_mem:.3f} GB",
        f"GPU Utilization percentage: {mean_gpu_util:.1f}%",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate training run results into a report.")
    parser.add_argument("run_names", nargs="*", help="Specific run names (e.g. stgcn_bimamba-37). Default: interactive selection.")
    parser.add_argument("--exp-dir", default="experiments", help="Path to experiments directory (default: experiments)")
    parser.add_argument("--basename", default=None, help="Model name to select (skips the interactive model prompt)")
    parser.add_argument("--prefixes", default=None, help="Prefix spec, e.g. '3-6, 8, 10, 14-19, 21' (skips the interactive prefix prompt; requires --basename)")
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"❌ Experiments directory not found: {args.exp_dir}")
        return

    if args.run_names:
        run_names = args.run_names
    elif args.basename and args.prefixes:
        chosen = parse_prefix_ranges(args.prefixes)
        run_names = resolve_run_names(args.exp_dir, args.basename, chosen)
        missing = chosen - {int(r.rpartition('-')[2]) for r in run_names}
        if missing:
            print(f"⚠️ No experiment folder found for prefixes: {sorted(missing)}")
    elif args.basename:
        available_prefixes = discover_prefixes_for_basename(args.exp_dir, args.basename)
        print(f"Available prefixes for '{args.basename}': {', '.join(str(p) for p in available_prefixes)}")
        spec = input("Enter prefixes (e.g. '3-6, 8, 10, 14-19, 21'): ").strip()
        chosen = parse_prefix_ranges(spec)
        run_names = resolve_run_names(args.exp_dir, args.basename, chosen)
    else:
        run_names = interactive_select(args.exp_dir)

    if not run_names:
        print("No runs selected.")
        return

    runs = []
    for run_name in run_names:
        run = load_run(args.exp_dir, run_name)
        if run is None:
            print(f"⚠️  Skipping '{run_name}' (missing/incomplete result files)")
            continue
        if args.basename and run["basename"] != args.basename:
            continue
        runs.append(run)

    if not runs:
        print("No valid runs found.")
        return

    groups = group_runs(runs)

    # Organize by basename -> list of (description, runs)
    by_basename = defaultdict(list)
    for (basename, _), group_runs_list in groups.items():
        description = group_runs_list[0]["description"]
        by_basename[basename].append((description, group_runs_list))

    output_lines = []
    for basename in sorted(by_basename.keys()):
        output_lines.append(f"# {basename}:\n")
        for description, group_runs_list in by_basename[basename]:
            output_lines.append(format_group(basename, description, group_runs_list))
        output_lines.append("")

    report = "\n".join(output_lines)
    print(report)


if __name__ == "__main__":
    main()