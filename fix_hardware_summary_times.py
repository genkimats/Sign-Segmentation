import os
import csv
import json

EXPERIMENTS_DIR = "experiments"

# Defaults to a dry run (prints what WOULD change, writes nothing).
# Review the printed diffs, then flip to False to actually apply them.
DRY_RUN = False


def compute_corrected_time(csv_path):
    """
    Reads training_metrics.csv's per-epoch 'epoch_time' column and recomputes
    the total training time as average_epoch_time * total_epochs_ran (which is
    mathematically just sum(epoch_time) -- the two are equivalent, matching how
    the fix was specified).

    This column was NEVER affected by the total_training_time bug: each row's
    epoch_time is a fresh local timer (epoch_end_time - epoch_start_time) taken
    once per epoch, and the CSV itself is truncated (mode='w') at the start of
    every restart attempt in train.py, so it already only ever contains the
    latest run's rows -- unlike the old total_training_time, which accumulated
    across every restart attempt before the fix.
    """
    epoch_times = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch_times.append(float(row["epoch_time"]))
            except (KeyError, ValueError, TypeError):
                continue

    if not epoch_times:
        return None

    total_epochs_ran = len(epoch_times)
    average_epoch_time = sum(epoch_times) / total_epochs_ran
    total_training_seconds = average_epoch_time * total_epochs_ran  # == sum(epoch_times)

    return total_epochs_ran, average_epoch_time, total_training_seconds


def fix_experiment(exp_dir):
    csv_path = os.path.join(exp_dir, "training_metrics.csv")
    summary_path = os.path.join(exp_dir, "hardware_summary.json")

    if not os.path.exists(csv_path) or not os.path.exists(summary_path):
        return None

    computed = compute_corrected_time(csv_path)
    if computed is None:
        return None
    total_epochs_ran, average_epoch_time, total_training_seconds = computed

    with open(summary_path, "r") as f:
        summary = json.load(f)

    old_total_seconds = summary.get("total_training_seconds")
    old_epochs = summary.get("total_epochs_ran")

    mismatch = None
    if old_epochs is not None and old_epochs != total_epochs_ran:
        mismatch = (f"total_epochs_ran mismatch: hardware_summary.json says {old_epochs}, "
                    f"but training_metrics.csv has {total_epochs_ran} rows. Using the CSV's "
                    f"count for the recalculation below; total_epochs_ran itself is left "
                    f"untouched (only the time fields are corrected here) -- worth a manual "
                    f"look if this experiment matters.")

    total_minutes, total_seconds_rem = divmod(int(round(total_training_seconds)), 60)
    avg_minutes, avg_seconds_rem = divmod(int(round(average_epoch_time)), 60)

    new_summary = dict(summary)  # only the three time fields below are touched
    new_summary["total_training_time"] = f"{int(total_minutes)}m {int(total_seconds_rem)}s"
    new_summary["total_training_seconds"] = round(total_training_seconds, 2)
    new_summary["average_time_per_epoch"] = f"{int(avg_minutes)}m {int(avg_seconds_rem)}s"
    # total_training_restarts and everything else (GPU memory/utilization, NaN count,
    # total_epochs_ran) are left exactly as they were -- those were already correctly
    # scoped to the latest run in train.py, this script only touches the time fields.

    if not DRY_RUN:
        with open(summary_path, "w") as f:
            json.dump(new_summary, f, indent=4)

    return {
        "exp_dir": exp_dir,
        "old_total_training_seconds": old_total_seconds,
        "new_total_training_seconds": round(total_training_seconds, 2),
        "total_epochs_ran": total_epochs_ran,
        "mismatch": mismatch,
    }


def main():
    if DRY_RUN:
        print("=== DRY RUN MODE -- nothing will actually be written. Set DRY_RUN = False to apply. ===\n")

    if not os.path.exists(EXPERIMENTS_DIR):
        print(f"Directory not found: {EXPERIMENTS_DIR}")
        return

    results = []
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        exp_dir = os.path.join(EXPERIMENTS_DIR, name)
        if not os.path.isdir(exp_dir):
            continue
        result = fix_experiment(exp_dir)
        if result:
            results.append(result)

    verb = "Would fix" if DRY_RUN else "Fixed"
    print(f"{verb} {len(results)} experiment(s):\n")

    for r in results:
        change = ""
        if r["old_total_training_seconds"] is not None:
            delta = r["new_total_training_seconds"] - r["old_total_training_seconds"]
            change = f" (was {r['old_total_training_seconds']}s, delta {delta:+.2f}s)"
        print(f"  {r['exp_dir']}: total_training_seconds -> {r['new_total_training_seconds']}s{change}")
        if r["mismatch"]:
            print(f"    ⚠️ {r['mismatch']}")


if __name__ == "__main__":
    main()