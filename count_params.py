"""
count_params.py -- number of parameters of the three compared models, read from their saved checkpoints.

    python count_params.py
    python count_params.py --runs multistream_bilstm-115 bilstm_2023-302 bilstm_2023-244
    python count_params.py --ckpt path/to/model.pt          # a checkpoint file directly

Defaults (seed 42 of each model; all seeds of one model have the same size):
  multistream_bilstm-115   feature-specific encoders (proposed)
  bilstm_2023-302          early fusion, same inputs (per-joint temporal conv)
  bilstm_2023-244          early fusion, Moryossef et al. (2023) re-implementation

How it counts: it loads each checkpoint on the CPU (no model code, no GPU), takes its state dict and sums the
size of every tensor that is a learned weight. Buffers that are saved but not trained are listed apart:
BatchNorm running statistics (running_mean / running_var / num_batches_tracked) and fixed graph adjacency
matrices (names ending in A / adj / adjacency). A learnable edge-importance mask, if your ST-GCN has one,
counts as a weight. The breakdown by first name part shows where the parameters sit (e.g. the body/face
streams, the HaMeR MLP, the fusion layer, the LSTM, the classifier).

Finding the checkpoint: for each run it searches the usual folders (checkpoints/, runs/, models/, outputs/,
experiments/, results/, evaluation_phrase/ and the repo root) for a folder or file whose name contains the run
name and ends in .pt / .pth / .ckpt. If several match it prefers names with "best", then the newest. If none
is found, pass --ckpt with the path.
"""
import argparse
import glob
import os
import re
import sys

import torch

DEFAULT_RUNS = {
    "multistream_bilstm-115": "Feature-specific encoders (proposed)",
    "bilstm_2023-302": "Early fusion, same inputs (temporal conv)",
    "bilstm_2023-244": "Early fusion, Moryossef et al. (2023)",
}
SEARCH_DIRS = ["checkpoints", "runs", "models", "outputs", "experiments", "results", "evaluation_phrase", "."]
EXTS = (".pt", ".pth", ".ckpt")
BUFFER_SUFFIX = re.compile(r"(running_mean|running_var|num_batches_tracked)$")
GRAPH_BUFFER = re.compile(r"(^|\.)(A|adj|adjacency|graph_A)$")


def find_checkpoint(run):
    hits = []
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for ext in EXTS:
            # file named after the run, or any checkpoint inside a folder named after the run
            hits += glob.glob(os.path.join(d, "**", f"*{run}*{ext}"), recursive=True)
            for folder in glob.glob(os.path.join(d, "**", f"*{run}*"), recursive=True):
                if os.path.isdir(folder):
                    hits += glob.glob(os.path.join(folder, "**", f"*{ext}"), recursive=True)
    # exact run name only (e.g. -115 must not match -1150)
    pat = re.compile(re.escape(run) + r"(?!\d)")
    hits = sorted({h for h in hits if pat.search(h) and "processed_data" not in h})
    if not hits:
        return None, []
    best = sorted(hits, key=lambda h: ("best" not in os.path.basename(h).lower(), -os.path.getmtime(h)))
    return best[0], hits


def state_dict_of(obj):
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model", "net", "module"):
            if k in obj and isinstance(obj[k], (dict, torch.nn.Module)):
                return state_dict_of(obj[k])
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError("no state dict found in the checkpoint (keys: "
                     + (", ".join(list(obj)[:10]) if isinstance(obj, dict) else type(obj).__name__) + ")")


def count(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = state_dict_of(obj)
    weights, buffers, by_part = 0, 0, {}
    for name, t in sd.items():
        if not torch.is_tensor(t):
            continue
        name = name.replace("module.", "", 1) if name.startswith("module.") else name
        if BUFFER_SUFFIX.search(name) or GRAPH_BUFFER.search(name):
            buffers += t.numel()
            continue
        weights += t.numel()
        part = name.split(".")[0]
        by_part[part] = by_part.get(part, 0) + t.numel()
    return weights, buffers, by_part


def fmt(n):
    return f"{n:,} ({n / 1e6:.2f}M)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--ckpt", nargs="+", default=None, help="checkpoint file(s) to count directly")
    args = ap.parse_args()

    items = [(p, p) for p in args.ckpt] if args.ckpt else []
    if not args.ckpt:
        for run in args.runs:
            path, all_hits = find_checkpoint(run)
            if path is None:
                print(f"{run}: no checkpoint found -- pass it with --ckpt <path>")
                continue
            if len(all_hits) > 1:
                print(f"{run}: {len(all_hits)} checkpoints match, using {path}")
            items.append((run, path))

    rows = []
    for label, path in items:
        try:
            w, b, parts = count(path)
        except Exception as e:                       # a bad file should not stop the others
            print(f"{label}: could not read {path}: {e}")
            continue
        rows.append((label, path, w, b, parts))

    if not rows:
        return 1
    print("\nParameters (learned weights; saved buffers listed apart)\n")
    for label, path, w, b, parts in rows:
        print(f"{DEFAULT_RUNS.get(label, label)}  [{label}]")
        print(f"  file:     {path}")
        print(f"  weights:  {fmt(w)}")
        if b:
            print(f"  buffers:  {b:,} (BatchNorm statistics / fixed graph; not trained)")
        for part, n in sorted(parts.items(), key=lambda kv: -kv[1]):
            print(f"    {part:<28s} {n:>12,}  ({100 * n / w:5.1f}%)")
        print()
    if len(rows) > 1:
        ref = rows[0][2]
        print("Relative to the first model:")
        for label, _, w, _, _ in rows:
            print(f"  {DEFAULT_RUNS.get(label, label):<45s} {w / ref:5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())