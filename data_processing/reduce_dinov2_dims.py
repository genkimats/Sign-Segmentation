"""
Reduces DINOv2 feature dimensionality via PCA -- a targeted, SMALL-scale
storage reduction. Unlike temporal downsampling (which cuts storage in a big,
coarse step by discarding whole frames), PCA trims redundancy WITHIN each
frame's descriptor, keeping every frame but making each one's representation
a bit more compact. Precisely tunable to a specific target size, rather than
only offering 2x/4x-style cuts.

Fits PCA on a random SUBSAMPLE of frame embeddings (not the full dataset --
a few hundred thousand samples is already far more than needed to estimate
principal components reliably for a 384-dim space, and keeps the fitting
step itself memory-safe), then applies that same fixed projection to EVERY
video's full feature set, one video at a time, streaming through files
rather than loading everything at once.

Uses torch.pca_lowrank (randomized SVD) -- no extra dependency beyond torch
itself.

Run this AFTER extract_dinov2_features.py has already produced the full
384-dim feature files. This does NOT re-run DINOv2 inference at all; it's a
pure post-processing step over already-extracted embeddings, so it's cheap
and fast by comparison. Writes to a SEPARATE output directory, so your
original 384-dim extraction is left untouched (in case you want to compare,
or revert).
"""
import os
import glob
import torch
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_DIR = "processed_data/dinov2_features"           # the already-extracted 384-dim features
OUTPUT_DIR = "processed_data/dinov2_features_reduced"   # PCA-reduced output goes here

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Target reduced dimension. 384 -> 340 saves ~6.2% of storage (~1.5GB off a 25GB
# extraction). Adjust to trade off more/less savings against more/less retained
# variance -- the script reports actual variance retained at the end so you can
# judge whether this target is a good tradeoff for your data before committing
# to it in a real experiment.
TARGET_DIM = 300

# How many frame-embeddings to sample (across all videos, both hands) to FIT
# the PCA. This is NOT how much data gets reduced -- every frame of every video
# still gets transformed at the end regardless of this number. This is purely
# for ESTIMATING the principal components, and a few hundred thousand samples
# is already far more than needed for a 384-dim space -- keeping this bounded
# keeps the FITTING step memory-safe too (the whole point of this exercise).
FIT_SAMPLE_SIZE = 300_000


def collect_fit_sample():
    """Randomly samples FIT_SAMPLE_SIZE frame-embeddings across all videos to fit
    the PCA on, without ever loading every video's full features into memory
    simultaneously -- one file at a time, sampling a small slice from each."""
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_dinov2.pt")))
    if not all_files:
        raise RuntimeError(f"No *_dinov2.pt files found in {INPUT_DIR}")

    samples_per_file = max(1, FIT_SAMPLE_SIZE // len(all_files))
    collected = []

    for path in tqdm(all_files, desc="Sampling for PCA fit"):
        data = torch.load(path, weights_only=False)
        feats = data["features"].float()  # (T, 2, D)
        flat = feats.reshape(-1, feats.shape[-1])  # (T*2, D)

        # Skip all-zero rows (hand not detected in that frame) -- these aren't
        # real appearance signal and would bias the PCA toward the origin.
        nonzero_mask = flat.abs().sum(dim=1) > 0
        flat = flat[nonzero_mask]
        if flat.shape[0] == 0:
            continue

        n = min(samples_per_file, flat.shape[0])
        idx = torch.randperm(flat.shape[0])[:n]
        collected.append(flat[idx])

    return torch.cat(collected, dim=0)  # (~FIT_SAMPLE_SIZE, D)


def fit_pca(sample):
    """Fits PCA via randomized SVD. Returns the mean (for centering), the
    projection matrix (D, TARGET_DIM), and the fraction of variance retained
    at TARGET_DIM."""
    mean = sample.mean(dim=0, keepdim=True)
    centered = sample - mean

    q = min(TARGET_DIM + 20, sample.shape[1])  # a little slack improves randomized-SVD accuracy
    U, S, V = torch.pca_lowrank(centered, q=q)
    projection = V[:, :TARGET_DIM]  # (D, TARGET_DIM)

    total_variance = (S ** 2).sum()
    retained_variance = (S[:TARGET_DIM] ** 2).sum()
    variance_ratio = (retained_variance / total_variance).item()

    return mean, projection, variance_ratio


def main():
    print(f"Fitting PCA to reduce {INPUT_DIR} -> {TARGET_DIM} dims...")
    sample = collect_fit_sample()
    print(f"Collected {sample.shape[0]} sample embeddings (dim={sample.shape[1]}) for fitting.")

    mean, projection, variance_ratio = fit_pca(sample)
    print(f"PCA fit complete: retaining {variance_ratio*100:.2f}% of variance at {TARGET_DIM} dims.")
    if variance_ratio < 0.95:
        print("Retained variance is below 95% -- consider a higher TARGET_DIM if this "
              "seems like too much information loss for your use case.")

    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_dinov2.pt")))
    existing_ids = {
        f[:-len("_dinov2.pt")] for f in os.listdir(OUTPUT_DIR) if f.endswith("_dinov2.pt")
    }

    for path in tqdm(all_files, desc="Applying PCA reduction"):
        fname = os.path.basename(path)
        vid = fname[:-len("_dinov2.pt")]
        if vid in existing_ids:
            continue

        data = torch.load(path, weights_only=False)
        feats = data["features"].float()  # (T, 2, D)
        T, num_hands, D = feats.shape

        flat = feats.reshape(-1, D)
        nonzero_mask = flat.abs().sum(dim=1) > 0

        # Preserve the "zero vector = hand not detected" convention: a zero row
        # run through (0 - mean) @ projection would come out NON-zero (since the
        # dataset mean generally isn't zero), which would inject a fake signal
        # where there should be none. Transform only the real detections; leave
        # everything else as zero.
        reduced_flat = torch.zeros(flat.shape[0], TARGET_DIM, dtype=torch.float32)
        if nonzero_mask.any():
            reduced_flat[nonzero_mask] = (flat[nonzero_mask] - mean) @ projection

        reduced = reduced_flat.reshape(T, num_hands, TARGET_DIM).half()

        data["features"] = reduced
        data["pca_variance_retained"] = variance_ratio
        data["original_dim"] = D

        torch.save(data, os.path.join(OUTPUT_DIR, fname))

    print(f"\nDone. Reduced features written to {OUTPUT_DIR}")
    print(f"To use these: point dinov2_dir at {OUTPUT_DIR!r} and set "
          f"\"dinov2_dim\": {2 * TARGET_DIM} explicitly in your experiment config "
          f"(2 x {TARGET_DIM}, since train.py's default assumes the un-reduced dimension).")


if __name__ == "__main__":
    main()