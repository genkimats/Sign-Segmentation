import shutil
from pathlib import Path


def flatten_filename(src_file, src_root):
    """Return a filename that includes ancestor directory names as prefixes."""
    rel_parts = src_file.relative_to(src_root).parts
    if len(rel_parts) <= 1:
        return src_file.name
    return "_".join(rel_parts[:-1] + (src_file.name,))


def copy_experiments_filtered(src_dir, dst_dir):
    """Copy experiment files into a flat destination folder."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    dst_path.mkdir(parents=True, exist_ok=True)

    for src_file in src_path.rglob("*"):
        if not src_file.is_file():
            continue

        if src_file.suffix.lower() == ".png" or src_file.name == "final_decoded_metrics.json":
            print(f"Skipping: {src_file}")
            continue

        flat_name = flatten_filename(src_file, src_path)
        dst_file = dst_path / flat_name
        shutil.copy2(src_file, dst_file)
        print(f"Copied: {src_file} -> {dst_file}")


if __name__ == "__main__":
    src = "/home/genki/GR/Sign-Segmentation/experiments"
    dst = "/home/genki/GR/Sign-Segmentation/experiments_filtered"

    print(f"Copying {src} to {dst}...")
    print("Excluding: .png files and final_decoded_metrics.json\n")

    copy_experiments_filtered(src, dst)

    print("\n✓ Copy complete!")
