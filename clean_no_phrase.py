"""Remove entries from dataset_splits.json by filename (without extension).

Usage:
    python clean_no_phrase.py missing_phrase.txt            # list file, one name per line
    python clean_no_phrase.py -n 2025500_A 1289462_B        # names given directly
    python clean_no_phrase.py -n "'1413451-11171532-11201836_B', '1584329-15450503-15475829_B'"
    (names may be separated by commas/spaces/newlines and wrapped in quotes)
    python clean_no_phrase.py missing_phrase.txt --dry-run  # preview only
"""
import argparse
import json
import os

SPLITS = ["train", "val", "test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("list_file", nargs="?", help="text file with one filename (no extension) per line")
    parser.add_argument("-n", "--names", nargs="*", default=[], help="filenames (no extension) given directly")
    parser.add_argument("--splits", default="dataset_splits.json")
    parser.add_argument("--dry-run", action="store_true", help="show what would be removed without writing")
    args = parser.parse_args()

    def split_names(text):
        # accept commas/whitespace as separators and strip quotes/brackets,
        # e.g. '1413451-11171532-11201836_B', '1584329-15450503-15475829_B'
        return [t for t in (t.strip("'\"[]()") for t in text.replace(",", " ").split()) if t]

    targets = set()
    for name in args.names:
        targets.update(split_names(name))
    if args.list_file:
        with open(args.list_file) as f:
            targets.update(split_names(f.read()))
    targets = {os.path.splitext(t)[0] for t in targets}  # tolerate accidental extensions
    if not targets:
        parser.error("no filenames given")

    with open(args.splits) as f:
        data = json.load(f)

    removed = set()
    for split in SPLITS:
        if split not in data:
            continue
        kept = []
        for entry in data[split]:
            stem = os.path.splitext(os.path.basename(entry))[0]
            if stem in targets:
                removed.add(stem)
                print(f"[{split}] removed {entry}")
            else:
                kept.append(entry)
        data[split] = kept

    not_found = sorted(targets - removed)
    if not_found:
        print(f"Not found ({len(not_found)}): {', '.join(not_found)}")
    print(f"Removed {len(removed)} of {len(targets)} requested names.")
    print({s: len(data[s]) for s in SPLITS if s in data})

    if args.dry_run:
        print("Dry run: file not modified.")
        return
    with open(args.splits, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {args.splits}")


if __name__ == "__main__":
    main()
