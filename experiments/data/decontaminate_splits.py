#!/usr/bin/env python3
"""
decontaminate_splits.py — Remove test-set sentences from baseline train/val splits.

Problem: The smart-selection baselines (TF-IDF, Length-Stratified, Vocab-Maximized)
were selected from the full 10K Tatoeba pool, which also sourced the test set.
This caused ~12-28% of test sentences to leak into training data.

Fix: Remove any sentence whose source text appears in test.tsv from both
train.tsv and val.tsv. Write cleaned files to a new directory.

Usage:
    python experiments/data/decontaminate_splits.py [--dry-run] [--in-place]

    --dry-run:  Print what would be removed without writing files
    --in-place: Overwrite original files (default: write to _clean/ suffix dirs)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR = REPO_ROOT / "experiments" / "data" / "splits"
TEST_PATH = SPLITS_DIR / "shared" / "test.tsv"

# All baseline conditions that may be contaminated
BASELINE_DIRS = {
    # Standalone baselines (already in paper Table 9)
    "baselines/TF-IDF-DIVERSE": SPLITS_DIR / "baselines" / "TF-IDF-DIVERSE",
    "baselines/LENGTH-STRATIFIED": SPLITS_DIR / "baselines" / "LENGTH-STRATIFIED",
    "baselines/VOCAB-MAXIMIZED": SPLITS_DIR / "baselines" / "VOCAB-MAXIMIZED",
    # Combo conditions (new for Table 9)
    "combos/STRUCT4K-TFIDF2K": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-TFIDF2K",
    "combos/STRUCT4K-LENGTH2K": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-LENGTH2K",
    "combos/STRUCT4K-VOCAB2K": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-VOCAB2K",
    "combos/STRUCT4K-ALL-BASELINES": SPLITS_DIR / "new_experiments" / "baselines" / "STRUCT4K-ALL-BASELINES",
}


def load_test_sources():
    """Load the set of source-side sentences in the test set."""
    test = pd.read_csv(TEST_PATH, sep="\t", header=None, names=["src_text", "tgt_text"])
    return set(test["src_text"].str.strip())


def decontaminate_file(path, test_src, dry_run=False):
    """Remove rows whose source text is in the test set. Returns (before, after, removed_count)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["src_text", "tgt_text"])
    before = len(df)

    mask = df["src_text"].str.strip().isin(test_src)
    removed = mask.sum()
    clean = df[~mask]

    if not dry_run:
        clean.to_csv(path, sep="\t", header=False, index=False)

    return before, len(clean), removed


def main():
    parser = argparse.ArgumentParser(description="Remove test sentences from baseline splits")
    parser.add_argument("--dry-run", action="store_true", help="Print report without modifying files")
    parser.add_argument("--in-place", action="store_true",
                        help="Modify original files (default: copy to *_clean/ dirs first)")
    args = parser.parse_args()

    if not TEST_PATH.exists():
        print(f"ERROR: Test file not found at {TEST_PATH}")
        sys.exit(1)

    test_src = load_test_sources()
    print(f"Test set: {len(test_src)} unique source sentences")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    total_removed = 0
    header = f"{'Condition':<35} {'File':<8} {'Before':>7} {'Removed':>8} {'After':>7}"
    print(header)
    print("-" * len(header))

    for label, src_dir in BASELINE_DIRS.items():
        if not src_dir.exists():
            print(f"{label:<35} SKIP (directory not found)")
            continue

        # Determine output directory
        if args.in_place or args.dry_run:
            work_dir = src_dir
        else:
            work_dir = src_dir.parent / (src_dir.name + "_clean")
            if not args.dry_run:
                if work_dir.exists():
                    shutil.rmtree(work_dir)
                shutil.copytree(src_dir, work_dir)

        for fname in ["train.tsv", "val.tsv"]:
            fpath = work_dir / fname
            if not fpath.exists():
                continue
            before, after, removed = decontaminate_file(fpath, test_src, dry_run=args.dry_run)
            total_removed += removed
            marker = " ***" if removed > 0 else ""
            print(f"{label:<35} {fname:<8} {before:>7} {removed:>8} {after:>7}{marker}")

    print()
    print(f"Total sentences removed: {total_removed}")
    if not args.dry_run and not args.in_place:
        print(f"\nCleaned files written to *_clean/ directories alongside originals.")
        print("To use them, either rename or update your training scripts to point there.")
    elif args.dry_run:
        print("\nDry run complete. No files were modified.")
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
